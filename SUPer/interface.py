#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 cibo
This file is part of SUPer <https://github.com/cubicibo/SUPer>.

SUPer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SUPer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SUPer.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import signal
import multiprocessing as mp

from itertools import chain
from pathlib import Path
from typing import Any, NoReturn, Self

from .bdnxml import BDNXML
from .internals import TC, LogFacility

from .bytestream.pgstreams import Epoch, PesMuiWriter, SUPWriter
from .bytestream.verifier import is_compliant, check_pts_dts_sanity, test_rx_bitrate, debug_stats
from .display.bdvideo import BDVideo
from .encoder.codecctx import PGStreamCtx
from .encoder.engine import EpochEncoderEngine
from .encoder.epochctx import EpochFinder, EventsPreprocessor, EpochData, LayoutMode
from .encoder.imgproc import BuiltinQuantizer, SSIMPW

logger = LogFacility.get_logger('SUPer')

#%%
class BDNEncoder:
    def __init__(self, bdn: BDNXML, kwargs: dict[str, Any]) -> None:
        self.bdn = bdn if isinstance(bdn, BDNXML) else BDNXML(bdn)
        self.kwargs = kwargs
        self._threads = kwargs.get('threads', 1)
        self._allow_pes_output = True

    def _prepare(self) -> BDVideo:
        log_filename = self.kwargs.get('log_filename', None)
        file_logging_level = self.kwargs.get('log_to_file', False)
        if file_logging_level > 0 and log_filename:
            logfile = str(log_filename) + ".txt"
            LogFacility.set_file_log(logger, logfile, file_logging_level)
            LogFacility.set_logger_level(logger.name, file_logging_level)

        str_kwa = ':'.join([f"{k}={v}" for k, v in self.kwargs.items() if not isinstance(v, dict)])

        if self.kwargs.get('quantize_lib', None) is None:
            self.kwargs['quantize_lib'] = BuiltinQuantizer.Qtzr
        else:
            for v, n, qtz in BuiltinQuantizer.get_all():
                if self.kwargs['quantize_lib'] == qtz or v == self.kwargs['quantize_lib']:
                    self.kwargs['quantize_lib'] = qtz
            assert not isinstance(self.kwargs['quantize_lib'], int)

        if self.kwargs.get('log_to_file', False) < 0:
            LogFacility.add_file_report(logger, str(self.outfile) + ".txt")

        logger.iinfo(f"Parameters: {str_kwa}")

        bdvideo = BDVideo(self.bdn.description.fmt,
                          self.bdn.description.fps,
                          self.kwargs.get('uhd_bd', False),
                          self.kwargs.get('matrix', None))

        if len(self.bdn.events) == 0:
            raise RuntimeError("No BDN event found, exiting.")

        fps_fmt = int(bdvideo.fps) if float(bdvideo.fps).is_integer() else round(float(bdvideo.fps), 3)
        logger.iinfo(f"BDN metadata: {'x'.join(map(str, bdvideo.fmt.value))}, FPS={fps_fmt}, DF={self.bdn.description.drop_frame}, {len(self.bdn.events)} valid events.")

        LogFacility.dissociate_log_and_report(logger)

        self.kwargs['adjust_ntsc'] = isinstance(fps_fmt, float)
        if self.kwargs['adjust_ntsc']:
            if self.bdn.description.drop_frame:
                logger.info("DF BDN detected: converting Timecodes to NDF and scaling all timestamps by 1.001.")
                #conversion was done internally in BDNXML already.
            else:
                logger.info("NDF NTSC detected: scaling all timestamps by 1.001.")

        self._adjust_thread_count()
        return bdvideo
    ####

    def _adjust_thread_count(self) -> None:
        n_threads = self._threads
        if (n_threads_auto := isinstance(n_threads, str)): # auto
            try:
                import psutil
            except (ModuleNotFoundError, NameError):
                #commonplace: logical = 2*physical cores
                n_threads = max(1, mp.cpu_count() >> 1)
            else:
                n_threads = psutil.cpu_count(logical=False)

        if n_threads_auto:
            logger.info(f"Using {n_threads} thread(s).")
        self._threads = n_threads
    ####

    def _convert_single(self, bdvideo: BDVideo) -> list[Epoch]:
        logger.info("Finding all epochs and their screen layout:")
        epochs_ctx = EpochFinder(bdn=self.bdn, threads=1,
                                 mode=self.kwargs.get('layout_mode', LayoutMode.GREEDY)).get_epochs()
        logger.info(f"Identified {len(epochs_ctx)} epochs to encode.")

        pg_stream_ctx = PGStreamCtx(bdvideo)
        output_pg_epochs = []
        for ke, e_ctx in enumerate(epochs_ctx):
            ee = EpochEncode(pg_stream_ctx, e_ctx, self.kwargs)
            logger.info(f"Encoding epoch {ke}: {e_ctx.events[0].inTC}->{e_ctx.events[-1].outTC} with {len(e_ctx.events)} event(s), {len(e_ctx.windows)} window(s).")
            output_pg_epochs.append(ee.preprocess().encode())
            logger.info(f"=> Encoded epoch {ke} as {len(output_pg_epochs[-1])} display sets.")
        return output_pg_epochs
    ####

    def _setup_mt_env(self) -> None:
        LogFacility.disable_tqdm()
        def sighandler(workers, snum, frame) -> NoReturn:
            for worker in workers:
                try:
                    if worker.is_alive():
                        worker.kill()
                except ValueError:
                    pass
            import sys, time
            time.sleep(0.005)
            for worker in workers:
                try:
                    worker.join()
                except (ValueError, RuntimeError, AssertionError):
                    pass
            sys.exit(1)

        f_term = lambda signal_num, frame: sighandler(self._workers, signal_num, frame)
        signal.signal(signal.SIGINT, f_term)
        signal.signal(signal.SIGTERM, f_term)
        if os.name == 'nt':
            signal.signal(signal.SIGBREAK, f_term)
        logger.debug("Registered signal handlers.")
    ####

    def _convert_mt(self, bd_video: BDVideo) -> list[Epoch]:
        import time
        BDNEpochWorker.reset_module()

        epochs_ctx = EpochFinder(bdn=self.bdn, threads=self._threads, mode=self.kwargs.get('layout_mode', LayoutMode.GREEDY)).get_epochs()

        # No point in having more workers than epochs
        n_threads = min(self._threads, len(epochs_ctx))
        as_deamon = self.kwargs.get('daemonize', True)
        workers = [BDNEpochWorker(bd_video, self.kwargs, as_deamon) for _ in range(n_threads)]

        self._setup_mt_env()

        logger.debug("Starting workers...")
        for worker in workers:
            worker.start()

        while not all(map(lambda worker: worker.is_available(), workers)):
            time.sleep(0.2)
        ###

        #Orchestrator starts here
        busy_flags = {worker.iid: False for worker in workers}
        g_epochs = enumerate(chain(epochs_ctx, (None,)))
        ep_timeline = []

        group_data = True
        healthy = True
        while group_data is not None and healthy:
            time.sleep(0.05)
            for free_worker in filter(lambda worker: worker.is_available(), workers):
                if (epoch_data := free_worker.get()) is not None:
                    ep_timeline[epoch_data[1]] = epoch_data[0]
                    busy_flags[free_worker.iid] = False
                if busy_flags[free_worker.iid] is False:
                    group_id, group_data = next(g_epochs)
                    if group_data is not None:
                        ep_timeline.append(None)
                        busy_flags[free_worker.iid] = True
                        free_worker.send((group_data, group_id))
                    else:
                        break
            healthy = all(map(lambda worker: worker.is_healthy(), workers))
            ####for
        ####while

        # Orchestrator is done distributing epochs, wait for everyone to finish
        if healthy:
            logger.info("Done distributing epochs, waiting for all workers to finish.")
        time.sleep(0.2)

        while any(busy_flags.values()) and healthy:
            for free_worker in filter(lambda worker: busy_flags[worker.iid] or worker.is_available(), workers):
                if free_worker.is_available() and (epoch_data := free_worker.get()) is not None:
                    ep_timeline[epoch_data[1]] = epoch_data[0]
                    busy_flags[free_worker.iid] = False
                if not busy_flags[free_worker.iid] or not free_worker.is_alive():
                    if free_worker.is_alive():
                        free_worker.send(None)
                        time.sleep(0.1)
                    else:
                        busy_flags[free_worker.iid] = False
                    logger.info(f"Worker {free_worker.iid} closed.")
                    free_worker.terminate()
                    free_worker.join(0.2)
                    free_worker.close()
            time.sleep(0.2)

        if healthy:
            logger.debug("All workers finished, cleaning-up.")
        time.sleep(0.01)
        for worker in workers:
            try: worker.terminate()
            except: ...
        time.sleep(0.05)
        for worker in workers:
            try: worker.kill()
            except: ...
        time.sleep(0.05)
        for worker in workers:
            try: worker.join()
            except: ...
        workers.clear()
        if not healthy:
            logger.warning("One worker had an unrecoverable error, giving up.")
            import sys
            sys.exit(1)
        return ep_timeline
    ####

    def encode(self) -> tuple[bool, list[Epoch]]:
        bd_video = self._prepare()

        if (threaded := self._threads > 1):
            epochs = self._convert_mt(bd_video)
        else:
            epochs = self._convert_single(bd_video)

        # with multithreading we have non deterministic generation order of DisplaySets
        # so fix the composition number here
        self.fix_composition_id(epochs, replace=threaded)

        is_valid = self.test_output(bd_video, epochs)
        return is_valid, epochs
    ####

    def test_output(self, bd_video: BDVideo, epochs: list[Epoch]) -> bool:
        if logger.level <= 10:
            logger.debug(debug_stats(epochs))

        logger.info("Checking stream consistency and compliancy...")
        LogFacility.associate_log_and_report(logger)

        final_fps = bd_video.fps
        compliant, warnings = is_compliant(epochs, final_fps)

        if compliant:
            logger.info("Checking PTS and DTS rules...")
            compliant &= check_pts_dts_sanity(epochs, final_fps)
            if not compliant:
                logger.error("=> Stream has a PTS/DTS issue!!")
                self._allow_pes_output = False
            elif (max_bitrate := self.kwargs.get('max_kbps', False)) > 0:
                logger.info(f"Checking PGS bitrate and buffer usage w.r.t user max bitrate: {max_bitrate} Kbps...")
                if not test_rx_bitrate(epochs, int(max_bitrate*1000/8), final_fps):
                    logger.warning("Detected buffer underflow(s) given the provided test bitrate.")
        if compliant:
            if warnings == 0:
                logger.info("=> Output PGS is compliant.")
            else:
                logger.info("=> Output PGS seems compliant but has minor issues (see warnings).")
        else:
            logger.error("=> Output PGS is not compliant. Expect display issues or decoder crash.")

        LogFacility.dissociate_log_and_report(logger)
        return compliant
    ####

    def fix_composition_id(self, epochs: list[Epoch], replace: bool = False) -> None:
        composition_num = 0
        for epoch in epochs:
            last_composition_num = epoch[0].pcs.composition_number-1
            composition_num_in_epoch = 0
            for kd, ds in enumerate(epoch):
                if not replace or kd > 0:
                    diff = (ds.pcs.composition_number - last_composition_num) & 0xFFFF
                    assert 0 <= diff <= 1
                else:
                    diff = 1 # always true for epoch start
                last_composition_num = ds.pcs.composition_number
                if replace:
                    ds.pcs.composition_number = (composition_num + composition_num_in_epoch) & 0xFFFF
                else:
                    assert ds.pcs.composition_number == (composition_num + composition_num_in_epoch) & 0xFFFF

                composition_num_in_epoch += diff
            composition_num += composition_num_in_epoch
    ####

    def write_output(self, output_file: Path | str, epochs: list[Epoch]) -> None:
        fp = output_file
        if not epochs:
            raise RuntimeError("No data to write.")

        is_pes = fp.lower().endswith('pes')
        is_sup = fp.lower().endswith('sup')
        if not (is_pes or is_sup):
            logger.warning("Unknown extension, assuming a .SUP file...")
            is_sup = True
        if self.kwargs.get('output_all_formats', False):
            is_pes = is_sup = True
        if len(filepath := fp.split('.')) > 1:
            fp_pes = '.'.join(filepath[:-1]) + '.pes'
            fp_sup = '.'.join(filepath[:-1]) + '.sup'
        else:
            fp_pes = filepath[0] + '.pes'
            fp_sup = filepath[0] + '.sup'

        if is_pes:
            if self._allow_pes_output:
                logger.info(f"Writing output file {fp_pes}")
                PesMuiWriter(fp_pes).write_epochs(epochs)
            else:
                logger.warning("PES+MUI not generated as the stream is not compliant.")
        ##if is_pes
        if is_sup:
            logger.info(f"Writing output file {fp_sup}")
            SUPWriter(fp_sup).write_epochs(epochs)
    ####def
####

class EpochEncode:
    def __init__(self, pg_stream_ctx: PGStreamCtx, epoch_data: EpochData, kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.pg_stream_ctx = pg_stream_ctx
        self.kwargs = kwargs
        self.epoch_data = epoch_data

    def preprocess(self, remove_dupes: bool = True, add_refreshes: float = 0) -> Self:
        if remove_dupes:
            self.epoch_data.events = EventsPreprocessor.remove_duplicates(self.epoch_data.events)

        if add_refreshes >= 1.0:
            for event in self.epoch_data.events:
                if (count := EventsPreprocessor.get_refresh_count(event)) > 0:
                    durTC = TC(event.fractional_fps, (event.outTC - event.inTC).frames)
                    step = int(durTC.frames / (count + 1))
                    for offset in range(step, durTC.frames, step):
                        event.repeated_inTC.append(TC(self.video_fmt.fps, frames=event.inTC.frames + offset))
        return self

    def encode(self) -> Epoch:
        engine = EpochEncoderEngine(self.epoch_data, self.pg_stream_ctx, self.kwargs)
        ctx = engine.analyze()
        ctx = engine.plan(ctx)
        return engine.encode(ctx)

class BDNEpochWorker(mp.Process):
    _instance_cnt = 0
    def __init__(self, video_fmt: BDVideo, kwargs: dict[str, Any], daemonize: bool = True) -> None:
        self._iid = __class__._instance_cnt
        __class__._instance_cnt += 1

        self.video_fmt = video_fmt
        self.kwargs = kwargs

        self._q_rx = mp.Queue()
        self._q_tx = mp.Queue()
        self._available = mp.Value('d', 0, lock=False)
        super().__init__(daemon=daemonize)

    @property
    def iid(self) -> int:
        return self._iid

    @classmethod
    def reset_module(cls) -> None:
        cls._instance_cnt = 0

    def setup_env(self) -> None:
        LogFacility.disable_tqdm()

        log_filename = self.kwargs.get('log_filename', None)
        file_logging_level = self.kwargs.get('log_to_file', False)
        if file_logging_level > 0 and log_filename:
            logfile = str(log_filename) + f"_{self._prefix}"  + ".txt"
            LogFacility.set_file_log(logger, logfile, file_logging_level)
            LogFacility.set_logger_level(logger.name, file_logging_level)

        libs_params = self.kwargs.get('ini_opts', {})
        logger.debug(f"INI parameters: {libs_params}")
        requested_qtz = self.kwargs['quantize_lib']
        if requested_qtz == BuiltinQuantizer.LIQ:
            if not BuiltinQuantizer.LIQ.value.configure(libs_params.get('quant', {})):
                self.kwargs['quantize_lib'] = BuiltinQuantizer.Qtzr
                logger.warning("Failed to configure PNG/ImageQuant, using Qtzr as a fallback.")

        if (sup_params := libs_params.get('super_cfg', None)) is not None:
            SSIMPW.use_gpu = bool(int(sup_params.get('use_gpu', True)))
        logger.debug(f"OpenCL enabled: {SSIMPW.use_gpu}.")
    ####

    def is_available(self) -> bool:
        return self._available.value > 0

    def is_healthy(self) -> bool:
        return self._available.value >= 0

    def send(self, data: Any) -> None:
        """
        Send data to the worker.
        """
        self._q_rx.put(data)

    def get(self, default: Any | None = None) -> Any:
        """
        Receive data from the worker.
        """
        try:
            return self._q_tx.get_nowait()
        except:
            return None

    @property
    def _prefix(self) -> str:
        return f"W{self.iid}"

    def _print_tb_except(self, e: Exception) -> None:
        tb = e.__traceback__
        while (next_tb := tb.tb_next) is not None:
            tb = next_tb
        logger.error(f"Encoder {self._prefix} died: {type(e).__name__}@{tb.tb_frame.f_code.co_name}::L{tb.tb_lineno}" + (f" - {e}." if len(e.args) else "."))

    def run(self) -> None:
        self.setup_env()
        logger.debug(f"{self._prefix} ready.")
        pg_stream_ctx = PGStreamCtx(self.video_fmt)
        self._available.value = 1
        while True:
            try:
                in_data = self._q_rx.get(timeout=0.1)
            except:
                continue
            else:
                self._available.value = 0
            if in_data is None:
                break
            ectx, epoch_id = in_data
            logger.info(f"{self._prefix} encoding epoch {epoch_id}: {ectx.events[0].inTC}->{ectx.events[-1].outTC} with {len(ectx.events)} event(s), {len(ectx.windows)} window(s).")
            ee = EpochEncode(pg_stream_ctx, ectx, self.kwargs)
            try:
                new_epoch = ee.preprocess().encode()
            except Exception as e:
                self._print_tb_except(e)
                self._available.value = -1
                break
            else:
                logger.info(f"{self._prefix} => encoded epoch {epoch_id} as {len(new_epoch)} display sets.")

            self._q_tx.put((new_epoch, epoch_id))
            self._available.value = 1
        ####
    ####
####
