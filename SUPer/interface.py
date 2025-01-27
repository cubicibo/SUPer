#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 cibo
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
import numpy as np
import multiprocessing as mp

from typing import Any, Optional, NoReturn, Union
from itertools import chain
from functools import partial
from enum import IntEnum

from scenaristream import EsMuiStream
from brule import LayoutEngine, Brule, HexTree, KDMeans

from .utils import LogFacility, Box, SSIMPW
from .pgraphics import PGDecoder
from .filestreams import BDNXML, BDNXMLEvent, remove_dupes
from .segments import Epoch, DisplaySet
from .optim import Quantizer
from .pgstream import is_compliant, check_pts_dts_sanity, test_rx_bitrate, EpochContext, debug_stats
from .render2 import PaddingEngine, EpochEncoder

class LayoutPreset(IntEnum):
    SAFE   = 0
    NORMAL = 1
    GREEDY = 2
#%%
logger = LogFacility.get_logger('SUPer')

_parent_id = os.getpid()
def _pool_worker_init():
    if os.name == 'nt':
        import psutil

        def sig_int(signal_num, frame):
            parent = psutil.Process(_parent_id)
            _cpid = os.getpid()
            for child in parent.children():
                if child.pid != _cpid:
                    child.kill()
            parent.kill()
            psutil.Process(_cpid).kill()
        signal.signal(signal.SIGINT, sig_int)
    #else, do nothing

def _decode_duration_base(cwdo, container_area) -> float:
    dd = PGDecoder.copy_gp_duration(container_area)
    t_dec = 0
    for wd in cwdo:
        t_dec += PGDecoder.decode_obj_duration(wd.area)
        dd = max(t_dec, dd) + PGDecoder.copy_gp_duration(wd.area)
    return dd

def _find_modify_layout(leng: LayoutEngine, container: Box, preset: LayoutPreset):
    cbox, w1, w2, is_vertical = leng.get_layout()
    cbox, w1, w2 = tuple(map(lambda b: Box.from_coords(*b), (cbox, w1, w2)))
    cwd = (w1, w2) if w1 != w2 else (w1,)
    cwd = PaddingEngine(cbox, container=container, n_groups=2).directional_pad(cwd, is_vertical)

    scores = []
    for cwdo in (cwd, reversed(cwd)):
        scores.append(_decode_duration_base(cwdo, container.area))

    if len(scores) > 1:
        flip_results = (preset == LayoutPreset.SAFE and scores[0] < scores[1])
        flip_results = flip_results or (preset != LayoutPreset.SAFE and scores[0] > scores[1])

        if flip_results:
            cwd = tuple(reversed(cwd))
            scores[0] = scores[1]

    base_box = Box.from_coords(*leng.get_raw_container())
    base_box_wd = Box(0, base_box.dy, 0, base_box.dx)
    score_container = _decode_duration_base((base_box,), container.area)
    is_bad_split = scores[0] >= max(1/PGDecoder.FREQ, score_container-10/PGDecoder.FREQ)
    #coded object buffer can fit at most 16 ODS: we need roughly +150 bytes in the buffer
    #note: technically we need to also consider the 2*height line-endings bytes, but let's assume there's *some* compression
    may_not_fit_buffer = any(map(lambda b: b.area >= (1 << 20)-150, cwd))
    is_greedysplit_worthwile = (score_container*0.85 < scores[0] and base_box_wd.area > 125000) or may_not_fit_buffer
    old_score = scores[0]

    #With greedy mode, anytime we're dealing with very big objects we abuse the 1/2 1/2. This also prevents coded buffer overflow.
    layout_modifier = 'N'
    if (preset == LayoutPreset.GREEDY or is_bad_split) and is_greedysplit_worthwile:
        cx, cy = (1, 0.5)
        box1 = Box(0, int(round(cy*base_box.dy)), 0, int(round(base_box.dx*cx)))
        box2 = Box.from_coords(0, box1.y2, base_box.dx, base_box.dy)
        assert base_box.area == Box.union(box1, box2).area
        assert abs(1-box1.area/box2.area) < 1e-1

        greedy_wds = (box1, box2)
        new_score = _decode_duration_base(greedy_wds, container.area)
        if scores[0] > new_score:
            cwd = greedy_wds
            scores[0] = new_score
            layout_modifier = 'G'
            cbox = base_box
        # Objects could still not fit in buffer at this point, but there's so much we can do to help authorers...
    if layout_modifier == 'N' and is_bad_split and not may_not_fit_buffer:
        cwd = (base_box_wd,)
        cbox = base_box
        scores[0] = score_container
        layout_modifier = 'S'
    return (cbox, cwd, layout_modifier, is_bad_split, is_greedysplit_worthwile, scores, old_score)

def _find_epochs_layouts(events: list[BDNXMLEvent], bdn: BDNXML, preset: Union[LayoutPreset, int] = LayoutPreset.GREEDY) -> list[EpochContext]:
    preset = LayoutPreset(preset)
    width, height = bdn.format.value
    container = Box.from_coords(0, 0, width, height)

    leng = LayoutEngine((width, height))
    ectx = []

    running_ev = []
    for k, ev in enumerate(reversed(events), 1):
        channel = np.ascontiguousarray(ev.image.getchannel('A'), dtype=np.uint8)
        if np.any(channel):
            leng.add_to_layout(ev.x, ev.y, channel)
            running_ev = [ev]
            break

    # All events in list are fully transparent, return empty context list.
    if 0 == len(running_ev):
        return ectx

    for ev in reversed(events[:-k]):
        # Remove empty bitmaps
        channel = np.ascontiguousarray(ev.image.getchannel('A'), dtype=np.uint8)
        ev.unload()
        if not np.any(channel):
            continue

        if ev.tc_out != running_ev[0].tc_in:
            cbox, cwd, layout_modifier, is_bad_split, is_greedysplit_worthwile, scores, old_score = _find_modify_layout(leng, container, preset)
            pts_out = ev.tc_out.to_pts()
            if pts_out + scores[0] + 1e-8 < running_ev[0].tc_in.to_pts():
                logger.debug(f"Epoch: {running_ev[0].tc_in}, modifier {layout_modifier}: {cwd} with b={is_bad_split}:g={is_greedysplit_worthwile}, {old_score:.03f}->{scores[0]:.03f}.")
                ectx.append(EpochContext(cbox, cwd, running_ev, pts_out))
                running_ev = []
                leng.reset()
        ####
        leng.add_to_layout(ev.x, ev.y, channel)
        running_ev.insert(0, ev)
    ####for ev
    assert len(running_ev)
    cbox, cwd, layout_modifier, is_bad_split, is_greedysplit_worthwile, scores, old_score = _find_modify_layout(leng, container, preset)

    leng.destroy() #Done with the layout engine
    logger.debug(f"Epoch: {running_ev[0].tc_in}, modifier {layout_modifier}: {cwd} with b={is_bad_split}:g={is_greedysplit_worthwile}, {old_score:.03f}->{scores[0]:.03f}.")

    ectx.append(EpochContext(cbox, cwd, running_ev, -np.inf))
    return ectx[::-1]
####

class BDNRender:
    def __init__(self, bdnf: str, kwargs: dict[str, Any], outfile: str) -> None:
        self.bdn_file = bdnf
        self.outfile = outfile
        self.kwargs = kwargs

        self._epochs = []
        self._first_pts = 0

    def prepare(self) -> BDNXML:
        stkw = '' + ':'.join([f"{k}={v}" for k, v in self.kwargs.items() if not isinstance(v, dict)])

        if self.kwargs.get('quantize_lib', None) is None:
            self.kwargs['quantize_lib'] = Quantizer.Libs.HEXTREE

        self._report_log_hdl = None
        if self.kwargs.get('log_to_file', False) < 0:
            self._report_log_hdl = LogFacility.get_logger('event_report', with_handler=False)
            LogFacility.set_file_log(self._report_log_hdl, str(self.outfile) + ".txt", simple_format=True)
            LogFacility.set_logger_level('event_report', self._report_log_hdl.level)
            logger.addHandler(self._report_log_hdl.handlers[0])

        logger.iinfo(f"Parameters: {stkw}")

        bdn = BDNXML(os.path.expanduser(self.bdn_file))
        fps_str = int(bdn.fps) if float(bdn.fps).is_integer() else round(float(bdn.fps), 3)
        logger.iinfo(f"BDN metadata: {'x'.join(map(str, bdn.format.value))}, FPS={fps_str}, DF={bdn.dropframe}, {len(bdn.events)} valid events.")

        # Pop succint handler
        if self._report_log_hdl:
            for ih, hdl in enumerate(logger.handlers):
                if hdl == self._report_log_hdl.handlers[0]:
                    logger.handlers.pop(ih)
                    break

        if len(bdn.events) == 0:
            logger.error("No BDN event found, exiting.")
            import sys
            sys.exit(1)

        self.kwargs['adjust_ntsc'] = isinstance(fps_str, float)
        if self.kwargs['adjust_ntsc']:
            if bdn.dropframe:
                logger.warning("DF BDN detected: converting Timecodes to NDF and scaling all timestamps by 1.001.")
                #conversion was done internally in BDNXML already.
            else:
                logger.info("NDF NTSC detected: scaling all timestamps by 1.001.")
        self._first_pts = bdn.events[0].tc_in.to_pts()

        if bdn.split_seen:
            logger.warning("BDN contains split graphics! Merging them back to find better ones, output quality may be affected.")
            logger.warning("Risk of excessive RAM usage: storing the merged images in RAM.")
        return bdn

    def find_all_layouts(self, bdn: BDNXML) -> list[EpochContext]:
        layout_mode = self.kwargs.get('ini_opts', {}).get('super_cfg', {}).get('layout_mode', LayoutPreset.NORMAL)
        layout_mode = int(layout_mode) if isinstance(layout_mode, (int, LayoutPreset)) or str.isnumeric(layout_mode) else LayoutPreset.NORMAL
        logger.debug(f"Layout engine preset: {layout_mode}, with capabilities: {', '.join(LayoutEngine.get_capabilities())}.")

        screen_area = np.multiply(*bdn.format.value)
        epochstart_dd_fn = lambda o_area: max(PGDecoder.copy_gp_duration(screen_area), PGDecoder.decode_obj_duration(o_area)) + PGDecoder.copy_gp_duration(o_area)

        pbar = LogFacility.get_progress_bar(logger, bdn.events)
        pbar.set_description("Finding epochs and layouts", True)

        p_find_epochs_layouts = partial(_find_epochs_layouts, bdn=bdn, preset=layout_mode)
        lectx = []
        if self.kwargs['threads'] > 1:
            with mp.Pool(self.kwargs['threads'], _pool_worker_init) as mpp:
                for r in mpp.imap_unordered(p_find_epochs_layouts, bdn.groups(epochstart_dd_fn(screen_area))):
                    pbar.update(sum(map(lambda ctx: len(ctx.events), r)))
                    lectx += r
            lectx = sorted(lectx, key=lambda ctx: ctx.events[0].tc_in)
        else:
            for r in map(p_find_epochs_layouts, bdn.groups(epochstart_dd_fn(screen_area))):
                pbar.update(sum(map(lambda ctx: len(ctx.events), r)))
                lectx += r
        pbar.update(len(bdn.events)-pbar.n+1)
        LogFacility.close_progress_bar(logger)

        wipe_margin = float(np.ceil(PGDecoder.copy_gp_duration(bdn.format.area)*bdn.fps + 1/bdn.fps/2)/bdn.fps)
        for ctx, ctx_next in zip(lectx, lectx[1:]):
            event_max_pts = ctx.events[-1].tc_out.to_pts()
            if np.isinf(ctx_next.min_dts):
                ctx_next.min_dts = max(event_max_pts, (ctx_next.events[0].tc_in.to_pts() - _decode_duration_base(ctx_next.windows, bdn.format.area)) - wipe_margin)
            ctx.max_pts = min(event_max_pts + wipe_margin, ctx_next.min_dts)
            ctx_next.min_dts += 1/PGDecoder.FREQ #safe: add one tick to min dts

        if logger.level <= 10:
            for ect in lectx:
                logger.debug(f"Epoch Context: {ect.events[0].tc_in}->{ect.events[-1].tc_out} {len(ect.events)}, RC={ect.box}, WDS={ect.windows}, [{ect.min_dts:.04f};{ect.max_pts:.04f}]")
        return lectx
    #####find_all

    def _convert_single(self, bdn: BDNXML) -> None:
        EpochRenderer.set_mt(False)
        renderer = EpochRenderer(bdn, self.kwargs, self.outfile)
        renderer.setup_env()

        pcs_id = 0
        logger.debug("Finding all epochs and their screen layout (this can take a while)...")
        epochs_ctx = self.find_all_layouts(bdn)
        logger.info(f"Identified {len(epochs_ctx)} epochs to render.")

        for ectx in epochs_ctx:
            epoch, pcs_id = renderer.convert2(ectx, pcs_id)
            self._epochs.append(epoch)
    ####

    def _setup_mt_env(self) -> None:
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
            logger.critical("Terminated.")
            sys.exit(1)

        f_term = lambda signal_num, frame: sighandler(self._workers, signal_num, frame)
        signal.signal(signal.SIGINT, f_term)
        signal.signal(signal.SIGTERM, f_term)
        if os.name == 'nt':
            signal.signal(signal.SIGBREAK, f_term)
        logger.debug("Registered signal handlers.")
    ####

    def _setup_mt_main_logging(self) -> None:
        file_logging_level = self.kwargs.get('log_to_file', False)
        if file_logging_level > 0:
            LogFacility.set_file_log(logger, str(self.outfile) + ".txt", file_logging_level)
            LogFacility.set_logger_level(logger.name, file_logging_level)

    def _convert_mt(self, bdn: BDNXML) -> None:
        import time
        EpochRenderer.set_mt(True)
        EpochRenderer.reset_module()
        self._setup_mt_main_logging()

        logger.debug("Finding all epochs and their screen layout (this can take a while)...")
        epochs_ctx = self.find_all_layouts(bdn)
        logger.info(f"Identified {len(epochs_ctx)} epochs.")

        as_deamon = self.kwargs.get('daemonize', True)
        n_threads = min(self.kwargs.get('threads', 2), len(epochs_ctx))
        renderers = [EpochRenderer(bdn, self.kwargs, self.outfile, as_deamon) for _ in range(n_threads)]

        self._workers = renderers
        self._setup_mt_env()

        logger.info("Starting workers...")
        for renderer in renderers:
            renderer.start()

        while not all(map(lambda renderer: renderer.is_available(), renderers)):
            time.sleep(0.2)

        def add_data(ep_timeline: list[bytes], epoch_data: tuple[bytes, int], hdl = None) -> None:
            if hdl is None:
                new_epoch, epoch_id = epoch_data
            else:
                new_epoch, epoch_id, fmsgs = epoch_data
                for msg in filter(lambda x: x[0] > 20, reversed(fmsgs)): hdl.info(msg[1])
            ep_timeline[epoch_id] = new_epoch
        ###

        #Orchestrator starts here
        busy_flags = {renderer.iid: False for renderer in renderers}
        g_epochs = enumerate(chain(epochs_ctx, (None,)))
        tc_inout, ep_timeline = [], []

        group_data = True
        healthy = True
        while group_data is not None and healthy:
            time.sleep(0.05)
            for free_renderer in filter(lambda renderer: renderer.is_available(), renderers):
                if (epoch_data := free_renderer.get()) is not None:
                    add_data(ep_timeline, epoch_data, self._report_log_hdl)
                    busy_flags[free_renderer.iid] = False
                if busy_flags[free_renderer.iid] is False:
                    group_id, group_data = next(g_epochs)
                    if group_data is not None:
                        ep_timeline.append(None)
                        tc_inout.append((group_data.events[0].tc_in, group_data.events[-1].tc_out))
                        busy_flags[free_renderer.iid] = True
                        free_renderer.send((group_data, group_id))
                    else:
                        break
            healthy = all(map(lambda encoder: encoder.is_healthy(), renderers))
            ####for
        ####while

        # Orchestrator is done distributing epochs, wait for everyone to finish
        if healthy:
            logger.info("Done distributing epochs, waiting for renderers to finish.")
        time.sleep(0.2)

        while any(busy_flags.values()) and healthy:
            for free_renderer in filter(lambda renderer: busy_flags[renderer.iid] or renderer.is_available(), renderers):
                if free_renderer.is_available() and (epoch_data := free_renderer.get()) is not None:
                    add_data(ep_timeline, epoch_data, self._report_log_hdl)
                    busy_flags[free_renderer.iid] = False
                if not busy_flags[free_renderer.iid] or not free_renderer.is_alive():
                    if free_renderer.is_alive():
                        free_renderer.send(None)
                        time.sleep(0.1)
                    else:
                        busy_flags[free_renderer.iid] = False
                    logger.info(f"Worker {free_renderer.iid} closed.")
                    free_renderer.terminate()
                    free_renderer.join(0.2)
                    free_renderer.close()
            time.sleep(0.2)

        if healthy:
            logger.info("All jobs finished, cleaning-up.")
        time.sleep(0.01)
        for renderer in renderers:
            try: renderer.terminate()
            except: ...
        time.sleep(0.05)
        for renderer in renderers:
            try: renderer.kill()
            except: ...
        time.sleep(0.05)
        for renderer in renderers:
            try: renderer.join()
            except: ...
        self._workers.clear()
        if not healthy:
            logger.critical("One encoder process had an unrecoverable error, giving up.")
            import sys
            sys.exit(1)
        logger.debug("Unserializing workers data.")
        for eid, epoch in enumerate(ep_timeline):
            ep_timeline[eid] = Epoch.from_bytes(epoch)
        self._epochs = ep_timeline
    ####

    def optimise(self) -> None:
        bdn = self.prepare()
        n_threads = self.kwargs.get('threads', 1)
        if (n_threads_auto := isinstance(n_threads, str)):
            try:
                import psutil
            except (ModuleNotFoundError, NameError):
                n_threads = max(1, mp.cpu_count() >> 1) #commonplace: logical = 2*physical cores
            else:
                n_threads = psutil.cpu_count(logical=False)

        if n_threads_auto:
            logger.info(f"Using {n_threads} thread(s).")
        self.kwargs['threads'] = n_threads

        if n_threads == 1:
            self._convert_single(bdn)
            self.fix_composition_id(False)
        else:
            self._convert_mt(bdn)
            logger.debug("Fixing PCS composition number.")
            self.fix_composition_id(True)

        self.test_output(bdn)
    ####

    def test_output(self, bdn: BDNXML) -> None:
        if logger.level <= 10:
            logger.debug(debug_stats(self._epochs))

        # Final checks
        logger.info("Checking stream consistency and compliancy...")

        #inject succint log handler to append the stats.
        if self._report_log_hdl:
            logger.addHandler(self._report_log_hdl.handlers[0])

        final_fps = round(bdn.fps, 3)
        compliant, warnings = is_compliant(self._epochs, final_fps)

        if compliant:
            logger.info("Checking PTS and DTS rules...")
            compliant &= check_pts_dts_sanity(self._epochs, final_fps)
            if not compliant:
                logger.error("=> Stream has a PTS/DTS issue!!")
            elif (max_bitrate := self.kwargs.get('max_kbps', False)) > 0:
                logger.info(f"Checking PGS bitrate and buffer usage w.r.t max bitrate: {max_bitrate} Kbps...")
                max_bitrate = int(max_bitrate*1000/8)
                warnings += not test_rx_bitrate(self._epochs, max_bitrate, final_fps)
        if compliant:
            if warnings == 0:
                logger.info("=> Output PGS seems compliant.")
            if warnings > 0:
                logger.warning("=> Excessive bandwidth detected, testing with mux required.")
        else:
            logger.error("=> Output PGS is not compliant. Expect display issues or decoder crash.")

        if self._report_log_hdl:
            for ih, hdl in enumerate(logger.handlers):
                if hdl == self._report_log_hdl.handlers[0]:
                    logger.handlers.pop(ih)
    ####

    def fix_composition_id(self, replace: bool = False) -> None:
        cnt = 0
        for epoch in self._epochs:
            for ds in epoch:
                if replace:
                    ds.pcs.composition_n = cnt & 0xFFFF
                else:
                    assert ds.pcs.composition_n == cnt & 0xFFFF
                cnt += 1
    ####

    def write_output(self) -> None:
        fp = self.outfile
        if self._epochs:
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
                logger.info(f"Writing output file {fp_pes}")

                decode_duration = (self._epochs[0][0].pcs.tpts - self._epochs[0][0].pcs.tdts) & ((1<<32) - 1)
                decode_duration /= PGDecoder.FREQ

                writer = EsMuiStream.segment_writer(fp_pes, first_dts=self._first_pts - decode_duration)
                next(writer) #init writer
                for epoch in self._epochs:
                    for ds in epoch:
                        for seg in ds:
                            writer.send(seg)
                # Close ESMUI writer
                writer.send(None)
                writer.close()
            if is_sup:
                logger.info(f"Writing output file {fp_sup}")

                with open(fp_sup, 'wb') as f:
                    f.write(b''.join(map(bytes, self._epochs)))
        else:
            raise RuntimeError("No data to write.")
####

class EpochRenderer(mp.Process):
    __threaded = True
    _instance_cnt = 0
    def __init__(self, bdn: BDNXML, kwargs: dict[str, Any], outfile: str, daemonize: bool = True) -> None:
        self.bdn = bdn
        self.outfile = outfile
        self.kwargs = kwargs
        self._iid = __class__._instance_cnt
        __class__._instance_cnt += 1

        if __class__.__threaded:
            self._q_rx = mp.Queue()
            self._q_tx = mp.Queue()
            self._available = mp.Value('d', 0, lock=False)
            super().__init__(daemon=daemonize)

    @property
    def iid(self) -> Optional[int]:
        return self._iid

    @classmethod
    def set_mt(cls, enable: bool = False) -> None:
        cls.__threaded = enable

    @classmethod
    def reset_module(cls) -> None:
        cls._instance_cnt = 0

    def setup_env(self) -> None:
        file_logging_level = self.kwargs.get('log_to_file', False)
        self._buffering_logs = False

        if __class__.__threaded:
            LogFacility.disable_tqdm()
        if file_logging_level < 0 and __class__.__threaded:
            LogFacility.set_logger_buffer(logger)
            self._buffering_logs = True
        elif file_logging_level > 0:
            logfile = str(self.outfile) + (f"_{self.iid}" if __class__.__threaded else '') + ".txt"
            LogFacility.set_file_log(logger, logfile, file_logging_level)
            LogFacility.set_logger_level(logger.name, file_logging_level)

        libs_params = self.kwargs.get('ini_opts', {})
        logger.debug(f"INI parameters: {libs_params}")
        requested_qtz = Quantizer.Libs(self.kwargs['quantize_lib'])
        if requested_qtz >= Quantizer.Libs.PILIQ:
            if not Quantizer.init_piliq(**libs_params.get('quant', {})):
                logger.warning("Failed to initialise imagequant/pngquant. Falling back to lower quality 'HexTree'.")
                self.kwargs['quantize_lib'] = Quantizer.Libs.HEXTREE.value
            else:
                self.kwargs['quantize_lib'] = Quantizer.select_quantizer(self.kwargs['quantize_lib'])
                logger.debug(f"Advanced image quantizer armed: {Quantizer.get_piliq().lib_name}")
        else:
            logger.debug(f"Image quantizer: '{requested_qtz.name}'.")
            self.kwargs['quantize_lib'] = requested_qtz.value

        if Quantizer.Libs.HEXTREE == self.kwargs['quantize_lib']:
            logger.debug(f"HexTree quantizer capabilities: {', '.join(HexTree.get_capabilities())}.")
        elif Quantizer.Libs.KMEANS == self.kwargs['quantize_lib']:
            logger.debug(f"KD-Means quantizer capabilities: {', '.join(KDMeans.get_capabilities())}.")

        logger.debug(f"Bitmap encoder capabilities: {', '.join(Brule.get_capabilities())}.")

        if (sup_params := libs_params.get('super_cfg', None)) is not None:
            SSIMPW.use_gpu = bool(int(sup_params.get('use_gpu', True)))
        logger.debug(f"OpenCL enabled: {SSIMPW.use_gpu}.")

        BDNXMLEvent.can_warn_palettized = True
    ####

    def convert2(self, ectx: EpochContext, pcs_id: int = 0) -> tuple[Epoch, DisplaySet, int]:
        ectx.events = remove_dupes(ectx.events)
        prefix = f"W{self.iid}: " if __class__.__threaded else ""
        logger.info(prefix + f"Encoding epoch {ectx.events[0].tc_in}->{ectx.events[-1].tc_out} with {len(ectx.events)} event(s), {len(ectx.windows)} window(s).")

        if logger.level <= 10:
            for w_id, wd in enumerate(ectx.windows):
                logger.debug(f"Window {w_id}: X={wd.x+ectx.box.x}, Y={wd.y+ectx.box.y}, W={wd.dx}, H={wd.dy}")

        wds_analyzer = EpochEncoder(ectx, self.bdn, pcs_id=pcs_id, **self.kwargs)
        new_epoch, pcs_id = wds_analyzer.analyze()
        logger.debug(prefix + f" => optimised as {len(new_epoch)} display sets.")
        return new_epoch, pcs_id

    def is_available(self) -> bool:
        assert __class__.__threaded
        return self._available.value > 0

    def is_healthy(self) -> bool:
        assert __class__.__threaded
        return self._available.value >= 0

    def send(self, data: Any):
        assert __class__.__threaded
        self._q_rx.put(data)

    def get(self, default: Optional[Any] = None) -> Any:
        assert __class__.__threaded
        try:
            return self._q_tx.get_nowait()
        except:
            return None

    def run(self):
        assert not (self._q_rx is None or self._q_tx is None)
        self.setup_env()
        logger.info(f"Worker {self.iid} ready.")

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
            logger.debug(f"WORKER {self.iid} on EPOCH {epoch_id}")
            try:
                new_epoch, _ = self.convert2(ectx) #discard pcs_id
            except Exception as e:
                tb = e.__traceback__
                while (next_tb := tb.tb_next) is not None:
                    tb = next_tb
                prefix = f"W{self.iid} " if __class__.__threaded else ""
                logger.critical(f"Encoder {prefix}died: {type(e).__name__}@{tb.tb_frame.f_code.co_name}::L{tb.tb_lineno}" + (f" - {e}." if len(e.args) else "."))
                self._available.value = -1
                break
            if self._buffering_logs:
                self._q_tx.put((bytes(new_epoch), epoch_id, LogFacility.get_buffered_msgs(logger)))
            else:
                self._q_tx.put((bytes(new_epoch), epoch_id))
            self._available.value = 1
        ####
    ####
####
