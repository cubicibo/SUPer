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

import gc
import numpy as np
from os import path

from scenaristream import EsMuiStream

from .utils import TimeConv as TC, LogFacility, Box, BDVideo
from .pgraphics import PGDecoder
from .filestreams import BDNXML, remove_dupes
from .optim import Quantizer
from .pgstream import is_compliant, check_pts_dts_sanity, test_rx_bitrate

logger = LogFacility.get_logger('SUPer')

class BDNRender:
    def __init__(self, bdnf: str, kwargs: dict[str, int], outfile: str) -> None:
        self.bdn_file = bdnf
        self.outfile = outfile
        self.kwargs = kwargs

        self._epochs = []
        self._first_pts = 0
        self.setup()

    def setup(self) -> None:
        if (libs_params := self.kwargs.pop('libs_path', {})):
            logger.info(f"Library parameters: {libs_params}")
            if self.kwargs.get('quantize_lib', Quantizer.Libs.PIL_CV2KM) == Quantizer.Libs.PILIQ:
                if (piq_params := libs_params.get('quant', None)) is not None:
                    if not Quantizer.init_piliq(*piq_params):
                        logger.info("Failed to initialise advanced image quantizer. Falling back to PIL+K-Means.")
                        self.kwargs['quantize_lib'] = Quantizer.Libs.PIL_CV2KM.value

        file_logging_level = self.kwargs.get('log_to_file', False)
        if file_logging_level > 0:
            logfile = str(self.outfile) + '.txt'
            LogFacility.set_file_log(logger, logfile, file_logging_level)
            LogFacility.set_logger_level(logger.name, file_logging_level)

    def optimise(self) -> None:
        from .render2 import GroupingEngine, WindowsAnalyzer

        stkw = '' + ':'.join([f"{k}={v}" for k, v in self.kwargs.items()])
        logger.iinfo(f"Parameters: {stkw}")

        bdn = BDNXML(path.expanduser(self.bdn_file))
        fps_str = bdn.fps if float(bdn.fps).is_integer() else round(bdn.fps, 3)
        logger.iinfo(f"BDN metadata: {'x'.join(map(str, bdn.format.value))}, FPS={fps_str}, DF={bdn.dropframe}, {len(bdn.events)} valid events.")

        if len(bdn.events) == 0:
            logger.error("No BDN event found, exiting.")
            import sys
            sys.exit(1)

        if self.kwargs.get('scale_fps', False) and bdn.fps > 30:
            logger.error("Incorrect XML FPS for subsampling operation: flag ignored.")
            self.kwargs['scale_fps'] = False

        self.kwargs['adjust_ntsc'] = isinstance(bdn.fps, float) and not bdn.dropframe
        if self.kwargs['adjust_ntsc']:
            logger.info("NDF NTSC detected: scaling all timestamps by 1.001.")

        self._first_pts = TC.tc2pts(bdn.events[0].tc_in, bdn.fps)

        logger.info("Finding epochs...")

        container = Box(0, bdn.format.value[1], 0, bdn.format.value[0])
        #In the worst case, there is a single composition object for the whole screen.
        screen_area = np.multiply(*bdn.format.value)
        epochstart_dd_fn = lambda o_area: max(PGDecoder.copy_gp_duration(screen_area), PGDecoder.decode_obj_duration(o_area)) + PGDecoder.copy_gp_duration(o_area)
        #Round up to tick
        epochstart_dd_fnr = lambda o_area: np.ceil(epochstart_dd_fn(o_area)*PGDecoder.FREQ)/PGDecoder.FREQ

        final_ds = None
        last_pts_out = None
        pcs_id = 0
        for group in bdn.groups(epochstart_dd_fn(screen_area)):
            if final_ds is not None:
                if TC.tc2s(group[0].tc_in, bdn.fps) - last_pts_out > 1.1:
                    logger.debug("Adding screen wipe since there was enough time between two epochs.")
                    self._epochs[-1].ds.append(final_ds)
                else:
                    #did not use an optional display set, subtract 1 to PCS id to have continuity
                    pcs_id -= 1

            subgroups = []
            areas = []
            offset = len(group)
            max_area = 0

            for k, event in enumerate(reversed(group)):
                if k == 0:
                    continue
                max_area = max(np.multiply(*event.shape), max_area)
                delay = TC.tc2s(group[len(group)-k].tc_in, bdn.fps) - TC.tc2s(event.tc_out, bdn.fps)

                if delay > epochstart_dd_fnr(max_area):
                    areas.append(max_area)
                    max_area = 0
                    subgroups.append(group[len(group)-k:offset])
                    offset -= len(subgroups[-1])
            if len(group[:offset]) > 0:
                areas.append(max_area)
                subgroups.append(group[:offset])
            else:
                assert offset == 0
            assert sum(map(len, subgroups)) == len(group)
            assert len(areas) == len(subgroups)

            #Epoch generation (each subgroup will be its own epoch)
            for ksub, subgroup in enumerate(reversed(subgroups), 1):
                logger.info(f"EPOCH {subgroup[0].tc_in}->{subgroup[-1].tc_out}, {len(subgroup)}->{len(subgroup := remove_dupes(subgroup))} event(s):")

                n_groups = 2 if (len(subgroup) > 1 or areas[-ksub]/screen_area > 0.1) else 1
                engine = GroupingEngine(Box.from_events(subgroup), container=container, n_groups=n_groups)
                box = engine.pad_box()
                windows = engine.group(subgroup)

                if logger.level <= 10:
                    for w_id, wd in enumerate(windows):
                        logger.debug(f"Window {w_id}: X={wd.x+box.x}, Y={wd.y+box.y}, W={wd.dx}, H={wd.dy}")
                else:
                    logger.info(f" => Screen layout: {len(windows)} window(s), processing...")

                wobz = WindowsAnalyzer(windows, subgroup, box, bdn, pcs_id=pcs_id, **self.kwargs)
                new_epoch, final_ds, pcs_id = wobz.analyze()
                self._epochs.append(new_epoch)
                logger.info(f" => optimised as {len(self._epochs[-1])} display sets.")
            gc.collect()
            last_pts_out = TC.tc2s(subgroups[0][-1].tc_out, bdn.fps)
        ####

        if final_ds is not None:
            logger.debug("Adding final displayset to the last epoch.")
            self._epochs[-1].ds.append(final_ds)

        scaled_fps = self.kwargs.get('scale_fps', False) and self.scale_pcsfps()

        # Final check
        logger.info("Checking stream consistency and compliancy...")
        final_fps = round(bdn.fps, 3) * int(1+scaled_fps)
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
    ####

    def scale_pcsfps(self) -> bool:
        pcs_fps = self._epochs[0].ds[0].pcs.fps.value
        real_fps = BDVideo.LUT_FPS_PCSFPS[pcs_fps]
        if (new_pcs_fps := BDVideo.LUT_PCS_FPS.get(2*real_fps, None)):
            for epoch in self._epochs:
                for ds in epoch.ds:
                    ds.pcs.fps = new_pcs_fps
            logger.info(f"Overwrote origin FPS {real_fps:.3f} to {2*real_fps:.3f} in stream.")
            return True
        else:
            logger.error(f"Expected input FPS of 25 or 29.97. Got '{BDVideo.LUT_FPS_PCSFPS[pcs_fps]}': no action taken.")
        return False
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
                fp_pes = ''.join(filepath[:-1]) + '.pes'
                fp_sup = ''.join(filepath[:-1]) + '.sup'
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
