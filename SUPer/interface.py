#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 cibo
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

from .utils import TimeConv as TC, get_super_logger
from .pgraphics import PGDecoder
from .filestreams import BDNXML, SUPFile
from .optim import Quantizer

logger = get_super_logger('SUPer')

class BDNRender:
    def __init__(self, bdnf: str, kwargs: dict[str, int]) -> None:
        self.bdn_file = bdnf
        self._epochs = []
        self.kwargs = kwargs
        if self.kwargs.get('quantize_lib', Quantizer.Libs.PIL_CV2KM) == Quantizer.Libs.PILIQ:
            if not Quantizer.init_piliq(kwargs.get('libs_path', {}).get('quant', None)):
                logger.info("Failed to initialise advanced image quantizer. Falling back to PIL+K-Means.")
                self.kwargs['quantize_lib'] = Quantizer.Libs.PIL_CV2KM.value

    def optimise(self) -> None:
        from .render2 import GroupingEngine, WOBSAnalyzer, is_compliant

        kwargs = self.kwargs
        stkw = ''
        stkw += ':'.join([f"{k}={v}" for k, v in kwargs.items()])
        logger.info(f"Parameters: {stkw}")

        bdn = BDNXML(path.expanduser(self.bdn_file))
        fps_str = bdn.fps if float(bdn.fps).is_integer() else round(bdn.fps, 3)
        logger.info(f"BDN metadata: {'x'.join(map(str, bdn.format.value))}, FPS={fps_str}, DF={bdn.dropframe}, {len(bdn.events)} valid events.")

        if self.kwargs.get('scale_fps', False):
            if bdn.fps >= 50:
                logger.critical("BDNXML is not subsampled, aborting!")
                import sys
                sys.exit(1)

        clip_framerate = bdn.fps
        if isinstance(bdn.fps, float) and not bdn.dropframe:
            bdn.fps = round(bdn.fps)
            self.kwargs['adjust_dropframe'] = True
            logger.info(f"NDF NTSC detected: scaling all timestamps by 1.001.")

        logger.info("Finding epochs...")

        #In the worst case, there is a single composition object for the whole screen.
        screen_area = np.multiply(*bdn.format.value)
        epochstart_dd_fn = lambda o_area: max(PGDecoder.copy_gp_duration(screen_area), PGDecoder.decode_obj_duration(o_area)) + PGDecoder.copy_gp_duration(o_area)
        #Round up to tick
        epochstart_dd_fnr = lambda o_area: np.ceil(epochstart_dd_fn(o_area)*PGDecoder.FREQ)/PGDecoder.FREQ

        final_ds = None
        last_pts_out = None
        for group in bdn.groups(epochstart_dd_fn(screen_area)):
            if last_pts_out is not None and TC.tc2s(group[0].tc_in, bdn.fps) - last_pts_out > 1.5:
                logger.debug("Adding screen wipe since there was enough time between two epochs.")
                assert final_ds is not None
                self._epochs[-1].ds.append(final_ds)

            subgroups = []
            offset = len(group)
            max_area = 0

            for k, event in enumerate(reversed(group)):
                if k == 0:
                    continue
                max_area = max(np.multiply(*event.shape), max_area)
                delay = TC.tc2s(group[len(group)-k].tc_in, bdn.fps) - TC.tc2s(event.tc_out, bdn.fps)

                if delay > epochstart_dd_fnr(max_area):
                    max_area = 0
                    subgroups.append(group[len(group)-k:offset])
                    offset -= len(subgroups[-1])
            if len(group[:offset]) > 0:
                subgroups.append(group[:offset])
            else:
                assert offset == 0
            assert sum(map(len, subgroups)) == len(group)

            #Epoch generation (each subgroup will be its own epoch)
            for subgroup in reversed(subgroups):
                logger.info(f"Identified epoch {subgroup[0].tc_in}->{subgroup[-1].tc_out}:")

                wob, box = GroupingEngine(n_groups=2, **kwargs).group(subgroup)
                logger.info(f" => Screen layout: {len(wob)} window(s), analyzing objects...")

                wobz = WOBSAnalyzer(wob, subgroup, box, clip_framerate, bdn, **kwargs)
                new_epoch, final_ds = wobz.analyze()
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
        is_compliant(self._epochs, bdn.fps * int(1+scaled_fps), self.kwargs.get('enforce_dts', True))
    ####

    def scale_pcsfps(self) -> bool:
        scaled_fps = False
        from SUPer.utils import BDVideo
        pcs_fps = self._epochs[0].ds[0].pcs.fps.value
        real_fps = BDVideo.LUT_FPS_PCSFPS[pcs_fps]
        if (new_pcs_fps := BDVideo.LUT_PCS_FPS.get(2*real_fps, None)):
            for epoch in self._epochs:
                for ds in epoch.ds:
                    ds.pcs.fps = new_pcs_fps
            scaled_fps = True
            logger.info(f"Overwrote origin FPS {real_fps:.3f} to {2*real_fps:.3f} in stream.")
        else:
            logger.error(f"Expected 25 or 30 fps for 2x scaling. Got '{BDVideo.LUT_FPS_PCSFPS[pcs_fps]}'.")
        return scaled_fps

    def merge(self, input_sup) -> None:
        epochs = SUPFile(input_sup).epochs()
        if not self._epochs:
            self._epochs = epochs
        else:
            in_pcs = epochs[0][0].pcs
            out_pcs = self._epochs[0][0].pcs
            if in_pcs.width != out_pcs.width or in_pcs.height != out_pcs.height or in_pcs.fps != out_pcs.fps:
                logger.error("Video properties mismatch between BDNXML and SUP to inject. Not performing inject.")
                return
            # Merge input sup with new content
            for k, epoch in enumerate(epochs.copy()):
                cnt = 0
                while len(self._epochs) > 0 and epoch.t_in >= self._epochs[0].t_in:
                    epochs[k+cnt:k+cnt] = [self._epochs.pop(0)]
                    cnt += 1
            epochs.extend(self._epochs)
            self._epochs = epochs

    def write_output(self, fp: str) -> None:
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

                writer = EsMuiStream.segment_writer(fp_pes)
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
                    for epoch in self._epochs:
                        f.write(bytes(epoch))
        else:
            raise RuntimeError("No data to write.")
 ####
