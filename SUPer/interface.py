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

from .utils import Shape, TimeConv as TC, _pinit_fn, get_super_logger
from .pgraphics import PGDecoder
from .render2 import GroupingEngine, WOBSAnalyzer, is_compliant
from .filestreams import BDNXML, SUPFile

logger = get_super_logger('SUPer')

class BDNRender:
    def __init__(self, bdnf: str, kwargs: dict[str, int]) -> None:
        self.bdn_file = bdnf
        self._epochs = []
        self.skip_errors = kwargs.pop("skip_errors", False)
        self.kwargs = kwargs

    def optimise(self) -> None:
        kwargs = self.kwargs
        stkw = ''
        stkw += ':'.join([f"{k}={v}" for k, v in kwargs.items()])
        logger.info(f"Parameters: {stkw}")

        bdn = BDNXML(path.expanduser(self.bdn_file))

        if self.kwargs.get('scale_fps', False):
            if bdn.fps >= 50:
                logger.critical("BDNXML is not subsampled, aborting!")
                import sys
                sys.exit(1)

        clip_framerate = bdn.fps
        if self.kwargs.get('adjust_dropframe', False):
            if isinstance(bdn.fps, float):
                bdn.fps = round(bdn.fps)
                logger.info(f"NTSC timing flag: using {round(bdn.fps)} for timestamps rather than BDNXML {clip_framerate:.03f}.")
            else:
                self.kwargs['adjust_dropframe'] = False
                logger.warning("Ignored NDF flag with integer framerate.")

        logger.info("Finding epochs...")

        #In the worst case, there is a single composition object for the whole screen.
        screen_area = np.multiply(*bdn.format.value)
        epochstart_dd_fn = lambda o_area: max(PGDecoder.copy_gp_duration(screen_area), PGDecoder.decode_obj_duration(o_area)) + PGDecoder.copy_gp_duration(o_area)
        #Round up to tick
        epochstart_dd_fnr = lambda o_area: round(epochstart_dd_fn(o_area)*PGDecoder.FREQ)/PGDecoder.FREQ

        for group in bdn.groups(epochstart_dd_fn(screen_area)):
            subgroups = []
            offset = len(group)
            max_area = 0

            for k, event in enumerate(reversed(group[1:]), 1):
                max_area = max(np.multiply(*event.shape), max_area)

                delay = TC.tc2s(event.tc_in, bdn.fps) - TC.tc2s(group[len(group)-k-1].tc_out, bdn.fps)
                if epochstart_dd_fnr(max_area) <= delay:
                    subgroups.append(group[offset-k:offset])
                    offset -= len(subgroups[-1])
                    max_area = 0
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
                epoch = wobz.analyze()
                self._epochs.append(epoch)
                logger.info(f" => optimised as {len(epoch)} display sets.")
            gc.collect()

        scaled_fps = False
        if self.kwargs.get('scale_fps', False):
            scaled_fps = self.scale_pcsfps()

        # Final check
        is_compliant(self._epochs, bdn.fps * int(1+scaled_fps), self.kwargs.get('enforce_dts', True))
    ####

    def scale_pcsfps(self) -> bool:
        from SUPer.utils import BDVideo
        pcs_fps = self._epochs[0].ds[0].pcs.fps.value
        if (new_pcs_fps := BDVideo.LUT_PCS_FPS.get(2*BDVideo.LUT_FPS_PCSFPS[pcs_fps], None)):
            for epoch in self._epochs:
                for ds in epoch.ds:
                    ds.pcs.fps = new_pcs_fps
            scaled_fps = True
        else:
            logger.error(f"Expexcted 25 or 30 fps for 2x scaling. Got '{BDVideo.LUT_FPS_PCSFPS[pcs_fps]}'.")
        return scaled_fps

    def merge(self, input_sup) -> None:
        epochs = SUPFile(input_sup).epochs()
        if not self._epochs:
            self._epochs = epochs
        else:
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
            if fp.lower().endswith('pes'):
                writer = EsMuiStream.segment_writer(fp)
                next(writer) #init writer
                for epoch in self._epochs:
                    for ds in epoch:
                        for seg in ds:
                            writer.send(seg)
                # Close ESMUI writer
                writer.send(None)
                writer.close()
            else:
                if not fp.lower().endswith('sup'):
                    logger.warning("Unknown extension, assuming a .SUP file...")
                with open(fp, 'wb') as f:
                    for epoch in self._epochs:
                        f.write(bytes(epoch))
        else:
            raise RuntimeError("No data to write.")
 ####
