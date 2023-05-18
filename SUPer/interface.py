#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2023 cibo
# This file is part of SUPer <https://github.com/cubicibo/SUPer>.
#
# SUPer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SUPer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SUPer.  If not, see <http://www.gnu.org/licenses/>.

import gc
import numpy as np
from os import path

from .utils import Shape, TimeConv as TC, _pinit_fn, get_super_logger
from .render2 import GroupingEngine, WOBSAnalyzer, is_compliant
from .filestreams import BDNXML, SUPFile

logger = get_super_logger('SUPer')

class BDNRender:
    def __init__(self, bdnf: str, kwargs: dict[str, int]) -> None:
        self.bdn_file = bdnf
        self._epochs = []
        self.skip_errors = kwargs.pop("skip_errors", False)
        #Leave norm threshold to zero, it can generate unexpected behaviours.
        #Colors should be 256. Anything above is illegal, anything below results in a
        # loss of quality.
        self.kwargs = {'colors': 256}
        self.kwargs |= kwargs

    def optimise(self) -> None:
        kwargs = self.kwargs
        stkw = ''
        stkw += ':'.join([f"{k}={v}" for k, v in kwargs.items()])
        logger.info(f"Parameters: {stkw}")

        bdn = BDNXML(path.expanduser(self.bdn_file))

        clip_framerate = bdn.fps
        if self.kwargs.pop('adjust_dropframe', False):
            if isinstance(bdn.fps, float):
                bdn.fps = round(bdn.fps)
                logger.info(f"NTSC NDF flag: using {bdn.fps} for timestamps rather than BDNXML {clip_framerate:.03f}.")
            else:
                logger.warning("Ignored NDF flag with integer framerate.")

        logger.info("Finding epochs...")

        #Empirical max: we need <=6 frames @23.976 to clear the buffers and windows.
        # This is doing coarse epoch definitions, without any consideration to
        # what's being displayed on screen.
        delay_refresh = 0.01+0.25*np.multiply(*bdn.format.value)/(1920*1080)
        for group in bdn.groups(delay_refresh):
            offset = len(group)-1
            subgroups = []
            last_split = len(group)
            largest_shape = Shape(0, 0)

            #Backward pass for fine epochs definition
            # We consider the delay between events and the size of the overall
            # graphic that we want to display.
            for k, event in enumerate(reversed(group[1:])):
                offset -= 1
                if np.multiply(*group[offset].shape) > np.multiply(*largest_shape):
                    largest_shape = event.shape
                nf = TC.tc2f(event.tc_in, bdn.fps) - TC.tc2f(group[offset].tc_out, bdn.fps)

                if nf > 0 and nf/bdn.fps > 3*_pinit_fn(largest_shape)/90e3:
                    subgroups.append(group[offset+1:last_split])
                    last_split = offset + 1
            if group[offset+1:last_split] != []:
                subgroups.append(group[offset+1:last_split])
            if subgroups:
                subgroups[-1].insert(0, group[0])
            else:
                subgroups = [[group[0]]]

            #Epoch generation (each subgroup will be its own epoch)
            for subgroup in reversed(subgroups):
                logger.info(f"Generating epoch {subgroup[0].tc_in}->{subgroup[-1].tc_out}...")
                wob, box = GroupingEngine(n_groups=2, **kwargs).group(subgroup)

                wobz = WOBSAnalyzer(wob, subgroup, box, clip_framerate, bdn, **kwargs)
                epoch = wobz.analyze()
                self._epochs.append(epoch)
                logger.info(f" => optimised as {len(epoch)} display sets on {len(wob)} window(s).")
            gc.collect()

        if clip_framerate != bdn.fps:
            adjustment_ratio = bdn.fps/round(clip_framerate, 2)
            for epoch in self._epochs:
                for ds in epoch:
                    for seg in ds:
                        seg.pts = seg.pts*adjustment_ratio - 3/90e3

        scaled_fps = False
        if self.kwargs.get('scale_fps', False):
            from SUPer.utils import BDVideo
            I_LUT_PCS_FPS = {v: k for k, v in BDVideo.LUT_PCS_FPS.items()}
            if (new_pcs_fps := BDVideo.LUT_PCS_FPS.get(2*I_LUT_PCS_FPS[self._epochs[0].ds[0].pcs.fps.value], None)):
                for epoch in self._epochs:
                    for ds in epoch.ds:
                        ds.pcs.fps = new_pcs_fps
                scaled_fps = True
            else:
                logger.error(f"Expexcted 25 or 30 fps for 2x scaling. Got '{I_LUT_PCS_FPS[self._epochs[0].ds[0].pcs.fps.value]}'.")

        # Final check
        is_compliant(self._epochs, bdn.fps * int(1+scaled_fps))

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
            with open(fp, 'wb') as f:
                for epoch in self._epochs:
                    f.write(bytes(epoch))
        else:
            raise RuntimeError("No data to write.")
 ####
