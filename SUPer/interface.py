#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 18:27:29 2022

@author: cibo
"""

import gc
import numpy as np
from os import path

from .utils import Shape, TimeConv as TC, _pinit_fn, get_super_logger
from .render import group_event_alikes, to_epoch2, is_compliant
from .filestreams import BDNXML, SupStream

logger = get_super_logger('SUPer')

class BDNRender:
    def __init__(self, bdnf: str, kwargs: dict[str, int]) -> None:
        self.bdn_file = bdnf
        self._epochs = []
        self.skip_errors = kwargs.pop("skip_errors", False)
        #Leave norm threshold to zero, it can generate unexpected behaviours.
        #Colors should be 256. Anything above is illegal, anything below results in a
        # loss of quality.
        self.kwargs = {'norm_thresh': 0, 'colors': 256}
        self.kwargs |= kwargs

    def optimise(self) -> None:
        kwargs = self.kwargs
        
        bdn = BDNXML(path.expanduser(self.bdn_file))
        logger.info("Finding epochs...")
        
        delay_refresh = 0.01+0.25*np.multiply(*bdn.format.value)/(1920*1080)
        
        for group in bdn.groups(delay_refresh):
            offset = len(group)-1
            subgroups = []
            last_split = len(group)
            largest_shape = Shape(0, 0)

            #Backward pass for fine epochs definition
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
                try:
                    regions_ods_mapping, box = group_event_alikes(subgroup, **kwargs)
                    outm, _ = to_epoch2(bdn, subgroup, regions_ods_mapping, box, **kwargs)
                except Exception as exc:
                    if self.skip_errors:
                        raise exc
                    logger.critical(f"!!! EXCEPTION: {exc}. Epoch starting at {subgroup[0].tc_in} was not generated !!!")
                else:
                    self._epochs.append(outm)
                    logger.info(f" => optimised as {len(outm.ds)} display sets.")
            gc.collect()

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

        if self.kwargs.get('adjust_dropframe', False):
            for epoch in self._epochs:
                for ds in epoch.ds:
                    for seg in ds.segments:
                        seg.pts = seg.pts/1.001
        # Final check
        is_compliant(self._epochs, bdn.fps * int(1+scaled_fps))

    def merge(self, input_sup) -> None:
        epochs = [epoch for epoch in SupStream(input_sup).epochs()]
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
