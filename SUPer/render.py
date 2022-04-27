#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""
from typing import Optional, Type
from numpy import typing as npt

import numpy as np
from PIL import Image
from SSIM_PIL import compare_ssim
from enum import IntEnum, auto
from dataclasses import dataclass

from .segments import DisplaySet, Epoch, PCS, CObject
from .filestreams import SeqIO
from .utils import BDVideo, Shape, TimeConv as TC

_pinit_fn = lambda shape: np.ceil(90e3*(shape.width*shape.height/(32*1e6)))

# As suggested in the US patent 2009/0185789 
def decode_duration(ds: DisplaySet, epoch: Optional[Epoch] = None) -> int:
    #pinit_fn = lambda shape: np.ceil(90e3*(shape.width*shape.height/(32*1e6)))
    
    def plane_initialization_time(ds: DisplaySet):
        if PCS.CompositionState.EPOCH_START & ds.pcs.composition_state:
            init_d = _pinit_fn(ds.pcs)
        else:
            init_d = 0
            for wds in ds.wds.windows:
                init_d += _pinit_fn(wds)
        return init_d

    def wait(ds: DisplaySet, obj: CObject, current_duration: int) -> int:
        wd = 0
        for ods in ds.ods:
            if ods.o_id == obj.o_id:
                obj_ready_time = obj.pts
                current_time = ds.pcs.dts + current_duration
                if current_time < obj_ready_time:
                    wd += obj_ready_time - current_time
                break
        return wd

    dd = plane_initialization_time(ds)
    
    if ds.pcs.n_objects > 0:
        dd += wait(ds, ds.pcs.cobjects[0], dd)
    
    if ds.pcs.n_objects == 2:
        if ds.pcs.cobjects[0].window_id == ds.pcs.cobjects[1].window_id:
            dd += wait(ds, ds.pcs.cobjects[1], dd)
            if epoch is not None:
                dd += _pinit_fn(epoch.get_wds(ds.pcs.cobjects[1].window_id), ds.pts)
            else:
                dd += _pinit_fn(ds.get_wds(ds.pcs.cobjects[1].window_id))
        else:
            if epoch is not None:
                dd += _pinit_fn(epoch.get_wds(ds.pcs.cobjects[0].window_id), ds.pts)
            else:
                dd += _pinit_fn(ds.get_wds(ds.pcs.cobjects[0].window_id))
            dd += wait(ds, ds.pcs.cobjects[1], dd)
            if epoch is not None:
                dd += _pinit_fn(epoch.get_wds(ds.pcs.cobjects[1].window_id), ds.pts)
            else:
                dd += _pinit_fn(ds.get_wds(ds.pcs.cobjects[1].window_id))
    elif ds.pcs.n_objects == 1:
        if epoch is not None:
            dd += _pinit_fn(epoch.get_wds(ds.pcs.cobjects[0].window_id), ds.pts)
        else:
            dd += _pinit_fn(ds.get_wds(ds.pcs.cobjects[0].window_id))
    return dd


class FadeType(IntEnum):
    FADE_IN = auto()
    FADE_OUT = auto()
    COMPLEX_FADE = auto()

@dataclass
class FadeEffect:
    type: FadeType
    duration: int
    coeffs: npt.NDArray[float]

def check_fade(imgs: list[Image]) -> Optional[list[FadeEffect]]:
    """
    For a given caption (estimated from object), this function seeks for a fade.
    This function applies https://doi.org/10.2991/iccasp-16.2017.55 but on alpha then
     use basic diff and math to find fade directions and duration.
    This function seeks for fade in and fade out at the beginning and end resp.
    """
    import pywt
    
    i_n = np.zeros((len(imgs),))
    for k, img in enumerate(imgs):
        c2 = pywt.dwt2(img.convert('RGBA').split()[-1], 'db6')[0]
        i_n[k] = np.sum(c2)/np.sum(c2.shape)
    
    i_n = np.divide(i_n, np.max(i_n))
    plateau = np.where(i_n == 1.0)[0]
    
    # No change in transparency or subtitle does not change at all
    if np.all(np.diff(i_n) == 0) or len(plateau) == len(i_n):
        return None # No fade effect at all, subtitle graphic is entirely constant
    
    fades = []
    
    # Distinguish between standard and complex fade so one can "linearize" the fade.
    # if complex fade, either use the coeff straight or use SUPer optimiser
    if np.all(plateau[1:]-plateau[:-1] == 1):
        f_fn = lambda ft: FadeEffect(ft, plateau[0], i_n[:plateau[0]])
        if np.all(np.diff(i_n[:plateau[0]]) > 0):
            fades.append(f_fn(FadeType.FADE_IN))
        else:
            fades.append(f_fn(FadeType.COMPLEX_FADE))
        
        f_fn = lambda ft: FadeEffect(ft, len(i_n) - plateau[-1], i_n[plateau[-1]:])
        if np.all(np.diff(i_n[plateau[-1]:]) < 0):
            fades.append(f_fn(FadeType.FADE_OUT))
        else:
            fades.append(f_fn(FadeType.COMPLEX_FADE))
    else:
        return FadeEffect(FadeType.COMPLEX_FADE, len(i_n), i_n)
        
def check_similar(imgs: list[Image]):
    # Measure RGB and RGBA similarity and decide if a split occur and a new object 
    # shoul dbe created.
    ...

def render_epochs(target: BDVideo, sequence: list[Type[SeqIO]]) -> npt.NDArray[np.uint8]:
    """
    This functions takes image events and put them on a plane
    """
    
    #This roughly performs grouping by epoch. If the said group does not fit within
    # an epoch it tries to apply SUPer optimiser over it. And if it is ugly/does not 
    # fit then it splits the group. The split can be ugly.
    
    for group in sequence.groups(4*_pinit_fn(Shape(*target.dim.value))/90e3+0.05):
        # Don't care about tc_out, graphics won't change once the time has passed
        # or if it does, it is by an automatic inline effect provided by BDNXML which
        # we will check later on.
        nf = abs(TC.tc2f(group[-1].tc_in, target.readable_fps) - \
                 TC.tc2f(group[ 0].tc_in, target.readable_fps))
        t0 = TC.tc2f(group[ 0].tc_in, target.readable_fps)
        # We will fill the region with True then perform some preliminary grouping
        #  (windowing & composition objects)
        omap = np.full((nf, *target.dim.value), False, dtype=bool)
        
        for event in group:
            f_pos = slice(TC.tc2f(event.tc_in , target.readable_fps) - t0, 
                          TC.tc2f(event.tc_out, target.readable_fps) - t0)
            x_pos = slice(event.x, event.x + event.width)
            y_pos = slice(event.y, event.y + event.height)
            omap[f_pos, x_pos, y_pos] = True
        
        
        
        
        #Check which event requires Optimisation or processing 
        for event in group:
            ...
            