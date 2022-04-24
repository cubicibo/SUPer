#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""
from typing import Optional
from numpy import ceil

from .segments import DisplaySet, Epoch, PCS, CObject

# As suggested in the US patent 2009/0185789 
def decode_duration(ds: DisplaySet, epoch: Optional[Epoch] = None) -> int:
    pinit_fn = lambda shape: ceil(90e3*(shape.width*shape.height/(32*1e6)))
    
    def plane_initialization_time(ds: DisplaySet):
        if PCS.CompositionState.EPOCH_START & ds.pcs.composition_state:
            init_d = pinit_fn(ds.pcs)
        else:
            init_d = 0
            for wds in ds.wds.windows:
                init_d += pinit_fn(wds)
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
                dd += pinit_fn(epoch.get_wds(ds.pcs.cobjects[1].window_id), ds.pts)
            else:
                dd += pinit_fn(ds.get_wds(ds.pcs.cobjects[1].window_id))
        else:
            if epoch is not None:
                dd += pinit_fn(epoch.get_wds(ds.pcs.cobjects[0].window_id), ds.pts)
            else:
                dd += pinit_fn(ds.get_wds(ds.pcs.cobjects[0].window_id))
            dd += wait(ds, ds.pcs.cobjects[1], dd)
            if epoch is not None:
                dd += pinit_fn(epoch.get_wds(ds.pcs.cobjects[1].window_id), ds.pts)
            else:
                dd += pinit_fn(ds.get_wds(ds.pcs.cobjects[1].window_id))
    elif ds.pcs.n_objects == 1:
        if epoch is not None:
            dd += pinit_fn(epoch.get_wds(ds.pcs.cobjects[0].window_id), ds.pts)
        else:
            dd += pinit_fn(ds.get_wds(ds.pcs.cobjects[0].window_id))
    return dd
        
def render_epoch():
    ...