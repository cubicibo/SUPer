#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""

from SUPer import TimeConv as TC, BDVideo, PGSTarget
from SUPer import SupStream, BDNXML, ImgSequence
from SUPer import PCS, WDS, PDS, ODS, ENDS, Epoch, CObject, WindowDefinition, DisplaySet
from SUPer import Optimise
from SUPer import PGraphics

from numpy import typing as npt
import numpy as np

def run():
    ##############
    ## CHANGE HERE

    # Video stream specifics
    target = PGSTarget(BDVideo.VideoFormat.HD1080, BDVideo.FPS.FILM_NTSC)

    seq = BDNXML('./examples/karaoke/bdn.xml').load_images()#('00:00:10:06', '00:00:13:01')
    #seq = ImgSequence('./examples/konosuba_op_animation/timing.csv').load_images()
    
    #if none, create new sup with just the optimised effect.
    original_sup = None #SupStream('./input.sup') # This is untested
    output_fname = './output3.sup'
    
    # colors: number of "sequences" used
    # norm_thresh: merge colors with dist(RGBA) smaller than value
    kwargs = {'colors': 256, 'norm_thresh': 20}
    
    ## CHANGE HERE
    ##############
    new_epochs = []
    
    for i_seq, animation in enumerate(seq):
        print(f"Optimising event {i_seq} : {animation[0].event.intc} -> {animation[0].event.outtc}")
        cmap, cluts = Optimise.solve_sequence(*Optimise.prepare_sequence(animation), **kwargs)
        
        # This should give the x,y position of bitmap. If wrong, overwrite those.
        target.h_pos, target.v_pos = animation[0].event.pos
        
        new_epochs.append(to_sup(target, cmap, cluts, animation))
    
    print("Writing to file...", end=' ')
    to_file(new_epochs, output_fname, original_sup)
    print("finished.")

#######################################
#######################################
def to_file(epochs: list[Epoch], outsup: str, insup: SupStream = None) -> None:
    st = bytearray()
    k = 0
    
    if insup is None:
        for epoch in epochs:
            for ds in epoch.ds:
                for seg in ds.segments:
                    st += seg._bytes
    else:
        for orig_epoch in insup.fetch_epoch():
            if k < len(epochs) and orig_epoch.t_in > epochs[k].t_in:
                for ds in epochs[k].ds:
                    for seg in ds.segments:
                        st += seg._bytes
                k+=1
            for ds in orig_epoch.ds:
                for seg in ds.segments:
                    st += seg._bytes
        
    with open(outsup, 'wb') as f:
        f.write(st)


def to_sup(pgs_target: PGSTarget, cmap: npt.NDArray[np.uint8], cluts: npt.NDArray[np.uint8], events):
    
    h, w = cmap.shape
    
    if pgs_target.h_pos < 0:
        pgs_target.h_pos = int(pgs_target.width//2-w//2)
    if pgs_target.v_pos < 0:
        pgs_target.v_pos = int(pgs_target.height-h-pgs_target.voffset)
        
    cobject = CObject.from_scratch(o_id=0, window_id=0, h_pos=pgs_target.h_pos,
                                   v_pos=pgs_target.v_pos, forced=pgs_target.forced)
    
    #Yes.
    pcs_fn = lambda cn,cs,pf,pts,dts=None,show=True : PCS.from_scratch(width=pgs_target.width,
                                                             height=pgs_target.height,
                                                             fps=pgs_target.fps,
                                                             composition_n=cn,
                                                             composition_state=cs,
                                                             pal_flag=pf,
                                                             pal_id=pgs_target.pal_id,
                                                             cobjects=[cobject]*show,
                                                             pts=pts, dts=dts)
    
    l_timestamps = [TC.tc2s(img.event.intc, pgs_target.readable_fps) for img in events]
    closing_ts = TC.tc2s(events[-1].event.outtc, pgs_target.readable_fps)
    
    l_pcs = [pcs_fn(k+1, PCS.CompositionState(0), True, ts) for k, ts in enumerate(l_timestamps[1:])]
    l_pcs.insert(0, pcs_fn(0, PCS.CompositionState.EPOCH_START, False, l_timestamps[0]))
    l_pcs.append(pcs_fn(len(l_pcs), PCS.CompositionState(0), False, closing_ts, show=False))
    
    l_pds = [PDS.from_scratch(pal, pts=ts) for ts, pal in zip(l_timestamps, Optimise.diff_cluts(cluts))]

    ods = ODS.from_scratch(0, 0, w, h, PGraphics.encode_rle(cmap), pts=l_timestamps[0])
    if type(ods) is not list:
        ods = [ods]
    
    window  = WindowDefinition.from_scratch(0, pgs_target.h_pos, pgs_target.v_pos, w, h)
    wds_in  = WDS.from_scratch([window], pts=l_timestamps[0])
    wds_out = WDS.from_scratch([], pts=closing_ts)
    
    ds = [DisplaySet([l_pcs[0], wds_in, l_pds[0], *ods, ENDS.from_scratch(l_pcs[0].pts)])]
    
    # for palette updates
    for pcs, pds in zip(l_pcs[1:-1], l_pds[1:]):
        ds.append(DisplaySet([pcs, pds, ENDS.from_scratch(pcs.pts)]))
    
    # Closing DS, clearing off display
    ds.append(DisplaySet([l_pcs[-1], wds_out, ENDS.from_scratch(l_pcs[-1].pts)]))
    
    return Epoch(ds)

if __name__ == '__main__':
    run()