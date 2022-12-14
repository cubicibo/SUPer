#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 cibo 
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

from typing import Optional, Union
import numpy as np
from numpy import typing as npt
from PIL import Image

from skimage.measure import regionprops
from skimage.measure import label
from skimage.filters import gaussian
from sklearn.cluster import KMeans

from anytree import Node, findall_by_attr

#%%
from .segments import DisplaySet, Epoch, PCS, CObject, PDS, ODS, ENDS, WindowDefinition, WDS
from .pgraphics import PGraphics
from .optim import Optimise
from .filestreams import BaseEvent, BDNXML, BDNXMLEvent
from .utils import min_enclosing_cube, Dim, Pos, _pinit_fn, BDVideo , TimeConv as TC, get_super_logger
from .palette import Palette

logging = get_super_logger('SUPer')

####### We need to fix the get_wds function missing when called with an Epoch
# As suggested in the US patent 2009/0185789 
def decode_duration(ds: DisplaySet, epoch: Optional[Epoch] = None) -> int:    
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

#%%
def box_hull(region: Union[tuple[Pos, Dim], tuple[slice, slice]],
             hull: Optional[tuple[Pos, Dim]] = None) -> tuple[Pos, Dim]:
    """
    This works but it would be better to use appropriately windows to  make the new hulls
    """
    if isinstance(region, tuple) and isinstance(region[0], Pos) and isinstance(region[1], Dim):
        if hull is None:
            return region
        #Are there no other way to make a stupid namespace??
        class Generic: ...
        rg = Generic()
        rg.slice = (..., slice(region[0].y, region[0].y+region[1].h),
                        slice(region[0].x, region[0].x+region[1].w))
        rg._label_image = Generic()
        rg._label_image.shape = [..., 2400, 2400]
        region = rg
        
    if hull is None:
        return (Pos(region.slice[2].start, region.slice[1].start),
                Dim(region.slice[2].stop - region.slice[2].start,
                    region.slice[1].stop - region.slice[1].start))
    a = np.zeros((2400, 2400), dtype=np.uint16)
    a[region.slice[1:]] = 1
    a[hull[0].y:hull[0].y+hull[1].h, hull[0].x:hull[0].x+hull[1].w] = 1
    
    vocc = np.where(a.sum(axis=1) > 0)[0]
    hocc = np.where(a.sum(axis=0) > 0)[0]
    return (Pos(hocc[0], vocc[0]), Dim(hocc[-1]-hocc[0]+1, vocc[-1]-vocc[0]+1))
    
#%%
def group_event_alikes(group: list[BaseEvent], **kwargs) -> list[list[list[...]]]: #YES
    regions, gs_map, gs_origs, box = coarse_grouping(group, **kwargs)        
    #Refine groups from the blurs (find smallest bounding box - agressively)
    for region in regions:
        cntXl = 0
        while np.all(gs_origs[region.slice[0], region.slice[1],
                              region.slice[2].start+cntXl:region.slice[2].start+1+cntXl] == 0):
            cntXl += 1
        
        cntXr = -1
        while np.all(gs_origs[region.slice[0], region.slice[1],
                              region.slice[2].stop+cntXr:region.slice[2].stop+1+cntXr] == 0):
            cntXr -= 1
        cntXr += 1
        cntYt = 0
        while np.all(gs_origs[region.slice[0], region.slice[1].start+cntYt:region.slice[1].start+cntYt+1,
                              region.slice[2]] == 0):
            cntYt += 1
        
        cntYb = -1
        while np.all(gs_origs[region.slice[0], region.slice[1].stop+cntYb:region.slice[1].stop+cntYb+1,
                              region.slice[2]] == 0):
            cntYb -= 1
        cntYb += 1
        
        region.slice = tuple([region.slice[0],
                             slice(region.slice[1].start+cntYt, region.slice[1].stop+cntYb),
                             slice(region.slice[2].start+cntXl, region.slice[2].stop+cntXr)])
    
    ebox = np.zeros((len(regions), *gs_map.shape[1:]), dtype=np.uint8)
    for k, region in enumerate(regions):
        ebox[k, region.slice[1], region.slice[2]] = 1
    
    ods_reserved = np.zeros((len(regions),len(gs_origs)), dtype=np.uint8)
    ods_id = -1*np.ones((len(regions),), dtype=np.uint8)
    ods_id[0] = 0 #First object always get ods 0
    ods_metadata = {0: box_hull(regions[0])}
    overlap = np.ones((len(gs_origs),), dtype=np.uint8)
    
    for k, region in enumerate(regions):
        did_set = []
        for l, other_region in enumerate(regions):
            if l == k:
                continue
            if np.any(ebox[l, region.slice[1], region.slice[2]]):
                if ods_id[k] == -1 and ods_id[l] >= 0:
                    ods_id[k] = ods_id[l]
                elif ods_id[k] == -1 and ods_id[l] == -1:
                    did_set.append(l)
                else:
                    ods_id[l] = ods_id[k]
                    ods_metadata[ods_id[k]] = box_hull(region, ods_metadata.get(ods_id[k], None))
        # New location on screen
        if -1 == ods_id[k]:
            ods_id[k] = np.max(ods_id)+1
            ods_metadata[ods_id[k]] = box_hull(region)
            for pset in did_set:
                ods_id[pset] = ods_id[k]
                ods_metadata[ods_id[k]] = box_hull(regions[pset], ods_metadata.get(ods_id[k], None))
        ods_reserved[ods_id[k], region.slice[0]] = 1
        overlap = overlap & ods_reserved[ods_id[k]]
    
    #A bit hacky
    if np.sum(overlap) == len(overlap):
        return ([[regions]], box)
    
    #Perform hard grouping. Higher quality output but constraining
    candidates = {}
    if np.max(ods_id) >= kwargs.get('num_ods_max', 2) and len(np.unique(ods_id)) != len(ods_id): #if we have more than 2 objects on screen, we need to be careful.
        for k, odsr in enumerate(ods_reserved):
            if np.size(np.unique(ods_reserved)) == 1:
                return ([[regions]], box)
            if np.all(odsr == 0):
                continue
            for l, other_odsr in enumerate(ods_reserved[k+1:]):
                if np.all(other_odsr == 0):
                    continue
                if hard_merge(odsr, other_odsr):
                    candidates[l+k+1] = candidates.get(l+k+1, []) + [(k, odsr | other_odsr)]
    
    elif np.max(ods_id) == 1 and hard_merge(ods_reserved[0], ods_reserved[1]):
        #If the two subtitles overlap by more than 30% of the time -> merge together in one bitmap
        if np.sum(ods_reserved[0].astype(np.bool_) & ods_reserved[1].astype(np.bool_)) > 0.3*len(ods_reserved[0]) \
            or kwargs.get('merge_nonoverlap', False):
            return ([[regions]], box) # Both ODS are compatible together -> merge because this is cheaper
        
    elif len(ods_id) == 1 or len(ods_id) > 1 and np.max(ods_id) == 0:
        return ([[regions]], box)
    
    elif len(np.unique(ods_id)) == len(ods_id):
        if np.sum(overlap) >= np.floor(0.8*len(overlap)):
            return [[regions]], box

        coords = np.zeros((len(regions), 3), dtype=np.uint16)

        for k, region in enumerate(regions):
            slt, sly, slx = region.slice
            coords[k] = np.asarray((80*(slt.stop-slt.start)/2, (sly.stop-sly.start)/2, (slx.stop-slx.start)/2), dtype=np.uint32)
            coords[k] += np.asarray((80*slt.start, sly.start, slx.start), dtype=np.uint32)


        kmeans = KMeans(kwargs.get('num_ods_max', 2), n_init=10).fit(coords)
        potential = np.zeros((kwargs.get('num_ods_max', 2), len(gs_origs)), dtype=np.uint8)
        regions_mapped = [[[]], [[]]]
        
        labels_ = kmeans.labels_.copy()

        assert kwargs.get('num_ods_max', 2) <= 2, "Next statements need to be modified to support N(ODS) > 2."

        if regions[np.argwhere(labels_ == 0)[0][0]].slice[0].start > regions[np.argwhere(labels_ == 1)[0][0]].slice[0].start:
            labels_ = 1 - labels_
        
        #hulls = [box_hull(regions[np.argwhere(kmeans.labels_ == k)[0][0]], None) for k in range(kwargs.get('num_ods_max', 2))],
        hulls = [box_hull(regions[np.argwhere(labels_ == 0)[0][0]], None),
                 box_hull(regions[np.argwhere(labels_ == 1)[0][0]], None)]
        
        for k, label_id in enumerate(labels_):
            potential[label_id] = potential[label_id] | ods_reserved[k]
            regions_mapped[label_id][0].append(regions[k])
            hulls[label_id] = box_hull(regions[k], hulls[label_id])
        
        area_max = np.zeros((len(hulls),))
        for k, hull in enumerate(hulls):
            area_max[k] = np.multiply(*hull[1])
        area_max = np.sum(area_max)
        
        if kwargs.get('merge_nonoverlap', False) \
            or np.sum(potential[0].astype(np.bool_) & potential[1].astype(np.bool_)) > 35*(1-2.5*area_max/(1920*1080)):
            logging.debug("Merging all events to a single bitmap.")
            return ([[regions]], box)
        else:
            logging.debug("Merged with KMeans clusters.")
            return (regions_mapped, box)

    ret = find_best_groups(cross_merge(ods_id, candidates, ods_reserved),
                            regions, ods_reserved, ods_metadata, ods_id)
    return ret, box

#%%
def hard_merge(odsr, other_odsr):
    nt_ours  = np.sum(np.diff(odsr).astype(np.bool_))
    nt_theirs= np.sum(np.diff(other_odsr).astype(np.bool_))
    idx_ours = np.where(odsr==1)[0]
    idx_theirs=np.where(other_odsr==1)[0]
    first_id = np.max([idx_ours[0], idx_theirs[0]])
    last_id  = np.min([idx_ours[-1], idx_theirs[-1]])
    
    #We use diff to find the number of transitions, as we can merge events that do not start
    # exactly at the same frame
    resulting_transitions = np.zeros(odsr.shape, dtype=np.bool_)
    resulting_transitions[first_id:last_id] = ~(odsr[first_id:last_id] & other_odsr[first_id:last_id]).astype(np.bool_)
    
    if np.sum(np.diff(~resulting_transitions & odsr).astype(np.bool_)) == nt_ours:
        if np.sum(np.diff(~resulting_transitions & other_odsr).astype(np.bool_)) == nt_theirs:
            return True
    return False

#%%
def cross_merge(ods_id, candidates, ods_reserved, **kwargs):
    """
    This function is the tricky part, we have to merge all these events to a limited number
    of objects (2 on screen for Blu-Ray). The key issue is finding:
        -compatible merge one-2-one (done before)
        -compatible merge all together (done here) - this use a binary tree
        -the merge that causes the smallest stress on the buffer.
           (chosen using all leaves at the bottom of tree)
    """
    histogram = np.zeros((np.max(ods_id)+1,), dtype=np.uint8)
    for id_other, candidate_merges in candidates.items():
        for id_main, candidate_merge in candidate_merges:   
            histogram[id_other] += 1
    
    num_ods_max = kwargs.get('num_ods_max', 2)
    assert len(histogram) - np.count_nonzero(histogram) <= num_ods_max,\
        "Too many different overlapping objects over time. Please synchronise more in/out times"\
        " or move slightly subtitles so they spatially overlap less with previous/next lines."\
        " THIS IS NOT A BUG, WHAT YOU WANT TO RENDER IS IMPOSSIBLE!"
    
    #Pick the base ODS to which we will merge all other events
    event_chain = Node(name='root', value=ods_reserved[np.any(ods_reserved, axis=1)][histogram==0])
    ods_ids_used = list(np.where(histogram==0)[0])
    event_chain.chain = [[k] for k in np.where(histogram==0)[0]]
    
    depth=0

    for ods_idx in ods_id:
        if ods_idx in ods_ids_used:
            continue
        for leaf in findall_by_attr(event_chain, depth, 'depth'): 
            for o_k, main_ods_id in enumerate(ods_ids_used):
                if hard_merge(leaf.value[o_k], ods_reserved[ods_idx]):
                    event_masks = leaf.value.copy()
                    event_masks[o_k] = event_masks[o_k] | ods_reserved[ods_idx]
                    node = Node(f"{ods_idx}on{main_ods_id}", leaf, value=event_masks)
                    node.chain = [ls.copy() for ls in leaf.chain]
                    node.chain[o_k].append(ods_idx)
        # Did not add a layer while we try to add ods to base ones, this is bad
        assert depth < event_chain.height, "Too many different objects overlap in time,"\
            " they cannot be merged to final objects without corruption or flickering."\
            " THIS IS NOT A BUG, WHAT YOU WANT TO RENDER IS IMPOSSIBLE WITHOUT HEAVY CORRUPTION."
        depth += 1

    return event_chain

#%%
def count_ods_updates(objplane, c_hull, regions, ods_id):
    updates = 1 #Showing the ODS on screen is an update
    emap = np.zeros((2000, 2000), dtype=np.uint8)
    groups, current_group = [], []
    prev_range = np.arange(-1, 0)
    isFirst = True
    for k, ods_idx in enumerate(ods_id):
        if ods_idx not in objplane:
            continue
        
        new_range = np.arange(regions[k].slice[0].start, regions[k].slice[0].stop)
        
        if np.any(emap[regions[k].slice[1:]]) or (np.intersect1d(prev_range, new_range).size == 0 and not isFirst):
            emap[:] = 0 #ODS update, clear plane
            updates += 1
            groups.append(current_group)
            current_group = []
            prev_range = new_range
        else:
            prev_range = np.union1d(new_range, prev_range)
        current_group.append(regions[k])
        emap[regions[k].slice[1:]] = 1
        isFirst = False
    if current_group:
        groups.append(current_group)
    return updates, groups

#%%    
def find_best_groups(tree, regions, ods_reserved, ods_metadata, ods_id):
    #We've got the end leaves, they contain the partial possible sequences -> ODS mappings  
    """
    This function is not really good as it does not take into account any time the plane
    may not be refreshed. This needs to be implemented before hand.
    """
    best_area = np.inf
    best_groups = []
    for leaf in findall_by_attr(tree, tree.height, 'depth'):
        areas = np.zeros((len(leaf.chain),))
        n_refresh = np.zeros((len(leaf.chain),))
        groups = [None] * len(leaf.chain)
        for k, objplane in enumerate(leaf.chain):
            c_hull = None
            for obj in objplane:
                c_hull = box_hull(ods_metadata[obj], c_hull)
            areas[k] = np.multiply(*c_hull[1])
            
            n_refresh[k], groups[k] = count_ods_updates(objplane, c_hull, regions, ods_id)
        if best_area > np.sum(np.multiply(areas, n_refresh)):
            best_area = np.sum(np.multiply(areas, n_refresh))
            best_groups = groups
    # We did EVERYTHING to get this damn best_groups
    return best_groups

#%%
def merge_events(group: list[BaseEvent], pos, dim, out_format: str = 'RGBA') -> Image.Image:
    img_plane = np.zeros((dim[1], dim[0], 4), dtype=np.uint8)
    for k, event in enumerate(group):
        slice_x = slice(event.x-pos.x, event.x-pos.x+event.width)
        slice_y = slice(event.y-pos.y, event.y-pos.y+event.height)
        img_plane[slice_y, slice_x, :] = event.astype(np.uint8)
    return Image.fromarray(img_plane).convert(out_format)

#%%
def coarse_grouping(group, blur_mul=1, blur_c=1.5, **kwargs):
    no_blur = kwargs.get('noblur_grouping', False)
    
    # SD content should be blurred with lower coeffs. Remove constant.
    if no_blur:
        blur_c = 0.0
        blur_mul = 1
    
    ptl, dims = min_enclosing_cube(group)
    (pxtl, pytl), (w, h) = ptl, dims
    ratio_woh = abs(w/h)
    ratio_how = 1/ratio_woh if 1/ratio_woh <= 1 else 1
    ratio_woh = ratio_woh if ratio_woh <= 1.3 else 1.3

    ne_imgs = []
    for event in group:
        imgg = np.asarray(event.img.getchannel('A'), dtype=np.uint8)
        img_blurred = (255*gaussian(imgg, (blur_c + blur_mul*ratio_how, blur_c + blur_mul*ratio_woh)))
        img_blurred[img_blurred <= 0.5] = 0
        img_blurred[img_blurred > 0.5] = 1
        ne_imgs.append(img_blurred)
        
    gs_graph = np.zeros((len(group), h, w), dtype=np.uint8)
    gs_orig = np.zeros((len(group), h, w), dtype=np.uint8)
    for k, (event, b_img) in enumerate(zip(group, ne_imgs)):
        slice_x = slice(event.x-pxtl, event.x-pxtl+event.width)
        slice_y = slice(event.y-pytl, event.y-pytl+event.height)
        gs_graph[k, slice_y, slice_x] = b_img.astype(np.uint8)
        gs_orig[k, slice_y, slice_x] = np.array(event.img.getchannel('A'))
    p = regionprops(label(gs_graph))
    return p, gs_graph, gs_orig, (ptl, dims)
####

def chain_epochs(epochs: list[Epoch]) -> Epoch:
    eo = []
    for k, tepoch in enumerate(epochs):
        #if we have more than one epoch, the ones at position n > 1 are all acquisitions
        if k > 0:
            tepoch.ds[0].pcs.composition_state = PCS.CompositionState.ACQUISITION
        else:
            assert tepoch.ds[0].pcs.composition_state & PCS.CompositionState.EPOCH_START
        #Remove END segments that are overwritten by the ACQUISITIONs.
        if eo != [] and eo[-1].pcs.pts == tepoch.ds[0].pcs.pts:
            prev = eo.pop()
            assert len(prev.segments) == 3 and prev.segments[1].type == 'WDS', "Popped actual data, oops!"
        eo.extend(tepoch.ds)
    return eo

def get_ods_dim(ods_region, box) -> tuple[list[range], Pos, Dim]:
    time_act_list = []
    posbr = np.array([0, 0])
    postl = np.array([np.inf, np.inf])

    for ods_object in ods_region:
        time_act = np.array([np.inf, -np.inf])
        for area in ods_object:
            time_act[:] = np.min([time_act[0], area.slice[0].start]), np.max([time_act[1], area.slice[0].stop])
            sly, slx = area.slice[1:]
            postl[:] = np.min([postl[0], slx.start]), np.min([postl[1], sly.start])
            posbr[:] = np.max([posbr[0], slx.stop]), np.max([posbr[1], sly.stop])           
        time_act_list.append(range(*time_act.astype(np.uint16)))
    postl = (postl + [box[0].x-1, box[0].y]).astype(np.uint16)
    posbr = (posbr + [box[0].x, box[0].y]).astype(np.uint16)
    return (time_act_list, Pos(*postl), Dim(*(posbr-postl)))

def is_compliant(epochs: list[Epoch], fps: float, *, _cnt_pts: bool = False) -> bool:
    prev_pts = -1
    last_cbbw = 0
    last_dbbw = 0
    compliant = True
    warnings = 0
    
    coded_bw_ra_pts = [-1] * round(fps)
    coded_bw_ra = [0] * round(fps)
    
    for ke, epoch in enumerate(epochs):
        ods_acc = 0
        window_area = {}

        for kd, ds in enumerate(epoch.ds):
            size_ds = 0
            decoded_this_ds = 0
            coded_this_ds = 0

            current_pts = ds.pcs.pts
            if epoch.ds[kd-1].pcs.pts != prev_pts and current_pts != epoch.ds[kd-1].pcs.pts:
                prev_pts = epoch.ds[kd-1].pcs.pts
                last_cbbw, last_dbbw, last_rc = [0]*3
            else:
                logging.warning(f"Two displaysets at {current_pts} [s] (internal rendering error?)")
                
            for seg in ds.segments:
                size_ds += len(bytes(seg))
                n_obj = 0
                if seg.pts != current_pts and current_pts != -1 and _cnt_pts:
                    logging.warning(f"Display set has non-constant pts at {seg.pts} or {current_pts} [s].")
                    current_pts = -1
                if seg.type == 'PCS' and int(seg.composition_state) != 0:
                    # On acquisition, the object buffer is flushed
                    ods_acc = 0
                    n_obj = len(seg.cobjects)
                elif seg.type == 'WDS':
                    for w in seg.windows:
                        window_area[w.window_id] = w.width*w.height
                elif seg.type == 'ODS' and int(seg.flags) & int(ODS.ODSFlags.SEQUENCE_FIRST):
                    decoded_this_ds += seg.width * seg.height
                    coded_this_ds += seg.rle_len
                elif seg.type == 'PDS':
                    if n_obj > 1 and seg.pal_flag:
                        logging.warning(f"Undefined behaviour: palette update with 2+ objects at {seg.pts}.")
                        compliant = False
                    if seg.p_id >= 8:
                        logging.warning(f"Using an undefined palette ID at {seg.pts} [s].")
                        compliant = False
                elif seg.type == 'END' and n_obj == 0 and ds.pcs.pal_flag \
                    and int(ds.pcs.composition_state) == 0 and ds.segments[1].type == 'WDS':
                    logging.warning(f"Bad END segment, graphics may not be undisplayed properly at {seg.pts} [s].")
    
            ####
            ods_acc += decoded_this_ds
            coded_this_ds *= 8
            decoded_this_ds *= 8
            
            coded_buffer_pts = last_cbbw + coded_this_ds
            decoded_buffer_pts = last_dbbw + decoded_this_ds
            
            if prev_pts != seg.pts:
                coded_buffer_bandwidth = coded_buffer_pts/abs(seg.pts-prev_pts)
                decoded_buffer_bandwidth = decoded_buffer_pts/abs(seg.pts-prev_pts)
                last_cbbw, last_dbbw = 0, 0
            else:
                # Same PTS, we can't do any calculation -> accumulate to next PTS
                last_cbbw = coded_buffer_pts
                last_dbbw = decoded_buffer_pts
                coded_buffer_bandwidth, decoded_buffer_bandwidth = 0, 0
                
            # This is probably the hardest constraint to meet: ts_packet are read at, at most Rx=16Mbps
            if coded_buffer_bandwidth > (max_rate := 16*(1024**2)):
                if coded_buffer_bandwidth/max_rate >= 2:
                    logging.warning(f"High instantaneous coded bandwidth at {seg.pts:.03f} [s] (not critical - fair warning)")
                else:
                    logging.info(f"High coded bandwidth at {seg.pts:.03f} [s] (not critical - fair warning).")
                # This is not an issue unless it happens very frequently, so we don't mark as not compliant
            
            if prev_pts != seg.pts:
                coded_bw_ra = coded_bw_ra[1:round(fps)]
                coded_bw_ra_pts = coded_bw_ra_pts[1:round(fps)]
                coded_bw_ra.append(coded_buffer_pts)
                coded_bw_ra_pts.append(seg.pts)
            
            if (rate:=sum(coded_bw_ra)/abs(coded_bw_ra_pts[-1]-coded_bw_ra_pts[0])) > (max_rate:=16*(1024**2)):
                logging.warning(f"Exceeding coded bandwidth at ~{seg.pts:.03f} [s] {100*rate/max_rate:.03f}%.")
                warnings += 1
            
            if decoded_buffer_bandwidth > 128*(1024**2):
                logging.warning(f"Exceeding decoded buffer bandwidth at {seg.pts} [s].")
                compliant = False
                
            # Decoded object plane is 4 MiB
            if ods_acc >= 4*(1024**2):
                logging.warning(f"Decoded obect buffer overrun at {seg.pts} [s].")
                compliant = False
            
            #We clear the plane (window area) and copy the objects to window. This is done at 32MiB/s
            Rc = fps*(sum(window_area.values()) + np.min([ods_acc, sum(window_area.values())]))
            nf = TC.s2f(seg.pts, fps) - TC.s2f(prev_pts, fps)
            if nf == 0:
                last_rc += Rc 
            elif (last_rc+Rc)/nf > 1920*1080/4*29.97*2:
                logging.warning(f"Graphic plane overloaded. Graphics may flicker at {seg.pts} [s].")
                warnings += 1
            
    if warnings == 0 and compliant:
        logging.info("Output PGS stream seems compliant.")
    if warnings > 0 and compliant:
        logging.warning(f"PGStream is pushing the limits. Requires HW testing ({warnings} warnings).")
    elif not compliant:
        logging.error("PGStream is not compliant. Will crash a HW decoder.")
    return compliant

def render(bdn, group, time_box, nb_ods_onscreen, **kwargs) -> list[BaseEvent]:
    time_ranges, pos, dim = time_box
    graphics = []
    n_colors = kwargs.get('colors')

    for time_range in time_ranges:
        nev = []
        for k, event in enumerate(group[time_range.start:time_range.stop]):
            if nev != [] and (nb_ods_onscreen[time_range.start+k] > 1 or nb_ods_onscreen[time_range.start+k-1] != nb_ods_onscreen[time_range.start+k]):
                gfx_colors = 256 if nb_ods_onscreen[time_range.start+k-1] == 0 else int(256/nb_ods_onscreen[time_range.start+k-1])
                kwargs['colors'] = gfx_colors
                graphics.append((Optimise.solve_sequence(*Optimise.prepare_sequence(nev, **kwargs), **kwargs), nev, gfx_colors, (pos, dim)))
                nev = []
            
            img_fs = np.zeros((*bdn.format.value[::-1], 4), dtype=np.uint8)
            img_fs[event.y:event.y+event.height, event.x:event.x+event.width, :] = np.asarray(event.img).astype(np.uint8)
            img = Image.fromarray(img_fs[pos.y:pos.y+dim.h, pos.x:pos.x+dim.w].astype(np.uint8), mode='RGBA')
            props = (Pos(event.x+pos.x, event.y+pos.y), Dim(*img.size))
            nev.append(BDNXMLEvent.copy_custom(event, img, props))
            
            if np.sum(np.sum(np.asarray(img).astype(np.uint32))) == 0:
                logging.warning(f"Renderer produced an additional empty frame at {event.tc_in}. Output should be OK.")
                nev.pop()
        ####for k, event
        if nev != []:
            gfx_colors = int(256/nb_ods_onscreen[time_range.start+k])
            kwargs['colors'] = gfx_colors
            graphics.append((Optimise.solve_sequence(*Optimise.prepare_sequence(nev, **kwargs), **kwargs), nev, gfx_colors, (pos, dim)))

    kwargs['colors'] = n_colors
    return graphics

#%%
def to_epoch2(bdn, group, regions_ods_mapping, box, **kwargs):
    def merge_epochs(epochs:list[Epoch]) -> Epoch:
        pcs_cnt, pds_cnt, p_vn = [0]*3
        window_map = []
        lds = []
        sds = epochs.copy()
        
        while np.any(np.array(sds, dtype=object)):
            collection = []
            first_pts = np.inf
            for l, ds in enumerate(sds):
                if not ds:
                    continue
                if ds[0].pcs.pts < first_pts:
                    first_pts = ds[0].pcs.pts
 
            for l, ds in enumerate(sds):
                if not ds:
                    continue
                if ds[0].pcs.pts == first_pts:
                    collection.append(ds.pop(0))
            # collected all equal PTS, generate Display Set
            
            #Two epochs at pts, merge them together
            if len(collection) > 1:
                pcs_new = PCS.from_scratch(**collection[0].pcs.__dict__)
                
                segs = {}
                for ds in collection:
                    for seg in ds.segments:
                        segs[seg.__class__] = segs.get(seg.__class__, []) + [seg]
                
                out_ds_dict = {}
                for seg_family, items in segs.items():
                    if seg_family._NAME == 'END':
                        #All end Segments are the same, take the first in the list
                        out_ds_dict['END'] = items[0]
                        continue
                    elif seg_family._NAME == 'ODS':
                        #Dump all ODS at once
                        out_ds_dict['ODS'] = [seg for seg in items]
                    elif seg_family._NAME == 'PDS':
                        # Merge all palettes using dict oring to one palette def
                        palette_dict = {}
                        for pds in items:
                            palette_dict |= pds.to_palette().palette
                        palette = Palette(0, 0, palette_dict)
                        out_ds_dict['PDS'] = [PDS.from_scratch(palette, pts=pcs_new.pts)]
                    elif seg_family._NAME == 'PCS':
                        # For PCS, we need to fill the composition objects
                        pcs_new.cobjects = []
                        for item in items:
                            for ncob in item.cobjects:
                                nf = True
                                for pcsnco in pcs_new.cobjects:
                                    if pcsnco.__dict__ == ncob.__dict__:
                                        nf = False
                                        break
                                if nf:
                                    pcs_new.cobjects.append(ncob)
                        pcs_new.update() # Update cobjects count and bytes
                    elif seg_family._NAME == 'WDS':
                        wds_new = WDS.from_scratch([], pts=pcs_new.pts)
                        for item in items:
                            for nwin in item.windows:
                                nf = True
                                for wdsnw in wds_new.windows:
                                    w1 = list(nwin.__dict__.values())[1:]
                                    w2 = list(wdsnw.__dict__.values())[1:]
                                    if w1 == w2:
                                        nf = False
                                        break
                                if nf: 
                                    wds_new.windows.append(nwin)
                        wds_new.update()
                        out_ds_dict['WDS'] = [wds_new]
                out_ds = [pcs_new] + out_ds_dict.get('WDS', []) + out_ds_dict.get('PDS', [])
                if out_ds_dict.get('ODS', False):
                    out_ds.extend(out_ds_dict['ODS'])
                out_ds.append(out_ds_dict['END'])
                out_ds = DisplaySet(out_ds)                
            else:
                out_ds = collection[0]
                            
            if out_ds.segments[1].type == 'WDS':
                for window in out_ds.segments[1].windows:
                    nf = True
                    for other_window in window_map:
                        w1 = list(other_window.__dict__.values())[1:]
                        w2 = list(window.__dict__.values())[1:]
                        if w1 == w2:
                            nf = False
                            break
                    if nf: 
                        window_map.append(window)
            
            if len(out_ds.segments) > 3 and lds != []:
                out_ds.pcs.composition_state = PCS.CompositionState.ACQUISITION
            lds.append(out_ds)
        ####while
        
        # Some corrections and asserts
        lds_out = []
        for ds in lds:
            seq = []
            pcs = ds.pcs
            addOut = True
            for seg in ds.segments.copy():
                seq.append(seg.type)
                if seg.type == 'PCS':
                    seg.composition_n = pcs_cnt & 0xFFFF
                    pcs_cnt += 1
                    assert not (len(ds.segments) > 3 and int(seg.composition_state) == 0)
                    assert not (len(ds.segments) == 3 and len(seg.cobjects) > 1), f"{len(ds.segments)} {seg.composition_state} {len(seg.cobjects)} {ds.segments[1].type} {ds.pcs.pal_flag} {ds.segments[1].size}"
                if seg.type == 'WDS':
                    seg.windows = window_map
                    seg.update()
                if seg.type == 'PDS':
                    if seg.size == 0:
                        if len(ds.segments) == 3:
                            assert pcs.pal_flag is True, "Critical rendering error, unknown reason (effect too complex?)"
                            addOut = False
                        else:
                            raise AssertionError("Critical internal error. Please report to the author.")
                    else:
                        seg.p_vn = pds_cnt
                        pds_cnt += 1
            if addOut:
                lds_out.append(ds)
            #PAL flag must be set only when the sequence has this format
            assert not addOut or not (seq != ['PCS', 'PDS', 'END'] and pcs.pal_flag), f"{seg.pts} {addOut} {seq} {pcs.pal_flag} {len(ds.segments)} {len(pcs.cobjects)} {ds.segments[2].size}"
        #Final sequence must have this format
        assert seq == ['PCS', 'WDS', 'END'] and not pcs.pal_flag
        return Epoch(lds_out)
    ####
    
    #Set missing parameters in kwargs
    params = {'colors': 256, 'norm_thresh': 0}
    params |= kwargs
    epochs = []
    
    nb_ods_onscreen = np.zeros((len(group),), dtype=np.uint16)
    for ods_map in regions_ods_mapping:
        for events_in_ods in ods_map:
            components = np.arange(events_in_ods[0].slice[0].start, events_in_ods[0].slice[0].stop)
            for event in events_in_ods[1:]:
                components = np.union1d(np.arange(event.slice[0].start, event.slice[0].stop), components)
            nb_ods_onscreen[components] += 1
    
    for ods_id, ods_region in enumerate(regions_ods_mapping):
        time_box = get_ods_dim(ods_region, box)
        #render function generates all graphics element for a given ODS ID in an epoch
        graphics = render(bdn, group, time_box, nb_ods_onscreen, **params)
        temp_epochs = []
        
        #to_sup2 generate individual epochs for all graphic object.
        # Those are then chained together in a single epoch
        for o_vn, graphic in enumerate(graphics):
            temp_epochs.append(to_sup(bdn, cmap=graphic[0][0], cluts=graphic[0][1],
                                      events=graphic[1], time_box=graphic[3],
                                      ods_id=ods_id, o_vn=o_vn, window=time_box[1:],
                                      offset_pal=ods_id*int(256/len(regions_ods_mapping)),
                                      gfx_colors=graphic[2],
                                      **kwargs))
        epochs.append(chain_epochs(temp_epochs))
    #Parallel merge of all objects ID in a single epoch.
    return merge_epochs(epochs), epochs
    
#%%
def to_sup(bdn: BDNXML, cmap: npt.NDArray[np.uint8], cluts: npt.NDArray[np.uint8], events, time_box, **kwargs):
    ods_id = kwargs.pop('ods_id', 0)
    w_id = kwargs.pop('w_id', ods_id)
    o_vn = kwargs.pop('o_vn', 0)
    
    cobject = CObject.from_scratch(o_id=ods_id, window_id=w_id,
                                   h_pos=time_box[0].x, v_pos=time_box[0].y,
                                   forced=kwargs.pop('forced', False))
    
    vw, vh = bdn.format.value
    pcs_fn = lambda cn,cs,pf,pts,dts=None,show=True : PCS.from_scratch(width=vw,
                                                             height=vh,
                                                             fps=BDVideo.PCSFPS.from_fps(bdn.fps),
                                                             composition_n=cn,
                                                             composition_state=cs,
                                                             pal_flag=pf,
                                                             pal_id=0,
                                                             cobjects=[cobject]*show,
                                                             pts=pts, dts=dts)
    
    l_timestamps = [TC.tc2s(img.tc_in, bdn.fps) for img in events]
    closing_ts = TC.tc2s(events[-1].tc_out, bdn.fps)
    
    l_pcs = [pcs_fn(k+1, PCS.CompositionState(0), True, ts) for k, ts in enumerate(l_timestamps[1:])]
    l_pcs.insert(0, pcs_fn(0, PCS.CompositionState.EPOCH_START, False, l_timestamps[0]))
    l_pcs.append(pcs_fn(len(l_pcs), PCS.CompositionState(0), False, closing_ts, show=False))
    
    gfx_colors = kwargs.get('gfx_colors')
    
    if gfx_colors != 256:
        if (pal_offset := kwargs.pop('offset_pal', 0)) > 0 and pal_offset < 255:
            assert 0 <= np.max(cmap) + pal_offset < 256, "Palette entries OOB. Can't export to SUP."
            cmap += pal_offset
    else:
        pal_offset = 0
    
    l_pds = [PDS.from_scratch(pal, pts=ts, offset=pal_offset) for ts, pal in zip(l_timestamps, Optimise.diff_cluts(cluts, matrix=kwargs.get('bt_colorspace', 'bt709')))]

    ods = ODS.from_scratch(ods_id, o_vn, time_box[1].w, time_box[1].h,
                           PGraphics.encode_rle(cmap), pts=l_timestamps[0])
    if type(ods) is not list:
        ods = [ods]
    
    w_data = kwargs.pop('window')
    window  = WindowDefinition.from_scratch(w_id, w_data[0].x, w_data[0].y, w_data[1].w, w_data[1].h)
    wds_in  = WDS.from_scratch([window], pts=l_timestamps[0])
    wds_out = WDS.from_scratch([], pts=closing_ts)
    
    ds = [DisplaySet([l_pcs[0], wds_in, l_pds[0], *ods, ENDS.from_scratch(l_pcs[0].pts)])]
    
    # for palette updates
    for pcs, pds in zip(l_pcs[1:-1], l_pds[1:]):
        ds.append(DisplaySet([pcs, pds, ENDS.from_scratch(pcs.pts)]))
    
    # Closing DS, clearing off display
    ds.append(DisplaySet([l_pcs[-1], wds_out, ENDS.from_scratch(l_pcs[-1].pts)]))
    
    return Epoch(ds)
