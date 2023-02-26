#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:45:37 2022

@author: cibo
"""

import numpy as np

from typing import Any, TypeVar, Optional
from dataclasses import dataclass
from itertools import combinations

from numpy import typing as npt
from anytree import Node

from skimage.filters import gaussian
from skimage.measure import regionprops, label

#%%
from SUPer.utils import get_super_logger, Pos, Dim, Shape
from SUPer.filestreams import BDNXMLEvent
from SUPer.segments import DisplaySet, PCS, WDS, ODS


_Region = TypeVar('Region')
_BaseEvent = TypeVar('BaseEvent')

logger = get_super_logger('SUPer')        

#%%
@dataclass(frozen=True)
class Box:
    y : int
    dy: int
    x : int
    dx: int
    
    @property
    def x2(self) -> int:
        return self.x + self.dx
    
    @property
    def y2(self) -> int:
        return self.y + self.dy
    
    @property
    def area(self) -> int:
        return self.dx * self.dy

    @property
    def coords(self) -> tuple[Pos, Pos]:
        return (Pos(self.x, self.y), Pos(self.x2, self.y2))
    
    @property
    def dims(self) -> Dim:
        return Dim(self.dx, self.dy)
    
    @property
    def shape(self) -> tuple[int, int]:
        return (self.dy, self.dx)
    
    @property
    def posdim(self) -> tuple[Pos, Dim]:
        return Pos(self.x, self.y), Dim(self.dx, self.dy)
    
    @property
    def slice(self) -> tuple[slice]:
        return (slice(self.y, self.y+self.dy),
                slice(self.x, self.x+self.dx))
    
    @property
    def slice_x(self) -> slice:
        return slice(self.x, self.x+self.dx)

    @property
    def slice_y(self) -> slice:
        return slice(self.y, self.y+self.dy)
    
    @classmethod
    def from_region(cls, region: _Region) -> 'Box':
        return cls.from_slices(region.slice)
    
    @classmethod
    def from_slices(cls, slices: tuple[slice]) -> 'Box':
        if len(slices) == 3:
            slyx = slices[1:]
        else:
            slyx = slices
        f_ZWz = lambda slz : (int(slz.start), int(slz.stop-slz.start))
        return cls(*f_ZWz(slyx[0]), *f_ZWz(slyx[1]))
    
    @classmethod
    def from_hulls(cls, *hulls: list[...]) -> 'Box':
        final_hull = cls(*([None]*4))
        for hull in hulls:
            final_hull
            raise NotImplementedError
        return final_hull
    
    @classmethod
    def from_events(cls, events: list[_BaseEvent]) -> 'Box':
        """
        From a chain of event, find the "working box" to minimise
        memory usage of the buffers while optimising.
        """
        if len(events) == 0:
            raise ValueError("No events given.")

        pxtl, pytl = np.inf, np.inf
        pxbr, pybr = 0, 0
        for event in events:
            pxtl = min(pxtl, event.x)
            pxbr = max(pxbr, event.x + event.width)
            pytl = min(pytl, event.y)
            pybr = max(pybr, event.y + event.height)
        return cls(int(pytl), int(pybr-pytl), int(pxtl), int(pxbr-pxtl))
    
    @classmethod
    def from_coords(cls, x1: int, y1: int, x2 : int, y2: int) -> 'Box':
        return cls(min(y1, y2), abs(y2-y1), min(x1, x2), abs(x2-x1))
#%%
####
@dataclass(frozen=True)
class ScreenRegion(Box):
    t:  int
    dt: int
    region: _Region

    @classmethod
    def from_slices(cls, slices: tuple[slice], region: Optional[_Region] = None) -> 'ScreenRegion':
        f_ZWz = lambda slz : (int(slz.start), int(slz.stop-slz.start))
        X, Y, T = f_ZWz(slices[2]), f_ZWz(slices[1]), f_ZWz(slices[0])
        
        if len(slices) != 3:
            raise ValueError("Expected 3 slices (t, y, x).")
        return cls(*Y, *X, *T, region)
    
    @property
    def temporal_slice(self) -> slice:
        return slice(self.t, self.t2)
    
    @property
    def temporal_range(self) -> range:
        return range(self.t, self.t2)
    
    @property
    def spatial_slice(self) -> tuple[slice]:
        return (slice(self.y, self.y2),
                slice(self.x, self.x2))
    
    @property
    def slice(self) -> tuple[slice]:
        return (slice(self.t, self.t2),
                slice(self.y, self.y2),
                slice(self.x, self.x2))
    
    @property
    def range(self) -> tuple[range]:
        return (range(self.t, self.t2),
                range(self.y, self.y2),
                range(self.x, self.x2))
    
    @property
    def t2(self) -> int:
        return self.t + self.dt
    
    @classmethod
    def from_region(cls, region: _Region) -> 'ScreenRegion':
        return cls.from_slices(region.slice, region)
####

class WindowOnBuffer:
    DURATION = None
    USE_DEFAULT_DURATION = False
    def __init__(self, screen_regions: list[ScreenRegion], id: Optional[int] = None) -> None:
        self.srs = screen_regions
        self.id = id
        
    @classmethod
    def set_default_duration(cls, duration: Optional[int]) -> None:
        if duration > 0:
            cls.DURATION = duration
        else:
            cls.DURATION = None
    
    @classmethod
    def get_default_duration(cls) -> Optional[int]:
        return cls.DURATION
    
    @property
    def duration(self) -> int:
        if __class__.USE_DEFAULT_DURATION:
            return __class__.get_default_duration()
        
        return max(map(lambda sr: sr.t2, self.srs))


    def bitmap_update_mask(self,
           main_box: Box,
           overlap_threshold: float = 0
        ) -> npt.NDArray[np.uint16]:
        """
        Find pixel collisions of different screen areas. Areas that don't collide can
        be optimised on the same bitmap without any visual artifact.
        """
        if not (0 <= overlap_threshold <= 1):
            raise ValueError(f"Overlap threshold not within [0;1], got '{overlap_threshold}'")
            
        update_mask = np.zeros(self.duration, np.uint8)
        buffer = np.zeros(main_box.shape, dtype=np.uint8)
        
        #we want to have the time of appearance in order
        srs = sorted(self.srs, key=lambda sr: sr.t)
        active_until = -1
        for ctime in range(self.duration):
            for sr in srs:
                if ctime not in sr.temporal_range:
                    continue
                percentage = np.sum(buffer[sr.spatial_slice] & sr.region.image[ctime-sr.t])/np.sum(sr.region.image[ctime-sr.t])
                if (sr.t > active_until or percentage >= overlap_threshold):
                    update_mask[ctime] = 1
                    buffer *= 0
                active_until = max(active_until, sr.t + sr.dt)
                buffer[sr.spatial_slice] |= sr.region.image[ctime-sr.t]
        return update_mask
        
            
    def delay_chain(self, events: list[_BaseEvent], fps, box: Box = None) -> npt.NDArray[np.uint8]:
        """
        Takes 
        """
        #imgs = np.zeros((2,*box.shape,4), dtype=np.int32)
        mask = np.zeros(self.duration, dtype=np.uint32)
        assert len(mask) == len(events)
        
        prev_fcnt = TC.tc2f(events[0].tc_in, fps)
        #imgs[0,:,:,:] = np.asarray(events[0].img, dtype=np.uint8)
        
        for k, event in enumerate(events[1:]):
            new_fcnt = TC.tc2f(event.tc_in, fps)
            mask[k] = new_fcnt - prev_fnct
            prev_fcnt = new_fcnt
        mask[-1] = TC.tc2f(events[-1].tc_out, fps) - new_fcnt
        return mask
        
        
    def event_mask(self, boolean: bool = True) -> npt.NDArray[np.uint8]:
        """
        event mask defines the times during which the window displays a composition.
        When zero, the window is just fully transparent, without any composition obj.
        """
        mask = np.zeros(self.duration, dtype=np.uint16)
        if boolean:
            for sr in self.srs:
                mask[sr.temporal_slice] = 1
        else:
            for sr in self.srs:
                mask[sr.temporal_slice] += 1
        return mask
    
    def get_window(self) -> Box:
        mxy = np.asarray([np.inf, np.inf])
        Mxy = np.asarray([-1, -1])
        for sr in self.srs:
            mxy[:] = np.min([np.asarray((sr.y,  sr.x)),  mxy], axis=0)
            Mxy[:] = np.max([np.asarray((sr.y2, sr.x2)), Mxy], axis=0)
        mxy, Mxy = np.uint32((mxy, Mxy))
        return Box(mxy[0], Mxy[0]-mxy[0], mxy[1], Mxy[1]-mxy[1])
    
    def area(self) -> int:
        return self.get_window().area
    
    def update_mask(self, boolean: bool = True) -> npt.NDArray[np.uint16]:
        """
        Update mask defines roughly when the buffer associated to the window should
        be updated. This is likely to catch false positives, we have to filter them.
        """
        mask = np.zeros((self.duration,), dtype=np.uint16)
        assert_str = "Caught an empty event."
        
        if boolean:
            for sr in self.srs:
                assert sr.dt > 0, assert_str
                # the event shows up at sr.t
                mask[sr.t] = 1
        else:
            #Usable to filter times when an update is needed and what not.
            for sr in self.srs:
                assert sr.dt > 0, assert_str
                mask[sr.t] += 1

class PGDecoder:
    RX =  2*1024**2
    RD = 16*1024**2
    RC = 32*1024**2
    FREQ = 90e3
    
    @classmethod
    def gplane_write_time(cls, *shape, coeff: int = 1):
        return cls.FREQ * np.ceil(coeff*shape[0]*shape[1]/cls.RC)

    @classmethod
    def plane_initilaization_time(cls, ds: DisplaySet) -> int:
        init_d = 0
        if PCS.CompositionState.EPOCH_START & ds.pcs.composition_state:
            #The patent gives the coeff 8 but does not explain where it comes from
            # and the statements in the documentation says it is just the size of the
            # graphic plane, so we clear once 1920*1080 at most (about 2 MiB).
            # for safety we assume a coefficient of two (window wipe + buffer wipe)
            init_d = cls.gplane_write_time(ds.pcs.width, ds.pcs.height, coeff=2)
        else:
            for window in ds.wds[0].windows:
                init_d += cls.gplane_write_time(ds.pcs.width, ds.pcs.height)
        return init_d

    @classmethod
    def wait(cls, ds: DisplaySet, obj_id: int, current_duration: int) -> int:
        wait_duration = 0
        for object_def in ds.ods:
            if object_def.o_id == obj_id:
                c_time = ds.pcs.dts + current_duration
                if c_time < object_def.pts:
                    wait_duration += object_def.pts - c_time
                return np.ceil(wait_duration*cls.FREQ)
        raise AssertionError("We should not be here (?)")
        return wait_duration
    ####
    @staticmethod
    def size(ds: DisplaySet, window_id: int) -> Shape:
        window = None
        for wd in ds.pcs.wds[0].window:
            if ds.pcs.cobjects[0].window_id == wd.window_id:
                window = wd
                break
        assert window is not None, "Did not find window definition."
        return Shape(window.width, window.height)
    
    @staticmethod
    def object_areas(ods: list[ODS]) -> int:
        return sum(map(lambda o: o.width*o.height if o.flags & o.ODSFlags.SEQUENCE_FIRST else 0, ods))
    
    @staticmethod
    def window_areas(wds: WDS) -> int:
        return sum(map(lambda window: window.width * window.height, wds.windows))
    
    @classmethod
    def rc_coeff(cls, ods: list[ODS], wds: WDS) -> int:
        return cls.object_areas(ods) + cls.window_areas(wds)        

    @classmethod
    def decode_duration(cls, ds: DisplaySet) -> int:
        decode_duration = cls.plane_initilaization_time(ds)
        if ds.pcs.n_objects == 2:
            if ds.pcs.cobjects[0].window_id == ds.pcs.cobjects[1].window_id:
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
            else:
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[1].window_id))

        elif ds.pcs.n_objects == 1:
            decode_duration += cls.wait(ds, ds.pcs.cobjects[0].o_id, decode_duration)
            decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
        return decode_duration
    ####
        
    @classmethod
    def displayset_transfer(cls, display_set: DisplaySet,
                            wds: Optional[WDS] = None,
                            ods: Optional[list[ODS]] = None) -> bool:
        """
        More precise, taking into account the transfer time at 128 Mbps.
        This function does not consider 
        """
        raise NotImplementedError("This function does not consider decoder multitasking.")
        dd = 0
        if display_set.ods != []:
            assert ods is None, "Provided displayset with ODS>=1 and a list of ODS..."
            ods = display_set.ods
            dd += cls.decode_duration(display_set)
            dd += cls.object_areas(display_set.ods)/cls.RD
        if wds is None:
            assert display_set.wds, "Cannot compute the transfer time without windows."
            wds = display_set.wds[0]            
        return dd + (cls.rc_coeff(ods, wds)/cls.RC)

#%%
class PGConvert:
	def __init__(self, wobs: tuple[WindowOnBuffer]) -> None:
		self.wobs = wobs

	def get_composition_states(self) -> tuple[PCS.CompositionState]:
		active_mask = self.wobs[0].event_mask()
		for wob in self.wobs[1:]:
			active_mask |= wob.event_mask()
		
		lcs = map(lambda t_mask: PCS.CompositionState.ACQUISITION if t_mask > 1 else PCS.CompositionState.NORMAL, active_mask)
		lcs[0] = PCS.CompostionState.EPOCH_START
		return lcs

#%%

class GroupingEngine:
    def __init__(self, options: dict[str, Any], n_groups: Optional[int] = None) -> None:
        if n_groups is not None and n_groups not in range(1, 3):
            logger.warning("Number of groups is not Blu-Ray compliant.")
        self.n_groups = n_groups
        self.options = options
    
    @staticmethod
    def coarse_grouping(group, /, *, blur_mul=1, blur_c=1.5, **kwargs):
        no_blur = kwargs.get('noblur_grouping', False)
        
        # SD content should be blurred with lower coeffs. Remove constant.
        if no_blur:
            blur_c = kwargs.get('noblur_bc_c', 0.0)
            blur_mul = kwargs.get('noblur_bm_c', 1.0)
        
        box = Box.from_events(group)
        (pxtl, pytl), (w, h) = box.posdim
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
        return regionprops(label(gs_graph)), gs_graph, gs_orig, box

    @staticmethod
    def group_and_sort2(srs: list[ScreenRegion]) -> list[tuple[WindowOnBuffer]]:
        """
        Seek for minimum areas from the regions, sort them and return them sorted,
        ascending area size. The caller will then choose the best area.
        """
        windows, areas = {}, {}

        n_regions = len(srs)
        region_ids = range(n_regions)
        
        if n_regions == 1:
            return [(WindowOnBuffer(srs),)]
        
        #If we have two composition objects, we want to find out the smallest 2 areas
        # that englobes all the screen regions. We generate all possible arrangement
        arrangements = map(lambda combination: set(filter(lambda region_id: region_id >= 0, combination)),
                           set(combinations(list(region_ids) + [-1]*(n_regions-2), n_regions-1)))
        
        union = set(region_ids)
        for key, arrangement in enumerate(arrangements):
            other = union - arrangement
            arr_sr, other_sr = [], []
            for k, sr in enumerate(srs):
                (arr_sr if k in arrangement else other_sr).append(sr)
            windows[key] = (WindowOnBuffer(arr_sr), WindowOnBuffer(other_sr))
            areas[key] = sum(map(lambda wb: wb.area(), windows[key]))
            
        #Here, we can sort by ascending area – the one that is the "cheapest" for the buffer
        return [windows[k] for k, _ in sorted(areas.items(), key=lambda x: x[1])]

    @staticmethod
    def group_and_sort(srs):
        """
        Seek for minimum areas from the regions, sort them and return them sorted,
        ascending area size. The caller will then choose the best area.
        """
        windows = {}
        areas = {}
        
        if len(srs) == 1:
            return [(WindowOnBuffer(srs),)]
        
        mask = set(range(len(srs)))
        processed = []

        for k, sr in enumerate(srs[1:], 1):            
            t_main = Node(0, chain=[0], srl=[srs[0]])
            t_other = Node(k, chain=[k], srl=[sr])
            
            for l, sr_other in enumerate(srs[1:], 1):
                if l == k: continue
                for root in [t_main, t_other]:
                    for descendant in root.descendants:
                        Node(l, parent=descendant, chain=descendant.chain + [l], srl=descendant.srl + [sr_other])
                    Node(l, parent=root, chain=root.chain + [l], srl=root.srl + [sr_other])

            #We can use t_main.leaves and t_other.descendants (?)
            main_leaves = list(t_main.descendants) + [t_main]
            other_leaves = list(t_other.descendants) + [t_other]
            
            for leaf in main_leaves:
                # Filter duplicate sets
                if (lset := set(leaf.chain)) in processed:
                    continue
                else:
                    processed.append(lset)
                leaf_other = next(filter(lambda other: mask - set(leaf.chain) == set(other.chain), other_leaves))
                windows[(leaf, leaf_other)] = (WindowOnBuffer(leaf.srl), WindowOnBuffer(leaf_other.srl))
                areas[(leaf, leaf_other)] = sum(map(lambda wb: wb.area(), windows[(leaf, leaf_other)]))

        #Here, we can sort by ascending area – the one that is the "cheapest" for the buffer
        return [windows[k] for k, _ in sorted(areas.items(), key=lambda x: x[1])]
        

    def group(self, subgroup: list[BDNXMLEvent], **kwargs) -> list[list[list[_Region]]]:
        cls = self.__class__
        regions, gs_map, gs_origs, box = cls.coarse_grouping(subgroup, **kwargs)
        
        tbox = []
        for region in regions:
            region.slice = cls.crop_region(region, gs_origs)
            tbox.append(ScreenRegion.from_region(region))
            
        wobs = cls.group_and_sort2(tbox)
        return cls.select_best_wob(wobs, box), box
                
    @classmethod
    def select_best_wob(cls, wobs: list[tuple[WindowOnBuffer]], box: Box) -> tuple[WindowOnBuffer]:
        scores = []
        
        #wobs is a list of pairs of wob
        for wobp in wobs[:20]:
            area_refreshed = 0
            for wob in wobp:
                mask = wob.bitmap_update_mask(box)
                area_refreshed += wob.area()*np.sum(mask)
            scores.append(area_refreshed)
        return wobs[scores.index(min(scores))]
        
    def feedback(self, sregs: list[ScreenRegion]) -> Any:
        assert len(sregs) <= self.n_groups, "Region count mismatch with settings."
        for sreg in sregs:
            ...
        return False
    
    @staticmethod
    def crop_region(region: _Region, gs_origs: npt.NDArray[np.uint8]) -> _Region:
        #Mask out object outside of the active region. 
        gs_origs = gs_origs.copy()
        #Apply blurred mask  so we don't catch nearby graphics by working with just rectangles
        gs_origs[region.slice] &= region.image
        
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
        
        f_region = tuple([region.slice[0],
                      slice(region.slice[1].start+cntYt, region.slice[1].stop+cntYb),
                      slice(region.slice[2].start+cntXl, region.slice[2].stop+cntXr)])
        
        # Refine image mask, this is a bit hacky as we modify the internal variable
        #(but it is what is returned by the .image property so we're good)
        region._cache['image'] = gs_origs[f_region] != 0
        return f_region

