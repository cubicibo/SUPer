#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:45:37 2022

@author: cibo
"""

import numpy as np

from typing import Any, TypeVar, Optional
from dataclasses import dataclass

from numpy import typing as npt
from anytree import Node

from skimage.filters import gaussian
from skimage.measure import regionprops, label

#%%
from SUPer.utils import get_super_logger, Pos, Dim, _pinit_fn, Shape
from SUPer.filestreams import BDNXMLEvent
from SUPer.segments import DisplaySet, PCS, WDS, ODS


#%%
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

    @classmethod
    def from_slices(cls, slices: tuple[slice]) -> 'ScreenRegion':
        f_ZWz = lambda slz : (int(slz.start), int(slz.stop-slz.start))
        X, Y, T = f_ZWz(slices[1]), f_ZWz(slices[2]), f_ZWz(slices[0])
        
        if len(slices) != 3:
            raise ValueError("Expected 3 slices (t, y, x).")
        return cls(*Y, *X, *T)
    
    @property
    def slice_t(self) -> slice:
        return slice(self.t, self.t2)
    
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
        return cls.from_slices(region.slice)
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
            
    def bitmap_update_mask(self, regions: _Region, gs_origs: npt.NDArray[np.uint8], boolean: bool = False) -> npt.NDArray[np.uint16]:
        mask = np.zeros((self.duration,), dtype=np.uint16)
    
    def temporal_mask(self, events: list[_BaseEvent]) -> npt.NDArray[np.uint8]:
        mask = np.zeros((self.duration,), dtype=np.uint16)
        
        area = ...
        
    
    def event_mask(self, boolean: bool = False) -> npt.NDArray[np.uint8]:
        """
        event mask defines the times during which the window displays a composition.
        When zero, the window is just fully transparent, without any composition obj.
        """
        mask = np.zeros((self.duration,), dtype=np.uint16)
        if boolean:
            for sr in self.srs:
                mask[sr.slice_t] = 1
        else:
            for sr in self.srs:
                mask[sr.slice_t] += 1
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
    
    @staticmethod
    def plane_initilaization_time(ds: DisplaySet) -> int:
        init_d = 0
        if PCS.CompositionState.EPOCH_START & ds.pcs.composition_state:
            init_d = _pinit_fn(Shape(ds.pcs.width, ds.pcs.height))
        else:
            for window in ds.wds[0].windows:
                init_d += _pinit_fn(Shape(window.width, window.height), 8)
        return init_d

    @staticmethod
    def wait(ds: DisplaySet, obj_id: int, current_duration: int) -> int:
        wait_duration = 0
        for object_def in ds.ods:
            if object_def.o_id == obj_id:
                c_time = ds.pcs.dts + current_duration
                if c_time < object_def.pts:
                    wait_duration += object_def.pts - c_time
                return np.ceil(wait_duration*90e3)
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
                decode_duration += _pinit_fn(cls.size(ds, ds.pcs.cobjects[0].window_id), 8)
            else:
                decode_duration += _pinit_fn(cls.size(ds, ds.pcs.cobjects[0].window_id), 8)
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += _pinit_fn(cls.size(ds, ds.pcs.cobjects[1].window_id), 8)

        elif ds.pcs.n_objects == 1:
            decode_duration += cls.wait(ds, ds.pcs.cobjects[0].o_id, decode_duration)
            decode_duration += _pinit_fn(cls.size(ds, ds.pcs.cobjects[0].window_id), 8)
        return decode_duration
    ####
        
    @classmethod
    def displayset_transfer(cls, display_set: DisplaySet, wds: Optional[WDS] = None, ods: Optional[list[ODS]] = None) -> bool:
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
    def group_and_sort(srs):
        """
        Seek for minimum areas from the regions, sort them and return them sorted,
        ascending area size. The caller will then choose the best area.
        """
        windows = {}
        areas = {}
        
        if len(srs) == 1:
            return {None: WindowOnBuffer(srs)}
        
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
        

    def group(self, group: list[BDNXMLEvent], **kwargs) -> list[list[list[_Region]]]:
        cls = self.__class__
        regions, gs_map, gs_origs, box = cls.coarse_grouping(group, **kwargs)  
        
        tbox = []
        for region in regions:
            region.slice = cls.crop_region(region, gs_origs)
            tbox.append(ScreenRegion.from_region(region))
            
        wobs = cls.group_and_sort(tbox)
        
    
    @staticmethod
    def filter_wobs(wobs) -> WindowOnBuffer:
        ...
    
    def feedback(self, sregs: list[ScreenRegion]) -> Any:
        assert len(sregs) <= self.n_groups, "Region count mismatch with settings."
        for sreg in sregs:
            ...
        return False
    
    @staticmethod
    def crop_region(region: _Region, gs_origs: npt.NDArray[np.uint8]) -> _Region:
        #Mask out object outside of the active region. 
        gs_origs = gs_origs.copy()
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
        
        return tuple([region.slice[0],
                      slice(region.slice[1].start+cntYt, region.slice[1].stop+cntYb),
                      slice(region.slice[2].start+cntXl, region.slice[2].stop+cntXr)])

