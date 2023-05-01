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



from typing import Any, TypeVar, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass
from itertools import combinations, chain, zip_longest

from PIL import Image
from SSIM_PIL import compare_ssim
from numpy import typing as npt
import numpy as np

from skimage.filters import gaussian
from skimage.measure import regionprops, label

#%%
from .utils import get_super_logger, Pos, Dim, Shape, BDVideo, TimeConv as TC
from .filestreams import BDNXMLEvent, BaseEvent
from .segments import DisplaySet, PCS, WDS, PDS, ODS, ENDS, WindowDefinition, CObject, Epoch
from .optim import Optimise, Preprocess
from .pgraphics import PGraphics, PGDecoder, PGObject, FadeEffect
from .palette import Palette

_Region = TypeVar('Region')
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

    def overlap_with(self, other) -> float:
        intersect = __class__.intersect(self, other)
        return intersect.area/min(self.area, other.area)

    @classmethod
    def intersect(cls, *box) -> 'Box':
        x2 = min(map(lambda b: b.x2, box))
        y2 = min(map(lambda b: b.y2, box))
        x1 = max(map(lambda b: b.x, box))
        y1 = max(map(lambda b: b.y, box))
        dx, dy = (x2-x1), (y2-y1)
        return cls(y1, dy * bool(dy > 0), x1, dx * bool(dx > 0))

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
    def union(cls, *box) -> 'Box':
        x2 = max(map(lambda b: b.x2, box))
        y2 = max(map(lambda b: b.y2, box))
        x1 = min(map(lambda b: b.x, box))
        y1 = min(map(lambda b: b.y, box))
        return cls(y1, y2-y1, x1, x2-x1)

    @classmethod
    def from_events(cls, events: list[BaseEvent]) -> 'Box':
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
    def __init__(self, screen_regions: list[ScreenRegion], duration: int) -> None:
        self.srs = screen_regions
        self.duration = duration

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
#%%

class PGConvert:
    def __init__(self, windows: dict[int, WindowOnBuffer], events: list[Type[BaseEvent]]) -> None:
        assert type(windows) is dict
        self.windows = windows

    def render_single(self, box: Box, event: Type[BaseEvent], window_id: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        assert window_id in self.windows, "Unknown window ID."

        bitmap, palette, padding = self._quantize_adapt_palette(event.img)

        bitmap_box = padding*np.ones(box.shape, dtype=np.uint8)
        bitmap_box[event.y-box.y:event.y-box.y+event.height, event.x-box.x:event.x-box.x+event.width] = bitmap

        return palette, bitmap_box[self.windows[window_id].get_window().slice]

    @staticmethod
    def _find_most_transparent(palette: dict[tuple[int], int]) -> tuple[npt.NDArray[np.uint8], Optional[int]]:
        padding_entry = None
        min_alpha = (256, None)
        if len(palette) >= 256:
            assert len(palette) == 256, "More than 256 colors."
            for entry, k in palette.items():
                if entry[-1] == 0 and padding_entry is None:
                    padding_entry = k
                if min_alpha[0] < entry[-1]:
                    min_alpha = (entry[-1], k)
            if padding_entry is None:
                return None, None #Do another turn with 255 colors
        else:
            if (0, 0, 0, 0) in palette:
                padding_entry = palette[(0, 0, 0, 0)]
            else:
                padding_entry = len(palette)
                palette[(0, 0, 0, 0)] = padding_entry
        return np.asarray(list(palette.keys()), dtype=np.uint8), padding_entry

    def _quantize_adapt_palette(self, img) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], int]:
        padding, k = None, 0
        while padding is None:
            bitmap, palette = Preprocess.quantize(img, 256-k, kmeans_quant=self.kwargs.get('kmeans_quant', False))
            palette, padding = __class__._find_most_transparent(palette)
            k+=1
        return bitmap, palette, padding

    def render_multiple(self,
          box: Box,
          event: Type[BaseEvent],
        ) -> tuple[npt.NDArray[np.uint8], tuple[npt.NDArray[np.uint8]]]:
        """
        Essentially find the PDS and CLU-bitmaps when 2+ compositions
        are on screen simultaneously (they share the same palette)
        """
        bitmap, palette, padding = self._quantize_adapt_palette(event.img)

        bitmap_box = padding*np.ones(box.shape, dtype=np.uint8)
        bitmap_box[event.y-box.y:event.y-box.y+event.height, event.x-box.x:event.x-box.x+event.width] = bitmap

        imgs = [bitmap_box[wob.get_window().slice] for wob in self.windows.values()]
        return (palette, tuple(imgs))

#%%
class GroupingEngine:
    class Mode(Enum):
        SMALLEST_WINDOWS = 'area'
        LEAST_ACQUISITIONS = 'acq'
        SPECIAL_EFFECTS = 'special'

        def __eq__(self, other: Any) -> bool:
            if isinstance(self, self.__class__):
                return self.value == other.value
            if isinstance(other, str):
                return self.value == other.lower()
            return NotImplemented

    def __init__(self, n_groups: int = 2, **kwargs) -> None:
        if n_groups not in range(1, 3):
            raise AssertionError(f"GroupingEngine expects 1 or 2 groups, not '{n_groups}'.")

        self.n_groups = n_groups
        self.candidates = kwargs.pop('candidates', 25)
        self.mode = kwargs.pop('mode', __class__.Mode.SMALLEST_WINDOWS)

        self.no_blur = kwargs.pop('noblur_grouping', False)
        self.blur_mul = kwargs.pop('blur_mul', 1.1)
        self.blur_c = kwargs.pop('blur_const', 1.5)

        self.kwargs = kwargs

    def coarse_grouping(self, group: list[Type[BaseEvent]]) -> tuple[_Region, npt.NDArray[np.uint8], Box]:
        # SD content should be blurred with lower coeffs. Remove constant.
        blur_mul = self.blur_mul
        blur_c = self.blur_c
        if self.no_blur:
            blur_c = self.kwargs.get('noblur_bc_c', 0.0)
            blur_mul = self.kwargs.get('noblur_bm_c', 1.0)

        box = Box.from_events(group)
        (pxtl, pytl), (w, h) = box.posdim
        ratio_woh = abs(w/h)
        ratio_how = 1/ratio_woh if 1/ratio_woh <= 1 else 1
        ratio_woh = ratio_woh if ratio_woh <= 1.3 else 1.3

        ne_imgs = []
        for event in group:
            imgg = np.asarray(event.img.getchannel('A'), dtype=np.uint8)
            img_blurred = (255*gaussian(imgg, (blur_c + blur_mul*ratio_how, blur_c + blur_mul*ratio_woh)))
            img_blurred[img_blurred <= 0.25] = 0
            img_blurred[img_blurred > 0.25] = 1
            ne_imgs.append(img_blurred)

        gs_graph = np.zeros((len(group), h, w), dtype=np.uint8)
        gs_orig = np.zeros((len(group), h, w), dtype=np.uint8)
        for k, (event, b_img) in enumerate(zip(group, ne_imgs)):
            slice_x = slice(event.x-pxtl, event.x-pxtl+event.width)
            slice_y = slice(event.y-pytl, event.y-pytl+event.height)
            gs_graph[k, slice_y, slice_x] = b_img.astype(np.uint8)
            gs_orig[k, slice_y, slice_x] = np.array(event.img.getchannel('A'))
        return regionprops(label(gs_graph)), gs_orig, box

    def group_and_sort_flat(self, srs: list[ScreenRegion], duration: int) -> list[tuple[WindowOnBuffer]]:
        """
        Seek for minimum areas from the regions, sort them and return them sorted,
        ascending area size. The caller will then choose the best area.
        This function flatten the 3D space to a 2D one by grouping similar regions
        """
        nsrs = len(srs)

        if nsrs == 1 or self.n_groups == 1:
            return [(WindowOnBuffer(srs, duration=duration),)]

        assert nsrs < 65536, "Too many regions."

        inters_coeff = np.zeros((nsrs, nsrs))
        for k, sr in enumerate(srs):
            inters_coeff[k,:] = list(map(sr.overlap_with, srs))
        inters_coeff -= np.eye(nsrs)

        success = False
        for thresh in np.arange(0.9, 0.4, -0.1):
            lut = {}
            groups = {}
            seen = set()
            mapping = np.argwhere(inters_coeff > thresh)
            for k, v in mapping:
                groups[k] = groups.get(k, []) + [v]
            z = 0
            for k, v in sorted(groups.items(), key=lambda e: len(e[1]), reverse=True):
                if k in seen:
                    continue
                ids = list(chain(v, [k]))
                lut[z] = [srid for srid in ids if srid not in seen]
                seen |= set(ids)
                z += 1
            for k, sr in enumerate(srs):
                if k not in seen:
                    lut[z] = [k]
                    z += 1
            if len(lut) <= np.ceil(len(srs)/3.5): #/3.5 number of groups
                success=True
                break
        ####
        if len(lut) >= 15:
            return None
        return self._group_and_sort_mapped(srs, duration, lut)

    @staticmethod
    def _get_combinations(n_regions: int) -> map:
        #If we have two composition objects, we want to find out the smallest 2 areas
        # that englobes all the screen regions. We generate all possible arrangement
        region_ids = range(n_regions)
        arrangements = map(lambda combination: set(filter(lambda region_id: region_id >= 0, combination)),
                                   set(combinations(list(region_ids) + [-1]*(n_regions-2), n_regions-1)))
        return arrangements

    def _group_and_sort_mapped(self,
           srs: list[ScreenRegion],
           duration: int,
           mapping: dict[int, list[int]]
        ) -> list[tuple[WindowOnBuffer]]:
        """
        Find the windows using pre-grouped srs according to some mapping.
        """
        combinations = __class__._get_combinations(len(mapping))

        windows, areas = {}, {}

        for key, arrangement in enumerate(combinations):
            arr_sr, other_sr = [], []
            for k, srl in mapping.items():
                (arr_sr if k in arrangement else other_sr).extend([sr for ksr, sr in enumerate(srs) if ksr in srl])
            windows[key] = (WindowOnBuffer(arr_sr, duration=duration), WindowOnBuffer(other_sr, duration=duration))
            areas[key] = sum(map(lambda wb: wb.area(), windows[key]))

        #Here, we can sort by ascending area – first has the smallest windows
        return [windows[k] for k, _ in sorted(areas.items(), key=lambda x: x[1])]

    def group_and_sort(self, srs: list[ScreenRegion], duration: int) -> list[tuple[WindowOnBuffer]]:
        """
        Seek for minimum areas from the regions, sort them and return them sorted,
        ascending area size. The caller will then choose the best area.
        This function performs an expensive 3D search, only suited for len(srs) <= 16
        """
        windows, areas = {}, {}
        n_regions = len(srs)

        if n_regions == 1 or self.n_groups == 1:
            return [(WindowOnBuffer(srs, duration=duration),)]

        for key, arrangement in enumerate(__class__._get_combinations(n_regions)):
            arr_sr, other_sr = [], []
            for k, sr in enumerate(srs):
                (arr_sr if k in arrangement else other_sr).append(sr)
            windows[key] = (WindowOnBuffer(arr_sr, duration=duration), WindowOnBuffer(other_sr, duration=duration))
            areas[key] = sum(map(lambda wb: wb.area(), windows[key]))

        #Here, we can sort by ascending area – first has the smallest windows
        return [windows[k] for k, _ in sorted(areas.items(), key=lambda x: x[1])]

    def group(self, subgroup: list[Type[BaseEvent]]) -> tuple[list[tuple[WindowOnBuffer]], Box]:
        cls = self.__class__

        trials = 15
        wobs = None
        while trials > 0 and wobs is None:
            trials -= 1
            regions, gs_origs, box = self.coarse_grouping(subgroup)

            tbox = []
            for region in regions:
                region.slice = cls.crop_region(region, gs_origs)
                tbox.append(ScreenRegion.from_region(region))

            if len(tbox) < 12:
                wobs = self.group_and_sort(tbox, len(subgroup))
            else:
                wobs = self.group_and_sort_flat(tbox, len(subgroup))
                if wobs is None:
                    self.no_blur = False
                    self.blur_mul += 0.5
                    self.blur_c += 0.5
        if wobs is None:
            logger.warning("Grouping Engine giving up optimising layout. Using a single window.")
            wobs = [(WindowOnBuffer(tbox, duration=len(subgroup)),)]
        return self.select_best_wob(wobs, box), box

    def select_best_wob(self, wobs: list[tuple[WindowOnBuffer]], box: Box) -> tuple[WindowOnBuffer]:
        """
        This function has three mode, depending of the GroupingEngine:
            - return the pair of WOBs with the minimum area
            - return the pair of WOBs with the least amount of acquisition (rough est)
            - TODOs: the pair of WOBs that would separate best special effects.

        :param wobs: list of wobs pair
        :param box: box containing all wobs
        :return: list of wobs pair ordered ascendingly by total refreshed area.
        """
        if self.mode == __class__.Mode.SPECIAL_EFFECTS:
            logger.warning("Not implemented yet, returning min area.")
        if self.mode in [__class__.Mode.SPECIAL_EFFECTS, __class__.Mode.SMALLEST_WINDOWS]:
            return tuple(sorted(wobs[0], key=lambda x: x.srs[0].t))
        elif self.mode == __class__.Mode.LEAST_ACQUISITIONS:
            scores = []
            #wobs is a list of pairs of wob
            for wobp in wobs[:self.candidates]:
                area_refreshed = 0
                for wob in wobp:
                    mask = wob.bitmap_update_mask(box)
                    area_refreshed += wob.area()*np.sum(mask)
                scores.append(area_refreshed)
            return next(map(lambda ws: ws[0], sorted(zip(wobs, scores), key=lambda ws : ws[1])))
        else:
            raise NotImplementedError("Unknown grouping choice mode.")

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

#%%
class WOBSAnalyzer:
    def __init__(self, wobs: ..., events: ..., box: ..., fps: ..., bdn: ..., **kwargs):
        self.wobs = wobs
        self.events = events
        self.box = box
        self.fps = fps
        self.bdn = bdn
        self.kwargs = kwargs

    def mask_event(self, window, event) -> npt.NDArray[np.uint8]:
        windowed_event = np.zeros((window.dy, window.dx, 4), dtype=np.uint8)
        work_plane = np.zeros((self.box.dy, self.box.dx, 4), dtype=np.uint8)

        hsi = slice(event.x-self.box.x, event.x-self.box.x+event.width)
        vsi = slice(event.y-self.box.y, event.y-self.box.y+event.height)
        work_plane[vsi, hsi, :] = np.array(event.img, dtype=np.uint8)

        return work_plane[window.y:window.y2, window.x:window.x2, :]

    def analyze(self):
        woba = []

        #Init
        gens, windows = [], []
        for k, swob in enumerate(self.wobs):
            woba.append(WOBAnalyzer(swob))
            windows.append(swob.get_window())
            gens.append(woba[k].analyze())
            next(gens[-1])

        #get all windowed bitmaps
        wevents = [[] for k in range(len(windows))]
        pgobjs = [[] for k in range(len(windows))]
        for event in chain(self.events, [None]*2):
            for wid, (window, gen) in enumerate(zip(windows, gens)):
                if event is not None:
                    wevents[wid].append(self.mask_event(window,  event))
                    pgobj = gen.send(wevents[wid][-1])
                else:
                    try:
                        pgobj = gen.send(None)
                    except StopIteration:
                        pgobj = None
                if pgobj is not None:
                    pgobjs[wid].append(pgobj)
        pgobjs_proc = [objs.copy() for objs in pgobjs]

        acqs, absolutes, margins, durs = self.find_acqs(pgobjs_proc, windows)
        states = [PCS.CompositionState.NORMAL] * len(acqs)
        states[0] = PCS.CompositionState.EPOCH_START
        drought = 0
        for k, (acq, forced, margin) in enumerate(zip(acqs[1:], absolutes[1:], margins[1:]), 1):
            if acq and (forced or margin > max(0.8-0.05*drought, 0)):
                states[k] = PCS.CompositionState.ACQUISITION
                drought = 0
            else:
                #try to not do too many acquisitions, as we want to compress the stream.
                drought += 1
        return self._convert(states, pgobjs, windows, durs)

    def _convert(self, states, pgobjs, windows, durs):
        event_mask = self.wobs[0].event_mask()
        for wb in self.wobs[1:]:
            event_mask += wb.event_mask()

        wd_base = [WindowDefinition.from_scratch(k, w.x+self.box.x, w.y+self.box.y, w.dx, w.dy) for k, w in enumerate(windows)]
        wds_base = WDS.from_scratch(wd_base, pts=0.0)
        n_actions = len(event_mask)
        displaysets = []

        def get_obj(frame, pgobjs) -> None:
            objs = {k: None for k, objs in enumerate(pgobjs)}

            for wid, pgobj in enumerate(pgobjs):
                for obj in pgobj:
                    if obj.is_active(frame):
                        objs[wid] = obj
            return objs

        def get_pts(c_pts):
            return c_pts - 2/90e3

        def get_undisplay(self, i, pcs_id, wds_base):
            c_pts = get_pts(TC.tc2s(self.events[i].tc_out, self.fps))
            pcs = pcs_fn(pcs_id, PCS.CompositionState.NORMAL, False, [], c_pts)
            wds = WDS(bytes(wds_base))
            wds.pts = c_pts
            return DisplaySet([pcs, wds, ENDS.from_scratch(pts=c_pts)])

        i = 0
        ods_reg = [0]*2
        pcs_id = 0
        pal_vn = 0
        c_pts = 0

        pcs_fn = lambda pcs_cnt, state, pal_flag, cl, pts:\
                    PCS.from_scratch(*self.bdn.format.value, BDVideo.LUT_PCS_FPS[round(self.fps, 3)], pcs_cnt, state, pal_flag, 0, cl, pts=pts)

        while i < n_actions:
            for k in range(i+1, n_actions):
                if states[k] != PCS.CompositionState.NORMAL:
                    break
            if i == n_actions-1:
                k = i + 1
            assert k > i
            if durs[i][1] != 0:
                assert i > 0
                displaysets.append(get_undisplay(self, i-1, pcs_id, wds_base))
                pcs_id += 1

            c_pts = get_pts(TC.tc2s(self.events[i].tc_in, self.fps))

            res, pals, o_ods, cobjs = [], [], [], []
            has_two_objs = event_mask[i] > 1
            for wid, pgo in get_obj(i, pgobjs).items():
                if pgo is None:
                    continue
                cobjs.append(CObject.from_scratch(wid, wid, windows[wid].x+self.box.x, windows[wid].y+self.box.y, False))
                res.append(Optimise.solve_sequence_fast([Image.fromarray(img) for img in pgo.gfx[i-pgo.f:k-pgo.f]],
                                                        128 if has_two_objs else 256,
                                                        self.kwargs.get('kmeans_quant', False)))
                pals.append(Optimise.diff_cluts(res[-1][1], matrix=self.kwargs.get('bt_colorspace', 'bt709')))

                ods_data = PGraphics.encode_rle(res[-1][0] + 128*(wid == 1 and has_two_objs))
                o_ods += ODS.from_scratch(wid, ods_reg[wid], res[-1][0].shape[1], res[-1][0].shape[0], ods_data, pts=c_pts)
                ods_reg[wid] += 1
            ####
            pal = pals[0][0]
            if has_two_objs:
                for p in pals[1]:
                    p.offset(128)
                pal |= pals[1][0]

            wds = WDS(bytes(wds_base))
            wds.pts = c_pts

            pds = PDS.from_scratch(pal, pal_vn, 0, pts=c_pts)
            pal_vn += 1
            pcs = pcs_fn(pcs_id, states[i], False, cobjs, c_pts)
            pcs_id += 1
            ends = ENDS.from_scratch(pts=c_pts)
            displaysets.append(DisplaySet([pcs, wds, pds] + o_ods + [ends]))

            if len(pals[0]) > 1:
                if len(pals) == 1:
                    assert not has_two_objs
                    pals.append([])
                else:
                    #assert len(pals[0]) == len(pals[1]) and has_two_objs
                    #assert has_two_objs
                    ...
                for z, (p1, p2) in enumerate(zip_longest(pals[0][1:], pals[1][1:], fillvalue=Palette()), i+1):
                    c_pts = get_pts(TC.tc2s(self.events[z].tc_in, self.fps))
                    pal |= (p1 | p2)
                    assert states[z] == PCS.CompositionState.NORMAL
                    if durs[z][1] != 0:
                        displaysets.append(get_undisplay(self, z-1, pcs_id, wds_base))
                        pcs_id += 1
                    if has_two_objs:
                        pcs = pcs_fn(pcs_id, states[z], False, cobjs, c_pts)
                        wds = WDS(bytes(wds_base))
                        wds.pts = c_pts
                        pds = PDS.from_scratch(pal, pal_vn, 0, pts=c_pts)
                        displaysets.append(DisplaySet([pcs, wds, pds, ENDS.from_scratch(pts=c_pts)]))
                    else:
                        pcs = pcs_fn(pcs_id, states[z], True, cobjs, c_pts)
                        pds = PDS.from_scratch(p1 | p2, pal_vn, 0, pts=c_pts)
                        displaysets.append(DisplaySet([pcs, pds, ENDS.from_scratch(pts=c_pts)]))
                    pal_vn += 1
                    pcs_id += 1
                #assert z+1 == k
                if z+1 != k:
                    print(f"{len(pals)} {has_two_objs}, {i}-{k}")
                    if len(pals) > 1:
                        print(f"{len(pals[0])} {len(pals[1])}")
            i = k
        ####while
        #final "undisplay" displayset
        displaysets.append(get_undisplay(self, -1, pcs_id, wds_base))
        pcs_id += 1
        return Epoch(displaysets)

    def find_acqs(self, pgobjs_proc: dict[..., list[...]], windows):
        #get the frame count between each screen update and find where we can do acqs
        gp_clear_dur = PGDecoder.copy_gp_duration(sum(map(lambda x: x.area, windows)))

        durs = self.get_durations()

        dtl = np.zeros((len(durs)), dtype=float)
        valid = np.zeros((len(durs),), dtype=np.bool_)
        absolutes = np.zeros_like(valid)

        objs = [None for objs in pgobjs_proc]

        prev_dt = 6
        for k, (dt, delay) in enumerate(durs):
            margin = (delay + prev_dt)*1/self.fps
            force_acq = False
            for wid, wd in enumerate(windows):
                if objs[wid] and not objs[wid].is_active(k):
                    objs[wid] = None
                    force_acq = True
                if len(pgobjs_proc[wid]):
                    if not objs[wid] and pgobjs_proc[wid][0].is_active(k):
                        objs[wid] = pgobjs_proc[wid].pop(0)
                        force_acq = True
                    else:
                        assert not pgobjs_proc[wid][0].is_active(k)

            areas = list(map(lambda obj: obj.area*obj.is_visible(k), filter(lambda x: x is not None, objs)))
            td = PGDecoder.decode_display_duration(gp_clear_dur, areas)
            valid[k] = td < margin
            dtl[k] = 1-td/margin
            absolutes[k] = force_acq
            prev_dt = dt
        return valid, absolutes, dtl, durs
    ####

    def get_durations(self) -> npt.NDArray[np.uint32]:
        """
        Returns the duration of each event in frames.
        Additionally, the offset from the previous event is also returned. This value
        is zero unless there are no PG objects shown at some point in the epoch.
        """
        top = TC.tc2f(self.events[0].tc_in, self.fps)
        delays = []
        for event in self.events:
            tic = TC.tc2f(event.tc_in, self.fps)
            toc = TC.tc2f(event.tc_out,self.fps)
            delays.append((toc-tic, top-tic))
            top = toc
        return delays
####
#%%

class WOBAnalyzer:
    def __init__(self, wob, ssim_threshold: float = 0.97, overlap_threshold: float = 0.995) -> None:
        self.wob = wob
        assert ssim_threshold < 1.0, "Not a valid SSIM threshold"
        self.ssim_threshold = ssim_threshold
        assert 0 < overlap_threshold < 1.0, "Not a valid overlap threshold."
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def get_grayscale(rgba: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rgba = rgba.astype(np.uint16)
        img = np.round(0.2989*rgba[:,:,0] + 0.587*rgba[:,:,1] + 0.114*rgba[:,:,2])
        return (img.clip(0, 255) & (255*(rgba[:,:,3] > 0))).astype(np.uint8)

    def compare(self, bitmap: Image.Image, current: Image.Image) -> float:
        """
        :param bitmap: (cropped or padded) aggregate of the previous bitmaps
        :param current: current bitmap under analysis
        :return: comparison score between the two
        """
        assert bitmap.width == current.width and bitmap.height == current.height, "Different shapes."

        # Intersect alpha planes
        a_bitmap = np.array(bitmap)
        a_current = np.array(current)
        inters_inv = np.logical_and(a_bitmap[:,:,3] == 0, a_current[:,:,3] == 0)
        inters = np.logical_and(a_bitmap[:,:,3] != 0, a_current[:,:,3] != 0)
        inters_area = np.sum(inters)
        #if the images have the exact same alpha channel, this measure is equal to 1
        overlap = (inters_area > 0) * (inters_area + np.sum(inters_inv))/inters.size

        if overlap < self.overlap_threshold and overlap > 0:
            #score = compare_ssim(bitmap.convert('L'), current.convert('L'))
            #Broadcast transparency mask of current on all channels of ref
            mask = 255*(np.logical_and((a_bitmap > 0), (a_current[:, :, 3, None] > 0)).astype(np.uint8))
            score = compare_ssim(Image.fromarray(a_bitmap & mask).convert('L'), current.convert('L'))
        else:
            #Perfect overlap or zero overlap, the current bitmap fits perfectly on the previous
            score = 1.0
        return score

    @staticmethod
    def get_patch(image1, image2) -> Image.Image:
        bbox1 = image1.getbbox()
        bbox2 = image2.getbbox()

        #One of the image is fully transparent.
        if bbox1 is None or bbox2 is None:
            return bbox1, bbox2

        f_area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        A1 = f_area(bbox1)
        A2 = f_area(bbox2)

        if A1 >= A2:
            return (image1.crop(bbox1), Pos(*bbox1[:2])), image2
        return image1, (image2.crop(bbox2), Pos(*bbox2[:2]))

    @staticmethod
    def correlate_patch(image, patch) -> tuple[float, Type['Box']]:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

        #pad image with patch dimension, so we can look at the edges.
        image = cv2.copyMakeBorder(image, patch.height-1, patch.height-1,
                                   patch.width-1, patch.width-1,
                                   cv2.BORDER_CONSTANT, value=[0]*4)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        patch_gray = cv2.cvtColor(np.asarray(patch, dtype=np.uint8), cv2.COLOR_RGBA2GRAY)

        #Correlate across entire surface and return the best scores
        res = cv2.matchTemplate(img_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
        score1, score2, p1, p2 = cv2.minMaxLoc(res)
        return (score1, p1), (score2, p2)

    def analyze(self):
        cls = self.__class__
        window = self.wob.get_window()

        bitmaps = []
        alpha_compo = Image.fromarray(np.zeros((window.dy, window.dx, 4), np.uint8))

        mask = []
        event_cnt = 0
        pgo_yield = None
        unseen = 0

        while True:
            rgba = yield pgo_yield
            pgo_yield = None

            if rgba is None:
                if len(bitmaps):
                    bbox = alpha_compo.getbbox()
                    if unseen > 0:
                        mask = mask[:-unseen]
                    pgo_yield = PGObject(np.stack(bitmaps).astype(np.uint8), Box.from_coords(*bbox), mask, f_start)
                    bitmaps = []
                    mask = []
                    continue
                else:
                    break

            has_content = np.any(rgba)
            if has_content or len(mask):
                if not len(mask):
                    f_start = event_cnt
                mask.append(has_content)
            if has_content:
                rgba_i = Image.fromarray(rgba)
                score = self.compare(alpha_compo, rgba_i)
                if score > self.ssim_threshold:
                    bitmaps.append(rgba)
                    alpha_compo.alpha_composite(rgba_i)
                else:
                    bbox = alpha_compo.getbbox()
                    pgo_yield = PGObject(np.stack(bitmaps).astype(np.uint8), Box.from_coords(*bbox), mask[:-1-unseen], f_start)

                    #new bitmap
                    mask = mask[-1:]
                    bitmaps = [rgba]
                    f_start = event_cnt
                    alpha_compo = Image.fromarray(rgba.copy())
                unseen = 0
            elif len(mask):
                unseen += 1
            event_cnt += 1
        ####while
        return # StopIteration


#%%
@dataclass
class PGDecoderStats:
    fps: Union[BDVideo.FPS, float]
    cb_usage: int = 0 #Coded buffer
    db_usage: int = 0 #Decoded buffer
    stream_usage: int = 0 #pg stream

    """
    This class implements a basic compliancy test using the standard bandwiths specified
    in the PGS patent. It is not perfect but is a good indicator of issues on the decoding
    side
    """

    def __post_init__(self) -> None:
        self.__is_valid = True
        self.__last_pcs_pts = -1 #PTS of each PCS should be unique (and PTS=DTS of END should)
        self.__objects = {}
        self.__cobjects ={}
        self.__windows = {}
        nfps = int(np.ceil(getattr(self.fps, 'value', self.fps)))
        self.__coded_past = [0] * nfps# coded_bw_ra[1:round(fps)]
        self.__decod_past = [0] * nfps
        self.__pgstream_past = [0] * nfps
        self.__past_pts = [-1] * nfps

    def stream_bandwidth(self) -> float:
        return sum(self.__pgstream_past)/abs(self.__past_pts[-1]-self.__past_pts[0])

    def coded_bandwidth(self) -> float:
        return sum(self.__coded_past)/abs(self.__past_pts[-1]-self.__past_pts[0])

    def decoded_bandwidth(self) -> float:
        return sum(self.__decod_past)/abs(self.__past_pts[-1]-self.__past_pts[0])

    def last_rc(self) -> int:
        #cast for overflows
        nf = TC.s2f(np.uint32(PGDecoder.FREQ*(self.__past_pts[-1] - self.__past_pts[-2]))/PGDecoder.FREQ, self.fps)
        w_area = sum([win.width * win.height for win in self.__windows])
        active_ods_area = sum(map(lambda x: x['width']*x['height'], filter(lambda x: x['display'], self.__cobjects.values())))
        return self.fps*(w_area + min(active_ods_area, w_area))

    def ds_comply(self, ds) -> bool:
        self._ds_action(ds)
        valid = True
        valid &= self.db_usage < PGDecoder.DECODED_BUF_SIZE
        if not valid:
            details += "ERR: Object buffer overrun, "

        valid &= self.stream_bandwidth() < PGDecoder.RX + PGDecoder.CODED_BUF_SIZE
        if not valid:
            details += "ERR: Excessive BDAV PG bandwidth, "
        else:
            #TODO: try to implement buffering here with the 1 MiB buffer at the input
            valid &= self.stream_bandwidth() < PGDecoder.RX + int(PGDecoder.CODED_BUF_SIZE/1.8)
            if not valid:
                # This is not a critical error because we have a 1 MiB PG buffer
                # if it triggers for a long time there will be an overrun.
                details += "WARN: High BDAV bandwidth, "

        valid &= self.coded_bandwidth() < PGDecoder.RD
        if not valid:
            details += "ERR: Excessive object bandwidth, "

        valid &= self.last_rc() < PGDecoder.RC
        if not valid:
            details += "WARN: Flickering"
        self.test_diplayset()
        logger.warning(details)
        return valid

    def _save_usage(self, pgbytes: int, coded: int, decoded: int, pts: float) -> None:
        if pts != self.__past_pts[-1]:
            #discard oldest entry and move next
            self.__coded_past = self.__coded_past[1:] + [coded]
            self.__decod_past = self.__decod_past[1:] + [decoded]
            self.__pgstream_past = self.__pgstream_past[1:] + [pgbytes]
        else:
            self.__coded_past[-1] += coded
            self.__decod_past[-1] += decoded
            self.__pgstream_past[-1] += pgbytes

    def _ds_action(self, ds: DisplaySet) -> None:
        ds_coded = 0
        ds_decod = 0

        if ds.pcs.composition_state & PCS.CompositionState.EPOCH_START:
            self.cb_usage = 0
            self.db_usage = 0
            self.__objects = {}
        elif ds.pcs.composition_state & PCS.CompositionState.ACQUISITION:
            self.cb_usage = 0
            self.db_usage = 0
            self.__cobjects = {}
        if ds.wds != []:
            for win in ds.wds.windows:
                self.windows[win.window_id] = win

        self.__is_valid |= (self.last_pcs_pts >= ds.pcs.pts)

        #NOTE: this is not exact, a buffer slot is identified by the object id but
        # the slot has fixed dims within an epoch (defined by the 1st object with this ID).
        # If this is not respected, buffer overruns can occur (esp. with FHD content)
        # effectively crashing the hardware decoder.
        for ods in ds.ods:
            if ods.flags & int(ODS.ODSFlags.SEQUENCE_FIRST):
                ds_decod += (ods.width * ods.height)
                ds_coded += ods.rle_len
                self.__cobjects[ods.o_id] = {'width': ods.width, 'height': ods.height}

        #Should never happen unless one defines >64 different CObj within an epoch
        # without ever doign an acquisitions.
        assert len(self.__cobjects) <= 64, "More than 64 composition objects defined."

        #Set all object to no display
        for cobj in self.__cobjects.values():
            cobj['display'] = False

        #assign the ones that are displayed.
        for cobj in ds.pcs.cobject:
            assert ods.o_id in self.__cobjects, "Displaying unknown object id."
            self.__cobjects[cobj.o_id] |= vars(cobj) | {'display': True}

        self.cb_usage += ds_coded
        self.db_usage += ds_decod
        self._save_usage(len(bytes(ds)), ds_coded, ds_decod, ds.pcs.pts)
        self.last_pcs_pts = ds.pcs.pts

    def test_diplayset(self, ds: DisplaySet) -> None:
        """
        This function performs hard check on the display set
        if its structure is bad, it raises an assertion error.
        This is preferred over a "return false" because a bad displayset
        will typically crash a hardware decoder and we don't want that.
        """
        current_pts = ds.pcs.pts
        if epoch.ds[kd-1].pcs.pts != prev_pts and current_pts != epoch.ds[kd-1].pcs.pts:
            prev_pts = epoch.ds[kd-1].pcs.pts
        else:
            logger.warning(f"Two displaysets at {current_pts} [s] (internal rendering error?)")

        if ds.pcs.composition_state != PCS.CompositionState.NORMAL:
            assert ds.pcs.pal_flag is False, "Palette update on epoch start or acquisition."
        if ds.wds:
            assert ds.pcs.pal_flag is False, "Manipulating windows on palette update."
            assert len(ds.wds.windows) <= 2, "More than two windows."
        if ds.ods:
            assert ds.pcs.pal_flag is False, "Defining ODS in palette update."
            start_cnt, close_cnt = 0, 0
            for ods in ds.ods:
                start_cnt += bool(int(ods.flags) & int(ODS.ODSFlags.SEQUENCE_FIRST))
                close_cnt += bool(int(ods.flags) & int(ODS.ODSFlags.SEQUENCE_LAST))
            assert start_cnt == close_cnt, "ODS segments flags mismatch."
        if ds.pds:
            for pds in ds.pds:
                if ds.pcs.pal_flag:
                    assert len(ds.pcs.cobjects) == 1, "Undefined behaviour: palette update with 2+ objects."
                    assert ds.pcs.pal_id == pds.p_id, "Palette ID mismatch between PCS and PDS on palette update."
                    assert len(ds) == 3, "Unusual display set structure for a palette update."
                assert pds.p_id < 8, "Using undefined palette ID."
                assert pds.n_entries <= 256, "Defining more than 256 palette entries."
        assert ds.end, "No END segment in DS."
        assert ds.end.pts >= ds.end.dts, "Not MPEG2 standard compliant."
        ####

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
                logger.warning(f"Two displaysets at {current_pts} [s] (internal rendering error?)")

            for seg in ds.segments:
                size_ds += len(bytes(seg))
                n_obj = 0
                if seg.pts != current_pts and current_pts != -1 and _cnt_pts:
                    logger.warning(f"Display set has non-constant pts at {seg.pts} or {current_pts} [s].")
                    current_pts = -1
                if isinstance(seg, PCS) and int(seg.composition_state) != 0:
                    # On acquisition, the object buffer is flushed
                    ods_acc = 0
                    n_obj = len(seg.cobjects)
                elif isinstance(seg, WDS):
                    for w in seg.windows:
                        window_area[w.window_id] = w.width*w.height
                elif isinstance(seg, ODS) and int(seg.flags) & int(ODS.ODSFlags.SEQUENCE_FIRST):
                    decoded_this_ds += seg.width * seg.height
                    coded_this_ds += seg.rle_len
                elif isinstance(seg, PDS):
                    if n_obj > 1 and seg.pal_flag:
                        logger.warning(f"Undefined behaviour: palette update with 2+ objects at {seg.pts}.")
                        compliant = False
                    if seg.p_id >= 8:
                        logger.warning(f"Using an undefined palette ID at {seg.pts} [s].")
                        compliant = False
                elif isinstance(seg, ENDS) and n_obj == 0 and ds.pcs.pal_flag \
                    and int(ds.pcs.composition_state) == 0 and isinstance(ds.segments[1], WDS):
                    logger.warning(f"Bad end displayset, graphics will not be undisplayed at {seg.pts} [s].")

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
                    logger.warning(f"High instantaneous coded bandwidth at {seg.pts:.03f} [s] (not critical - fair warning)")
                else:
                    logger.info(f"High coded bandwidth at {seg.pts:.03f} [s] (not critical - fair warning).")
                # This is not an issue unless it happens very frequently, so we don't mark as not compliant

            if prev_pts != seg.pts:
                coded_bw_ra = coded_bw_ra[1:round(fps)]
                coded_bw_ra_pts = coded_bw_ra_pts[1:round(fps)]
                coded_bw_ra.append(coded_buffer_pts)
                coded_bw_ra_pts.append(seg.pts)

            if (rate:=sum(coded_bw_ra)/abs(coded_bw_ra_pts[-1]-coded_bw_ra_pts[0])) > (max_rate:=16*(1024**2)):
                logger.warning(f"Exceeding coded bandwidth at ~{seg.pts:.03f} [s] {100*rate/max_rate:.03f}%.")
                warnings += 1

            if decoded_buffer_bandwidth > 128*(1024**2):
                logger.warning(f"Exceeding decoded buffer bandwidth at {seg.pts} [s].")
                warnings +=1

            # Decoded object plane is 4 MiB
            if ods_acc >= 4*(1024**2):
                logger.warning(f"Decoded obect buffer overrun at {seg.pts} [s].")
                compliant = False

            #We clear the plane (window area) and copy the objects to window. This is done at 32MiB/s
            Rc = fps*(sum(window_area.values()) + np.min([ods_acc, sum(window_area.values())]))
            nf = TC.s2f(seg.pts, fps) - TC.s2f(prev_pts, fps)
            if nf == 0:
                last_rc += Rc
            elif (last_rc+Rc)/nf > 1920*1080/4*29.97*2:
                logger.warning(f"Graphic plane overloaded. Graphics may flicker at {seg.pts} [s].")
                warnings += 1

    if warnings == 0 and compliant:
        logger.info("Output PGS seems compliant.")
    if warnings > 0 and compliant:
        logger.warning(f"Excessive bandwidth detected, requires HW testing (PGS may go out of sync).")
    elif not compliant:
        logger.error("PGStream will crash a HW decoder.")
    return compliant
