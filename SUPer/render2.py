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

from typing import TypeVar, Optional, Type, Union, Callable
from dataclasses import dataclass
from itertools import combinations, chain, zip_longest

from PIL import Image
from SSIM_PIL import compare_ssim
from numpy import typing as npt
import numpy as np

from skimage.filters import gaussian
from skimage.measure import regionprops, label

#%%
from .utils import get_super_logger, Pos, Dim, BDVideo, TimeConv as TC
from .filestreams import BDNXMLEvent, BaseEvent
from .segments import DisplaySet, PCS, WDS, PDS, ODS, ENDS, WindowDefinition, CObject, Epoch
from .optim import Optimise
from .pgraphics import PGraphics, PGDecoder, PGObject, PGObjectBuffer, PaletteManager
from .palette import Palette, PaletteEntry

_Region = TypeVar('Region')
logger = get_super_logger('SUPer')
skip_dts = False

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
    def coords(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)

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

    @classmethod
    def from_coords(cls, x1: int, y1: int, t1: int, x2: int, y2: int, t2: int, region: _Region) -> 'ScreenRegion':
        return cls(min(y1, y2), abs(y2-y1), min(x1, x2), abs(x2-x1), min(t1, t2), abs(t2-t1), region=region)
####

class WindowOnBuffer:
    def __init__(self, screen_regions: list[ScreenRegion], duration: int) -> None:
        self.srs = screen_regions
        self.duration = duration

    def get_window(self) -> Box:
        mxy = np.asarray([np.inf, np.inf])
        Mxy = np.asarray([-1, -1])
        for sr in self.srs:
            mxy[:] = np.min([np.asarray((sr.y,  sr.x)),  mxy], axis=0)
            Mxy[:] = np.max([np.asarray((sr.y2, sr.x2)), Mxy], axis=0)
        mxy, Mxy = np.int32((mxy, Mxy))
        return Box(mxy[0], max(Mxy[0]-mxy[0], 8), mxy[1], max(Mxy[1]-mxy[1], 8))

    def area(self) -> int:
        return self.get_window().area
####

#%%
class GroupingEngine:
    def __init__(self, n_groups: int = 2, **kwargs) -> None:
        if n_groups not in range(1, 3):
            raise AssertionError(f"GroupingEngine expects 1 or 2 groups, not '{n_groups}'.")

        self.n_groups = n_groups
        self.candidates = kwargs.pop('candidates', 25)

        self.no_blur = True
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

        gs_blur = np.zeros((1, h, w), dtype=np.uint8)
        gs_orig = np.zeros_like(gs_blur)

        for k, event in enumerate(group):
            slice_x = slice(event.x-pxtl, event.x-pxtl+event.width)
            slice_y = slice(event.y-pytl, event.y-pytl+event.height)
            alpha = np.array(event.img.getchannel('A'), dtype=np.uint8)
            event.unload()
            blurred = (255*gaussian(alpha, (blur_c + blur_mul*ratio_how, blur_c + blur_mul*ratio_woh)))
            blurred[blurred <= 0.25] = 0
            blurred[blurred > 0.25] = 1
            alpha[alpha > 0] = 1
            gs_blur[0, slice_y, slice_x] |= (blurred > 0)
            gs_orig[0, slice_y, slice_x] |= (alpha > 0)
        return regionprops(label(gs_blur)), gs_orig, box

    @staticmethod
    def _get_combinations(n_regions: int) -> map:
        #If we have two composition objects, we want to find out the smallest 2 areas
        # that englobes all the screen regions. We generate all possible arrangement
        region_ids = range(n_regions)
        arrangements = map(lambda combination: set(filter(lambda region_id: region_id >= 0, combination)),
                                   set(combinations(list(region_ids) + [-1]*(n_regions-2), n_regions-1)))
        return arrangements

    def group_and_sort(self, srs: list[ScreenRegion], duration: int) -> list[tuple[WindowOnBuffer]]:
        """
        Seek for minimum areas from the regions, sort them and return them sorted,
        ascending area size. The caller will then choose the best area.
        """
        windows, areas = {}, {}
        n_regions = len(srs)

        if n_regions == 1 or self.n_groups == 1:
            return [(WindowOnBuffer(srs, duration=duration),)]
        elif n_regions > 16:
            return None

        for key, arrangement in enumerate(__class__._get_combinations(n_regions)):
            arr_sr, other_sr = [], []
            for k, sr in enumerate(srs):
                (arr_sr if k in arrangement else other_sr).append(sr)
            windows[key] = (WindowOnBuffer(arr_sr, duration=duration), WindowOnBuffer(other_sr, duration=duration))
            areas[key] = sum(map(lambda wb: wb.area(), windows[key]))

        output = []
        #Here, we can sort by ascending area – first has the smallest windows
        # we also discard overlapping windows
        for k, _ in sorted(areas.items(), key=lambda x: x[1]):
            if len(windows[k]) == 1 or 0 == windows[k][0].get_window().overlap_with(windows[k][1].get_window()):
                output.append(windows[k])
        return output if len(output) else None

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

            wobs = self.group_and_sort(tbox, len(subgroup))
            if wobs is None:
                if self.no_blur:
                    self.no_blur = False
                else:
                    self.blur_mul += 0.33
                    self.blur_c += 0.33
        if wobs is None:
            logger.warning("Grouping Engine giving up optimising layout. Using a single window.")
            wobs = [(WindowOnBuffer(tbox, duration=len(subgroup)),)]
        return tuple(sorted(wobs[0], key=lambda x: x.srs[0].t)), box

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
    def __init__(self, wobs: tuple[WindowOnBuffer], events: list[BDNXMLEvent], box: Box, fps: Union[float, int], bdn: ..., **kwargs):
        self.wobs = wobs
        self.events = events
        self.box = box
        self.target_fps = fps
        self.bdn = bdn
        self.kwargs = kwargs
        self.buffer = PGObjectBuffer()

    def mask_event(self, window, event) -> Optional[npt.NDArray[np.uint8]]:
        if event is not None:
            #+8 for minimum object width and height
            work_plane = np.zeros((self.box.dy+8, self.box.dx+8, 4), dtype=np.uint8)

            hsi = slice(event.x-self.box.x, event.x-self.box.x+event.width)
            vsi = slice(event.y-self.box.y, event.y-self.box.y+event.height)
            work_plane[vsi, hsi, :] = np.array(event.img, dtype=np.uint8)
            event.unload() #Help a bit to save on RAM

            return work_plane[window.y:window.y2, window.x:window.x2, :]
        return None

    def analyze(self):
        global skip_dts
        skip_dts = not self.kwargs.get('enforce_dts', True)
        allow_normal_case = self.kwargs.get('normal_case_ok', False)
        woba = []
        pm = PaletteManager()

        #Init
        gens, windows = [], []
        for k, swob in enumerate(self.wobs):
            woba.append(WOBAnalyzer(swob))
            windows.append(swob.get_window())
            gens.append(woba[k].analyze())
            next(gens[-1])

        #get all windowed bitmaps
        pgobjs = [[] for k in range(len(windows))]
        for event in chain(self.events, [None]*2):
            for wid, (window, gen) in enumerate(zip(windows, gens)):
                try:
                    pgobj = gen.send(self.mask_event(window,  event))
                except StopIteration:
                    pgobj = None
                if pgobj is not None:
                    pgobjs[wid].append(pgobj)
        pgobjs_proc = [objs.copy() for objs in pgobjs]

        acqs, absolutes, margins, durs, nodes, flags = self.find_acqs(pgobjs_proc, windows)
        states = [PCS.CompositionState.NORMAL] * len(acqs)
        states[0] = PCS.CompositionState.EPOCH_START
        drought = 0

        thresh = self.kwargs.get('quality_factor', 0.75)
        dthresh = self.kwargs.get('dquality_factor', 0.035)
        refresh_rate = max(0, min(self.kwargs.get('refresh_rate', 1.0), 1.0))

        for k, (acq, forced, margin) in enumerate(zip(acqs[1:], absolutes[1:], margins[1:]), 1):
            if thresh < 0:
                states[k] = PCS.CompositionState.ACQUISITION
                absolutes[k] = True
                continue
            if (forced or (acq and margin > max(thresh-dthresh*drought, 0))):
                states[k] = PCS.CompositionState.ACQUISITION
                drought = 0
            else:
                #try to not do too many acquisitions, as we want to compress the stream.
                drought += 1*refresh_rate

        allow_overlaps = not self.kwargs.get('no_overlap', False)

        #At this point, we have the stream acquisition shaped nicely
        # except that some of them may be impossible. We apply a final filtering
        # step to either discard the impossible events or shift some PG operations
        k = len(states)-1
        while k > 0 and not skip_dts:
            if not (absolutes[k] and not acqs[k] and flags[k] == 0):
                k -= 1
                continue

            mask = nodes[k].new_mask.copy()
            dts_start_nc = dts_start = nodes[k].dts()
            dropped_nc = dropped = 0
            j_nc = j = k - 1
            while j > 0 and nodes[j].dts_end() >= dts_start:
                if absolutes[j] is True:
                    assert len(nodes[j].new_mask) == len(mask)
                    for km, mask_v in enumerate(nodes[j].new_mask):
                        mask[km] |= mask_v
                    dropped += 1
                j -= 1

            #Normal case is only possible if we discard past acquisitions that redefined the same object
            normal_case_possible = sum(nodes[k].new_mask) == sum(mask) == 1 and sum(map(lambda x: x is not None, nodes[k].objects)) == 2
            normal_case_possible &= allow_normal_case
            if normal_case_possible:
                nodes[k].partial = True
                dts_start_nc = nodes[k].dts()
                while j_nc > 0 and nodes[j_nc].dts_end() >= dts_start:
                    if absolutes[j_nc] is True and nodes[j_nc].dts_end() >= dts_start_nc:
                        dropped_nc += 1
                    j_nc -= 1
                nodes[k].partial = False
            nc_not_ok = normal_case_possible and j_nc == 0 and nodes[j_nc].dts_end() >= dts_start_nc
            # we can't delete or move epoch start -> delete self if we can't shift it forward
            if nc_not_ok or (not normal_case_possible and j == 0 and nodes[j].dts_end() >= dts_start):
                if durs[k][1] > 0:
                    logger.info(f"Discarded screen wipe before {self.events[k].tc_in} as it collides with epoch start.")
                    durs[k] = (durs[k][0], 0) # Drop screen wipe (we can't do it!)

                #If this event is long enough, we shift it forward in time.
                wipe_area = nodes[j].wipe_duration()
                worst_dur = (np.ceil(wipe_area*2) + 3)
                if durs[k][0]*1/self.bdn.fps > np.ceil(worst_dur*2+PGDecoder.FREQ/self.bdn.fps)/PGDecoder.FREQ:
                    nodes[k].tc_shift = int(np.ceil(worst_dur/PGDecoder.FREQ*self.bdn.fps))
                    logger.warning(f"Shifted event at {self.events[k].tc_in} by +{nodes[k].tc_shift} frames to account for epoch start and compliancy.")
                    #wipe all events in between epoch start and this point
                    for ze in range(j+1, k):
                        logger.warning(f"Discarded event at {self.events[ze].tc_in} to perform a mendatory acquisition right after epoch start.")
                        flags[ze] = -1
                else:
                    # event is short, we can't shift it so we just discard it.
                    logger.warning(f"Discarded event at {self.events[k].tc_in} colliding with epoch start.")
                    flags[k] = -1
                ze = k
                #We may have discarded an acquisition followed by NCs, we must find the new acquisition point.
                while (ze := ze+1) < len(states) and states[ze] != PCS.CompositionState.ACQUISITION:
                    if flags[ze] != -1 and nodes[ze].dts() > dts_start:
                        logger.info(f"Epoch start collision: promoted normal case to acquisition at {self.events[ze].tc_in}.")
                        states[ze] = PCS.CompositionState.ACQUISITION
                        for zek in range(k+1, ze):
                            if nodes[zek].parent is not None and nodes[zek].parent.dts_end() >= nodes[ze].dts():
                                durs[zek] = (durs[zek][0], 0) # Drop screen wipe (we can't do it!)
                                logger.info(f"Dropped screen wipe before {self.events[zek].tc_in} as it hinders the promoted acquisition point.")
                            if nodes[zek].dts_end() >= nodes[ze].dts():
                                flags[zek] = -1
                                logger.info(f"Dropped event at {self.events[zek].tc_in} as it hinders the promoted acquisition point.")
                        break
                k -= 1
                continue

            # Analyze normal case only if it is worthwile
            if dts_start_nc > dts_start and normal_case_possible and nodes[j].dts_end() > dts_start:
                num_pal_nc = 0
                objs = list(map(lambda x: x is not None, nodes[j_nc].objects))
                for l in range(j_nc+1, k):
                    if nodes[l].parent is not None:
                        if num_pal_nc < 7 and allow_overlaps:
                            if nodes[l].parent.dts() >= dts_start_nc:
                                nodes[l].parent.set_dts(dts_start_nc - 1/PGDecoder.FREQ)
                            num_pal_nc += 1
                        else:
                            logger.warning(f"Discarded screen wipe before {self.events[l].tc_in} to perform a mendatory acquisition.")
                            durs[l] = (durs[l][0], 0) # Drop screen wipe (we can't do it!)
                    for ko, (obj, mask) in enumerate(zip(nodes[l].objects, nodes[l].new_mask)):
                        objs[ko] &= (obj is not None) & (not mask)
                    # We ran out of palette or the objects are too different -> drop
                    if sum(objs) == 0 or num_pal_nc >= 7 or not allow_overlaps:
                        logger.warning(f"Discarded event at {self.events[l].tc_in} to perform a mendatory acquisition.")
                        flags[l] = -1
                    else:
                        num_pal_nc += 1
                        nodes[l].objects = []
                        if nodes[l].dts() >= dts_start_nc:
                            nodes[l].set_dts(dts_start_nc - 1/PGDecoder.FREQ)
                    states[l] = PCS.CompositionState.NORMAL
                states[k] = PCS.CompositionState.NORMAL
                nodes[k].partial = True
                flags[k] = 1
                logger.info(f"Screen refreshed with a NORMAL CASE at {self.events[k].tc_in} (tight timing).")
            else:
                num_pal = 0
                objs = list(map(lambda x: x is not None, nodes[j].objects))
                for l in range(j+1, k):
                    if nodes[l].parent is not None:
                        if num_pal < 7 and allow_overlaps:
                            if nodes[l].parent.dts() >= dts_start:
                                nodes[l].parent.set_dts(dts_start - 1/PGDecoder.FREQ)
                            num_pal += 1
                        else:
                            logger.warning(f"Discarded screen wipe before {self.events[l].tc_in} to perform a mendatory acquisition.")
                            durs[l] = (durs[l][0], 0) #Drop screen wipe (we can't do it!)
                    for ko, (obj, mask) in enumerate(zip(nodes[l].objects, nodes[l].new_mask)):
                        objs[ko] &= (obj is not None) & (not mask)
                    # We ran out of palette or the objects are too different -> drop
                    if sum(objs) == 0 or num_pal >= 7 or not allow_overlaps:
                        logger.warning(f"Discarded event at {self.events[l].tc_in} to perform a mendatory acquisition.")
                        flags[l] = -1
                    else:
                        num_pal += 1
                        nodes[l].objects = []
                        if nodes[l].dts() >= dts_start:
                            nodes[l].set_dts(dts_start - 1/PGDecoder.FREQ)
                        states[l] = PCS.CompositionState.NORMAL
                states[k] = PCS.CompositionState.ACQUISITION
            k -= 1
        ####

        #Allocate palettes as a test, this is essentially doing a final sanity check
        #on the selected display sets. The palette values generated here are not used.
        for k, (node, state, flag) in enumerate(zip(nodes, states, flags)):
            if flag == 0 and state == PCS.CompositionState.NORMAL:
                node.objects = []
                assert allow_overlaps or not node.is_custom_dts()
            if node.parent and durs[k][1] > 0:
                assert allow_overlaps or not node.parent.is_custom_dts()
                node.parent.palette_id = pm.get_palette(node.parent.dts())
                if not pm.lock_palette(node.parent.palette_id, node.parent.pts(), node.parent.dts()):
                    logger.error(f"Cannot acquire palette (rendering error) at {node.parent.pts():.05f}, discarding.")
                    assert durs[k][1] > 0
                    durs[k] = (durs[k][0], 0)
                else:
                    node.parent.pal_vn = pm.get_palette_version(node.parent.palette_id)
                logger.debug(f"{state} {flag} - {node.parent.partial} DTS={node.parent.dts():.05f} PTS={node.parent.pts():.05f} {node.parent.palette_id} {node.parent.pal_vn}")
            #Deleted event are skipped
            if flag == -1:
                continue
            node.palette_id = pm.get_palette(node.dts())
            if not pm.lock_palette(node.palette_id, node.pts(), node.dts()):
                logger.error(f"Cannot acquire palette (rendering error) at {nodes[k].pts()}, discarding.")
                flags[k] = -1
            else:
                node.pal_vn = pm.get_palette_version(node.palette_id)

            logger.debug(f"{state:02X} {flag} - {node.partial} DTS={node.dts():.05f}->{node.dts_end():.05f} PTS={node.pts():.05f} {node.palette_id} {node.pal_vn}")
        return self._convert(states, pgobjs, windows, durs, flags, nodes)

    @staticmethod
    def _get_stack_direction(*box) -> tuple[npt.NDArray[np.uint16], tuple[int, int]]:
        widths = list(map(lambda b: b.dx, box))
        heights = list(map(lambda b: b.dy, box))

        if max(heights)*sum(widths) <= max(widths)*sum(heights):
            return np.array([widths[0], 0], np.int32), (sum(widths), max(heights))
        return np.array([0, heights[0]], np.int32), (max(widths), sum(heights))

    def _generate_acquisition_ds(self, i: int, k: int, pgobs_items, windows: list[Box],
                                    double_buffering: int, has_two_objs: bool, is_compat_mode: bool,
                                    ods_reg: list[int], c_pts: float, normal_case_refresh: bool) -> ...:
        box_to_crop = lambda cbox: {'hc_pos': cbox.x, 'vc_pos': cbox.y, 'c_w': cbox.dx, 'c_h': cbox.dy}
        cobjs, cobjs_cropped = [], []
        pals, o_ods = [], []

        #In this mode, we re-combine the two objects in a smaller areas than in the original box
        # and then pass that to the optimiser. Colors are efficiently distributed on the objects.
        # In the future, this will be the default behaviour unless there's a NORMAL CASE to update
        # to redefine an object in the middle.
        if has_two_objs and normal_case_refresh is False:
            compositions = [pgo for _, pgo in pgobs_items if not (pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f]))]

            offset, dims = self.__class__._get_stack_direction(*list(map(lambda x: x.box, compositions)))
            imgs_chain = []

            for j in range(i, k):
                coords = np.zeros((2,), np.int32)
                a_img = Image.new('RGBA', dims, (0, 0, 0, 0))
                for pgo in compositions:
                    if len(pgo.mask[j-pgo.f:j+1-pgo.f]) == 1:
                        paste_box = (coords[0], coords[1], coords[0]+pgo.box.dx, coords[1]+pgo.box.dy)
                        a_img.paste(Image.fromarray(pgo.gfx[j-pgo.f, :, : ,:], 'RGBA').crop(pgo.box.coords), paste_box)
                    coords += offset
                imgs_chain.append(a_img)
            ####
            #We have the "packed" object, let's optimise it
            # 254 colors because 0x00 encoding is annoying and 0xFF is reserved for Scenarist
            bitmap, palettes = Optimise.solve_sequence_fast(imgs_chain, 254, **self.kwargs)
            pals.append(Optimise.diff_cluts(palettes, matrix=self.kwargs.get('bt_colorspace', 'bt709')))

            #Discard palette entry 0
            for pal in pals[-1]:
                pal.offset(1)
            bitmap += 1

            coords = np.zeros((2,), np.int32)

            for wid, pgo in pgobs_items:
                if not (pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f])):
                    oid = wid + double_buffering[wid]
                    double_buffering[wid] = abs(2 - double_buffering[wid])
                    #get bitmap
                    window_bitmap = np.zeros((windows[wid].dy, windows[wid].dx), np.uint8)
                    nx, ny = coords
                    window_bitmap[pgo.box.slice] = bitmap[ny:ny+pgo.box.dy, nx:nx+pgo.box.dx]

                    #Generate object related segments objects
                    cobjs.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x, windows[wid].y+self.box.y, False))
                    cparams = box_to_crop(pgo.box)
                    cobjs_cropped.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x+cparams['hc_pos'], windows[wid].y+self.box.y+cparams['vc_pos'], False,
                                                              cropped=True, **cparams))

                    ods_data = PGraphics.encode_rle(window_bitmap)
                    o_ods += ODS.from_scratch(oid, ods_reg[oid] & 0xFF, window_bitmap.shape[1], window_bitmap.shape[0], ods_data, pts=c_pts)
                    ods_reg[oid] += 1
                    coords += offset
            pals.append([Palette()] * len(pals[0]))
            ####for wid, pgo
        else:
            # If in the chain there's a NORMAL CASE redefinition, we
            # must work with separate palette for each object (127 colors per window)
            id_skipped = None
            for wid, pgo in pgobs_items:
                if pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f]):
                    continue
                n_colors = 127 if has_two_objs else 254

                if isinstance(normal_case_refresh, list) and not normal_case_refresh[wid]:
                    assert sum(normal_case_refresh) == 1
                    #Take latest used object id
                    oid = wid + abs(2 - double_buffering[wid])
                    cobjs.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x, windows[wid].y+self.box.y, False))
                    cparams = box_to_crop(pgo.box)
                    cobjs_cropped.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x+cparams['hc_pos'], windows[wid].y+self.box.y+cparams['vc_pos'],
                                                              False, cropped=True, **cparams))
                    pals.append([Palette()] * (k-i))
                    id_skipped = oid
                    continue
                oid = wid + double_buffering[wid]
                double_buffering[wid] = abs(2 - double_buffering[wid])
                imgs_chain = [Image.fromarray(img) for img in pgo.gfx[i-pgo.f:k-pgo.f]]
                cobjs.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x, windows[wid].y+self.box.y, False))

                cparams = box_to_crop(pgo.box)
                cobjs_cropped.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x+cparams['hc_pos'], windows[wid].y+self.box.y+cparams['vc_pos'], False,
                                                          cropped=True, **cparams))

                wd_bitmap, wd_pal = Optimise.solve_sequence_fast(imgs_chain, n_colors, **self.kwargs)
                wd_bitmap += 1 + (127*(wid == 1 and has_two_objs))

                pals.append(Optimise.diff_cluts(wd_pal, matrix=self.kwargs.get('bt_colorspace', 'bt709')))
                ods_data = PGraphics.encode_rle(wd_bitmap)

                if wid == 1 and has_two_objs:
                    assert len(pals) == 2
                    for p in pals[-1]:
                        p.offset(128)
                elif wid == 0 or not has_two_objs:
                    for p in pals[-1]:
                        p.offset(1)

                #On normal case, we generate one chain of palette update and
                #add in a screen wipe if necessary. This is not used if the object is changed.
                if normal_case_refresh and len(pals[-1]) < k-i:
                    mibm, mabm = np.min(wd_bitmap), np.max(wd_bitmap)
                    pals[-1].append(Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(mibm, mabm+1)}))
                    pals[-1].extend([Palette()] * ((k-i)-len(pals[-1])))

                o_ods += ODS.from_scratch(oid, ods_reg[oid] & 0xFF, wd_bitmap.shape[1], wd_bitmap.shape[0], ods_data, pts=c_pts)
                ods_reg[oid] += 1
            if id_skipped is not None:
                assert isinstance(normal_case_refresh, list)
                #The existing object should be the first composition (so key has to eval to False for it to be first)
                cobjs_cropped = sorted(cobjs_cropped, key=lambda cobj: cobj.o_id != id_skipped)
                cobjs = sorted(cobjs, key=lambda cobj: cobj.o_id != id_skipped)

        #Set the 0x00 entry once. It should never change during the epoch anyway.
        pals[0][0][0] = PaletteEntry(16, 128, 128, 0)
        pal = pals[0][0]
        if has_two_objs:
            pal |= pals[1][0]
        else:
            pals.append([Palette()] * len(pals[0]))

        return cobjs if is_compat_mode else cobjs_cropped, pals, o_ods, pal

    def _get_undisplay(self, c_pts: float, pcs_id: int, wds_base: WDS, palette_id: int, pcs_fn: Callable[[...], PCS]) -> DisplaySet:
        pcs = pcs_fn(pcs_id, PCS.CompositionState.NORMAL, False, palette_id, [], c_pts)
        wds = wds_base.copy(pts=c_pts, in_ticks=False)
        uds = DisplaySet([pcs, wds, ENDS.from_scratch(pts=c_pts)])
        apply_pts_dts(uds, set_pts_dts_sc(uds, self.buffer, wds))
        return uds

    def _get_undisplay_pds(self, c_pts: float, pcs_id: int, node: 'DSNode', cobjs: list[CObject],
                           pcs_fn: Callable[[...], PCS], n_colors: int, wds_base: WDS) -> tuple[DisplaySet, int, int]:
        pcs = pcs_fn(pcs_id, PCS.CompositionState.NORMAL, True, node.palette_id, cobjs, c_pts)
        tsp_e = PaletteEntry(16, 128, 128, 0)
        pds = PDS.from_scratch(Palette({k: tsp_e for k in range(n_colors)}), p_vn=node.pal_vn, p_id=node.palette_id, pts=c_pts)
        uds = DisplaySet([pcs, pds, ENDS.from_scratch(pts=c_pts)])
        apply_pts_dts(uds, set_pts_dts_sc(uds, self.buffer, wds_base, node))
        return uds, pcs_id+1

    def _convert(self, states, pgobjs, windows, durs, flags, nodes):
        wd_base = [WindowDefinition.from_scratch(k, w.x+self.box.x, w.y+self.box.y, w.dx, w.dy) for k, w in enumerate(windows)]
        wds_base = WDS.from_scratch(wd_base, pts=0.0)
        n_actions = len(durs)
        is_compat_mode = self.kwargs.get('pgs_compatibility', True)
        displaysets = []
        time_scale = 1.001 if self.kwargs.get('adjust_dropframe', False) else 1
        use_full_pal = self.kwargs.get('full_palette', False)
        palette_manager = PaletteManager()

        ## Internal helper function
        def get_obj(frame, pgobjs: dict[int, list[PGObject]]) -> dict[int, Optional[PGObject]]:
            objs = {k: None for k, objs in enumerate(pgobjs)}

            for wid, pgobj in enumerate(pgobjs):
                for obj in pgobj:
                    if obj.is_active(frame):
                        objs[wid] = obj
            return objs

        def get_palette_data(pal_manager: PaletteManager, node: DSNode) -> tuple[int, int]:
            pal_id = pal_manager.get_palette(node.dts())
            assert pal_manager.lock_palette(pal_id, node.pts(), node.dts())
            return pal_id, pal_manager.get_palette_version(pal_id)

        ####

        i = 0
        double_buffering = [0]*2
        ods_reg = [0]*4
        pcs_id = 0
        c_pts = 0
        last_cobjs = []
        last_palette_id = -1

        get_pts: Callable[[float], float] = lambda c_pts: max(c_pts - (1/3)/PGDecoder.FREQ, 0) * time_scale
        pcs_fn = lambda pcs_cnt, state, pal_flag, palette_id, cl, pts:\
                    PCS.from_scratch(*self.bdn.format.value, BDVideo.LUT_PCS_FPS[round(self.target_fps, 3)], pcs_cnt & 0xFFFF, state, pal_flag, palette_id, cl, pts=pts)

        final_node = DSNode([None, None], windows, TC.tc2s(self.events[-1].tc_out, self.bdn.fps), None, scale_pts=time_scale)
        # last_legal_pts = final_node.pts() - final_node.write_duration()

        try:
            use_pbar = False
            from tqdm import tqdm
        except ModuleNotFoundError:
            from contextlib import nullcontext as tqdm
        else:
            use_pbar = True

        pbar = tqdm(range(n_actions))
        while i < n_actions:
            if flags[i] == -1:
                i+=1
                continue
            assert states[i] != PCS.CompositionState.NORMAL
            normal_case_refresh = False
            for k in range(i+1, n_actions+1):
                if k < n_actions:
                    normal_case_refresh |= (flags[k] == 1)
                if k == n_actions or states[k] != PCS.CompositionState.NORMAL:
                    break
            assert k > i

            if durs[i][1] != 0:
                assert i > 0
                assert nodes[i].parent is not None
                p_id, p_vn = get_palette_data(palette_manager, nodes[i].parent)
                nodes[i].parent.palette_id = p_id
                nodes[i].parent.pal_vn = p_vn
                uds, pcs_id = self._get_undisplay_pds(get_pts(TC.tc2s(self.events[i-1].tc_out, self.bdn.fps)), pcs_id, nodes[i].parent, last_cobjs, pcs_fn, 255, wds_base)
                displaysets.append(uds)

            if nodes[i].tc_shift == 0:
                c_pts = get_pts(TC.tc2s(self.events[i].tc_in, self.bdn.fps))
            else:
                c_pts = get_pts(TC.add_frames(self.events[i].tc_in, self.bdn.fps, nodes[i].tc_shift))
            nodes[i].tc_pts = c_pts

            pgobs_items = get_obj(i, pgobjs).items()
            has_two_objs = 0
            for wid, pgo in pgobs_items:
                if pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f]):
                    continue
                has_two_objs += 1

            #Normal case refresh implies we are refreshing one object out of two displayed.
            has_two_objs = has_two_objs > 1 or normal_case_refresh

            r = self._generate_acquisition_ds(i, k, pgobs_items, windows, double_buffering, has_two_objs, is_compat_mode, ods_reg, c_pts, normal_case_refresh)
            cobjs, pals, o_ods, pal = r

            wds = wds_base.copy(pts=c_pts, in_ticks=False)
            p_id, p_vn = get_palette_data(palette_manager, nodes[i])
            pds = PDS.from_scratch(pal, p_vn=p_vn, p_id=p_id, pts=c_pts)
            pcs = pcs_fn(pcs_id, states[i], False, p_id, cobjs, c_pts)

            nds = DisplaySet([pcs, wds, pds] + o_ods + [ENDS.from_scratch(pts=c_pts)])
            apply_pts_dts(nds, set_pts_dts_sc(nds, self.buffer, wds, nodes[i]))
            displaysets.append(nds)

            pcs_id += 1
            last_palette_id = p_id

            if len(pals[0]) > 1:
                # Pad palette chains
                if not normal_case_refresh:
                    zip_length = max(map(len, pals))
                    if len(pals[0]) < zip_length:
                        pals[0] += [Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(min(pals[0][0].palette), max(pals[0][0].palette)+1)})]
                    if has_two_objs and len(pals[1]) < zip_length:
                        pals[1] += [Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(min(pals[1][0].palette), max(pals[1][0].palette)+1)})]
                pals[0] += [Palette()] * (k-i - len(pals[0]))
                pals[1] += [Palette()] * (k-i - len(pals[1]))

                for z, (p1, p2) in enumerate(zip_longest(pals[0][1:], pals[1][1:], fillvalue=Palette()), i+1):
                    c_pts = get_pts(TC.tc2s(self.events[z].tc_in, self.bdn.fps))
                    assert states[z] == PCS.CompositionState.NORMAL
                    pal |= pals[0][z-i] | pals[1][z-i]

                    #Is there a know screen clear in the chain? then use palette screen clear here
                    if durs[z][1] != 0:
                        assert nodes[z].parent is not None
                        p_id, p_vn = get_palette_data(palette_manager, nodes[z].parent)
                        nodes[z].parent.palette_id = p_id
                        nodes[z].parent.pal_vn = p_vn
                        uds, pcs_id = self._get_undisplay_pds(get_pts(TC.tc2s(self.events[z-1].tc_out, self.bdn.fps)), pcs_id, nodes[z].parent, cobjs, pcs_fn, max(pal.palette)+1, wds_base)
                        displaysets.append(uds)
                        #We just wipped a palette, whatever the next palette id, rewrite it fully
                        last_palette_id = None

                    if flags[z] == 1:
                        normal_case_refresh = nodes[z].new_mask
                        r = self._generate_acquisition_ds(z, k, get_obj(z, pgobjs).items(), windows, double_buffering, has_two_objs, is_compat_mode, ods_reg, c_pts, normal_case_refresh)
                        cobjs, n_pals, o_ods, new_pal = r
                        pal |= new_pal
                        for nz, (new_p1, new_p2) in enumerate(zip_longest(n_pals[0], n_pals[1], fillvalue=Palette()), z):
                            pals[0][nz-i] |= new_p1
                            pals[1][nz-i] |= new_p2
                        normal_case_refresh = True
                        last_palette_id = None
                    elif flags[z] == -1:
                        continue

                    p_write = (pals[0][z-i] | pals[1][z-i])
                    #Skip empty palette updates
                    if len(p_write) == 0:
                        logger.debug("Skipped an empty palette.")
                        continue

                    p_id, p_vn = get_palette_data(palette_manager, nodes[z])
                    #If the palette ID change, we must give the full palette.
                    if last_palette_id != p_id or use_full_pal:
                        p_write = pal

                    pcs = pcs_fn(pcs_id, states[z], flags[z] != 1, p_id, cobjs, c_pts)
                    pds = PDS.from_scratch(p_write, p_vn=p_vn, p_id=p_id, pts=c_pts)
                    wds_upd = [wds_base.copy(pts=c_pts, in_ticks=False)] if flags[z] == 1 else []
                    ods_upd = o_ods if flags[z] == 1 else []

                    nds = DisplaySet([pcs] + wds_upd + [pds] + ods_upd +[ENDS.from_scratch(pts=c_pts)])
                    apply_pts_dts(nds, set_pts_dts_sc(nds, self.buffer, wds_base, nodes[z]))
                    displaysets.append(nds)

                    pcs_id += 1
                    last_palette_id = p_id
                    if z+1 == k:
                        break
                assert z+1 == k
            i = k
            last_cobjs = cobjs
            if use_pbar:
                pbar.n = i
                pbar.update()
        if use_pbar:
            pbar.close()
        ####while
        #final "undisplay" displayset
        p_id, p_vn = get_palette_data(palette_manager, final_node)
        final_node.palette_id = p_id
        final_node.pal_vn = p_vn
        uds, _ = self._get_undisplay_pds(get_pts(TC.tc2s(self.events[-1].tc_out, self.bdn.fps)), pcs_id, final_node, last_cobjs, pcs_fn, 255, wds_base)
        displaysets.append(uds)
        #displaysets.append(self._get_undisplay(get_pts(TC.tc2s(self.events[-1].tc_out, self.bdn.fps)), pcs_id, wds_base, last_palette_id, pcs_fn))
        return Epoch(displaysets)

    def find_acqs(self, pgobjs_proc: dict[..., list[...]], windows: list[Box]):
        #get the frame count between each screen update and find where we can do acqs
        durs, nodes = self.get_durations(windows)

        dtl = np.zeros((len(durs)), dtype=float)
        valid = np.zeros((len(durs),), dtype=np.bool_)
        absolutes = np.zeros_like(valid)
        flags = [0] * len(durs)

        objs = [None for objs in pgobjs_proc]

        prev_dt = 6
        for k, (dt, delay) in enumerate(durs):
            is_new = [False]*len(windows)
            margin = (delay + prev_dt)/self.bdn.fps
            force_acq = False
            for wid, wd in enumerate(windows):
                is_new[wid] = False
                if objs[wid] and not objs[wid].is_active(k):
                    objs[wid] = None
                if len(pgobjs_proc[wid]):
                    if not objs[wid] and pgobjs_proc[wid][0].is_active(k):
                        objs[wid] = pgobjs_proc[wid].pop(0)
                        force_acq = True
                        is_new[wid] = True
                    else:
                        assert not pgobjs_proc[wid][0].is_active(k)

            nodes[k].objects = objs.copy()
            nodes[k].new_mask = is_new

            if k == 0:
                prev_dts = -np.inf
            elif nodes[k].parent:
                prev_dts = nodes[k].parent.dts_end()
            else:
                prev_dts = nodes[k-1].dts_end()
            valid[k] = nodes[k].dts() > prev_dts
            absolutes[k] = force_acq
            dtl[k] = (nodes[k].dts() - prev_dts)/margin if valid[k] and k > 0 else (-1 + 2*(k==0))
            prev_dt = dt
        return valid, absolutes, dtl, durs, nodes, flags
    ####

    def get_durations(self, windows: list[Box]) -> npt.NDArray[np.uint32]:
        """
        Returns the duration of each event in frames.
        Additionally, the offset from the previous event is also returned. This value
        is zero unless there are no PG objects shown at some point in the epoch.
        """
        scale_pts = 1.001 if self.kwargs.get('adjust_dropframe', False) else 1
        top = TC.tc2f(self.events[0].tc_in, self.bdn.fps)
        delays = []
        nodes = []
        for ne, event in enumerate(self.events):
            tic = TC.tc2f(event.tc_in, self.bdn.fps)
            toc = TC.tc2f(event.tc_out,self.bdn.fps)
            delays.append((toc-tic, tic-top))
            parent = None
            if tic-top > 0:
                parent = DSNode([], windows, TC.tc2s(self.events[ne-1].tc_out, self.bdn.fps), scale_pts=scale_pts)
            nodes.append(DSNode([], windows, TC.tc2s(event.tc_in, self.bdn.fps), parent=parent, scale_pts=scale_pts))
            top = toc
        return delays, nodes
####
#%%

class WOBAnalyzer:
    def __init__(self, wob, ssim_threshold: float = 0.95, overlap_threshold: float = 0.995) -> None:
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

    def compare(self, bitmap: Image.Image, current: Image.Image) -> tuple[float, float]:
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

        if overlap > 0: #and overlap < self.overlap_threshold:
            #score = compare_ssim(bitmap.convert('L'), current.convert('L'))
            #Broadcast transparency mask of current on all channels of ref
            mask = 255*(np.logical_and((a_bitmap[:, :, 3] > 0), (a_current[:, :, 3] > 0)).astype(np.uint8))
            score = compare_ssim(Image.fromarray(a_bitmap & mask[:, :, None]).convert('L'), current.convert('L'))
            cross_percentage = np.sum(mask > 0)/mask.size
        else:
            #Perfect overlap or zero overlap, the current bitmap fits perfectly on the previous
            score = 1.0
            cross_percentage = 1.0
        return score, cross_percentage

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

                rgba_i = Image.fromarray(rgba)
                score, cross_percentage = self.compare(alpha_compo, rgba_i)
                if score >= max(1.0, self.ssim_threshold + (1-self.ssim_threshold)*(1-cross_percentage)) - 0.008333:
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
                unseen = (not has_content)*(unseen + 1)
            event_cnt += 1
        ####while
        return # StopIteration
####
class DSNode:
    def __init__(self,
            objects: list[Optional[PGObject]],
            windows: list[Box],
            tc_pts: float,
            parent: Optional['DSNode'] = None,
            scale_pts: float = 1
        ) -> None:
        self.objects = objects
        self.windows = windows
        self.tc_pts = max(tc_pts - (1/3)/PGDecoder.FREQ, 0) * scale_pts
        self.new_mask = []
        self.parent = parent
        self.partial = False
        self.tc_shift = 0

        self._pal_id = None
        self.pal_vn = 0
        self._dts = None

    def wipe_duration(self) -> int:
        return np.ceil(sum(map(lambda w: PGDecoder.FREQ*w.dy*w.dx/PGDecoder.RC, self.windows)))

    def write_duration(self) -> int:
        return sum(map(lambda w: np.ceil(PGDecoder.FREQ*w.dy*w.dx/PGDecoder.RC), self.windows))

    def set_dts(self, dts: Optional[float]) -> None:
        assert dts is None or dts <= self.dts()
        self._dts = round(dts*PGDecoder.FREQ) if dts is not None else None

    @property
    def palette_id(self) -> Optional[int]:
        return self._pal_id

    @palette_id.setter
    def palette_id(self, palette_id: Optional[int]) -> None:
        self._pal_id = palette_id

    def dts_end(self) -> float:
        if self._dts is not None:
            return (self.get_dts_markers()[1] + self._dts)/PGDecoder.FREQ
        return sum(self.get_dts_markers())/PGDecoder.FREQ

    def dts(self) -> float:
        if self._dts is not None:
            return self._dts/PGDecoder.FREQ
        return self.get_dts_markers()[0]/PGDecoder.FREQ

    def delta_dts(self) -> float:
        return self.get_dts_markers()[1]/PGDecoder.FREQ

    def pts(self) -> float:
        return self.tc_pts

    def is_custom_dts(self) -> bool:
        return not (self._dts is None)

    def get_dts_markers(self) -> float:
        t_decoding = 0
        decode_duration = self.wipe_duration()

        for wid, obj in enumerate(self.objects):
            if obj is None:
                continue
            box = self.windows[wid]
            read = box.dy*box.dx*PGDecoder.FREQ
            if not self.partial or (self.partial and self.new_mask[wid]):
                dec = np.ceil(read/PGDecoder.RD)
                t_decoding += dec

            cop = np.ceil(read/PGDecoder.RC)
            decode_duration = max(decode_duration, t_decoding) + cop
        ####

        if t_decoding == 0:
            decode_duration = self.write_duration() + 1
        return (round(self.tc_pts*PGDecoder.FREQ) - decode_duration, t_decoding)

#%%
def get_wipe_duration(wds: WDS) -> int:
    return np.ceil(sum(map(lambda w: PGDecoder.FREQ*w.height*w.width/PGDecoder.RC, wds.windows)))

#%%
def set_pts_dts_sc(ds: DisplaySet, buffer: PGObjectBuffer, wds: WDS, node: Optional['DSNode'] = None) -> list[tuple[int, int]]:
    """
    This function generates the timestamps (PTS and DTS) associated to a given DisplaySet.

    :param ds: DisplaySet, PTS of PCS must be set to the right value.
    :param buffer: Object buffer that supports allocation and returning a size of allocated slots.
    :param lwd: list of windows used in the epoch.
    :return: Pairs of timestamps in ticks for each segment in the displayset.
    """
    ddurs = {}
    for ods in ds.ods:
        if ods.flags & ods.ODSFlags.SEQUENCE_FIRST:
            assert ods.o_id not in ddurs, f"Object {ods.o_id} defined twice in DS."
            if (shape := buffer.get(ods.o_id)) is not None:
                assert (ods.height, ods.width) == shape, "Dimension mismatch, buffer corruption."
            else:
                # Allocate a buffer slot for this object
                assert buffer.allocate_id(ods.o_id, ods.height, ods.width) is True, "Slot already allocated or buffer overflow."
            ddurs[ods.o_id] = np.ceil(ods.height*ods.width*PGDecoder.FREQ/PGDecoder.RD)

    t_decoding = 0
    decode_duration = 0
    wipe_duration = get_wipe_duration(wds)

    if ds.ods:
        if ds.wds:
            if ds.pcs.composition_state == ds.pcs.CompositionState.EPOCH_START:
                decode_duration = np.ceil(ds.pcs.width*ds.pcs.height*PGDecoder.FREQ/PGDecoder.RC)
            else:
                decode_duration = wipe_duration
            object_decode_duration = ddurs.copy()

            windows = {wd.window_id: (wd.height, wd.width) for wd in ds.wds.windows}

            #For every composition object, compute the transfer time
            for k, cobj in enumerate(ds.pcs.cobjects):
                shape = buffer.get(cobj.o_id)
                assert shape is not None, "Object does not exist in buffer."
                w, h = windows[cobj.window_id][0], windows[cobj.window_id][1]

                t_decoding += object_decode_duration.pop(cobj.o_id, 0)

                # Same window -> patent claims the plane is written only once after the two cobj are processed.
                if k == 0 and ds.pcs.n_objects > 1 and ds.pcs.cobjects[1] == cobj.window_id:
                    continue
                copy_dur = np.ceil(w*h*PGDecoder.FREQ/PGDecoder.RC)
                decode_duration = max(decode_duration, t_decoding) + copy_dur
        #This DS defines new objects without displaying them
        elif not ds.pcs.pal_flag:
            decode_duration = sum(ddurs.values())
        #pal flag with ODS, not desirable.
        else:
            raise AssertionError("Illegal DS (palette update with ODS!)")
    else:
        #No decoding required, just set the apart PTS and DTS according to the graphic plane access time.
        assert ds.pcs.composition_state == ds.pcs.CompositionState.NORMAL, "DS refreshes the screen but no valid object exists."
        decode_duration = sum(map(lambda w: np.ceil(PGDecoder.FREQ*w.height*w.width/PGDecoder.RC), wds.windows)) + 1

    mask = ((1 << 32) - 1)
    dts = int(ds.pcs.tpts - decode_duration) & mask
    if node is not None and node.is_custom_dts():
        new_dts = round(node.dts()*PGDecoder.FREQ)
        assert new_dts <= dts
        dts = new_dts

    #PCS always exist
    ts_pairs = [(ds.pcs.tpts, dts)]

    if ds.wds:
        ts_pairs.append((int(ds.pcs.tpts - wipe_duration) & mask, dts))
    for pds in ds.pds:
        ts_pairs.append((dts, dts))

    for ods in ds.ods:
        ods_pts = int(dts + ddurs.get(ods.o_id)) & mask
        ts_pairs.append((ods_pts, dts))
        if ods.flags & ods.ODSFlags.SEQUENCE_LAST:
            dts = ods_pts
    ts_pairs.append((dts, dts))
    return ts_pairs

def apply_pts_dts(ds: DisplaySet, ts: tuple[int, int]) -> None:
    global skip_dts
    nullify_dts = lambda x: x*(0 if skip_dts else 1)
    select_pts = lambda x: ts[0][0] if skip_dts else x

    assert len(ds) == len(ts), "Timestamps-DS size mismatch."
    for seg, (pts, dts) in zip(ds, ts):
        seg.tpts, seg.tdts = select_pts(pts), nullify_dts(dts)

def test_diplayset(ds: DisplaySet) -> bool:
    """
    This function performs hard check on the display set
    if its structure is bad, it raises an assertion error.
    This is preferred over a "return false" because a bad displayset
    will typically crash a hardware decoder and we don't want that.

    :param ds: Display Set to test for structural compliancy
    """
    comply = ds.pcs is not None
    if ds.pcs.composition_state != PCS.CompositionState.NORMAL:
        comply &= ds.pcs.pal_flag is False # "Palette update on epoch start or acquisition."
        comply &= ds.pcs.pal_id < 8 # "Using undefined palette ID."
    if ds.wds:
        comply &= ds.pcs.pal_flag is False # "Manipulating windows on palette update (conflicting display updates)."
        comply &= len(ds.wds.windows) <= 2 # "More than two windows."
    if ds.pds:
        for pds in ds.pds:
            if ds.pcs.pal_flag:
                comply &= len(ds) == 3 # "Unusual display set structure for a palette update."
                comply &= ds.pcs.pal_id == pds.p_id # "Palette ID mismatch between PCS and PDS on palette update."
            comply &= pds.p_id < 8 # "Using undefined palette ID."
            comply &= pds.n_entries <= 256 # "Defining more than 256 palette entries."
    if ds.ods:
        start_cnt = close_cnt = 0
        for ods in ds.ods:
            start_cnt += bool(ods.flags & ODS.ODSFlags.SEQUENCE_FIRST)
            close_cnt += bool(ods.flags & ODS.ODSFlags.SEQUENCE_LAST)
        comply &= start_cnt == close_cnt # "ODS segments flags mismatch."
    return comply & (ds.end is not None) # "No END segment in DS."
####

def is_compliant(epochs: list[Epoch], fps: float, has_dts: bool = False) -> bool:
    ts_mask = ((1 << 32) - 1)
    prev_pts = -1
    last_cbbw = 0
    last_dbbw = 0
    compliant = True
    warnings = 0
    pal_id = 0
    cumulated_ods_size = 0

    coded_bw_ra_pts = [-1] * round(fps)
    coded_bw_ra = [0] * round(fps)

    to_tc = lambda pts: TC.s2tc(pts, fps)

    for ke, epoch in enumerate(epochs):
        prev_pcs_id = -1
        window_area = {}
        objects_sizes = {}
        ods_vn = {}
        pds_vn = [-1] * 8
        pals = [Palette() for _ in range(8)]
        buffer = PGObjectBuffer()

        compliant &= bool(epoch[0].pcs.composition_state & epoch[0].pcs.CompositionState.EPOCH_START)

        for kd, ds in enumerate(epoch.ds):
            compliant &= test_diplayset(ds)
            decoded_this_ds = 0
            coded_this_ds = 0

            current_pts = ds.pcs.pts
            if epoch.ds[kd-1].pcs.pts != prev_pts and current_pts != epoch.ds[kd-1].pcs.pts:
                prev_pts = epoch.ds[kd-1].pcs.pts
                last_cbbw, last_dbbw, last_rc = [0]*3
            else:
                logger.warning(f"Two displaysets at {to_tc(current_pts)} (internal rendering error?)")

            for ks, seg in enumerate(ds.segments):
                areas2gp = {}
                if has_dts and ((seg.tpts - seg.tdts) & ts_mask >= PGDecoder.FREQ):
                    logger.warning(f"Too large PTS-DTS difference for seg._type at {to_tc(current_pts)}.")

                if isinstance(seg, PCS):
                    pal_id = seg.pal_id
                    compliant &= (ks == 0) #PCS is not first in DisplaySet
                    if seg.composition_n == prev_pcs_id:
                        logger.warning(f"Displayset does not increment composition number. Composition will be ignored by HW decoder at {to_tc(seg.pts)}.")
                    prev_pcs_id = seg.composition_n
                    if int(seg.composition_state) != 0:
                        # On acquisition, the object buffer is flushed
                        objects_sizes = {}
                        pals = [Palette() for _ in range(8)]
                    for cobj in seg.cobjects:
                        areas2gp[cobj.o_id] = -1 if not cobj.cropped else cobj.c_w*cobj.c_h

                elif isinstance(seg, WDS):
                    compliant &= (ks == 1) #WDS is not second segment of DS, if present
                    for w in seg.windows:
                        window_area[w.window_id] = w.width*w.height
                elif isinstance(seg, PDS):
                    if (pds_vn[seg.p_id] + 1) & 0xFF != seg.p_vn:
                        logger.warning(f"Palette version not incremented by one, may be discarded by decoder. Palette {seg.p_id} at DTS {to_tc(seg.pts)}.")
                    pds_vn[seg.p_id] = seg.p_vn
                    pals[seg.p_id] |= seg.to_palette()
                elif isinstance(seg, ODS):
                    if seg.flags & ODS.ODSFlags.SEQUENCE_FIRST:
                        ods_data = bytearray()
                        ods_width = seg.width
                        ods_height = seg.height
                        if (slot := buffer.get(seg.o_id)) is None:
                            if not buffer.allocate_id(seg.o_id, seg.height, seg.width):
                                logger.error("Object buffer overflow (not enough memory for all object slots).")
                                compliant = False
                            ods_vn[seg.o_id] = seg.o_vn
                        elif slot != (seg.height, seg.width):
                            logger.error(f"Object-slot {seg.o_id} dimensions mismatch. Slot: {slot}, object: {(seg.height, seg.width)}")
                            compliant = False
                        elif ods_vn[seg.o_id] == seg.o_vn:
                            logger.warning(f"Object version not incremented, will be discarded by decoder. ODS {seg.o_id} at {to_tc(seg.pts)}.")
                        ods_vn[seg.o_id] = seg.o_vn
                        if cumulated_ods_size > 0:
                            logger.error("A past ODS was not properly terminated! Stream is critically corrupted!")
                            compliant = False
                        decoded_this_ds += seg.width * seg.height
                        coded_this_ds += seg.rle_len
                        objects_sizes[seg.o_id] = seg.width * seg.height

                    cumulated_ods_size += len(bytes(seg)[2:])
                    ods_data += seg.data

                    if seg.flags & ODS.ODSFlags.SEQUENCE_LAST:
                        if cumulated_ods_size > PGDecoder.CODED_BUF_SIZE:
                            logger.warning(f"Object size >1 MiB at {to_tc(seg.pts)} is unsupported by oldest decoders. UHD BD will be OK.")
                            warnings += 1
                        try:
                            next(filter(lambda x: len(x) >= ods_width + 16, PGraphics.get_rle_lines(ods_data, ods_width)))
                        except StopIteration:
                            ...
                        else:
                            logger.warning("ODS at {to_tc(seg.pts)} has too long RLE line(s). Oldest decoders may have issues.")
                            warnings += 1
                        for pe in np.unique(PGraphics.decode_rle(ods_data, width=ods_width, height=ods_height)):
                            if pe not in pals[pal_id].palette:
                                logger.warning("ODS at {to_tc(seg.pts)} uses undefined palette entries. Some pixels will not display.")
                                warnings += 1
                                break
                        cumulated_ods_size = 0
                elif isinstance(seg, ENDS):
                    compliant &= ks == len(ds)-1 # END segment is not last or not alone
                ####elif
            ####for
            if ds.pcs.pal_flag or ds.wds is not None:
                compliant &= pds_vn[ds.pcs.pal_id] >= 0 #Check that the used palette is indeed set in the decoder.

            area_copied = 0
            for idx, area in areas2gp.items():
                area_copied += area if area >= 0 else objects_sizes[idx]
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

            # This is probably the hardest constraint to meet: ts_packet are read at Rx=16Mbps
            if coded_buffer_bandwidth > (max_rate := PGDecoder.RX) and not has_dts:
                if coded_buffer_bandwidth/max_rate >= 2:
                    logger.warning(f"High instantaneous coded bandwidth at {to_tc(seg.pts)} (not critical - fair warning)")
                # This is not an issue unless it happens very frequently, so we don't mark as not compliant

            if prev_pts != seg.pts:
                coded_bw_ra = coded_bw_ra[1:round(fps)]
                coded_bw_ra_pts = coded_bw_ra_pts[1:round(fps)]
                coded_bw_ra.append(coded_buffer_pts)
                coded_bw_ra_pts.append(seg.pts)

            if (rate:=sum(coded_bw_ra)/abs(coded_bw_ra_pts[-1]-coded_bw_ra_pts[0])) > PGDecoder.RX and not has_dts:
                logger.warning(f"Exceeding coded bandwidth at ~{to_tc(seg.pts)}, {100*rate/PGDecoder.RX:.03f}%.")
                warnings += 1

            if decoded_buffer_bandwidth > PGDecoder.RD and not has_dts:
                logger.warning(f"Exceeding decoded buffer bandwidth at {to_tc(seg.pts)}.")
                warnings += 1

            #On palette update, we re-evaluate the existing graphic plane with a new CLUT.
            # so we are not subject to the Rc constraint.
            if ds.pcs.pal_flag:
                continue

            #We clear the plane (window area) and copy the objects to window. This is done at 32MB/s
            Rc = fps*(sum(window_area.values()) + np.min([area_copied, sum(window_area.values())]))
            nf = TC.s2f(seg.pts, fps) - TC.s2f(prev_pts, fps)
            if nf == 0:
                last_rc += Rc
            elif (last_rc+Rc)/nf > PGDecoder.RC and not has_dts:
                logger.warning(f"Graphic plane overloaded. Display is not ensured at {to_tc(seg.pts)}.")
                warnings += 1

    if warnings == 0 and compliant:
        logger.info("Output PGS seems compliant.")
    if warnings > 0 and compliant:
        logger.warning("Excessive bandwidth detected, requires HW testing (PGS may go out of sync).")
    elif not compliant:
        logger.error("Output PGS is not compliant. Expect display issues or decoder crash.")
    return compliant
