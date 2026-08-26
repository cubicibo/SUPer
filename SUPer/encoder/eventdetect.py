#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 cibo
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

import cv2
import numpy as np

from dataclasses import dataclass, field
from itertools import chain, count
from PIL import Image
from typing import Sequence, Generator, Self

from brule import LayoutEngine

from .epochctx import EpochEvent, PaddingEngine
from .imgproc import SSIMPW

from ..display.bdvideo import Format
from ..geometry import Box
from ..internals import LogFacility

logger = LogFacility.get_logger('SUPer')

#%%
@dataclass
class ProspectiveObject:
    f:     int
    mask:  list[bool]
    boxes: list[Box]
    box: Box
    wid: int = None
    # Must not be set by the user
    __uuid: int = field(default_factory=count().__next__)

    def __post_init__(self) -> None:
        assert len(self.mask) == len(self.boxes)
        assert any(self.mask)
        assert self.box.area > 0
        assert self.f >= 0
        self.ext_range = self.f + len(self.mask)

    def __hash__(self) -> int:
        return self.__uuid

    @property
    def uuid(self) -> int:
        return self.__uuid

    def is_active(self, frame: int) -> bool:
        """
        Returns if the object may be buffered at the given epoch event id.
        """
        return frame in range(self.f, self.f+len(self.mask))

    def is_visible(self, frame: int) -> bool:
        """
        Returns if the object is visible at the given epoch event id.
        """
        if self.is_active(frame):
            return self.mask[frame-self.f]
        return False

    def get_bbox_at(self, frame: int) -> Box | None:
        """
        Return the bounding box at the given event frame.
        """
        if self.is_active(frame):
            return self.boxes[frame-self.f]
        return None

    def pad_left(self, padding: int) -> None:
        """
        Activate an object earlier by padding it to the left on the event grid.
        All structures must be extended accordingly.
        """
        assert padding > 0
        self.f -= padding
        self.mask[0:0] = [False] * padding
        self.boxes[0:0] = [self.boxes[0]] * padding

    def set_extended_visibility_limit(self, f_max: int) -> None:
        self.ext_range = f_max

    def is_visible_extended(self, frame: int) -> bool:
        assert frame > self.f and not self.is_active(frame), f"{frame} < {self.f} ? act={self.is_active(frame)}"
        return frame < self.ext_range

    def copy(self) -> Self:
        # DO NOT set the uuid
        return self.__class__(self.f, self.mask.copy(), self.boxes.copy(), self.box, self.wid)

def _get_windowed_image(window: Box, event: EpochEvent, img: Image.Image) -> np.ndarray[tuple[int, int, int], np.uint8]:
    event_box = event.get_bbox()
    zone_overlap = Box.intersect(window, event_box)
    work_plane = np.zeros((window.dy, window.dx, 4), dtype=np.uint8)
    empty = False
    if zone_overlap.area > 0:
        #plane slices
        psy = slice(zone_overlap.y - window.y, zone_overlap.y2 - window.y)
        psx = slice(zone_overlap.x - window.x, zone_overlap.x2 - window.x)
        #image slices
        isy = slice(zone_overlap.y - event_box.y, zone_overlap.y2 - event_box.y)
        isx = slice(zone_overlap.x - event_box.x, zone_overlap.x2 - event_box.x)
        work_plane[psy, psx, :] = np.asarray(img, dtype=np.uint8)[isy, isx, :]
    else:
        empty = True
    return work_plane, empty

#%%
####
class TreeAnalyzer:
    MAX_DEPTH = 4
    def __init__(self, region: Box, *, _depth: int = 0):
        self.region = Box(0, region.dy, 0, region.dx)
        self.depth = _depth

    def _get_layout(self, composite: Image.Image, frame: Image.Image) -> tuple[bool, tuple[Box, Box, Box]]:
        leng = LayoutEngine((self.region.dx, self.region.dy))
        leng.add_to_layout(0, 0, np.asarray(composite.getchannel('A')))
        leng.add_to_layout(0, 0, np.asarray(frame.getchannel('A')))
        cbox, reg1, reg2, is_vertical = leng.get_layout()
        leng.destroy()
        box_factory = lambda x: Box.from_coords(x[1], x[3], x[0], x[2])
        cbox, reg1, reg2 = tuple(map(lambda b: box_factory(b), (cbox, reg1, reg2)))
        return self._validate_layout(cbox, is_vertical, reg1, reg2)

    def _validate_layout(self, cbox: Box, is_vertical: int, reg1: Box, reg2: Box) -> ...:
        is_valid = is_vertical >= 0
        if is_valid:
            lwds = PaddingEngine(cbox, self.region).directional_pad((reg1, reg2), is_vertical)
            is_valid &= (2 == len(lwds))
            #become more and more demanding on the gain to justify the split
            is_valid &= (lwds[0].area + lwds[1].area)/self.region.area < 0.9 - self.depth/13
        return is_valid, cbox, (lwds if is_valid else None)

    def _get_score(self, composite: Image.Image, frame: Image.Image) -> ...:
        assert composite.size == frame.size == (self.region.dx, self.region.dy)
        split_valid = self.depth < __class__.MAX_DEPTH
        if split_valid:
            split_valid, cbox, split_layout = self._get_layout(composite, frame)
            assert Box.intersect(cbox, self.region) == cbox, f"{cbox}, {self.region}, {split_valid}, {split_layout}"

        #perform recursion up to depth
        if split_valid:
            nd = self.depth+1
            costs = map(lambda r: __class__(r, _depth=nd)._get_score(composite.crop((cbox.x+r.x, cbox.y+r.y, cbox.x+r.x2, cbox.y+r.y2)),
                                                                     frame.crop((cbox.x+r.x, cbox.y+r.y, cbox.x+r.x2, cbox.y+r.y2))), split_layout)
            #recursion happens here
            return list(costs)
        else:
            return self.get_region_cost(composite, frame)

    def evaluate(self, composite: Image.Image, frame: Image.Image) -> tuple[float, float]:
        ls_costs = __class__._flatten_costs(self._get_score(composite, frame))
        sum_area = 0
        for cost in ls_costs:
            sum_area += cost[1]
        if sum_area == 0: #no active area, everything fits
            return 1.0, 1.0
        score, cross_p = 0, 0
        for cost in ls_costs:
            score += cost[1]*cost[0][0]
            cross_p += cost[1]*cost[0][1]
        return score/sum_area, cross_p/sum_area

    def get_region_cost(self, composite: Image.Image, frame: Image.Image) -> tuple[float, float]:
        cropbbox = (self.region.x, self.region.y, self.region.x2, self.region.y2)
        scores = __class__._compare_f(composite.crop(cropbbox), frame.crop(cropbbox))
        area_coeff = 0.325 if all(map(lambda s: s == 1.0, scores)) else 1.0
        return scores, self.region.area*area_coeff

    @classmethod
    def _flatten_costs(cls, costs) -> tuple[float, float, int]:
        ls_costs = []
        if isinstance(costs, tuple):
            return [costs]
        for cost in costs:
            if isinstance(cost, list):
                ls_costs.extend(cls._flatten_costs(cost))
            elif isinstance(cost, tuple):
                ls_costs.append(cost)
        return ls_costs

    @staticmethod
    def _compare_f(bitmap: Image.Image, current: Image.Image) -> tuple[float, float]:
        """
        :param bitmap: (cropped or padded) aggregate of the previous bitmaps
        :param current: current bitmap under analysis
        :return: comparison score between the two
        """
        assert bitmap.size == current.size, "Different shapes."

        # Intersect alpha planes
        a_bitmap = np.array(bitmap)
        a_current = np.array(current)
        inters_inv = np.logical_and(a_bitmap[:,:,3] == 0, a_current[:,:,3] == 0)
        inters = np.logical_and(a_bitmap[:,:,3] != 0, a_current[:,:,3] != 0)
        inters_area = np.sum(inters)
        #if the images have the exact same alpha channel, this measure is equal to 1
        overlap = (inters_area > 0) * (inters_area + np.sum(inters_inv))/inters.size

        if overlap > 0:
            mask = 255*(np.logical_and((a_bitmap[:, :, 3] > 0), (a_current[:, :, 3] > 0)).astype(np.uint8))
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            mask[mask > 0] = 255

            score = SSIMPW.compare(Image.fromarray(a_bitmap & mask[:, :, None]).convert('L'), Image.fromarray(a_current & mask[: , :, None]).convert('L'))
            cross_percentage = np.sum(mask > 0)/mask.size

            ksize = 3
            kernel = (ksize, ksize)
            img_comp = cv2.GaussianBlur(np.array(bitmap.convert('L')), kernel, 0)
            img_curr = cv2.GaussianBlur(np.array(current.convert('L')), kernel, 0)

            ksize = 5
            sobel_compo = cv2.Sobel(src=img_comp, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=ksize)
            sobel_curr = cv2.Sobel(src=img_curr, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=ksize)
            score_edge = SSIMPW.compare(Image.fromarray(sobel_compo & mask), Image.fromarray(sobel_curr & mask))

            score = min(score, score_edge)
        else:
            cross_percentage = 1.0
            score = 1.0
        return score, cross_percentage

class ObjectDetector:
    def __init__(self,
         window: Box,
         ssim_threshold: float = 0.986,
         ssim_offset: float = 0.0,
         overlap_threshold: float = 0.995
    ) -> None:
        self.window = window
        assert ssim_threshold < 1.0, "Not a valid SSIM threshold"
        self.ssim_threshold = ssim_threshold
        assert abs(ssim_offset) <= 1.0
        self.ssim_offset = ssim_offset
        assert 0 < overlap_threshold < 1.0, "Not a valid overlap threshold."
        self.overlap_threshold = overlap_threshold

    def mask_event(self, event: EpochEvent, img: Image.Image) -> tuple[np.ndarray[tuple[int, int, int], np.uint8], bool]:
        return _get_windowed_image(self.window, event, img)

    @staticmethod
    def _generate_object(
        alpha_compo: Image.Image,
        mask: list[bool],
        unseen: int,
        containers: list[Box],
        f_start: int
    ) -> ProspectiveObject:
        """
        Helper function to generate a prospective object, from the metadata collected
        in analyze(...)

        Always returns a valid object that should be encoded.
        """
        x, y, x2, y2 = alpha_compo.getbbox()
        if unseen > 0:
            mask = mask[:-unseen]
            containers = containers[:-unseen]
        return ProspectiveObject(f_start, mask, containers, Box.from_coords(y, y2, x, x2))

    def analyze(self) -> Generator[ProspectiveObject, None, None]:
        alpha_compo = Image.new('RGBA', (self.window.dx, self.window.dy), (0, 0, 0, 0))

        unseen = f_start = event_cnt = 0
        pgo_yield = None
        containers, mask = [], []

        while True:
            event, img = yield pgo_yield
            pgo_yield = None

            if event is None:
                if len(mask):
                    pgo_yield = __class__._generate_object(alpha_compo, mask, unseen, containers, f_start)
                    mask, containers = [], []
                    continue
                else:
                    break
            rgba, is_empty = self.mask_event(event, img)

            #only look at alpha as RGB channels may be random
            has_content = not is_empty and np.any(rgba[:, :, 3])
            if has_content or len(mask):
                if not len(mask):
                    f_start = event_cnt

                rgba_i = Image.fromarray(rgba)

                #If no content, bounding box keeps the last value
                #TODO: maybe do NOT use the bbox when the object is masked.
                if has_content:
                    x, y, x2, y2 = rgba_i.getbbox()
                    event_container = Box.from_coords(y, y2, x, x2)

                if len(mask) and has_content:
                    score, cross_percentage = TreeAnalyzer(self.window).evaluate(alpha_compo, rgba_i)
                else:
                    score, cross_percentage = 1.0, 1.0

                # fine tuned, self correct based on the percentage of overlap between running compo and event (cross_percentage)
                thr_score = min(1.0, self.ssim_threshold + (1-self.ssim_threshold)*(1-cross_percentage) - 0.008333*(1.0-self.ssim_offset))
                if score >= thr_score:
                    alpha_compo.alpha_composite(rgba_i)
                    mask.append(has_content)
                    containers.append(event_container)
                else:
                    assert has_content, "New PGObject must have visible content!!"
                    pgo_yield = __class__._generate_object(alpha_compo, mask, unseen, containers, f_start)
                    #prepare for the next bitmap
                    mask = [has_content]
                    containers = [event_container]
                    f_start = event_cnt
                    alpha_compo = Image.fromarray(rgba.copy())
                unseen = (unseen + 1) if (not has_content) else 0
            event_cnt += 1
        ####while
        return # StopIteration
####

class WindowsObjectDetector:
    def __init__(self, fmt: Format, windows: Sequence[Box], ssim_tol: float = 0, nested_analysis: bool = False):
        self.windows = windows
        self.fmt = fmt
        self.ssim_tol = ssim_tol
        self.nested_analysis = nested_analysis

    def identify_primary_objects(self,
        events: list[EpochEvent],
        ssim_offset: float,
        ssim_threshold: float
    ) -> list[list[ProspectiveObject]]:
        #Init the detectors
        detectors = []
        for k, window in enumerate(self.windows):
            detectors.append(ObjectDetector(window, ssim_threshold=ssim_threshold, ssim_offset=ssim_offset).analyze())
            next(detectors[-1])

        # run the analysis on both windows, event per event. Collect all objects returned in a list, for each window
        pgobjs = [[] for k in range(len(self.windows))]

        pbar = LogFacility.get_progress_bar(logger, range(len(events)))
        pbar.set_description("Analyzing", False)

        # to flush a detector, two consecutives None must be sent.
        for event in chain(events, [None]*2):
            # load image once, regardless of the window count.
            ev_img = event.image if event else None
            for wid, (window, detector) in enumerate(zip(self.windows, detectors)):
                try:
                    pgobj = detector.send((event, ev_img))
                except StopIteration:
                    pgobj = None
                if pgobj is not None:
                    pgobj.wid = wid
                    logger.debug(f"Window={wid} has new PGObject: f={pgobj.f}, S(mask)={len(pgobj.mask)}, mask={pgobj.mask}")
                    pgobjs[wid].append(pgobj)
            if event is not None:
                pbar.n += 1
                if pbar.n & 0xF == 0 or pbar.n == len(events):
                    pbar.refresh()
        pbar.clear()
        return pgobjs

    def get_objects(self, events: list[EpochEvent]) -> list[list[ProspectiveObject]]:
        ssim_threshold = 0.014 * min(1, max(-1, self.ssim_tol))
        #Adjust slightly SSIM threshold depending of res
        ssim_score = min(0.9999, 0.9608 + self.fmt.value[1]*(0.986-0.972)/(1080-480))

        pgobjs = self.identify_primary_objects(events, ssim_threshold, ssim_score)

        if self.nested_analysis:
            raise NotImplementedError("Nested analysis currently not implemented.")
        return pgobjs
####
