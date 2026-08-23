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

import os
import numpy as np
import multiprocessing as mp

from dataclasses import dataclass, field
from enum import IntEnum
from functools import reduce, partial
from pathlib import Path
from typing import Sequence, Generator, Iterable

from brule import LayoutEngine

from .bdnxml import BDNXML, BDNEvent
from .bdvideo import Format
from .geometry import Box, Point, Shape
from .internals import TC, GraphicsDecoder, GfxCompositor, LogFacility

logger = LogFacility.get_logger('SUPer')

class LayoutMode(IntEnum):
    SAFE   = 0
    NORMAL = 1
    GREEDY = 2       

@dataclass
class Graphic:
    point: Point
    shape: Shape
    filepath: Path
    
    def __post_init__(self) -> None:
        assert self.box.area > 0
        assert self.filepath.exists()
    
    @property
    def box(self) -> Box:
        return Box(self.point.y, self.shape.height, self.point.x, self.shape.width)
####

@dataclass
class EpochEvent(GfxCompositor):
    inTC: TC
    outTC: TC
    graphics: tuple[Graphic]
    repeated_inTC: list[TC] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        assert isinstance(self.inTC, TC) and isinstance(self.outTC, TC)
        assert self.inTC < self.outTC
        assert isinstance(self.graphics, (tuple, list)) and len(self.graphics) in range(1, 3)
        if len(self.repeated_inTC) > 0:
            assert all(map(lambda p: p[0] < p[1], zip(self.repeated_inTC, self.repeated_inTC[1:])))
            assert self.inTC < self.repeated_inTC[0] <= self.repeated_inTC[-1] < self.outTC
####

@dataclass
class EpochData:
    windows: list[Box]
    events: list[EpochEvent]
    min_dts: int | float = -np.inf
    max_pts: int | float =  np.inf
    
def _minmax(iterable: Iterable):
    return reduce(lambda x, y: (min(x[0], np.floor(y)), max(x[1], np.ceil(y))),
                  iterable, (np.inf, -np.inf))
####

class PaddingEngine:
    def __init__(self, box: Box, container: Box) -> None:
        self.box = box
        self.container = container

    @staticmethod
    def _pad_any_box(box: Box, container: Box, min_dx: int, min_dy: int) -> Box:
        if box.dx >= min_dx and box.dy >= min_dy:
            return box

        diff_y = max(0, min_dy - box.dy)
        diff_x = max(0, min_dx - box.dx)
        dv = np.array([[diff_y*(2*box.y + (box.dy - container.dy))/container.dy,
                        diff_x*(2*box.x + (box.dx - container.dx))/container.dx]])

        pu, pd = _minmax(map(lambda y: -diff_y/2 + dv[0, 0] + y, range(diff_y)))
        pl, pr = _minmax(map(lambda x: -diff_x/2 + dv[0, 1] + x, range(diff_x)))

        new_x1 = max(0, int(box.x  + (pl if (pl < pr) else 0)))
        new_x2 = min(container.dx, int(box.x2 + (pr if (pl < pr) else 0)))
        new_y1 = max(0, int(box.y  + (pu if (pu < pd) else 0)))
        new_y2 = min(container.dy, int(box.y2 + (pd if (pu < pd) else 0)))

        #Try to pad right and bottom first due to range rounding
        if (missing := diff_x - (box.x - new_x1 + new_x2 - box.x2)) > 0:
            new_x2 += (did_offset := (new_x2 + missing <= container.dx))*missing
            new_x1 -= (not did_offset)*missing
        if (missing := diff_y - (box.y - new_y1 + new_y2 - box.y2)) > 0:
            new_y2 += (did_offset := (new_y2 + missing <= container.dy))*missing
            new_y1 -= (not did_offset)*missing
        return Box.from_coords(new_y1, new_y2, new_x1, new_x2)

    @staticmethod
    def check_best(new_lwd: tuple[Box], prev_lwd: tuple[Box]) -> tuple[Box]:
        lwd = (Box.union(*new_lwd),) if (Box.intersect(*new_lwd).area > 0) else new_lwd
        if sum(map(lambda wd: wd.area, lwd)) < sum(map(lambda wd: wd.area, prev_lwd)):
            return lwd
        return prev_lwd

    def directional_pad(self, lwd: tuple[Box], vertical: bool | None = None) -> tuple[Box]:
        """
        Tries really hard to pad the windows in a smart way.
        :param lwd: list of 1 or 2 window(s)
        :param vertical: the 2 windows are from a vertical split (one window above the other).
        """
        bad_wds = tuple(filter(lambda wd: wd.dx < 8 or wd.dy < 8, lwd))
        if len(bad_wds) == 0:
            return lwd
        if len(lwd) == 1:
            return (__class__._pad_any_box(lwd[0], self.box, 8, 8),)
        assert len(lwd) == 2, "Expected 1 or 2 windows." # case == 1 handled above
        assert isinstance(vertical, (bool, int))

        new_lwd = []
        inter_margin = abs((lwd[0].y2 - lwd[1].y) if vertical else (lwd[0].x2 - lwd[1].x))

        for wid, wd in enumerate(lwd):
            missing = left_pad = right_pad = top_pad = bot_pad = 0
            if vertical:
                #Pad horizontally (no constraint)
                diff = max(0, 8 - wd.dx)
                left_pad = max(0, min(diff, wd.x + wd.dx + diff - self.box.dx))
                right_pad = diff - left_pad
                #Pad vertically (constrained)
                diff = max(0, 8 - wd.dy)
                top_pad = wd.y if wid == 0 and diff > 0 else 0
                bot_pad = 0 if wid == 0 or diff == 0 else (self.box.dy - wd.y2)
                if top_pad + bot_pad < diff:
                    missing = (diff - top_pad - bot_pad)
                    if wid == 0:
                        bot_pad += missing
                    else:
                        top_pad += missing
            else:
                diff = max(0, 8 - wd.dy)
                top_pad = max(0, min(diff, wd.y + wd.dy + diff - self.box.dy))
                bot_pad = diff - top_pad
                diff = max(0, 8 - wd.dx)
                left_pad = wd.x if wid == 0 and diff > 0 else 0
                right_pad = 0 if wid == 0 or diff == 0 else (self.box.dx - wd.x2)
                if left_pad + right_pad < diff:
                    missing = (diff - left_pad - right_pad)
                    if wid == 0:
                        right_pad += missing
                    else:
                        left_pad += missing
            inter_margin -= missing
            new_lwd.append(Box.from_coords(wd.x-left_pad, wd.y-top_pad, wd.x2+right_pad, wd.y2+bot_pad))
            # logger.debug(f"Padded window ID={wid}: {new_lwd[-1]} from {lwd[wid]}")
            assert Box(self.box.y + new_lwd[-1].y, new_lwd[-1].dy, new_lwd[-1].x + self.box.x, new_lwd[-1].dx).overlap_with(self.container) == 1.0, f"Window does not overlap with renderer container: {self.container}"

        #No suitable padding -> merge
        if inter_margin < 0:
            # logger.debug("No padding marging available: merge to a single window.")
            return (Box.union(*lwd),)
        assert Box.intersect(*new_lwd).area == 0, f"Padded windows overlap: {new_lwd} from {lwd}."
        return new_lwd
    ####
####

def _pool_worker_init() -> None:
    if os.name == 'nt':
        _parent_id = os.getpid()
        import psutil, signal

        def sig_int(signal_num, frame):
            parent = psutil.Process(_parent_id)
            _cpid = os.getpid()
            for child in parent.children():
                if child.pid != _cpid:
                    child.kill()
            parent.kill()
            psutil.Process(_cpid).kill()
        signal.signal(signal.SIGINT, sig_int)
    #else, do nothing

def _get_epoch_start_duration(windows: Sequence[Box], plane_area: int) -> int:
    dd = GraphicsDecoder.get_composition_duration(plane_area)
    t_dec = 0
    for wd in windows:
        t_dec += GraphicsDecoder.get_object_duration(wd.area)
        dd = max(t_dec, dd) + GraphicsDecoder.get_composition_duration(wd.area)
    return dd

def _to_absolute_coordinates(windows: Sequence[Box], container: Box) -> tuple[Box, ...]:
    return tuple(map(lambda w: Box(container.y + w.y, w.dy, container.x + w.x, w.dx), windows))

def _find_modify_layout(leng: LayoutEngine, container: Box, mode: LayoutMode) -> tuple[Box, ...]:
    cbox, w1, w2, is_vertical = leng.get_layout()
    box_factory = lambda x: Box.from_coords(x[1], x[3], x[0], x[2])

    cbox, w1, w2 = tuple(map(box_factory, (cbox, w1, w2)))

    cwd = (w1, w2) if w1 != w2 else (w1,)
    cwd = PaddingEngine(cbox, container=container).directional_pad(cwd, is_vertical)
    scores = [_get_epoch_start_duration(cwo, container.area) for cwo in (cwd, reversed(cwd))]

    if len(cwd) > 1:
        # safety: we want highest margin, preferred window order is the reversed one.
        if mode == LayoutMode.SAFE:
            flip_results = (scores[0] < scores[1])
        else: # take smallest
            flip_results = scores[0] > scores[1]

        if flip_results:
            cwd = tuple(reversed(cwd))
            scores[0] = scores[1]
    
    cwd = _to_absolute_coordinates(cwd, cbox)
    decode_duration = scores[0]        

    # given this, evaluate the single window layout (this epoch bounding box)
    base_box = box_factory(leng.get_raw_container())
    decode_duration_single = _get_epoch_start_duration((base_box,), container.area)
    is_bad_split = decode_duration >= max(1, decode_duration_single-10)

    #coded object buffer can fit at most 16 ODS: we need roughly +150 bytes in the buffer
    #note: technically we need to also consider the 2*height line-endings bytes, but let's assume there's *some* compression
    may_not_fit_buffer = any(map(lambda b: b.area >= (1 << 20)-150, cwd))
    is_greedysplit_worthwile = (decode_duration_single*0.85 < decode_duration and base_box.area > 125000) or may_not_fit_buffer

    #With greedy mode, anytime we're dealing with very big objects we abuse the 1/2 1/2. This also prevents coded buffer overflow.
    layout_modifier = 'N'
    if (mode == LayoutMode.GREEDY or is_bad_split) and is_greedysplit_worthwile:
        cx, cy = (1, 0.5)
        box1 = Box(0, int(round(cy*base_box.dy)), 0, int(round(base_box.dx*cx)))
        box2 = Box.from_coords(box1.y2, base_box.dy, 0, base_box.dx)
        assert base_box.area == Box.union(box1, box2).area and abs(1-box1.area/box2.area) < 1e-1

        greedy_wds = (box1, box2)
        greedy_duration = _get_epoch_start_duration(greedy_wds, container.area)
        if decode_duration > greedy_duration:
            cwd = _to_absolute_coordinates(greedy_wds, base_box)
            layout_modifier = 'G'
        # Objects could still not fit in buffer at this point, but there's so much we can do to help authorers...
    if layout_modifier == 'N' and is_bad_split and not may_not_fit_buffer:
        cwd = (base_box,)
        layout_modifier = 'S'
    return (cwd)
####

def _bdnev_to_epochevent(bdnev: BDNEvent) -> EpochEvent:
    return EpochEvent(bdnev.inTC, bdnev.outTC,
                      list(map(lambda g: Graphic(Point(y=g.y, x=g.x),
                                                 Shape(h=g.height, w=g.width),
                                                 g.filepath), bdnev.graphics)))

def _perform_fine_epoch_detection(
    events: Sequence[BDNEvent],
    plane_format: Format,
    mode: LayoutMode = LayoutMode.GREEDY
) -> list[EpochData]:
    """
    Perform fine analysis of the events to find the epoch splits.
    """
    screen = Box(0, plane_format.height, 0, plane_format.width)
    layout = LayoutEngine(plane_format.value)
    
    minimum_epoch_start = _get_epoch_start_duration((Box(0, 0, 0, 0),), screen.area)
    
    epochs = []
    events_in_epoch = []
    for ev in reversed(events):
        channel = np.ascontiguousarray(ev.image.getchannel('A'), dtype=np.uint8)
        if not np.any(channel):
            continue
        
        if len(events_in_epoch) > 0:
            delta_ticks = (events_in_epoch[0].inTC.to_pts() - ev.outTC.to_pts())
            if delta_ticks > minimum_epoch_start:
                cwd = _find_modify_layout(layout, screen, mode)
                if delta_ticks > _get_epoch_start_duration(cwd, screen.area):
                    epochs.insert(0, EpochData(cwd, events_in_epoch, min_dts=ev.inTC.to_pts()+1))
                    events_in_epoch = []
                    layout.reset()
        
        events_in_epoch.insert(0, _bdnev_to_epochevent(ev))
        bbox = ev.get_bbox()
        layout.add_to_layout(bbox.x, bbox.y, channel)
    #### for
    if len(events_in_epoch) > 0:
        cwd = _find_modify_layout(layout, screen, mode)
        epochs.insert(0, EpochData(cwd, events_in_epoch, min_dts=-np.inf))
    layout.destroy()
    return epochs
####

class EpochFinder:
    """
    This class finds epoch in a continuous stream of events.
    - Rough first pass: find obvious epochs (single threaded)
    - Fine grained second pass: perform detailed anaylsis of the input data
        (one thread per split obtained in the 1st pass)
        The fine grained pass assumes that:
            - The 2 windows are always occupied from epoch start
            - The said objects have the same size as the windows (worst case)
    """
    def __init__(self, bdn: BDNXML, threads: int = 1, mode: LayoutMode = LayoutMode.GREEDY) -> None:
        self.mode = LayoutMode(mode)
        self.bdn = bdn
        self.threads = threads
        
    def _get_rough_split(self) -> Generator[Sequence[BDNEvent], None, None]:
        plane = self.bdn.description.fmt
        window = Box(0, plane.height, 0, plane.width)
        screen_transfer_max = _get_epoch_start_duration((window,), plane.area)
        
        epoch = [self.bdn.events[0]]
        for event in self.bdn.events[1:]:
            # + 1 to avoid PTS(last event) = DTS(epoch start) which would not guarantee a display of the last event
            if (event.inTC.to_pts() - epoch[-1].outTC.to_pts() + 1) > screen_transfer_max:
                yield epoch
                epoch = []
            epoch.append(event)
        yield epoch

    def get_epochs(self) -> list[EpochData]:
        plane = self.bdn.description.fmt
        p_find_epochs_layouts = partial(_perform_fine_epoch_detection, plane_format=plane, mode=self.mode)
        epochs_data = []
        
        pbar = LogFacility.get_progress_bar(logger, self.bdn.events)
        pbar.set_description("Finding epochs and layouts", True)
        
        if self.threads > 1:
            with mp.Pool(self.threads, _pool_worker_init) as mpp:
                for r in mpp.imap_unordered(p_find_epochs_layouts, self._get_rough_split()):
                    pbar.update(sum(map(lambda ctx: len(ctx.events), r)))
                    epochs_data += r
            epochs_data = sorted(epochs_data, key=lambda e: e.events[0].inTC.frames)
        else:
            for r in map(p_find_epochs_layouts, self._get_rough_split()):
                pbar.update(sum(map(lambda ctx: len(ctx.events), r)))
                epochs_data += r

        pbar.update(len(self.bdn.events)-pbar.n+1)
        LogFacility.close_progress_bar(logger)

        get_composition_time = lambda windows: 1 + GraphicsDecoder.get_composition_duration(sum(map(lambda x: x.area, windows)))
        for ed, ed_next in zip(epochs_data, epochs_data[1:]):
            if np.isinf(ed_next.min_dts):
                #inflate a bit the epoch start to allow for WDS stacking at epoch start.
                ed_next.min_dts = max(ed.events[-1].outTC.to_pts() + 1,
                                      ed_next.events[0].inTC.to_pts() - _get_epoch_start_duration(ed_next.windows, plane.area) - get_composition_time(ed_next.windows))
            
            ed.max_pts = min(ed_next.min_dts - 1, ed.events[-1].outTC.to_pts() + get_composition_time(ed.windows))
            assert ed.min_dts < ed.max_pts < ed_next.min_dts < ed_next.max_pts
        if logger.level <= 10:
            for ect in epochs_data:
                logger.debug(f"Epoch Context: {ect.events[0].inTC}->{ect.events[-1].outTC} {len(ect.events)}, WDS={ect.windows}, [{ect.min_dts};{ect.max_pts}]")
        return epochs_data
####

class EventsPreprocessor:
    """
    Class to handle basic event(s) pre-processing or to compute side metadata.
    """
    @staticmethod
    def get_refresh_count(event: BDNEvent | EpochEvent, period: float = -1) -> int:
        """
        Compute the number of time this event shall be redrawn onto the display
        given the period.
        """
        if period < 1:
            return 0
        n_refreshes = int(((event.outTC - event.inTC).to_realtime(as_float=True) - period) // period)
        return max(0, n_refreshes)
    
    @staticmethod
    def remove_duplicates(events: list[BDNEvent | EpochEvent]) -> list[BDNEvent | EpochEvent]:
        trimmed = [events[0]]
        for event in events[1:]:
            add = True
            if trimmed[-1].outTC == event.inTC and trimmed[-1].get_bbox() == event.get_bbox():
                if np.array_equal(np.asarray(event.image), np.asarray(trimmed[-1].image)):
                    trimmed[-1].set_outTC(event.outTC)
                    add = False
            if add:
                trimmed.append(event)

        assert trimmed[0].inTC == events[0].inTC and trimmed[-1].outTC == events[-1].outTC
        logger.debug(f"Removed {len(events) - len(trimmed)} duplicate event(s).")
        return trimmed
####
