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


import numpy as np

from typing import Any, TypeVar, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass
from itertools import combinations, chain

from numpy import typing as npt
from anytree import Node

from skimage.filters import gaussian
from skimage.measure import regionprops, label

#%%
from SUPer.utils import get_super_logger, Pos, Dim, Shape, BDVideo
from SUPer.filestreams import BDNXMLEvent, BaseEvent
from SUPer.segments import DisplaySet, PCS, WDS, ODS, WindowDefinition
from SUPer.optim import Optimise, Preprocess
from SUPer.pgraphics import PGraphics, PGDecoder

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


    def delay_chain(self, events: list[BaseEvent], fps, box: Box = None) -> npt.NDArray[np.uint8]:
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
        details = f"{TC.s2tc(ds.pcs.pts, LUT_FPS_PCSFPS[ds.pcs.fps])}: "
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
        logging.warning(details)
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
            logging.warning(f"Two displaysets at {current_pts} [s] (internal rendering error?)")

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

class PGConvert:
    def __init__(self, windows: dict[int, WindowOnBuffer], events: list[Type[BaseEvent]]) -> None:
        assert type(windows) is dict
        self.windows = windows

    @property
    def windows(self) -> dict[int, WindowOnBuffer]:
        return self._windows

    @windows.setter
    def windows(self, windows: dict[int, WindowOnBuffer]) -> None:
        #Always keep windows sorted, and then generate all objects to show up
        self._windows = dict(sorted(windows.items()))

    def get_wds(self, pts: float) -> WDS:
        windows = []
        for w_id, window in self.windows.items():
            box = window.get_window()
            windows.append(WindowDefinition.from_scratch(w_id, box.x, box.y, box.dx, box.dy))
        return WDS.from_scratch(windows, pts=pts)

    def get_composition_states(self, acq_on_change: bool = True) -> tuple[PCS.CompositionState]:
        active_mask = self.windows[0].event_mask()
        for wob in self.windows[1:]:
            active_mask += wob.event_mask()

        lcs = map(lambda t_mask: PCS.CompositionState.ACQUISITION if t_mask > 1 else PCS.CompositionState.NORMAL, active_mask)
        if acq_on_change:
            #Force an acquisition when the object count decreases to 1.
            cs_prev = lcs[0] #at this point t0 is supposed to be an acquisition or a normal case
            for k, cs in enumerate(lcs[1:], start=1):
                if cs_prev == PCS.CompositionState.ACQUISITION and cs == PCS.CompositionState.NORMAL:
                    lcs[k] = PCS.CompositionState.ACQUISITION
                cs_prev = cs
        lcs[0] = PCS.CompostionState.EPOCH_START #t0 is epoch start (set now to simplify above's logic)
        #TODO: return acq_on_change mask. In case remaining bitmap does not change, we can just do a normal case.
        return lcs

    def render(self,
           events: list[Type[BaseEvent]],
           box: Box
        ) -> list[list[...], list[Type[BaseEvent]]]:
        """
        Performs the conversion from T[BaseEvents] to bitmaps with palette.
        """
        window_active_mask = np.zeros((len(self.window), len(events)), dtype=np.uint8)
        min_update_mask = window_active_mask.copy() #Bare minimum
        interm_update_mask = min_update_mask.copy() #Add some updates here and there when we can
        max_update_mask = interm_update_mask.copy() #Update aggressively bitmaps

        delays = self.windows[0].delay_chain()

        for w_id, window in self.windows.items():
            window_active_mask[w_id] = window.event_mask()
            min_update_mask[w_id] = window.bitmap_update_mask() | (delays > 20).astype(np.uint8)
            interm_update_mask[w_id] = min_update_mask[w_id] | (delays > 9).astype(np.uint8) #seek for 6 frames for refresh
            max_update_mask[w_id] = interm_update_mask[w_id] | window.update_mask()

        k = 0
        output = []
        output_events = []
        tc_prev = events[0].tc_in

        while k < len(events):
            if tc_prev != events[k].tc_in:
                #prev(out) != current(in) -> clear display at prev(out)
                output.append((None, None))
                output_events.append(tc_prev)

            n_objs = np.sum(window_active_mask[:, k])
            if n_objs > 1:
                output.append(self.render_multiple(box, events[k]))
                output_events.append(events[k])
                ke = 1
            elif n_objs == 1:
                active_window = np.argmax(window_active_mask[:, k])
                ke = 0
                while k+ke < len(events) and np.sum(window_active_mask[:, k+ke]) == 1 and window_active_mask[active_window, k+ke] == 1:
                    ke += 1
                    if interm_update_mask[w_id, k+ke] != 0 or events[k+ke].tc_in != events[k+ke-1].tc_out:
                        break
                e_group = events[k:k+ke]
                if len(e_group) == 1:
                    output.append(self.render_single(box, e_group[0], active_window))
                    output_events.append(e_group[0])
                else:
                    #Create an event chain and send it to the palette sequence optimiser
                    nev = []
                    wd = self.windows[active_window].get_window()
                    for eig in e_group:
                        img_fs = np.zeros((wd.dy, wd.dx, 4), dtype=np.uint8)
                        img_fs[eig.y-wd.y:eig.y+eig.height-wd.y, eig.x-wd.x:eig.x+eig.width-wd.x, :] = np.asarray(eig.img, np.uint8)
                        img = Image.fromarray(img_fs, mode='RGBA')
                        props = (Pos(wd.x, wd.y), Dim(*img.size))
                        nev.append(BDNXMLEvent.copy_custom(eig, img, props))
                    output.append(Optimise.solve_sequence(*Optimise.prepare_sequence(nev))[::-1])
                    output_events.append(nev)
            else:
                logging.warning("This code should not be executed. Did you load an empty bitmap?")
                output.append((None, None))
                output_events.append(tc_prev)
            ####
            tc_prev = events[k+ke-1].tc_out
            k += ke
        return output, output_events


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

    # def convert(self, events: list[Type[BaseEvent]], box: Box, bdv: BDVideo) -> Epoch:
    #     output_ip, nevs = self.render(events, box)
    #     windows = {w_id: wobw.get_window() for w_id, wobw in self.windows.items()}

    #     epoch = []
    #     pds_v = 0
    #     kds = 0

    #     for (palette, images), nev in zip(output_ip, nevs):
    #         #ACQUISITION
    #         if type(image) is tuple or kds == 0 or type(palette) == type(image) is not tuple:
    #             pts = TC.tc2s(nev.tc_in, bdv.fps.value)
    #             if type(image) is not tuple:
    #                 image == (image,)
    #             cobjects = []
    #             ods = []
    #             for o_id, img in enumerate(images):
    #                 ods.extend(PGraphics.bitmap_to_ods(image, o_id,
    #                                                    pts=TC.tc2s(nev.tc_in, bdv.fps.value)))
    #                 cobjects.append(CObject.from_scratch(o_id, o_id,
    #                                                      h_pos=windows[o_id].x,
    #                                                      v_pos=windows[o_id].y,
    #                                                      forced=getattr(nev, 'forced', False))

    #             pcs = PCS.from_scratch(bdv.width, bdv.height, bdv.pcsfps, kds,
    #                                    PCS.CompositionState.ACQUISITION, False, 0,
    #                                    cobjects=cobjects, pts=pts)
    #             end = ENDS.from_scratch(pts=pts)
    #             pds = PDS.from_scratch(dict(zip(range(len(palette)), palette)), pds_v, 0, pts=pts)
    #             epoch.append(DisplaySet([pcs, self.get_wds(pts)] + [pds] + ods + [end]))

    #             kds += 1
    #             pds_v += 1
    #         elif type(palette) is tuple: #NORMAL CASE with palette effect
    #             palups = Optimise.diff_cluts(cluts, matrix=self.kwargs.get('bt_colorspace', 'bt709'))

    #             pcs_f = lambda pts, kdsu: PCS.from_scratch(bdv.width, bdv.height, bdv.pcsfps, kdsu,
    #                                                  PCS.CompositionState.NORMAL, True, 0,
    #                                                  cobjects=cobjects, pts=pts)

    #             cobject
    #             for kdp, (palup, snev) in enumerate(zip(palups, nev)):
    #                 pts = TC.tc2s(snev.tc_in, bdv.fps.value)
    #                 pds = PDS.from_scratch(palup, pds_v, 0, pts=pts)

    #                 epoch.append(DisplaySet([pcs_f(pts, kdp+kds), pds, ENDS.from_scratch(pts=pts)]))
    #             kds += kdp + 1
    #         else:
    #             pts = TC.tc2s(nev.tc_in, bdv.fps.value)
    #             assert palette is None and images is None, "Not an empty NORMAL case."
    #             pcs = PCS.from_scratch(bdv.width, bdv.height, bdv.pcsfps, kds,
    #                                    PCS.CompositionState.NORMAL, False, 0,
    #                                    cobjects=[], pts=pts)
    #             epoch.append(DisplaySet([pcs, self.get_wds(pts), ENDS.from_scratch(pts=pts)]))

    #         pts = TC.tc2s(nev.tc_out, bdv.fps.value)
    #         pcs = PCS.from_scratch(bdv.width, bdv.height, bdv.pcsfps, kds,
    #                                PCS.CompositionState.NORMAL, False, 0,
    #                                cobjects=[], pts=pts)
            # epoch.append(DisplaySet([pcs, WDS.from_scratch([], pts=pts), ENDS.from_scratch(pts=pts)]))
        # return Epoch(epoch)

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
        if len(lut) >= 16:
            if len(lut) > 19:
                raise AssertionError("Cannot optimise, too many isolated graphics in epoch.")
            logging.warning("Grouping: long brute force situation: this will be SLOW!")
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
        regions, gs_origs, box = self.coarse_grouping(subgroup)

        tbox = []
        for region in regions:
            region.slice = cls.crop_region(region, gs_origs)
            tbox.append(ScreenRegion.from_region(region))

        if len(tbox) < 12:
            wobs = self.group_and_sort(tbox, len(subgroup))
        else:
            wobs = self.group_and_sort_flat(tbox, len(subgroup))
        wob = self.select_best_wob(wobs, box)
        return wob, box

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
