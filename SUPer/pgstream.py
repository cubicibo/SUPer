#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 cibo
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

from typing import Type, Optional
from dataclasses import dataclass
import numpy as np

from .segments import PGSegment, PCS, WDS, PDS, ODS, ENDS, Epoch, DisplaySet
from .pgraphics import PGDecoder, PGraphics, PGObjectBuffer, PaletteManager
from .palette import Palette
from .utils import LogFacility, TimeConv as TC, Box

logger = LogFacility.get_logger('SUPer')

@dataclass
class BufferStats:
    min: float = np.inf
    avg: float = 0
    maxrate: float = 0
    count: int = 0
    tsmaxrate: int = 0
    tsmin: int = 0
    tsavg: int = 0
    max1s: float = 0

class LeakyBuffer:
    SIZE = (1 << 20)
    TS_MASK = (1 << 32) - 1

    def __init__(self, first_ts: int, bitrate: Optional[int] = None) -> None:
        self.used_bytes = self.__class__.SIZE
        self.bitrate = (PGDecoder.RX) if bitrate is None else int(bitrate)
        self.last_ts = first_ts
        self.stats = BufferStats()
        self.good_ds = True
        self.rate_past = []

    def set_tc_func(self, tc_func) -> None:
        self._tc_func = tc_func

    def step(self, segment: Type[PGSegment]) -> bool:
        if isinstance(segment, PCS):
            self.good_ds = True

        new_ts = segment.tdts
        dticks = (new_ts - self.last_ts) & self.__class__.TS_MASK

        self.used_bytes = min(self.used_bytes + round(dticks*self.bitrate/PGDecoder.FREQ), self.__class__.SIZE)
        self.used_bytes -= (len(segment) - 2)

        self.set_stats(segment.tdts)
        self.last_ts = new_ts

        self.good_ds &= self.used_bytes >= 0

        if not self.good_ds and isinstance(segment, ENDS):
            logger.error(f"PG stream underflow at {self._tc_func(segment.tpts)}: {self.used_bytes} bytes.")
        return self.used_bytes >= 0

    def set_bitrate(self, size_ds: int, curr_ts: int, prev_ts: int) -> None:
        dticks = (curr_ts - prev_ts) & self.__class__.TS_MASK
        if dticks > 0:
            rate = size_ds*PGDecoder.FREQ/dticks
            if rate >= self.stats.maxrate:
                self.stats.maxrate = rate
                self.stats.tsmaxrate = self._tc_func(curr_ts)
        self.rate_past = list(filter(lambda x: (curr_ts - x[1]) & self.__class__.TS_MASK <= PGDecoder.FREQ, self.rate_past))
        self.rate_past.append((size_ds, curr_ts))

        crate = sum(map(lambda x: x[0], self.rate_past))/(128*1024)
        if crate >= self.stats.max1s:
            self.stats.tsavg = self._tc_func(curr_ts)
            self.stats.max1s = crate

    def get_usage(self) -> float:
        return self.used_bytes/self.__class__.SIZE

    def set_stats(self, ts: int) -> None:
        if self.get_usage() <= self.stats.min:
            self.stats.min = self.get_usage()
            self.stats.tsmin = self._tc_func(ts)
        self.stats.avg = ((self.stats.avg*self.stats.count) + self.get_usage())/(self.stats.count + 1)
        self.stats.count += 1

    def get_stats(self) -> BufferStats:
        return (100*self.stats.min, 100*self.stats.avg, round(self.stats.max1s, 3))
####

def test_rx_bitrate(epochs: list[Epoch], bitrate: int, fps: float) -> bool:
    prev_ts = (epochs[0][0][0].tdts-int(PGDecoder.FREQ)) & ((1<<32)-1)
    is_ok = True
    leaky = LeakyBuffer(prev_ts, bitrate)

    leaky.set_tc_func(lambda pts: TC.s2tc(pts/PGDecoder.FREQ/(1 if float(fps).is_integer() else 1.001), fps))

    dur_offset = 0
    ts_first = prev_ts
    for epoch in epochs:
        for ds in epoch:
            for seg in ds:
                is_ok &= leaky.step(seg)
            leaky.set_bitrate(len(bytes(ds)), ds.pcs.tpts, prev_ts)
            if ds.pcs.tpts < prev_ts and ts_first < np.inf and ts_first != prev_ts:
                dur_offset += LeakyBuffer.TS_MASK + 1
                ts_first = np.inf
            prev_ts = ds.pcs.tpts
    ##for epoch
    dur_offset += (epochs[-1][-1].pcs.tpts - epochs[0][0].pcs.tpts)
    stats = leaky.get_stats()

    avg_bitrate = sum(map(lambda x: len(bytes(x)), epochs))/(dur_offset/PGDecoder.FREQ)
    logger.iinfo(f"Bitrate: AVG={avg_bitrate/(128*1024):.04f} Mbps, PEAK_1s={stats[2]:.03f} Mbps @ {leaky.stats.tsavg}.")

    f_log_fun = logger.iinfo if is_ok else logger.error
    f_log_fun(f"Buffer margin: AVG={stats[1]:.02f}%, MIN={stats[0]:.02f}% @ {leaky.stats.tsmin}")
    return is_ok
####
#%%
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

def is_compliant(epochs: list[Epoch], fps: float) -> bool:
    ts_mask = ((1 << 32) - 1)
    prev_pts = -1
    last_cbbw = 0
    last_dbbw = 0
    compliant = True
    warnings = 0
    pal_id = 0
    cumulated_ods_size = 0
    prev_pcs_id = 0xFFFF

    coded_bw_ra_pts = [-1] * round(fps)
    coded_bw_ra = [0] * round(fps)

    to_tc = lambda pts: TC.s2tc(pts/(1 if float(fps).is_integer() else 1.001), fps)

    for ke, epoch in enumerate(epochs):
        windows = {}
        window_area = {}
        objects_sizes = {}
        ods_vn = {}
        pds_vn = [-1] * 8
        pals = [Palette() for _ in range(8)]
        buffer = PGObjectBuffer()

        compliant &= bool(epoch[0].pcs.composition_state & epoch[0].pcs.CompositionState.EPOCH_START)
        
        if epoch[0].wds:
            for wd in epoch[0].wds.windows:
                if wd.h_pos + wd.width > epoch[0].pcs.width or wd.v_pos + wd.height > epoch[0].pcs.height:
                    logger.error(f"Window {wd.window_id} out of screen in epoch starting at {to_tc(epoch[0].pcs.pts)}.")

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
                if ((seg.tpts - seg.tdts) & ts_mask >= PGDecoder.FREQ):
                    logger.warning(f"Too large PTS-DTS difference for seg._type at {to_tc(current_pts)}.")

                if isinstance(seg, PCS):
                    pal_id = seg.pal_id
                    compliant &= (ks == 0) #PCS is first in DisplaySet
                    if seg.composition_n != (prev_pcs_id + 1) & 0xFFFF and seg.composition_state != PCS.CompositionState.EPOCH_START:
                        logger.warning(f"Displayset does not increment composition number normally at {to_tc(seg.pts)}.")
                    prev_pcs_id = seg.composition_n
                    if int(seg.composition_state) != 0:
                        # On acquisition, the object buffer is flushed
                        objects_sizes = {}
                        pals = [Palette() for _ in range(8)]
                    for cobj in seg.cobjects:
                        areas2gp[cobj.o_id] = -1 if not cobj.cropped else cobj.c_w*cobj.c_h

                elif isinstance(seg, WDS):
                    compliant &= (ks == 1) #WDS is not second segment of DS, if present
                    if len(windows) == 0:
                        for w in seg.windows:
                            windows[w.window_id] = (w.h_pos, w.v_pos, w.width, w.height)
                        lwdb = list(map(lambda w: Box(w[1], w[3], w[0], w[2]), windows.values()))
                        if len(windows) == 2 and Box.intersect(*lwdb).area > 0:
                            logger.error(f"Overlapping windows in epoch starting at {to_tc(current_pts)}.")
                            compliant = False
                    else:
                        for w in seg.windows:
                            if windows[w.window_id] != (w.h_pos, w.v_pos, w.width, w.height):
                                logger.error(f"Window change mid-epoch at {to_tc(current_pts)}, this is strictly prohibited.")
                                compliant = False
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
                        if 8 > min(ods_width, ods_height) or 4096 < max(ods_width, ods_height):
                            logger.error(f"Illegal object dimensions at {to_tc(current_pts)}, object id={seg.o_id}: {ods_width}x{ods_height}.")
                            compliant = False
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
                            #Hypothesis: the graphic controller processes one RLE command (byte) per Rd tick
                            # To avoid decode time > object write time, RLE line must be smaller or equal to width + marker.
                            next(filter(lambda x: len(x) > ods_width + 2, PGraphics.get_rle_lines(ods_data, ods_width)))
                        except StopIteration:
                            ...
                        else:
                            logger.warning("ODS at {to_tc(seg.pts)} has too long RLE line(s). Oldest decoders may have issues.")
                            warnings += 1
                        
                        for pe in np.unique(PGraphics.decode_rle(ods_data, width=ods_width, height=ods_height)):
                            if pe != 0xFF and pe not in pals[pal_id].palette:
                                logger.warning(f"ODS at {to_tc(seg.pts)} uses undefined palette entries (first: {pe:02X}). Some pixels will not display.")
                                warnings += 1
                                break
                        cumulated_ods_size = 0
                elif isinstance(seg, ENDS):
                    compliant &= ks == len(ds)-1 # END segment is not last or not alone
                    
                    # Control the spatial values of the composition w.r.t. object
                    if ds.wds:
                        for cobj in ds.pcs.cobjects:
                            obj_dims = buffer.get(cobj.o_id)
                            if obj_dims == None:
                                logger.error(f"Trying to use object {cobj.o_id} not stored in buffer at {to_tc(current_pts)}.")
                                compliant = False
                            else:
                                h, w = obj_dims
                                if cobj.cropped:
                                    if h < cobj.c_h or w < cobj.c_w or h < cobj.c_h + cobj.vc_pos or w < cobj.c_w + cobj.hc_pos:
                                        logger.error(f"Cropped dimension exceeed object {cobj.o_id} size at {to_tc(current_pts)}.")
                                        compliant = False
                                    else:
                                        w, h = cobj.c_w, cobj.c_h
                                        if w == 0 or h == 0:
                                            logger.warning("Zero cropping width or height at {to_tc(current_pts)}.")
                                            warnings += 1
                                ####if cropped
                            wd = ds.wds.windows[cobj.window_id]
                            if cobj.h_pos < wd.h_pos or cobj.h_pos + w > wd.h_pos + wd.width or\
                               cobj.v_pos < wd.v_pos or cobj.v_pos + h > wd.v_pos + wd.height:
                                logger.error(f"Composition object {cobj.o_id} misplaced outside of window {wd.window_id} at {to_tc(current_pts)}.")
                                compliant = False
                        ####for cobj
                    ####if wds
                ####elif END
                        
                ####elif
            ####for
            if ds.pcs.pal_flag or ds.wds is not None:
                compliant &= pds_vn[ds.pcs.pal_id] >= 0 #Check that the used palette is indeed set in the decoder.
        #### for ds
    ####for epoch
    return compliant, warnings

def check_pts_dts_sanity(epochs: list[Epoch], fps: float) -> bool:
    PTS_MASK = (1 << 32) - 1
    PTS_DIFF_BOUND = PTS_MASK >> 1
    is_compliant = True
    prev_pts = prev_dts = -99999999
    min_dts_delta = 99999999

    to_tc = lambda pts: TC.s2tc(pts/(1 if float(fps).is_integer() else 1.001), fps)
    frame_duration = np.floor(PGDecoder.FREQ/fps)

    for k, epoch in enumerate(epochs):
        pts_delta = int(sum(map(lambda w: np.ceil(w.width*w.height*PGDecoder.FREQ/PGDecoder.RC), epoch[0].wds)))
        wipe_duration = int(np.ceil(epoch[0].pcs.width*epoch[0].pcs.height*PGDecoder.FREQ/PGDecoder.RC))
        is_compliant &= (epoch[0].pcs.tpts - epoch[0].pcs.tdts) & PTS_MASK > wipe_duration
        #Must not decode epoch start before previous epoch is fully finished (at PTS)
        diff = (epoch[0].pcs.tdts - prev_pts) & PTS_MASK
        is_compliant &= diff > 0 and diff < PTS_DIFF_BOUND

        for l, ds in enumerate(epoch):
            ds_comply = True
            #Check for minimum DTS delta between DS, ideally this should be bigger than 0
            min_dts_delta = min((ds.pcs.tdts - prev_dts) & PTS_MASK, min_dts_delta)
            if ds.wds:
                # WDS action requires pts_delta margin from previous DS
                diff = (ds.pcs.tpts - prev_pts) & PTS_MASK
                ds_comply &= diff > pts_delta and diff < PTS_DIFF_BOUND
                #WDS deadline is pts_delta close to final pts
                ds_comply &= (ds.pcs.tpts - ds.wds.tpts) & PTS_MASK <= pts_delta
                #WDS decoding should be realistic (epoch start is worst case)
                ds_comply &= (ds.wds.tpts - ds.wds.tdts) & PTS_MASK <= wipe_duration*2
            else:
                # Palette update and others requires one frame duration as margin
                ds_comply &= (ds.pcs.tpts - prev_pts) & PTS_MASK >= frame_duration
            for pds in ds.pds:
                ds_comply &= pds.tpts == pds.tdts
            for seg in ds:
                diff = (seg.tdts - prev_dts) & PTS_MASK
                ds_comply &= diff >= 0 and diff < PTS_DIFF_BOUND
                prev_dts = seg.tdts
            prev_pts = ds.pcs.tpts
            if not ds_comply:
                logger.error(f"Incorrect PTS-DTS at {to_tc(ds.pcs.pts)}, DS:S={ds.pcs.composition_state:02X}:PU={ds.pcs.pal_flag > 0}, stream is out of spec.")
            is_compliant &= ds_comply
        ####ds
    ####epochs
    is_compliant &= min_dts_delta >= 0
    return is_compliant
