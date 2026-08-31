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

from dataclasses import dataclass

import numpy as np
from brule import Brule

from ..display.palette import Palette
from ..encoder.codecctx import PGEpochContext
from ..geometry import Box, Shape
from ..internals import TC, GraphicsDecoder, LogFacility
from .pgstreams import DisplaySet, Epoch
from .segments import END, ODS, PCS, GraphicSegment, PGSegmentType

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

    def __init__(self, first_ts: int, bitrate: int | None = None) -> None:
        self.used_bytes = self.__class__.SIZE
        self.bitrate = (2000000) if bitrate is None else int(bitrate)
        self.last_ts = first_ts
        self.stats = BufferStats()
        self.good_ds = True
        self.rate_past = []

    def set_tc_func(self, tc_func) -> None:
        self._tc_func = tc_func

    def step(self, segment: GraphicSegment) -> bool:
        if isinstance(segment, PCS):
            self.good_ds = True

        new_ts = segment.dts
        dticks = (new_ts - self.last_ts) & self.__class__.TS_MASK

        self.used_bytes = min(self.used_bytes + round(dticks*self.bitrate/GraphicsDecoder.FREQ), self.__class__.SIZE)
        #+ PTS + DTS + some PES overhead
        self.used_bytes -= (len(segment.get_payload()) + 13)

        self.set_stats(segment.dts)
        self.last_ts = new_ts

        self.good_ds &= self.used_bytes >= 0

        if not self.good_ds and isinstance(segment, END):
            logger.warning(f"PG stream underflow at {self._tc_func(segment.pts)}: {self.used_bytes} bytes.")
        return self.used_bytes >= 0

    def set_bitrate(self, size_ds: int, curr_ts: int, prev_ts: int) -> None:
        dticks = (curr_ts - prev_ts) & self.__class__.TS_MASK
        if dticks > 0:
            rate = size_ds*GraphicsDecoder.FREQ/dticks
            if rate >= self.stats.maxrate:
                self.stats.maxrate = rate
                self.stats.tsmaxrate = self._tc_func(curr_ts)
        self.rate_past = list(filter(lambda x: (curr_ts - x[1]) & self.__class__.TS_MASK <= GraphicsDecoder.FREQ, self.rate_past))
        self.rate_past.append((size_ds, curr_ts))

        crate = sum(x[0] for x in self.rate_past)/(128*1024)
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
    prev_ts = (epochs[0][0][0].dts-int(GraphicsDecoder.FREQ)) & ((1<<32)-1)
    is_ok = True
    leaky = LeakyBuffer(prev_ts, bitrate)

    f_print_tc = lambda pts: str(TC.s2tc(pts/GraphicsDecoder.FREQ, fps)) + ('' if (float(fps).is_integer() or fps < 25) else (', DF=' + str(TC.s2tc(pts/GraphicsDecoder.FREQ, fps, True))))

    leaky.set_tc_func(f_print_tc)

    total_duration = 0
    ts_first = prev_ts
    total_bytes = 0
    for epoch in epochs:
        for ds in epoch:
            bytes_in_ds = 0
            for seg in ds:
                is_ok &= leaky.step(seg)
                bytes_in_ds += seg.length + 13
            leaky.set_bitrate(bytes_in_ds, ds.pcs.pts, prev_ts)
            if ds.pcs.pts < prev_ts and ts_first < np.inf and ts_first != prev_ts:
                total_duration += LeakyBuffer.TS_MASK + 1
                ts_first = np.inf
            total_bytes += bytes_in_ds
            prev_ts = ds.pcs.pts
    ##for epoch
    total_duration += (epochs[-1][-1].pcs.pts - epochs[0][0].pcs.pts)
    stats = leaky.get_stats()

    avg_bitrate = total_bytes/(total_duration/GraphicsDecoder.FREQ)
    logger.iinfo(f"Bitrate: AVG={avg_bitrate/(128*1024):.04f} Mbps, PEAK(1s)={stats[2]:.03f} Mbps @ {leaky.stats.tsavg}.")

    f_log_fun = logger.iinfo if is_ok else logger.warning
    f_log_fun(f"Target bitrate underflow margin (higher is better): AVG={stats[1]:.02f}%, MIN={stats[0]:.02f}% @ {leaky.stats.tsmin}")
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
    comply = ds.pcs is not None and PGSegmentType.PCS == ds.pcs.type
    comply = comply and 0 <= len(ds.pcs.composition_objects) <= 2
    if ds.pcs.composition_state != PCS.CompositionState.NORMAL_CASE:
        comply &= ds.pcs.palette_update is False # "Palette update on epoch start or acquisition."
        comply &= ds.wds is not None
    comply = comply and ds.pcs.palette_id < 8 # "Using undefined palette ID."

    if ds.wds:
        comply &= PGSegmentType.WDS == ds.wds.type
        comply &= ds.pcs.palette_update is False # "Manipulating windows on palette update (conflicting display updates)."
        comply &= 1 <= len(ds.wds.windows) <= 2 # "Unusual window count."

    if ds.pds:
        pds_ids = set()
        for pds in ds.pds:
            comply &= pds.palette_id not in pds_ids
            pds_ids.add(pds.palette_id)
            if ds.pcs.palette_update:
                comply &= ds.pcs.palette_id == pds.palette_id # "Palette ID mismatch between PCS and PDS on palette update."
            comply &= pds.palette_id < 8 # "Using undefined palette ID."
            comply &= len(pds.palette) <= 256 # "Defining more than 256 palette entries."
    if ds.ods:
        ctx_cnt = 0
        for ods in ds.ods:
            ctx_cnt += bool(ods.flag & ODS.DataFlag.FIRST)
            ctx_cnt -= bool(ods.flag & ODS.DataFlag.LAST)
        comply &= 0 == ctx_cnt # "ODS segments flags mismatch."
    return comply and (ds.end is not None) and ds[-1].type == PGSegmentType.END# "No END segment in DS."
####

def is_compliant(epochs: list[Epoch], fps: float) -> bool:
    prev_pts = -np.inf
    compliant = True
    warnings = 0
    cumulated_ods_size = 0
    prev_pcs_id = 0xFFFF

    to_tc = lambda pts: str(TC.s2tc(pts/GraphicsDecoder.FREQ, fps)) + ('' if (float(fps).is_integer() or fps < 25) else (', DF=' + str(TC.s2tc(pts/GraphicsDecoder.FREQ, fps, True))))

    for ke, epoch in enumerate(epochs):
        windows = {}
        ods_vn = {}
        ods_hash = {}
        ods_filled = set()
        pds_vn = [-1] * 8
        pals = [Palette() for _ in range(8)]

        if epoch[0].pcs.composition_state & PCS.CompositionState.EPOCH_START == 0:
            logger.warning(f"First display set in epoch is not an Epoch Start at {to_tc(epoch[0].pcs.pts)}.")
            compliant = False

        if epoch[0].wds is None:
            logger.critical("An epoch cannot start without defining windows.")
            return False, np.inf

        for wd in epoch[0].wds.windows:
            if wd.h_pos + wd.width > epoch[0].pcs.width or wd.v_pos + wd.height > epoch[0].pcs.height:
                logger.error(f"Window {wd.window_id} out of screen in epoch starting at {to_tc(epoch[0].pcs.pts)}.")
                compliant = False
            windows[wd.window_id] = (wd.h_pos, wd.v_pos, wd.width, wd.height)
        lwdb = [Box(w[1], w[3], w[0], w[2]) for w in  windows.values()]
        if len(lwdb) == 2 and Box.intersect(*lwdb).area > 0:
            logger.error(f"Overlapping windows in epoch starting at {to_tc(epoch[0].pcs.pts)}.")
            compliant = False

        epoch_ctx = PGEpochContext(None, lwdb)

        last_ds = []
        for kd, ds in enumerate(epoch.ds):
            if not test_diplayset(ds):
                logger.error(f"DS {kd} of Epoch {ke} failed basic test.")
                compliant = False
                return 0, 0
            current_pts = ds.pcs.pts

            if epoch.ds[kd-1].pcs.pts != prev_pts and current_pts != epoch.ds[kd-1].pcs.pts:
                prev_pts = epoch.ds[kd-1].pcs.pts
            else:
                logger.warning(f"Two display sets at {to_tc(current_pts)}.")
            if kd > 0 and ds.pcs.composition_state == PCS.CompositionState.EPOCH_START:
                logger.error(f"Found an Epoch Start at {to_tc(current_pts)} in the middle of an epoch.")
                compliant = False

            is_dupe = False
            if len(ds) == len(last_ds) and ds.pcs.composition_number == prev_pcs_id:
                # Make copies and wipe fields that could change (timestamps, composition state)

                different = False
                for s1, s2 in zip(ds[1:], last_ds[1:]):
                    if s1.get_payload() != s2.get_payload():
                        different = True
                if not different:
                    is_dupe = ds.pcs.composition_state == PCS.CompositionState.ACQUISITION
                    is_dupe = is_dupe and last_ds.pcs.composition_state in [PCS.CompositionState.ACQUISITION, PCS.CompositionState.EPOCH_START]
                    is_dupe = is_dupe and ds.pcs.get_payload()[:7] == last_ds.pcs.get_payload()[:7]
                    is_dupe = is_dupe and ds.pcs.get_payload()[8:] == last_ds.pcs.get_payload()[8:]
            last_ds = ds

            current_ods_processed = None
            for ks, seg in enumerate(ds.segments):
                if seg.type == PGSegmentType.PCS:
                    if not is_dupe and seg.composition_number != (prev_pcs_id + 1) & 0xFFFF and seg.composition_state != PCS.CompositionState.EPOCH_START:
                        logger.warning(f"Displayset does not increment composition number normally at {to_tc(current_pts)}.")
                    prev_pcs_id = seg.composition_number
                    if int(seg.composition_state) != 0:
                        # On acquisition, past palettes and objects should not be accessed
                        ods_filled.clear()
                        for pal in pals:
                            pal.clear()
                        pds_vn = [-1]*8
                    # Not sure about that one, if there's no PDS it's probably fine?
                    if not epoch_ctx.update_palette_reservation(seg.palette_id, seg.pts, seg.dts) and len(seg.composition_objects) and len(ds.pds) == 0:
                        logger.error("Cannot decode to the specified palette, it is taken by a preceding display set.")
                        compliant = False

                elif seg.type == PGSegmentType.WDS:
                    for w in seg.windows:
                        if windows[w.window_id] != (w.h_pos, w.v_pos, w.width, w.height):
                            logger.error(f"Window change mid-epoch at {to_tc(current_pts)}, this is strictly prohibited.")
                            compliant = False

                elif seg.type == PGSegmentType.PDS:
                    if pds_vn[seg.palette_id] != -1 and (pds_vn[seg.palette_id] + 1) & 0xFF != seg.palette_version and not is_dupe:
                        logger.warning(f"Palette version not incremented by one, may be discarded by decoder. Palette {seg.p_id} at DTS {to_tc(seg.pts)}.")
                    pds_vn[seg.palette_id] = seg.palette_version
                    new_pal = seg.palette
                    pals[seg.palette_id] |= new_pal
                    if next(filter(lambda x: not (16 <= x.Y <= 235 and 16 <= x.Cb <= 240 and 16 <= x.Cr <= 240), new_pal.values()), None) is not None:
                        logger.warning(f"Palette is not limited range at {to_tc(current_pts)}.")
                        compliant = False
                    if (pal_ff_entry := pals[seg.palette_id].get(0xFF, None)) is not None and pal_ff_entry.alpha != 0:
                        logger.warning(f"Palette entry 0xFF is set and not transparent at {to_tc(current_pts)}.")
                        compliant = False

                elif seg.type == PGSegmentType.ODS:
                    if seg.flag & ODS.DataFlag.FIRST:
                        ods_data = bytearray()
                        ods_width = seg.width
                        ods_height = seg.height
                        current_ods_processed = ods_object_id = seg.object_id
                        ods_object_vn = seg.object_version
                        if 8 > min(ods_width, ods_height) or 4096 < max(ods_width, ods_height):
                            logger.error(f"Illegal object dimensions at {to_tc(current_pts)}, object id={seg.object_id}: {ods_width}x{ods_height}.")
                            compliant = False
                            continue #We can't do the buffer allocation below with the illegal dimension
                        object_shape = Shape(h=seg.height, w=seg.width)
                        if (slot := epoch_ctx.buffer.get_indexed(seg.object_id)) is None:
                            if not epoch_ctx.buffer.allocate_indexed(object_shape, seg.object_id):
                                logger.error(f"Object buffer overflow (not enough memory for all object slots) at {to_tc(current_pts)}.")
                                compliant = False
                        elif slot.shape != object_shape:
                            logger.error(f"Object-slot {seg.object_id} dimensions mismatch. Slot: {slot.shape}, object: {(seg.width, seg.height)} at {to_tc(current_pts)}.")
                            compliant = False
                        if compliant and not epoch_ctx.update_object_reservation(seg.object_id, ds.pcs.pts, ds.pcs.dts):
                            logger.error("Object ID being decoded is reserved by a preceding Display Set. Possible screen corruption.")
                            compliant = False
                        if cumulated_ods_size > 0:
                            logger.error("A past ODS was not properly terminated! Stream is critically corrupted!")
                            compliant = False
                    elif ods_object_id != seg.object_id or ods_object_vn != seg.object_version or current_ods_processed is None:
                        logger.error("Object definition header mismatch in a chain of segments! Stream is critically corrupted!")
                        compliant = False

                    cumulated_ods_size += seg.length
                    ods_data += seg.data

                    if seg.flag & ODS.DataFlag.LAST and current_ods_processed is not None:
                        data_hash = hash(bytes(ods_data))
                        # +6 (PES header) +13 (Optional PES header), +1 (type) +2 (length) +2 (object_id) +1 (object_vn) +1 (flags) = 26
                        # +13 (header(2) + PTS(4) + DTS(4) + type(1) + length(2)) +2 (object_id) +1 (object_vn) +1 (flags) = 17
                        # The Coded Object Buffer can hold up to 1 MiB of raw PES data
                        # This is roughly: "16 full PES packets" or "16 full b'PG' segments + 16*9 bytes"
                        if cumulated_ods_size >= (1 << 20)-(16*9):
                            logger.error(f"Coded buffer overflow at {to_tc(current_pts)}: coded object size exceed 1 MiB.")
                            compliant = False
                        cumulated_ods_size = 0

                        if seg.object_id in ods_filled and ods_hash.get(seg.object_id, None) != data_hash and ods_vn[seg.object_id] == seg.object_version:
                            logger.warning(f"Object {seg.o_id} at {to_tc(current_pts)} differs from previous but does not increment version number. It will be discarded.")
                        ods_filled.add(seg.object_id)
                        ods_vn[seg.object_id] = seg.object_version
                        ods_hash[seg.object_id] = data_hash

                        #Hypothesis: the graphic controller processes one RLE command (byte) per Rd tick
                        # To avoid decode time > object write time, RLE line must be smaller or equal to width + marker.
                        try:
                            dec_bitmap = Brule.decode(ods_data, width=ods_width, height=ods_height, check_rle=True)
                        except AssertionError:
                            dec_bitmap = Brule.decode(ods_data, width=ods_width, height=ods_height, check_rle=False)
                            logger.warning(f"ODS at {to_tc(current_pts)} has too long RLE line(s). Older decoders may have issues.")
                            warnings += 1

                        for pe in np.unique(dec_bitmap):
                            if pe != 0xFF and pe not in pals[ds.pcs.palette_id]:
                                logger.warning(f"ODS at {to_tc(current_pts)} uses undefined palette entries (first: {pe:02X}). Some pixels will not display.")
                                warnings += 1
                                break
                        current_ods_processed = None
                    elif current_ods_processed is None:
                        logger.error("Object definition header mismatch in a chain of segments! Stream is critically corrupted!")
                        compliant = False

                    #### if seg.flags
                elif seg.type == PGSegmentType.END:
                    # Control the spatial values of the composition w.r.t. object
                    if ds.wds:
                        for cobj in ds.pcs.composition_objects:
                            slot = epoch_ctx.buffer.get_indexed(cobj.object_id)
                            if slot is None:
                                logger.error(f"Using an unknown slot {cobj.object_id} in buffer at {to_tc(current_pts)}.")
                                compliant = False
                            elif cobj.object_id not in ods_filled:
                                logger.error(f"Using expired memory for object {cobj.object_id} at {to_tc(current_pts)}.")
                                compliant = False
                            else:
                                w, h = slot.shape.width, slot.shape.height
                                if cobj.cropped_flag:
                                    if h < cobj.c_h or w < cobj.c_w or h < cobj.c_h + cobj.vc_pos or w < cobj.c_w + cobj.hc_pos:
                                        logger.error(f"Cropped dimension exceeed object {cobj.object_id} size at {to_tc(current_pts)}.")
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
                                logger.error(f"Composition object {cobj.object_id} misplaced outside of window {wd.window_id} at {to_tc(current_pts)}.")
                                compliant = False
                        ####for cobj
                    ####if wds
                ####elif END
            ####for
            if not (pds_vn[ds.pcs.palette_id] >= 0) and (len(ds.pcs.composition_objects) or ds.pcs.palette_update):
                logger.error(f"Palette {ds.pcs.palette_id} unused or unset in decoder.")
                compliant = False
        #### for ds
    ####for epoch
    return compliant, warnings

def debug_stats(epochs: list[Epoch]) -> str:
    cnt_acq = cnt_nc = cnt_es = 0
    cnt_nc_ods = 0
    cnt_pu = cnt_buffered_pu = 0
    for epoch in epochs:
        pts_delta_w = [int(np.ceil(w.width*w.height*GraphicsDecoder.FREQ/GraphicsDecoder.RC)) for w in epoch[0].wds.windows]
        pts_delta = sum(pts_delta_w)
        for ds in epoch:
            if ds.pcs.composition_state == PCS.CompositionState.NORMAL_CASE:
                cnt_nc += 1
            elif ds.pcs.composition_state & PCS.CompositionState.ACQUISITION:
                cnt_acq += 1
            elif ds.pcs.composition_state & PCS.CompositionState.EPOCH_START:
                cnt_es += 1
            if ds.pcs.palette_update:
                cnt_pu += 1
                if (ds.pcs.pts - ds.pcs.dts) > pts_delta+1:
                    cnt_buffered_pu += 1
            elif ds.wds and ds.pcs.composition_state == 0 and ds.ods:
                cnt_nc_ods += 1
    return f"n(ES)={cnt_es}, n(ACQ)={cnt_acq}, n(NC)={cnt_nc} (n(ODS)={cnt_nc_ods}, n(PU)={cnt_pu}). Buffered: n(PU)={cnt_buffered_pu}."

def check_pts_dts_sanity(epochs: list[Epoch], fps: float) -> bool:
    is_compliant = True
    prev_pts = prev_dts = epochs[0][0].pcs.pts - GraphicsDecoder.FREQ

    to_tc = lambda pts: str(TC.s2tc(pts, fps)) + ('' if (float(fps).is_integer() or fps < 25) else (', DF=' + str(TC.s2tc(pts, fps, True))))
    frame_duration = np.floor(GraphicsDecoder.FREQ/fps)

    for k, epoch in enumerate(epochs):
        pts_delta = int(sum(np.ceil(w.width*w.height*GraphicsDecoder.FREQ/GraphicsDecoder.RC) for w in epoch[0].wds.windows))
        wipe_duration = int(np.ceil(epoch[0].pcs.width*epoch[0].pcs.height*GraphicsDecoder.FREQ/GraphicsDecoder.RC))
        pts_dts_delta_epoch_start = (epoch[0].pcs.pts - epoch[0].pcs.dts) > wipe_duration
        #Must not decode epoch start before previous epoch is fully finished (at PTS)
        diff = (epoch[0].pcs.dts - prev_pts)
        pts_last_dts_epoch_start = diff > 0
        if not pts_last_dts_epoch_start:
            logger.error(f"DTS of {to_tc(epoch[0].pcs.pts)} predates PTS of previous epoch by {epoch[0].pcs.dts - prev_pts} ticks.")
        if not pts_dts_delta_epoch_start:
            logger.error(f"Incorrect PTS-DTS values for epoch start DS @ {to_tc(epoch[0].pcs.pts)}.")

        is_compliant &= pts_last_dts_epoch_start & pts_dts_delta_epoch_start

        for l, ds in enumerate(epoch):
            ds_comply = (ds.pcs.dts - prev_dts) >= 0
            ds_comply &= (ds.pcs.pts - prev_pts) > 0
            if ds.wds:
                # WDS action requires pts_delta margin from previous DS
                diff = (ds.pcs.pts - prev_pts)
                ds_comply &= diff > pts_delta
                #WDS deadline is pts_delta close to final pts
                ds_comply &= (ds.pcs.pts - ds.wds.pts) <= pts_delta
                #WDS decoding should be realistic (epoch start is worst case)
                ds_comply &= (ds.wds.pts - ds.wds.dts) <= wipe_duration*2
                ds_comply &= (ds.wds.pts != ds.wds.dts)
            else:
                # Palette update and others requires one frame duration as margin
                ds_comply &= (ds.pcs.pts - prev_pts) >= frame_duration
            for pds in ds.pds:
                ds_comply &= pds.pts == pds.dts
            for ods in ds.ods:
                ds_comply &= ods.pts != ods.dts
            for seg in ds:
                #All PTS shall be smaller than the PCS PTS
                ds_comply &= (ds.pcs.pts - seg.pts) >= 0
                #All DTS shall be larger or equal to the PCS DTS
                ds_comply &= (seg.dts - ds.pcs.dts) >= 0
                #All PTS shall be larger or equal to the PCS DTS
                ds_comply &= (seg.pts - ds.pcs.dts) >= 0
                #All PTS shall be larger or equal to the DTS
                ds_comply &= (seg.pts - seg.dts) >= 0

                diff = (seg.dts - prev_dts)
                ds_comply &= diff >= 0
                #Segment lifetime in decoder should be less than one second
                ds_comply &= (seg.pts - seg.dts) < GraphicsDecoder.FREQ
                prev_dts = seg.dts
            prev_pts = ds.pcs.pts
            if not ds_comply:
                logger.error(f"Incorrect PTS-DTS at {to_tc(ds.pcs.pts)}, DS:S={ds.pcs.composition_state:02X}:PU={ds.pcs.palette_update > 0}, stream is out of spec.")
            is_compliant &= ds_comply
        ####ds
    ####epochs
    return is_compliant
