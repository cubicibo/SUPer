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
import numpy as np
from PIL import Image
from itertools import zip_longest
from typing import Self

from .geometry import Box, Rectangle
from .internals import GraphicsDecoder, LogFacility, TC
from .codecctx import PGStreamCtx, PGEpochContext, PGObjectBuffer
from .imgproc import ImageSequence
from .eventdetect import WindowsObjectDetector, ProspectiveObject

from .palette import Palette, PaletteEntry
from .pgstreams import Epoch, DisplaySet
from .segments import PCS, WDS, END, CompositionObject

logger = LogFacility.get_logger('SUPer')

class DSNode:
    def __init__(self,
            objects: list[ProspectiveObject | None],
            windows: list[Box],
            tc_pts: TC,
            is_palette_update: bool = False,
            new_mask: list[bool] = []
        ) -> None:
        assert len(objects) == len(new_mask)
        self.objects = objects
        self.windows = windows
        self.tc_pts = tc_pts
        self.is_palette_update = is_palette_update
        self.new_mask = new_mask

        self.slots = [None] * len(self.objects)
        self.pos = [None] * len(self.objects)
        self.partial = False
        self.idx = 0

        # These fields are not copied over
        self.parent = None
        self.palette_id = None
        self._dts = None

    def copy(self) -> 'DSNode':
        new_node = self.__class__(self.objects.copy(), self.windows, self.tc_pts,
                                  is_palette_update=self.is_palette_update, new_mask=self.new_mask.copy())
        new_node.slots = self.slots.copy()
        new_node.pos = self.pos.copy()
        new_node.partial = self.partial
        new_node.idx = self.idx
        return new_node

    def wipe_duration(self) -> int:
        return np.ceil(sum(map(lambda w: GraphicsDecoder.FREQ*w.dy*w.dx/GraphicsDecoder.RC, self.windows)))

    def write_duration(self) -> int:
        return sum(map(lambda w: np.ceil(GraphicsDecoder.FREQ*w.dy*w.dx/GraphicsDecoder.RC), self.windows))

    def set_dts(self, dts: int | None) -> None:
        assert dts is None or dts <= self.dts() or (self.is_palette_update and dts < self.pts())
        self._dts = dts

    def dts_end(self) -> float:
        if self._dts is not None:
            return (sum(self.get_dts_markers()[1]) + self._dts)
        decode_ts, t_dec = self.get_dts_markers()
        return (decode_ts + sum(t_dec))

    def dts(self) -> float:
        if self._dts is not None:
            return self._dts
        return self.get_dts_markers()[0]

    def delta_dts(self) -> float:
        return sum(self.get_dts_markers()[1])

    def pts(self) -> float:
        return self.tc_pts.to_pts()

    def is_custom_dts(self) -> bool:
        return not (self._dts is None)

    def get_decode_duration(self) -> tuple[int, int]:
        t_decoding = []

        if not self.is_palette_update:
            assert any(self.objects)
            target_windows = list(map(lambda o: o.wid, filter(lambda o: o is not None, self.objects)))
            assigned_wids = set(target_windows)
            decode_duration = sum([np.ceil(self.windows[wid].dy*self.windows[wid].dx*GraphicsDecoder.FREQ/GraphicsDecoder.RC) for wid in range(len(self.windows)) if wid not in assigned_wids])

            #writing twice to the same window?
            delay_write = 2 == len(target_windows) and 1 == len(assigned_wids)

            t_other_copy = 0
            for obj in filter(lambda o: o is not None, self.objects):
                wid = obj.wid

                box = self.windows[wid]
                write = box.dy*box.dx*GraphicsDecoder.FREQ
                if self.slots[wid] is not None:
                    read = int(self.slots[wid][0])*int(self.slots[wid][1])*GraphicsDecoder.FREQ
                else:
                    #no slot -> buffer is sized to the window
                    read = write
                if not self.partial or (self.partial and self.new_mask[wid]):
                    t_decoding.append(np.ceil(read/GraphicsDecoder.RD))
                elif self.partial and not self.new_mask[wid]:
                    #the other object is copied at the end.
                    assert sum(self.new_mask) == 1 and t_other_copy == 0
                    t_other_copy = np.ceil(write/GraphicsDecoder.RC)
                    continue

                decode_duration = max(decode_duration, sum(t_decoding)) + np.ceil(write/GraphicsDecoder.RC)*int(not delay_write)
            ####
            if delay_write:
                write_duration = np.ceil(write/GraphicsDecoder.RC)
                assert t_other_copy == 0 or write_duration == t_other_copy
                decode_duration += write_duration
            else:
                assert t_other_copy == 0 or self.partial
                decode_duration += t_other_copy
        else:
            decode_duration = self.write_duration() + 1
        return (decode_duration, t_decoding)

    def get_dts_markers(self) -> tuple[int, int]:
        decode_duration, t_decoding = self.get_decode_duration()
        return (self.pts() - decode_duration, t_decoding)
    ####

    @classmethod
    def set_pts_dts_sc(cls, ds: DisplaySet, buffer: PGObjectBuffer, wds: WDS, node: Self) -> list[tuple[int, int]]:
        """
        This function generates the timestamps (PTS and DTS) associated to a given DisplaySet.

        :param ds: DisplaySet, PTS of PCS must be set to the right value.
        :param buffer: Object buffer that supports allocation and returning a size of allocated slots.
        :param wds: WDS of the epoch.
        :return: Pairs of timestamps in ticks for each segment in the displayset.
        """
        ddurs = {}
        for ods in ds.ods:
            if ods.flag & ods.DataFlag.FIRST:
                object_shape = Rectangle(h=ods.height, w=ods.width)
                assert ods.object_id not in ddurs, f"Object {ods.object_id} defined twice in DS."
                if (slot := buffer.get_indexed(ods.object_id)) is not None:
                    assert object_shape == slot.shape, "Dimension mismatch, buffer corruption."
                else:
                    # Allocate a buffer slot for this object
                    assert buffer.allocate_indexed(object_shape, ods.object_id) is True, "Critical error: object buffer overflow."
                ddurs[ods.object_id] = np.ceil(ods.height*ods.width*GraphicsDecoder.FREQ/GraphicsDecoder.RD)

        t_decoding = 0
        decode_duration = 0
        wipe_duration = __class__.get_wipe_duration(wds)

        windows = {wd.window_id: (wd.height, wd.width) for wd in wds.windows}

        if ds.pcs.composition_state == ds.pcs.CompositionState.EPOCH_START:
            decode_duration = np.ceil(ds.pcs.width*ds.pcs.height*GraphicsDecoder.FREQ/GraphicsDecoder.RC)
        else:
            assigned_windows = list(map(lambda x: x.window_id, ds.pcs.composition_objects))
            unassigned_windows = [wd for wd in windows if wd not in assigned_windows]
            decode_duration = sum([np.ceil(windows[wid][0]*windows[wid][1]*GraphicsDecoder.FREQ/GraphicsDecoder.RC) for wid in unassigned_windows])

        object_decode_duration = ddurs.copy()

        if not ds.pcs.palette_update:
            #For every composition object, compute the transfer time
            for k, cobj in enumerate(ds.pcs.composition_objects):
                assert buffer.get_indexed(cobj.object_id) is not None, "Object does not exist in buffer."
                w, h = windows[cobj.window_id][0], windows[cobj.window_id][1]

                t_dec_obj = object_decode_duration.pop(cobj.object_id, 0)
                t_decoding += t_dec_obj

                # Same window -> patent claims a window is written only once after the two cobj are processed.
                if k == 0 and ds.pcs.num_composition_objects > 1 and ds.pcs.composition_objects[1].window_id == cobj.window_id:
                    continue
                copy_dur = np.ceil(w*h*GraphicsDecoder.FREQ/GraphicsDecoder.RC)
                decode_duration = max(decode_duration, t_decoding) + copy_dur

        #Prevent PTS(WDS) = DTS(WDS) (a WDS shall have a DTS and it shall differ from the PTS)
        decode_duration = max(decode_duration, sum(map(lambda w: np.ceil(GraphicsDecoder.FREQ*w[0]*w[1]/GraphicsDecoder.RC), windows.values())) + 1)

        dts = int(ds.pcs.pts - decode_duration)
        assert node.pts() == ds.pcs.pts

        if node.is_custom_dts():
            new_dts = node.dts()
            assert new_dts <= dts or ds.pcs.palette_update, f"new={new_dts}, min={dts}, {node.tc_pts}"
            dts = new_dts

        #PCS always exist
        ts_pairs = [(ds.pcs.pts, dts)]

        if ds.wds:
            ts_pairs.append((int(ds.pcs.pts - wipe_duration), dts))
        for pds in ds.pds:
            ts_pairs.append((dts, dts))

        for ods in ds.ods:
            ods_pts = int(dts + ddurs.get(ods.object_id))
            ts_pairs.append((ods_pts, dts))
            if ods.flag & ods.DataFlag.LAST:
                dts = ods_pts
        ts_pairs.append((dts, dts))
        return ts_pairs
    ####

    @staticmethod
    def get_wipe_duration(wds: WDS) -> int:
        return np.ceil(sum(map(lambda w: GraphicsDecoder.FREQ*w.height*w.width/GraphicsDecoder.RC, wds.windows)))

    @classmethod
    def apply_pts_dts(cls, ds: DisplaySet, ts: tuple[int, int]) -> None:
        assert len(ds) == len(ts), "Timestamps-DS size mismatch."
        for seg, (pts, dts) in zip(ds, ts):
            seg.pts, seg.dts = pts, dts
    ####
####

class EpochEncoderEngine:
    def __init__(self, ectx, stream_ctx: PGStreamCtx, kwargs) -> None:
        self.ectx = ectx
        self.kwargs = kwargs
        self._codec = PGEpochContext(stream_ctx, self.ectx.windows)
        
    def analyze(self) -> tuple[...]:
        ssim_tol = self.kwargs.get('ssim_tol', 0)
        detector = WindowsObjectDetector(self._codec.bd_video.fmt, self.ectx.windows, ssim_tol)
        
        pgobjs = detector.get_objects(self.ectx.events)
        
        # Create all potential display sets in the epoch
        durs, nodes = self.create_displaysets_nodes([objs.copy() for objs in pgobjs])
        return pgobjs, durs, nodes

    def plan(self, ctx: tuple[...]) -> tuple[...]:
        pgobjs, durs, nodes = ctx
        #Plan datastream
        states, flags, cboxes = self.shape_stream(durs, nodes)

        #Set-up datastructures for bytestream generation
        self.set_pgobjects_extended_visibilities(nodes)
        r_states, r_durs, r_nodes, r_flags = self.roll_nodes(nodes, durs, flags, states)
        
        return r_states, pgobjs, r_durs, r_flags, r_nodes

    def encode(self, ctx: tuple[...]) -> Epoch:
        r_states, pgobjs, r_durs, r_flags, r_nodes = ctx
        #Generate datastream according to plan
        return self._convert(r_states, pgobjs, r_durs, r_flags, r_nodes)

    def shape_stream(self,
         durs: list[int],
         nodes: list[DSNode],
    ) -> tuple[list[PCS.CompositionState], list[int], list[list[Box]]]:

        allow_normal_case = self.kwargs.get('allow_normal_case', False)
        allow_overlaps = self.kwargs.get('allow_overlaps', False)

        acqs, absolutes, margins, bslots, cboxes = self.find_acqs(durs, nodes)
        flags = [0] * len(durs)

        states = [PCS.CompositionState.NORMAL_CASE] * len(acqs)
        states[0] = PCS.CompositionState.EPOCH_START
        drought = 0

        thresh = self.kwargs.get('quality_factor', 0.75)
        dthresh = self.kwargs.get('dquality_factor', 0.035)
        refresh_rate = max(0, min(self.kwargs.get('refresh_rate', 1.0), 1.0))

        positions = cboxes[0].copy()
        k = last_acq = 0
        for k, (acq, forced, margin, node) in enumerate(zip(acqs[1:], absolutes[1:], margins[1:], nodes[1:]), 1):
            if not node.is_palette_update:
                for wid in range(len(self.ectx.windows)):
                    box_assets = list(filter(lambda x: x is not None, [positions[wid], cboxes[k][wid]]))
                    if len(box_assets) > 0:
                        cont = Box.union(*box_assets)

                        if cont.dx > bslots[wid][1] or cont.dy > bslots[wid][0]:
                            assert cboxes[k][wid] is not None
                            states[k] = PCS.CompositionState.ACQUISITION
                            absolutes[k] = True
                            node.new_mask[wid] = True #For possible Normal case update
                            drought = 0
                        else:
                            positions[wid] = cont
                #### for wid
            #### if not nc
            if thresh == 0 and not node.is_palette_update:
                states[k] = PCS.CompositionState.ACQUISITION
                absolutes[k] = True
            if states[k] != PCS.CompositionState.ACQUISITION:
                if (forced or (acq and margin > max(thresh-dthresh*drought, 0))) and not node.is_palette_update:
                    states[k] = PCS.CompositionState.ACQUISITION
                    drought = 0
                else:
                    #prevent excessive acquisitions, as we want to compress the stream.
                    drought += 1*refresh_rate
                if states[k] == PCS.CompositionState.NORMAL_CASE:
                    nodes[k].is_palette_update = True
            if states[k] > 0:
                for zk in range(last_acq, k):
                    nodes[zk].pos = positions
                positions = cboxes[k].copy()
                last_acq = k
        for zk in range(last_acq, k+1):
            nodes[zk].pos = positions

        pts_delta = nodes[0].write_duration()

        if 2 == len(self.ectx.windows):
            self.shift_forward_overlay(nodes, states, absolutes, acqs, allow_overlaps, pts_delta)

        self.filter_events(nodes, states, flags, absolutes, durs, pts_delta, allow_normal_case, allow_overlaps)
        cls = __class__
        if allow_overlaps:
            cls.align_palette_updates(nodes, states, flags)
        cls.verify_palette_usage(nodes, states, flags, allow_overlaps)
        return states, flags, cboxes

    def align_palette_updates(
        nodes: list[DSNode],
        states: list[PCS.CompositionState],
        flags: list[int],
    ) -> None:
        first_possible_dts = (nodes[0].dts_end())
        for ck, node in enumerate(nodes[1:], 1):
            if flags[ck] < 0:
                continue
            if states[ck] == 0 and node.is_palette_update:
                last_possible_dts = np.inf
                current_dts = node.dts()
                if (current_dts) < first_possible_dts and first_possible_dts < (node.pts()):
                    for ckf, flag in enumerate(flags[ck+1:], ck+1):
                        if flag >= 0:
                            last_possible_dts = (nodes[ckf].dts())
                            break
                    if first_possible_dts < last_possible_dts:
                        node.set_dts(min((first_possible_dts+1), last_possible_dts))
                        logger.debug(f"Shifted DTS of PU at {node.tc_pts}={node.pts():.03f} from {current_dts:.04f} to {node.dts():.04f}.")
                    else:
                        logger.error(f"Required to drop a PU at {node.tc_pts}={node.pts():.03f} to ensure a monotonic DTS.")
                        flags[ck] = -1
                        continue
            first_possible_dts = (node.dts_end())
        ####for ck
    ####

    def shift_forward_overlay(self,
          nodes: list[DSNode],
          states: list[PCS.CompositionState],
          absolutes: list[bool],
          acqs: list[bool],
          allow_overlaps: bool,
          pts_delta: float,
      ) -> None:
        #First backtrack: remove acquisitions to display one window after the other
        for k, node in enumerate(nodes):
            if acqs[k] or node.objects == [] or sum(node.new_mask) != 1:
                continue
            assert absolutes[k]
            future_obj_idx = node.new_mask.index(True)

            scores = []
            drop_pal_ups_def = 0
            drop_abs_acq_def = False
            j = k
            while (j := j-1) and (nodes[j].dts_end() >= node.dts() or nodes[j].pts() + pts_delta >= node.pts()):
                drop_abs_acq_def |= absolutes[j]
                drop_pal_ups_def += int(not allow_overlaps and nodes[j].is_palette_update)

            other_new_mask = 0
            for pk, pnode in enumerate(reversed(nodes[:k]), 1):
                if pnode.objects == []:
                    continue
                redefine_same_object = next(filter(lambda x: x > 1, map(sum, zip(node.new_mask, pnode.new_mask))), None) is not None
                overlap_in_window = sum(map(lambda x: x is not None, [node.objects[future_obj_idx], pnode.objects[future_obj_idx]])) > 1
                other_new_mask += pnode.new_mask[1-future_obj_idx]
                #Same object is redefined in the previous DS, give up
                if redefine_same_object or overlap_in_window or other_new_mask > 1 or pk > 15:
                    break

                new_node = pnode.copy()
                new_node.new_mask[future_obj_idx] = True
                new_node.objects[future_obj_idx] = node.objects[future_obj_idx]
                new_node.pos[future_obj_idx] = node.pos[future_obj_idx]
                new_node.is_palette_update = False

                drop_abs_acq = False
                drop_pal_ups = 0
                j = k - pk
                while (j := j-1) >= 0 and (nodes[j].dts_end() >= new_node.dts() or nodes[j].pts() + pts_delta >= new_node.pts()):
                    drop_abs_acq |= absolutes[j]
                    drop_pal_ups += int(not allow_overlaps and nodes[j].is_palette_update)

                if not drop_abs_acq:
                    #Shifting up to epoch start and acquisition at j=1 is not possible?
                    if j == 0 and (nodes[j].dts_end() >= new_node.dts() or nodes[j].pts() + pts_delta >= new_node.pts()) and\
                       next(filter(lambda x: x > 1, map(sum, zip(node.new_mask, pnode.new_mask))), None) is None:
                        scores.append((0, drop_pal_ups, new_node, 0))
                        break #Hit epoch start, can't go any closer

                    elif nodes[j].dts_end() < new_node.dts() and nodes[j].pts() + pts_delta < new_node.pts():
                        scores.append((k - pk, drop_pal_ups, new_node, j+1))

                    #quick exit
                    if 0 == drop_pal_ups or (allow_overlaps and len(scores)):
                        break
            ####for pk, node
            if scores:
                #jk: preceeding nodes, best_pk: promoted node
                best_pk, drop_palups, new_node, jk = min(scores, key=lambda x: x[1] + 0.1249*(k - x[0]))
                #Only do the shift if worthwile
                if drop_pal_ups_def > drop_palups or drop_abs_acq_def:
                    new_node.objects[future_obj_idx].pad_left(node.idx - new_node.idx)

                    logger.debug(f"Merged acquisition at {nodes[best_pk].tc_pts} from {node.tc_pts}, NM={new_node.new_mask}, shift={node.idx - new_node.idx}")

                    if best_pk > 0:
                        states[best_pk] = PCS.CompositionState.ACQUISITION

                    absolutes[best_pk]   =   True
                    node.new_mask[future_obj_idx] = False

                    for j in range(jk, best_pk):
                        assert not absolutes[j]
                        states[j] = PCS.CompositionState.NORMAL_CASE
                        nodes[j].is_palette_update = True
                    for j in range(best_pk+1, k+1):
                        nodes[j].objects[future_obj_idx] = new_node.objects[future_obj_idx]
                        nodes[j].pos[future_obj_idx] = new_node.pos[future_obj_idx]
                        assert not absolutes[j] or j == k
                        states[j] = PCS.CompositionState.NORMAL_CASE
                        nodes[j].is_palette_update = True
                        absolutes[j] = False
                    #Apply new node to output
                    nodes[best_pk] = new_node
                ####if drop_pal_
            ####if scores
        ####for k, node
    ####

    def set_pgobjects_extended_visibilities(self, nodes: list[DSNode]) -> None:
        """
        Set the _ext_range visibility of a PGObject, used in partial screen refreshes

        Brief: Buffered palette updates are basic screen updates that are stacked up
         before a mandatory acquisition that takes a long time to decode.
         This makes some animation smoother (such as fades) and drop fewer events
         however these palette updates must operate on consistent data. some PGObject
         may be outdated but shall remain on screen for consistency (no blinking!)

         This function essentially look at the window occupancies and determine the lifetime of
         every pgobject. An empty window enforces them to be undisplayed!

         Everything done here is relevant solely if both --ahead and --allow-normal are used.
        """
        running_objs = [[] for _ in range(len(self.ectx.windows))]
        nk = 0
        for node in nodes:
            # skip wipes: composition objects encoding is agnostic to them
            if 0 == len(node.objects):
                continue
            nk = node.idx
            assert nk >= 0
            wipe_everything = [False] * len(self.ectx.windows)
            for wid, obj in enumerate(node.objects):
                empty_wd = obj is None
                assert empty_wd or obj.is_active(nk), (nk, obj.f, len(obj.mask), obj.mask)
                if empty_wd or not obj.is_visible(nk): # ext range of a masked object will be updated several time
                    wipe_everything[wid] = True
            for wid, _ in filter(lambda tw: tw[1], enumerate(wipe_everything)):
                for past_object in running_objs[wid]:
                    past_object.set_extended_visibility_limit(nk)
                running_objs[wid].clear()
            for wid, obj in enumerate(node.objects):
                if obj is not None and (0 == len(running_objs[wid]) or running_objs[wid][-1] != obj):
                    running_objs[wid].append(obj)
        #overly careful - set extended visibility of remaining objects to epoch length
        for past_object in running_objs[0] + (running_objs[1] if (2 == len(running_objs)) else []):
            past_object.set_extended_visibility_limit(nk+1)

        for nk, node in enumerate(nodes):
            for obj in filter(lambda x: x is not None, node.objects):
                assert obj.ext_range >= obj.f + len(obj.mask)
    ####

    def filter_events(self,
          nodes: list[DSNode],
          states: list[PCS.CompositionState],
          flags: list[int],
          absolutes: list[bool],
          durs: list[int],
          pts_delta: float,
          allow_normal_case: bool,
          allow_overlaps: bool,
    ) -> None:
        prefer_normal_case = self.kwargs.get('prefer_normal_case', False)
        #At this point, we have the stream acquisitions. Some may be impossible,
        # so we have to filter out some less relevant events.
        logger.debug("Backtracking to filter acquisitions and events.")
        k = len(states)
        last_dts = nodes[-1].dts() + 1.0

        while (k := k - 1) > 0:
            if flags[k] < 0:
                logger.debug(f"Not analyzing event at {nodes[k].tc_pts} due to filtering (f={absolutes[k]}, a={flags[k]}).")
                continue
            if states[k] == PCS.CompositionState.NORMAL_CASE:
                last_dts = nodes[k].dts()
                continue

            #look-up to next acquisition to see which objects are relevant in our case
            zk = k
            while (zk := zk + 1) < len(states) and states[zk] == PCS.CompositionState.NORMAL_CASE: pass
            upper_bound = (nodes[-1].idx+1) if zk == len(states) else nodes[zk].idx
            assert upper_bound > nodes[k].idx > 0
            dec_objs = [obj if __class__._object_is_relevant(obj, flags, slice(nodes[k].idx, upper_bound)) else None for obj in nodes[k].objects]
            diff = sum(map(lambda x: x is not None, nodes[k].objects)) - sum(map(lambda x: x is not None, dec_objs))

            if 1 == diff and 2 == len(self.ectx.windows):
                old_object_list = nodes[k].objects
                nodes[k].objects = dec_objs
                real_dts_end = nodes[k].dts_end()
                real_margin = last_dts - real_dts_end
                if real_margin < 0:
                    if allow_overlaps:
                        new_dts = nodes[k].dts()+real_margin
                        logger.debug(f"Shifted DTS of Acq at {nodes[k].tc_pts} from {nodes[k].dts()} to {new_dts} (collision due to reduced ODS count).")
                        nodes[k].set_dts(new_dts)
                    else:
                        # force object to be encoded despite having nothing visible at all
                        logger.debug(f"Adding an empty object in Acq at {nodes[k].tc_pts} due to DTS collision with a reduced ODS count.")
                        nodes[k].objects = old_object_list
                        discarded_obj_id = dec_objs.index(None)
                        nodes[k].objects[discarded_obj_id].mask[nodes[k].idx - nodes[k].objects[discarded_obj_id].f] = True
                ### if round
            #### if 1 ==

            assert states[k] == PCS.CompositionState.ACQUISITION, f"Filtering error: {nodes[k].tc_pts} k={nodes[k].idx} is not an acquisition. NM={nodes[k].new_mask} OM={list(map(lambda x: x is not None, nodes[k].objects))}."
            dts_start_nc = dts_start = nodes[k].dts()
            j = j_nc = k - 1
            while j > 0 and (nodes[j].dts_end() >= dts_start or nodes[j].pts() + pts_delta >= nodes[k].pts()):
                j -= 1

            #Normal case is only possible if we discard past acquisitions that redefined the same object
            normal_case_possible = sum(nodes[k].new_mask) == 1 and sum(map(lambda x: x is not None, nodes[k].objects)) == 2
            normal_case_possible &= allow_normal_case
            if normal_case_possible:
                mask = nodes[k].new_mask.copy()
                nodes[k].partial = True
                dts_start_nc = nodes[k].dts()

                while j_nc > 0 and (nodes[j_nc].dts_end() >= dts_start_nc or nodes[j_nc].pts() + pts_delta >= nodes[k].pts()):
                    if absolutes[j_nc]:
                        for km, mask_v in enumerate(nodes[j_nc].new_mask):
                            mask[km] |= mask_v
                    j_nc -= 1
                # Normal case
                normal_case_possible &= sum(mask) == 1
                nodes[k].partial = False

            #Normal case is not possible (collision with epoch start)
            nc_not_ok = normal_case_possible and j_nc == 0 and (nodes[j_nc].dts_end() >= dts_start_nc or nodes[j_nc].pts() + pts_delta >= nodes[k].pts())
            #Impossible normal case (could be disabled) or Not a normal case and collide with epoch start
            if nc_not_ok or (not normal_case_possible and j == 0 and (nodes[j].dts_end() >= dts_start or nodes[j].pts() + pts_delta >= nodes[k].pts())):
                #epoch start up to k are all cluttered together... we just move epoch start to k.
                logger.info(f"Epoch Start squeeze: dropping {k} event(s) before new ES at {nodes[k].tc_pts} (old ES: {nodes[0].tc_pts}).")
                for zk in range(0, k):
                    logger.info(f"Discarded event at {nodes[zk].tc_pts} preceeding new Epoch Start.")
                    flags[zk] = -1
                states[k] = PCS.CompositionState.EPOCH_START
                # the encoder function initializes the iterator to the index of epoch start - remove unused one.
                states[0] = PCS.CompositionState.ACQUISITION
                continue #(or break)

            #Filter the events
            is_normal_case = normal_case_possible and dts_start_nc > dts_start and (j_nc > j or prefer_normal_case or (j_nc == 0 and nodes[j].dts_end() >= dts_start))
            j_iter = j_nc if is_normal_case else j
            dts_iter = dts_start_nc if is_normal_case else dts_start
            dts_end_iter = nodes[max(j_iter-1, 0)].dts_end()

            num_pcs_buffered = 0

            #screen wipes don't contain objects, take the previous list
            objs = nodes[j_iter if len(nodes[j_iter].objects) else (j_iter-1)].objects
            objs = list(map(lambda obj: obj is not None, objs))
            for l in range(j_iter+1, k):
                # We ran out of PCS to buffer or the objects are too different or min delta PTS -> drop
                if flags[l] >= 0 and (not allow_overlaps or sum(objs) == 0 or num_pcs_buffered >= 7 or (nodes[l].pts() + pts_delta + 1 >= nodes[k].pts())):
                    logger.info(f"Discarded event at {nodes[l].tc_pts} to perform a mendatory acquisition.")
                    flags[l] = -1
                elif flags[l] == 0:
                    absolutes[l] = False
                    num_pcs_buffered += 1
                    nodes[l].is_palette_update = True

                    if nodes[l].dts() >= dts_iter:
                        logger.debug(f"Shift DTS {dts_iter:.04f}, {nodes[l].dts()}, {nodes[l].pts()}={nodes[l].tc_pts} {dts_end_iter}")
                        nodes[l].set_dts(max(dts_iter - 1, dts_end_iter))
                assert flags[l] != 1
                if any(nodes[l].new_mask) and flags[l] == 0 and allow_overlaps:
                    wd_occupied_count = sum(map(lambda x: x is not None, nodes[l].objects))
                    if wd_occupied_count == 2:
                        logger.info(f"Downgraded event at {nodes[l].tc_pts} to a palette update to perform a mendatory acquisition.")
                    else:
                        logger.info(f"Discarded event at {nodes[l].tc_pts} to perform a mendatory acquisition.")
                states[l] = PCS.CompositionState.NORMAL_CASE
                #Update object mask on which PUs are performed.
                # evaluated at the end since the above could be a valid wipe
                if allow_overlaps:
                    for ko, (obj, mask) in enumerate(zip(nodes[l].objects, nodes[l].new_mask)):
                        objs[ko] &= (obj is not None) and (not mask)

            nodes[k].partial = is_normal_case
            flags[k] = int(is_normal_case)
            if is_normal_case:
                states[k] = PCS.CompositionState.NORMAL_CASE #else equals Acquisition
                logger.info(f"Object refreshed with a Normal Case at {nodes[k].tc_pts}.")
            last_dts = nodes[k].dts()

        assert 1 == sum(map(lambda cs: cs == PCS.CompositionState.EPOCH_START, states))
        ####while (k := k - 1) > 0
    ####filter_events

    @staticmethod
    def verify_palette_usage(
         nodes: list[DSNode],
         states: list[PCS.CompositionState],
         flags: list[int],
         allow_overlaps: bool = False
    ) -> None:
        #Allocate palettes as a test, this is essentially doing a final sanity check
        #on the selected display sets. The palette values generated here are not used.

        prev_idx = -1
        for k, (node, state, flag) in enumerate(zip(nodes, states, flags)):
            assert (node.objects == [] and node.idx == -1) or len(node.objects) and node.idx > prev_idx
            if len(node.objects):
                prev_idx = node.idx
            if flag == 0 and state == PCS.CompositionState.NORMAL_CASE:
                #Palette update
                assert nodes[k].is_palette_update, f"{node.tc_pts} palette update k-node {k} not configured, NM={node.new_mask} P={node.partial}."
                assert allow_overlaps or not node.is_custom_dts()
            elif flag == 1:
                #Normal Case redefinition
                assert state == PCS.CompositionState.NORMAL_CASE
                assert nodes[k].objects != [] and sum(nodes[k].new_mask) == 1
            logger.debug(f"{k}: {state:02X} {flag:02}-{node.partial} DTS={node.dts()}->{node.dts_end()} PTS={node.pts()}={node.tc_pts} OM={node.new_mask} cdts={node.is_custom_dts()}")
        ####
    ####

    @staticmethod
    def _get_stack_direction(*box) -> tuple[np.ndarray[tuple[int, int], np.uint8], tuple[int, int]]:
        widths = list(map(lambda b: b.dx, box))
        heights = list(map(lambda b: b.dy, box))

        if max(heights)*sum(widths) <= max(widths)*sum(heights):
            return np.array([widths[0], 0], np.int32), (sum(widths), max(heights))
        return np.array([0, heights[0]], np.int32), (max(widths), sum(heights))

    @staticmethod
    def _object_is_relevant(pgo: ProspectiveObject, flags: list[int], sl: slice) -> bool:
        if pgo is None:
            return False
        assert len(flags) >= len(pgo.mask)
        return any(map(lambda z: z[0] and z[1] >= 0, zip(pgo.mask[sl.start-pgo.f:sl.stop-pgo.f], flags[sl])))

    def mask_event(self, wid: int, event_id: int) -> np.ndarray[tuple[int, int, int], np.uint8]:
        event = self.ectx.events[event_id]
        wd = self.ectx.windows[wid]
        event_box = event.get_bbox()
        zone_overlap = Box.intersect(wd, event_box)
        work_plane = np.zeros((wd.dy, wd.dx, 4), dtype=np.uint8)
        if zone_overlap.area > 0:
            #plane slices
            psy = slice(zone_overlap.y - wd.y, zone_overlap.y2 - wd.y)
            psx = slice(zone_overlap.x - wd.x, zone_overlap.x2 - wd.x)
            #image slices
            isy = slice(zone_overlap.y - event_box.y, zone_overlap.y2 - event_box.y)
            isx = slice(zone_overlap.x - event_box.x, zone_overlap.x2 - event_box.x)
            work_plane[psy, psx, :] = np.asarray(event.image, dtype=np.uint8)[isy, isx, :]
        return work_plane

    def _encode_composition_objects(self,
        i: int, k: int, pgobs_items, node: DSNode, has_two_objs: bool,
        c_pts: float, normal_case_refresh: bool | list[bool], flags: list[int], prev_cobjs: list[CompositionObject] | None = None
    ) -> ...:
        cobjs, pals, o_ods = [], [], []

        #In this mode, we re-combine the two objects in a smaller areas than in the original box
        # and then pass that to the optimiser. Colors are efficiently distributed on the objects.
        if has_two_objs and normal_case_refresh is False:
            bitmap, palettes = None, None
            for trial in range(2):
                compositions = [(oix, pgo) for oix, pgo in pgobs_items if __class__._object_is_relevant(pgo, flags, slice(i, k))]
                assert len(compositions) == 2
                #todo: stack using slot dimensions?
                offset, dims = self.__class__._get_stack_direction(*list(map(lambda x: x[1].box, compositions)))
                last_imgs = [None] * len(compositions)
                img_seq = ImageSequence(k-i, self.kwargs['quantize_lib'], self._codec.bd_video.matrix)
    
                for j in range(i, k):
                    coords = np.zeros((2,), np.int32)
                    a_img = Image.new('RGBA', dims, (0, 0, 0, 0))
                    for oix, pgo in compositions:
                        multiplier = np.uint8(flags[j] >= 0)
                        if len(pgo.mask[j-pgo.f:j+1-pgo.f]) == 1:
                            paste_box = (coords[0], coords[1], coords[0]+pgo.box.dx, coords[1]+pgo.box.dy)
                            crop_coords = (pgo.box.x, pgo.box.y, pgo.box.x2, pgo.box.y2)
                            last_imgs[oix] = (self.mask_event(pgo.wid, j), paste_box, crop_coords)
                        else:
                            multiplier &= pgo.is_visible_extended(j)
                        a_img.paste(Image.fromarray(multiplier*last_imgs[oix][0], 'RGBA').crop(last_imgs[oix][2]), last_imgs[oix][1])
                        coords += offset
                    img_seq.add_to_stack(a_img, 255-trial)
                ####
                #We have the "packed" object, the entire palette is usable
                bitmap, palettes = img_seq.flatten(255-trial).remap(1)
                if bitmap is not None:
                    break
            assert bitmap is not None
            pals.append(palettes)

            coords = np.zeros((2,), np.int32)
            for oix, pgo in pgobs_items:
                if __class__._object_is_relevant(pgo, flags, slice(i, k)):
                    #get bitmap
                    window_bitmap = 0xFF*np.ones((self.ectx.windows[pgo.wid].dy, self.ectx.windows[pgo.wid].dx), np.uint8)
                    nx, ny = coords
                    window_bitmap[pgo.box.slice] = bitmap[(slice(ny, ny+pgo.box.dy), slice(nx, nx+pgo.box.dx))]
                    #Generate object related segments objects
                    oxl = max(0, node.pos[oix].x2 - node.slots[oix][1])
                    oyl = max(0, node.pos[oix].y2 - node.slots[oix][0])
                    cpx = self.ectx.windows[pgo.wid].x + oxl
                    cpy = self.ectx.windows[pgo.wid].y + oyl

                    window_bitmap = window_bitmap[oyl:oyl+node.slots[oix][0], oxl:oxl+node.slots[oix][1]]
                    new_ods = self._codec.register_object(c_pts, node.dts(), window_bitmap)

                    cobjs.append(CompositionObject(new_ods[0].object_id, pgo.wid, cpx, cpy, False))
                    assert window_bitmap.shape == node.slots[pgo.wid]
                    coords += offset
                    o_ods += new_ods
            pals.append([Palette()] * len(pals[0]))
            ####for wid, pgo
        else:
            # If in the chain there's a NORMAL_CASE redefinition, we
            # must work with separate palette for each object (127+1 colors per window by default)
            n_colors = 255
            bias = 0
            if has_two_objs:
                assert normal_case_refresh
                assert not any(filter(lambda x: x[0] < 0 or x[0] > self._codec.bd_video.fmt.height or x[1] < 0 or x[1] > self._codec.bd_video.fmt.width, node.slots)) and sum(map(lambda x: x is not None, node.slots)) == 2
                f_slot_area = lambda slot: int(slot[0])*int(slot[1])
                ratio_area = (f_slot_area(node.slots[0]) - f_slot_area(node.slots[1]))/sum(map(f_slot_area, node.slots))
                bias = 0 if abs(ratio_area) < 0.5 else int(67*(ratio_area-np.sign(ratio_area)*0.25))
                n_colors = 128
                assert n_colors > abs(bias) + 10
                logger.debug(f"Split colour distribution: r={ratio_area:.03f}, b={bias} -> w0={n_colors+bias}, w1={n_colors-bias}")

            id_skipped = None
            for oix, pgo in pgobs_items:
                if not __class__._object_is_relevant(pgo, flags, slice(i, k)):
                    if normal_case_refresh:
                        #An object may exist but be masked for the whole acquisition: pad palette.
                        pals.append([Palette()] * (k-i))                
                elif isinstance(normal_case_refresh, list) and not normal_case_refresh[oix]:
                    assert 1 == sum(normal_case_refresh) and id_skipped is None and prev_cobjs is not None
                    composition = next(filter(lambda c: c.window_id == pgo.wid, prev_cobjs))
                    cobjs.append(composition)
                    pals.append([Palette()] * (k-i))
                    id_skipped = oix

            for oix, pgo in pgobs_items:
                if not __class__._object_is_relevant(pgo, flags, slice(i, k)) or oix == id_skipped:
                    continue

                oxl = max(0, node.pos[oix].x2 - node.slots[oix][1])
                oyl = max(0, node.pos[oix].y2 - node.slots[oix][0])
                cpx = self.ectx.windows[pgo.wid].x + oxl
                cpy = self.ectx.windows[pgo.wid].y + oyl

                assert len(flags[i:k]) >= len(pgo.mask[i-pgo.f:k-pgo.f])

                n_colors_qtz = n_colors + (-1 if oix == 1 else 1)*bias
                clut_offset = 1 + (n_colors - 1 + bias)*(oix == 1 and has_two_objs)
                
                wd_bitmap, wd_pal = None, None
                for trial in range(2):
                    last_img = None
                    img_seq = ImageSequence(k-i, self.kwargs['quantize_lib'], self._codec.bd_video.matrix)
                    for j in range(i, k):
                        multiplier = np.uint8(flags[j] >= 0)
                        if pgo.is_active(j):
                            last_img = self.mask_event(pgo.wid, j)
                        else:
                            multiplier &= pgo.is_visible_extended(j)
                        crop_coords = (pgo.box.x, pgo.box.y, pgo.box.x2, pgo.box.y2)
                        img_seq.add_to_stack(Image.fromarray(multiplier*last_img, 'RGBA').crop(crop_coords), n_colors_qtz-trial)

                
                    wd_bitmap, wd_pal = img_seq.flatten(n_colors_qtz-trial).remap(clut_offset)
                    if wd_bitmap is not None:
                        break
                assert wd_bitmap is not None
                window_bitmap = 0xFF*np.ones((self.ectx.windows[pgo.wid].dy, self.ectx.windows[pgo.wid].dx), np.uint8)
                window_bitmap[pgo.box.slice] = wd_bitmap
                wd_bitmap = window_bitmap[oyl:oyl+node.slots[oix][0], oxl:oxl+node.slots[oix][1]]
                pals.append(wd_pal)

                #On normal case, we generate one chain of palette update and
                #add in a screen wipe if necessary. This is not used if the object is changed.
                if normal_case_refresh and len(pals[-1]) < k-i:
                    mibm, mabm = min(wd_pal[0].palette), max(wd_pal[0].palette)
                    pals[-1].append(Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(mibm, mabm+1)}))
                    pals[-1].extend([Palette()] * ((k-i)-len(pals[-1])))

                new_ods = self._codec.register_object(c_pts, node.dts(), wd_bitmap)
                cobjs.append(CompositionObject(new_ods[0].object_id, pgo.wid, cpx, cpy, False))
                o_ods += new_ods

            if id_skipped is not None:
                assert isinstance(normal_case_refresh, list)
                #The END segment tells the decoder to use whatever it has in the buffer
                # for the composition without an ODS in the current display set.
                #The refreshed object has to come first in the composition list.
                cobjs = cobjs[::-1]

        pal = pals[0][0]
        if has_two_objs:
            pal |= pals[1][0]
        else:
            pals.append([Palette()] * len(pals[0]))
        return cobjs, pals, o_ods, pal

    def _get_undisplay(self, c_pts: int, palette_id: int, node: DSNode) -> DisplaySet:
        dts = node.dts()
        self._codec.update_palette_reservation(palette_id, c_pts, dts)
        pcs = self._codec.register_composition(c_pts, node.dts(), PCS.CompositionState.NORMAL_CASE, palette_id, False, [])
        wds = self._codec.get_window_definition_segment(c_pts, c_pts)
        uds = DisplaySet([pcs, wds, END(pts=c_pts, dts=c_pts)])
        DSNode.apply_pts_dts(uds, DSNode.set_pts_dts_sc(uds, self._codec.buffer, wds, node))
        return uds

    def _get_undisplay_pds(self, c_pts: int, node: DSNode, cobjs: list[CompositionObject], n_colors: int) -> DisplaySet:
        palette = Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(n_colors)})
        # todo: fixme
        pds = self._codec.register_palette(c_pts, node.dts(), palette, write_full_palette=True)
        pcs = self._codec.register_composition(c_pts, node.dts(), PCS.CompositionState.NORMAL_CASE,
                                               pds.palette_id, True, cobjs)
        uds = DisplaySet([pcs, pds, END(pts=c_pts, dts=c_pts)])
        DSNode.apply_pts_dts(uds, DSNode.set_pts_dts_sc(uds, self._codec.buffer, self._codec.get_window_definition_segment(0, 0), node))
        
        for cobj in cobjs:
            self._codec.update_object_reservation(cobj.object_id, c_pts)
        
        return uds

    def _convert(self, states, pgobjs, durs, flags, nodes):
        n_actions = len(durs)
        insert_acqs = self.kwargs.get('insert_acquisitions', 0)
        displaysets = []
        use_full_pal = self.kwargs.get('full_palette', False)

        ## Internal helper function
        def get_obj(frame, pgobjs: dict[int, list[ProspectiveObject]]) -> dict[int, ProspectiveObject | None]:
            objs = {}
            for ix, pgobjl in enumerate(pgobjs):
                objs[ix] = None
                #objs[ix] = next(filter(lambda obj: obj.is_active(frame), pgobjl), None)
                for obj in pgobjl:
                    if obj.is_active(frame):
                        assert objs[ix] is None
                        objs[ix] = obj
            return objs
        ####
        
        i = states.index(PCS.CompositionState.EPOCH_START)
        c_pts = 0
        last_cobjs = []
        last_palette_id = -1

        final_node = DSNode([], self.ectx.windows, self.ectx.events[-1].outTC, is_palette_update=True)
        #Do we have time to redraw the window (with some margin)?
        perform_wds_end = durs[-1][0]*GraphicsDecoder.FREQ / self._codec.bd_video.fps >= np.ceil((final_node.write_duration() + 10))

        #Generate datastream according to all assets
        while i < n_actions:
            if durs[i][1] != 0:
                assert i > 0
                assert nodes[i].parent is not None
                w_pts = self.ectx.events[i-1].outTC.to_pts()
                wds_doable = (nodes[i].parent.write_duration() + 3) < 1/self._codec.bd_video.fps
                if wds_doable and not nodes[i].parent.is_custom_dts():
                    uds = self._get_undisplay(w_pts, last_palette_id, nodes[i].parent)
                    logger.debug(f"Writing screen clear with WDS at PTS={self.ectx.events[i-1].outTC} before an acquisition.")
                else:
                    uds = self._get_undisplay_pds(w_pts, nodes[i].parent, last_cobjs, 255)
                    last_palette_id = uds.pcs.palette_id
                    logger.debug(f"Writing screen clear with palette update before an acquisition at PTS={self.ectx.events[i-1].outTC}")
                displaysets.append(uds)

            if flags[i] == -1:
                logger.debug(f"Skipping discarded event at PTS={self.events[i].inTC}")
                i+=1
                continue

            assert states[i] != PCS.CompositionState.NORMAL_CASE
            normal_case_refresh = False
            for k in range(i+1, n_actions+1):
                if k < n_actions:
                    normal_case_refresh |= (flags[k] == 1)
                if k == n_actions or states[k] != PCS.CompositionState.NORMAL_CASE:
                    break
            assert k > i

            c_pts = nodes[i].tc_pts.to_pts()
            pgobs_items = get_obj(i, pgobjs).items()
            has_two_objs = 0
            for _, pgo in pgobs_items:
                if __class__._object_is_relevant(pgo, flags, slice(i, k)):
                    has_two_objs += 1

            #Normal case refresh implies we are refreshing one object out of two displayed.
            has_two_objs = has_two_objs > 1 or normal_case_refresh
            r = self._encode_composition_objects(i, k, pgobs_items, nodes[i],
                                                 has_two_objs, c_pts, normal_case_refresh, flags)
            cobjs, pals, o_ods, pal = r

            pds = self._codec.register_palette(c_pts, nodes[i].dts(), pal, write_full_palette=True)
            last_palette_id = pds.palette_id

            pcs = self._codec.register_composition(c_pts, nodes[i].dts(), states[i], pds.palette_id, False, cobjs)
            wds = self._codec.get_window_definition_segment(c_pts, nodes[i].dts())

            nds = DisplaySet([pcs, wds, pds] + o_ods + [END(dts=c_pts, pts=c_pts)])
            DSNode.apply_pts_dts(nds, DSNode.set_pts_dts_sc(nds, self._codec.buffer, wds, nodes[i]))
            displaysets.append(nds)

            logger.debug(f"Acquisition: PTS={nodes[i].tc_pts}={c_pts}, 2OBJs={has_two_objs}, NC={normal_case_refresh} Npalups={len(pals[0])-1} S(ODS)={sum(map(lambda x: len(bytes(x)), o_ods))}, L(ODS)={len(o_ods)}, f: {i}->{k}")

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
                    c_pts = self.ectx.events[z].inTC.to_pts()
                    assert states[z] == PCS.CompositionState.NORMAL_CASE
                    pal |= pals[0][z-i] | pals[1][z-i]

                    #Is there a know screen clear in the chain? then use palette screen clear here
                    if durs[z][1] != 0:
                        assert nodes[z].parent is not None
                        logger.debug(f"Writing screen wipe in palette update chain at PTS={self.ectx.events[z-1].outTC}={c_pts:.03f}")
                        uds = self._get_undisplay_pds(self.ectx.events[z-1].outTC.to_pts(), nodes[z].parent, cobjs, max(pal.palette)+1)
                        displaysets.append(uds)
                        #We just wipped a palette, whatever the next palette id, rewrite it fully
                        last_palette_id = None
                        #Should not be necessary but in any case...
                        durs[z] = (durs[z][0], 0)

                    if flags[z] == 1:
                        normal_case_refresh = nodes[z].new_mask
                        r = self._encode_composition_objects(z, k, get_obj(z, pgobjs).items(), nodes[z],
                                                             has_two_objs, c_pts, normal_case_refresh, flags, cobjs)
                        cobjs, n_pals, o_ods, new_pal = r

                        logger.debug(f"Normal Case: PTS={self.ectx.events[z].inTC}={c_pts:.03f}, NM={nodes[z].new_mask} S(ODS)={sum(map(lambda x: len(bytes(x)), o_ods))}")

                        pal |= new_pal
                        idxnc = nodes[z].new_mask.index(True)
                        for nz, new_p in enumerate(n_pals[idxnc], z):
                            pals[idxnc][nz-i] = new_p
                        normal_case_refresh = True
                        last_palette_id = None
                    elif flags[z] == -1:
                        logger.debug(f"Skipped discarded event at PTS={self.ectx.events[z].inTC}={c_pts:.03f}.")
                        continue

                    p_write = (pals[0][z-i] | pals[1][z-i])
                    #Skip empty palette updates
                    if len(p_write) == 0 and last_palette_id is not None:
                        logger.debug(f"Skipped an empty palette at PTS={self.ectx.events[z].inTC}={c_pts:.03f}.")
                        continue

                    pds = self._codec.register_palette(c_pts, nodes[z].dts(), p_write, write_full_palette=True)
                    last_palette_id = pds.palette_id

                    pcs = self._codec.register_composition(c_pts, nodes[z].dts(), states[z], pds.palette_id, flags[z] != 1, cobjs)
                    wds_upd = [self._codec.get_window_definition_segment(c_pts, nodes[z].dts())] if flags[z] == 1 else []
                    ods_upd = o_ods if flags[z] == 1 else []

                    if flags[z] != 1:
                        for cobj in cobjs:
                            self._codec.update_object_reservation(cobj.object_id, c_pts)

                    nds = DisplaySet([pcs] + wds_upd + [pds] + ods_upd +[END(dts=c_pts, pts=c_pts)])
                    DSNode.apply_pts_dts(nds, DSNode.set_pts_dts_sc(nds, self._codec.buffer, self._codec.get_window_definition_segment(0, 0), nodes[z]))
                    displaysets.append(nds)

                    if z+1 == k:
                        break
                assert z+1 == k

            if insert_acqs > 0 and len(pals[0]) > insert_acqs and flags[k-1] != -1:
                t_diff = (self.ectx.events[k-1].outTC - self.ectx.events[k-1].inTC).to_realtime(as_float=True)
                #Worst decoding time is twice the write duration. The next display set should also have as much margin.
                if t_diff > 4.5*nodes[k-1].write_duration():
                    dts_end = nodes[k-1].dts_end() + 2
                    npts = nodes[k-1].pts() + 2
                    nodes[k-1].is_palette_update = nodes[k-1].partial = False
                    frame_added = 0
                    while nodes[k-1].dts() < dts_end or nodes[k-1].pts() < npts + nodes[k-1].write_duration():
                        nodes[k-1].tc_pts = nodes[k-1].tc_pts + 1
                        frame_added += 1
                    # Subtract one frame to durs to ensure we have enough time for the next real acquisition.
                    # 22500 = 0.25 * 90 kHz
                    if nodes[k-1].dts() - dts_end < 22500 and frame_added <= (durs[k-1][0]-1) >> 1:
                        pgobs_items = get_obj(k-1, pgobjs).items()
                        has_two_objs = 0
                        for _, pgo in pgobs_items:
                            if __class__._object_is_relevant(pgo, flags, slice(k-1, k)):
                                has_two_objs += 1

                        c_pts = nodes[k-1].pts()
                        logger.debug(f"INS Acquisition: PTS={nodes[k-1].tc_pts}={c_pts:.03f} from event at {self.ectx.events[k-1].inTC}.")

                        r = self._encode_composition_objects(k-1, k, pgobs_items, nodes[k-1],
                                                             has_two_objs > 1, c_pts, False, flags)
                        cobjs, _, o_ods, pal = r
                        wds = self._codec.get_window_definition_segment(0, 0)
                        pds = self._codec.register_palette(c_pts, nodes[k-1].dts(), pal, write_full_palette=True)
                        last_palette_id = pds.palette_id

                        pcs = self._codec.register_composition(c_pts, nodes[k-1].dts(), PCS.CompositionState.ACQUISITION, pds.palette_id, False, cobjs)
                        nds = DisplaySet([pcs, wds, pds] + o_ods + [END(pts=c_pts, dts=c_pts)])
                        DSNode.apply_pts_dts(nds, DSNode.set_pts_dts_sc(nds, self._codec.buffer, wds, nodes[k-1]))
                        displaysets.append(nds)
                    ####if nodes[k-1
                ####if t_diff >
            i = k
            last_cobjs = cobjs
        ####while

        #We can't undraw the screen due to delta PTS constraint, we clear it with a palette update and will undraw optionally at +N frames
        if not perform_wds_end:
            logger.debug(f"Performing palette wipe (delta PTS too short) at {self.ectx.events[-1].outTC} (end of epoch).")
            uds = self._get_undisplay_pds(self.ectx.events[-1].outTC.to_pts(), final_node, last_cobjs, 255)
            displaysets.append(uds)
            
            #Prepare an additional display set to undraw the screen if it can fit (< self.ectx.max_pts)
            nf_shift = max(1, int(np.ceil(((final_node.write_duration()+10)*self._codec.bd_video.fps))))
            tc_final_pts = self.ectx.events[-1].outTC + nf_shift
            final_pts = tc_final_pts.to_pts()
            perform_wds_end = final_pts < self.ectx.max_pts
        else:
            tc_final_pts = self.ectx.events[-1].outTC
            final_pts = self.ectx.events[-1].outTC.to_pts()

        if perform_wds_end:
            final_ds = self._get_undisplay(final_pts, last_palette_id, final_node)
            logger.debug(f"Performing standard screen wipe at {tc_final_pts} (end of epoch).")
            displaysets.append(final_ds)
        return Epoch(displaysets)
    ####

    def find_acqs(self, durs: list[int], nodes: list['DSNode']):
        dtl = np.zeros((len(durs)), dtype=float)
        valid = np.zeros((len(durs),), dtype=np.bool_)
        absolutes = np.zeros_like(valid)

        chain_boxes = []
        min_boxes = 8*np.ones((len(self.ectx.windows), 2), np.int32)

        running_bbox = [None, None]
        for k, node in enumerate(nodes):
            boxes = [None] * len(self.ectx.windows)
            #NC (screen wipes at this stage of the encoding process) don't need to know
            if node.is_palette_update is False:
                for wid in filter(lambda oix: node.objects[oix] is not None, range(len(self.ectx.windows))):
                    if node.objects[wid].is_visible(node.idx):
                        ob = node.objects[wid].get_bbox_at(node.idx)
                        min_boxes[wid] = np.max((min_boxes[wid], (ob.dy, ob.dx)), axis=0)
                        running_bbox[wid] = ob
                    elif node.objects[wid].is_active(node.idx): # we never fall here on the first frame an object is active (is_visible is true)
                        assert (k > 0) and (None != running_bbox[wid])
                        ob = running_bbox[wid]
                    else:
                        raise RuntimeError("Critical encoding error, getting bbox of object that is neither visible or active.")
                    boxes[wid] = ob
            ####!is_palette_update
            chain_boxes.append(boxes)
            absolutes[k] = any(node.new_mask)

        write_duration = nodes[0].write_duration()
        min_boxes = list(map(tuple, min_boxes))
        prev_dt = 0
        for k, (dt, node) in enumerate(zip(durs, nodes)):
            if not node.is_palette_update:
                node.slots = min_boxes
            if k == 0:
                prev_pts = prev_dts = -np.inf
            else:
                margin = prev_dt/self._codec.bd_video.fps
                prev_dts = nodes[k-1].dts_end()
                prev_pts = nodes[k-1].pts()
            valid[k] = (node.dts() > prev_dts) and (node.pts() - prev_pts > write_duration)
            dtl[k] = (node.dts() - prev_dts)/margin if (valid[k] and k > 0) else (-1 + 2*int(k==0))
            prev_dt = dt
        return valid, absolutes, dtl, min_boxes, chain_boxes
    ####

    def create_displaysets_nodes(self, pgobjs_proc: dict[int, list[ProspectiveObject]]) -> tuple[list['DSNode'], list[int]]:
        objs = [None for objs in pgobjs_proc]
        top = self.ectx.events[0].inTC.frames
        delays, nodes = [], []

        for ne, event in enumerate(self.ectx.events):
            tic = event.inTC.frames
            toc = event.outTC.frames
            # gap between two events in an epoch: add a screen wipe
            if (clear_duration := tic-top) > 0:
                delays += [clear_duration]
                nodes.append(DSNode([], self.ectx.windows, self.ectx.events[ne-1].outTC, is_palette_update=True))
                nodes[-1].idx = -1 # screen wipes do not refer to valid events in the array
            ####
            delays += [toc-tic]

            is_new = [False] * len(self.ectx.windows)
            for wid, _ in enumerate(self.ectx.windows):
                is_new[wid] = False
                if objs[wid] is not None and not objs[wid].is_active(ne):
                    objs[wid] = None
                if len(pgobjs_proc[wid]):
                    if not objs[wid] and pgobjs_proc[wid][0].is_active(ne):
                        objs[wid] = pgobjs_proc[wid].pop(0)
                        objs[wid].wid = wid
                        is_new[wid] = True
                    else:
                        assert not pgobjs_proc[wid][0].is_active(ne)

            nodes.append(DSNode(objs.copy(), self.ectx.windows, event.inTC, new_mask=is_new))
            nodes[-1].idx = ne
            top = toc
        return delays, nodes

    def get_events_nodes(self) -> tuple[list[int], list['DSNode']]:
        """
        Returns the duration of each event in frames.
        Additionally, the offset from the previous event is also returned. This value
        is zero unless there are no PG objects shown at some point in the epoch.
        """
        top = self.ectx.events[0].inTC.frames
        delays = []
        nodes = []
        for ne, event in enumerate(self.ectx.events):
            tic = event.inTC.frames
            toc = event.outTC.frames
            clear_duration = tic-top
            if clear_duration > 0:
                delays += [clear_duration]
                nodes.append(DSNode([], self.ectx.windows, self.ectx.events[ne-1].outTC, is_palette_update=True))
                nodes[-1].idx = -1
            delays += [toc-tic]
            nodes.append(DSNode([], self.ectx.windows, event.inTC))
            nodes[-1].idx = ne
            top = toc
        return delays, nodes
    ####

    def roll_nodes(self, nodes, durs, flags, states) -> tuple[list, list, list, list]:
        k = 0
        r_nodes = []
        r_durs = []
        r_states = []
        r_flags = []
        for ne, event in enumerate(self.ectx.events):
            parent = nodes[k] if nodes[k].objects == [] else None
            valid_parent = parent is not None and flags[k] == 0
            k += parent is not None
            nodes[k].parent = parent

            assert parent is None or self.ectx.events[ne-1].outTC == nodes[k].parent.tc_pts

            r_durs.append((durs[k], 0 if not valid_parent else durs[k-1]))
            r_nodes.append(nodes[k])
            r_flags.append(flags[k])
            r_states.append(states[k])
            k += 1
        assert k == len(nodes)
        return r_states, r_durs, r_nodes, r_flags
    ####
####
    