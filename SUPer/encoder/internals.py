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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from PIL import Image

from ..bytestream.pgstreams import DisplaySet
from ..bytestream.segments import PCS, CompositionObject
from ..display.palette import Palette, PaletteEntry
from ..display.bdvideo import BDVideo
from ..geometry import Box, Shape
from ..internals import TC, GraphicsDecoder, LogFacility, _classproperty
from .codecctx import PGObjectBuffer
from .epochctx import EpochEvent
from .eventdetect import ProspectiveObject
from .imgproc import PaletteSequenceEffect


logger = LogFacility.get_logger('SUPer')

class DSNode(ABC):
    def __init__(self,
            objects: list[ProspectiveObject | None],
            tc_pts: TC,
            is_palette_update: bool = False,
            new_mask: list[bool] | None = None
        ) -> None:
        if new_mask is None:
            new_mask = []
        else:
            assert len(objects) == len(new_mask)
        self.objects = objects
        self.tc_pts = tc_pts
        self.is_palette_update = is_palette_update
        self.new_mask = new_mask

        self.slots = [None] * len(self.objects)
        self.pos = [None] * len(self.objects)
        self.partial = False
        #index in the event list space
        self.idx = 0
        self._state = PCS.CompositionState.NORMAL_CASE

        # These fields are not copied over
        self.parent = None
        self.discard = False
        self._dts = None
        self._planned_compositions = None

    @_classproperty
    @abstractmethod
    def windows(cls) -> Sequence[Box]:
        pass

    def copy(self) -> Self:
        new_node = self.__class__(self.objects.copy(), self.tc_pts,
                                  is_palette_update=self.is_palette_update, new_mask=self.new_mask.copy())
        new_node.slots = self.slots.copy()
        new_node.pos = self.pos.copy()
        new_node.partial = self.partial
        new_node.idx = self.idx
        new_node._state = self._state
        return new_node

    @property
    def flag(self) -> int:
        if self.discard:
            return -1
        if self.partial:
            return 1
        return 0

    @property
    def planned_compositions(self) -> tuple[bool]:
        if self._planned_compositions is not None:
            return self._planned_compositions
        return tuple(x is not None for x in self.objects)

    @planned_compositions.setter
    def planned_compositions(self, planned: tuple[bool]) -> None:
        self._planned_compositions = planned

    @property
    def state(self) -> PCS.CompositionState:
        return self._state

    @state.setter
    def state(self, state: PCS.CompositionState) -> None:
        state = PCS.CompositionState(state)
        assert state == PCS.CompositionState.NORMAL_CASE or self.is_palette_update is False
        self._state = state

    def wipe_duration(self) -> int:
        """
        Duration to erase the area of both windows (one operation for the decoder)
        """
        return int(np.ceil(sum(GraphicsDecoder.FREQ*w.area/GraphicsDecoder.RC for w in self.windows)))

    def write_duration(self) -> int:
        """
        Duration to write both window, one after the other
        """
        return int(sum(np.ceil(GraphicsDecoder.FREQ*w.area/GraphicsDecoder.RC) for w in self.windows))

    def set_dts(self, dts: int | None) -> None:
        if dts is not None:
            # we can only shift dts further away from its minimum, except for a palette update
            assert dts <= self.dts() or (self.is_palette_update and dts < self.pts())
            assert isinstance(dts, int)
        self._dts = dts

    def dts_end(self) -> int:
        if self._dts is not None:
            return (sum(self.get_dts_markers()[1]) + self._dts)
        decode_ts, t_dec = self.get_dts_markers()
        return (decode_ts + sum(t_dec))

    def dts(self) -> int:
        if self._dts is not None:
            return self._dts
        return self.get_dts_markers()[0]

    def delta_dts(self) -> int:
        return sum(self.get_dts_markers()[1])

    def pts(self) -> int:
        return self.tc_pts.to_pts()

    def is_custom_dts(self) -> bool:
        return self._dts is not None

    def get_decode_duration(self) -> tuple[int, int]:
        t_decoding = []

        if not self.is_palette_update:
            assert any(self.objects)
            target_windows = [o.wid for o in filter(lambda o: o is not None, self.objects)]
            assigned_wids = set(target_windows)
            if self.state == PCS.CompositionState.EPOCH_START:
                decode_duration = int(np.ceil(self.plane.area * GraphicsDecoder.FREQ / GraphicsDecoder.RC))
            else:
                decode_duration = int(sum([np.ceil(self.windows[wid].dy*self.windows[wid].dx*GraphicsDecoder.FREQ/GraphicsDecoder.RC) for wid in range(len(self.windows)) if wid not in assigned_wids]))

            #writing twice to the same window?
            delay_write = 2 == len(target_windows) and 1 == len(assigned_wids)

            t_other_copy = 0
            for oix, obj in filter(lambda o: o[1] is not None, enumerate(self.objects)):
                write = self.windows[obj.wid].area*GraphicsDecoder.FREQ
                if self.slots[oix] is not None:
                    read = self.slots[oix].area*GraphicsDecoder.FREQ
                else:
                    logger.error(f"Node at {self.tc_pts} lacks an object slot, assuming windows-size.")
                    read = write
                if not self.partial or (self.partial and self.new_mask[oix]):
                    t_decoding.append(int(np.ceil(read/GraphicsDecoder.RD)))
                elif self.partial and not self.new_mask[oix]:
                    #the other object is copied at the end.
                    assert sum(self.new_mask) == 1 and t_other_copy == 0
                    t_other_copy = int(np.ceil(write/GraphicsDecoder.RC))
                    continue

                decode_duration = max(decode_duration, sum(t_decoding)) + int(np.ceil(write/GraphicsDecoder.RC))*int(not delay_write)
            ####
            if delay_write:
                write_duration = int(np.ceil(write/GraphicsDecoder.RC))
                assert t_other_copy == 0 or write_duration == t_other_copy
                decode_duration += write_duration
            else:
                assert t_other_copy == 0 or self.partial
                decode_duration += t_other_copy
            decode_duration = max(decode_duration, self.write_duration() + 1)
        else:
            decode_duration = self.write_duration() + 1
        return (decode_duration, t_decoding)

    def get_dts_markers(self) -> tuple[int, int]:
        decode_duration, t_decoding = self.get_decode_duration()
        return (self.pts() - decode_duration, t_decoding)
    ####

    def set_pts_dts_sc(self, ds: DisplaySet, buffer: PGObjectBuffer) -> list[tuple[int, int]]:
        """
        This function generates the timestamps (PTS and DTS) associated to a given DisplaySet.

        :param ds: DisplaySet, PTS of PCS must be set to the right value.
        :param buffer: Object buffer that supports allocation and returning a size of allocated slots.
        :return: Pairs of timestamps in ticks for each segment in the displayset.
        """
        ddurs = {}
        for ods in ds.ods:
            if ods.flag & ods.DataFlag.FIRST:
                object_shape = Shape(h=ods.height, w=ods.width)
                assert ods.object_id not in ddurs, f"Object {ods.object_id} defined twice in DS."
                if (slot := buffer.get_indexed(ods.object_id)) is not None:
                    assert object_shape == slot.shape, "Dimension mismatch, buffer corruption."
                else:
                    # Allocate a buffer slot for this object
                    assert buffer.allocate_indexed(object_shape, ods.object_id) is True, "Critical error: object buffer overflow."
                    slot = buffer.get_indexed(ods.object_id)
                # make sure the encoder has reserved this object ID up to PTS.
                assert not slot.is_available_at(ds.pcs.pts)

                ddurs[ods.object_id] = int(np.ceil(ods.height*ods.width*GraphicsDecoder.FREQ/GraphicsDecoder.RD))
        t_decoding = 0
        decode_duration = 0
        wipe_duration = self.wipe_duration()

        if ds.pcs.composition_state == ds.pcs.CompositionState.EPOCH_START:
            decode_duration = int(np.ceil(ds.pcs.width*ds.pcs.height*GraphicsDecoder.FREQ/GraphicsDecoder.RC))
        else:
            assigned_windows = [x.window_id for x in ds.pcs.composition_objects]
            unassigned_windows = [wd for wid, wd in enumerate(self.windows) if wid not in assigned_windows]
            decode_duration = int(sum(np.ceil(wd.area*GraphicsDecoder.FREQ/GraphicsDecoder.RC) for wd in unassigned_windows))

        object_decode_duration = ddurs.copy()

        if not ds.pcs.palette_update:
            #For every composition object, compute the transfer time
            for k, cobj in enumerate(ds.pcs.composition_objects):
                assert buffer.get_indexed(cobj.object_id) is not None, "Object does not exist in buffer."
                t_dec_obj = object_decode_duration.pop(cobj.object_id, 0)
                t_decoding += t_dec_obj

                # Same window -> patent claims a window is written only once after the two cobj are processed.
                if k == 0 and ds.pcs.num_composition_objects > 1 and ds.pcs.composition_objects[1].window_id == cobj.window_id:
                    continue
                copy_dur = np.ceil(self.windows[cobj.window_id].area*GraphicsDecoder.FREQ/GraphicsDecoder.RC)
                decode_duration = max(decode_duration, t_decoding) + int(copy_dur)

        #Prevent PTS(WDS) = DTS(WDS) (a WDS shall have a DTS and it shall differ from the PTS)
        decode_duration = max(decode_duration, self.write_duration() + 1)

        dts = int(ds.pcs.pts - decode_duration)
        assert self.pts() == ds.pcs.pts, f"{self.pts()}, {ds.pcs.pts}, {self.idx}, {self.tc_pts}"

        if self.is_custom_dts():
            new_dts = self.dts()
            assert new_dts <= dts or ds.pcs.palette_update, f"new={new_dts}, min={dts}, {self.tc_pts}"
            dts = new_dts
        elif ds.pcs.composition_state != PCS.CompositionState.EPOCH_START:
            assert self.dts() == dts, f"node {self.tc_pts}: expected: {self.dts()}, actual={dts}, n_objs: {len(ds.pcs.composition_objects)}, PU={ds.pcs.palette_update}, nno={sum(o is not None for o in self.objects)}."

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

    @classmethod
    def apply_pts_dts(cls, ds: DisplaySet, ts: tuple[int, int]) -> None:
        assert len(ds) == len(ts), "Timestamps-DS size mismatch."
        for seg, (pts, dts) in zip(ds, ts):
            seg.pts, seg.dts = pts, dts
####

@dataclass(frozen=True, slots=True)
class PGAsset:
    i: int #start ix
    k: int #end ix
    composition_objects: CompositionObject
    palettes: list[list[Palette]]
    bitmaps: list[np.ndarray[[int, int], np.uint8]]
    cobjs_refs: list[tuple[ProspectiveObject, CompositionObject]]

@dataclass(frozen=True, slots=True)
class DisplayChain:
    start: int
    stop: int
    pgobjs: tuple[tuple[int, ProspectiveObject]]
    has_two_objs: bool
    normal_case_refresh: bool


def object_is_displayed_within(pgo: ProspectiveObject, nodes: list[DSNode], sl: slice) -> bool:
    if pgo is None:
         return False
    assert len(nodes) >= len(pgo.mask)
    return next((True for node in nodes[sl] if (node.flag >= 0 and pgo.is_visible(node.idx))), False)

class PGAssetEncoder:
    def __init__(self, nodes: list[DSNode], windows: tuple[Box], events: list[EpochEvent], bd_video: BDVideo, kwargs: dict[str, Any]) -> None:
        self.nodes = nodes
        self.windows = windows
        self.events = events
        self.bd_video = bd_video
        self.kwargs = kwargs
    
    def mask_event(self, wid: int, event_id: int) -> np.ndarray[tuple[int, int, int], np.uint8]:
        event = self.events[event_id]
        wd = self.windows[wid]
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

    @staticmethod
    def _get_stack_direction(*box) -> tuple[np.ndarray[tuple[int, int], np.uint8], tuple[int, int]]:
        widths = [b.dx for b in box]
        heights = [b.dy for b in box]

        if max(heights)*sum(widths) <= max(widths)*sum(heights):
            return np.array([widths[0], 0], np.int32), (sum(widths), max(heights))
        return np.array([0, heights[0]], np.int32), (max(widths), sum(heights))

    def encode_composition_objects(self,
        i: int, k: int,
        pgobs_items: list[tuple[int, ProspectiveObject]],
        has_two_objs: bool,
        normal_case_refresh: bool | list[bool],
        prev_cobjs_refs: list[tuple[ProspectiveObject, CompositionObject]] | None = None,
    ) -> ...:
        """
        Encodes the composition objects (plus its palette sequence) within interval
        i->k. The pgobs_items contains the objects to refer to (at most 2)
        
        has_two_objs: True whenever there's two composition objects or a normal case
        normal_case_refresh: True or a list of boolean.
            True: this encodes an acquisition which is to be followed up by a normal case object refresh
            list[bool]: pair of bool (only one is True) that gives the the object to normal case refresh.
        prev_cobjs_refs: the prior composition objects, only relevant to normal case path
        """
        cls = self.__class__
        bitmaps, cobjs, pals, cobjs_refs = [], [], [], []
        node = self.nodes[i]

        #In this mode, we re-combine the two objects in a smaller areas than in the original box
        # and then pass that to the optimiser. Colors are efficiently distributed on the objects.
        if has_two_objs and normal_case_refresh is False:
            compositions = [(oix, pgo) for oix, pgo in pgobs_items if object_is_displayed_within(pgo, self.nodes, slice(i, k))]
            assert len(compositions) == 2
            #todo: stack using slot dimensions?
            offset, dims = cls._get_stack_direction(*[x[1].box for x in compositions])
            last_imgs = [None] * len(compositions)

            imgs_chain = []
            for j in range(i, k):
                coords = np.zeros((2,), np.int32)
                a_img = Image.new('RGBA', dims, (0, 0, 0, 0))
                for oix, pgo in compositions:
                    multiplier = np.uint8(self.nodes[j].flag >= 0)
                    if pgo.is_active(self.nodes[j].idx):
                        paste_box = (coords[0], coords[1], coords[0]+pgo.box.dx, coords[1]+pgo.box.dy)
                        crop_coords = (pgo.box.x, pgo.box.y, pgo.box.x2, pgo.box.y2)
                        last_imgs[oix] = (self.mask_event(pgo.wid, self.nodes[j].idx), paste_box, crop_coords)
                    else:
                        multiplier &= pgo.is_visible_extended(self.nodes[j].idx)
                    a_img.paste(Image.fromarray(multiplier*last_imgs[oix][0], 'RGBA').crop(last_imgs[oix][2]), last_imgs[oix][1])
                    coords += offset
                imgs_chain.append(a_img)
            ####
            #We have the "packed" object, the entire palette is usable
            palette_effect = PaletteSequenceEffect(self.kwargs['quantize_lib'], self.bd_video.matrix)
            bitmap, palettes = palette_effect.solve_and_remap(imgs_chain, 255, 1, **self.kwargs)
            pals.append(palettes)

            coords = np.zeros((2,), np.int32)
            for oix, pgo in pgobs_items:
                if object_is_displayed_within(pgo, self.nodes, slice(i, k)):
                    #get bitmap
                    window_bitmap = 0xFF*np.ones((self.windows[pgo.wid].dy, self.windows[pgo.wid].dx), np.uint8)
                    nx, ny = coords
                    window_bitmap[pgo.box.slice] = bitmap[(slice(ny, ny+pgo.box.dy), slice(nx, nx+pgo.box.dx))]
                    #Generate object related segments objects
                    oxl = max(0, node.pos[oix].x2 - node.slots[oix].w)
                    oyl = max(0, node.pos[oix].y2 - node.slots[oix].h)
                    cpx = self.windows[pgo.wid].x + oxl
                    cpy = self.windows[pgo.wid].y + oyl

                    slot_bitmap = window_bitmap[oyl:oyl+node.slots[oix].h, oxl:oxl+node.slots[oix].w]

                    cobjs.append((pgo.wid, cpx, cpy, False))
                    cobjs_refs.append((pgo, cobjs[-1]))
                    assert slot_bitmap.shape == node.slots[oix], f"{bitmap.shape}, {node.slots[oix]}"
                    coords += offset
                    bitmaps.append(slot_bitmap)
            pals.append([Palette() for _ in range(len(pals[0]))])
            ####for oix, pgo
        else:
            # If in the chain there's a NORMAL_CASE redefinition, we
            # must work with separate palette for each object (127+1 colors per window by default)
            n_colors = 255
            bias = 0
            if has_two_objs and normal_case_refresh:
                assert all((x.h in range(self.bd_video.fmt.height) and x.w in range(self.bd_video.fmt.width)) for x in node.slots)
                assert sum(x is not None for x in node.slots) == 2
                ratio_area = (node.slots[0].area - node.slots[1].area)/sum(s.area for s in node.slots)
                bias = 0 if abs(ratio_area) < 0.5 else int(67*(ratio_area-np.sign(ratio_area)*0.25))
                n_colors = 128
                assert n_colors > abs(bias) + 10
                logger.debug(f"Split colour distribution: r={ratio_area:.03f}, b={bias} -> w0={n_colors+bias}, w1={n_colors-bias}")

            id_skipped = None
            for oix, pgo in pgobs_items:
                if not object_is_displayed_within(pgo, self.nodes, slice(i, k)):
                    continue
                if isinstance(normal_case_refresh, list) and not normal_case_refresh[oix]:
                    assert 1 == sum(normal_case_refresh) and id_skipped is None and prev_cobjs_refs is not None and pgo is not None
                    composition = next(filter(lambda x: x[0] == pgo, prev_cobjs_refs))[1]
                    cobjs.append(composition)
                    cobjs_refs.append((pgo, composition))
                    id_skipped = oix

            for oix, pgo in pgobs_items:
                if not object_is_displayed_within(pgo, self.nodes, slice(i, k)):
                    if normal_case_refresh:
                        #An object may be suggested but it is masked for the entire acquisition and
                        # subsequent normal cases: pad palette as it won't be in the composition list.
                        pals.append([Palette() for _ in range(k-i)])
                    continue
                if oix == id_skipped:
                    pals.append([Palette() for _ in range(k-i)])
                    continue
                oxl = max(0, node.pos[oix].x2 - node.slots[oix].w)
                oyl = max(0, node.pos[oix].y2 - node.slots[oix].h)
                cpx = self.windows[pgo.wid].x + oxl
                cpy = self.windows[pgo.wid].y + oyl

                assert self.nodes[i].idx >= pgo.f
                assert len(self.nodes[i:k]) >= len(pgo.mask[self.nodes[i].idx-pgo.f:self.nodes[k].idx-pgo.f])

                n_colors_qtz = n_colors + (-1 if oix == 1 else 1)*bias
                clut_offset = 1 + (n_colors - 1 + bias)*(oix == 1 and has_two_objs)

                last_img = None
                imgs_chain = []
                for j in range(i, k):
                    multiplier = np.uint8(self.nodes[j].flag >= 0)
                    if pgo.is_active(self.nodes[j].idx):
                        last_img = self.mask_event(pgo.wid, self.nodes[j].idx)
                    else:
                        multiplier &= pgo.is_visible_extended(self.nodes[j].idx)
                    crop_coords = (pgo.box.x, pgo.box.y, pgo.box.x2, pgo.box.y2)
                    imgs_chain.append(Image.fromarray(multiplier*last_img, 'RGBA').crop(crop_coords))

                palette_effect = PaletteSequenceEffect(self.kwargs['quantize_lib'], self.bd_video.matrix)
                wd_bitmap, wd_pal = palette_effect.solve_and_remap(imgs_chain, n_colors_qtz, clut_offset, **self.kwargs)

                window_bitmap = 0xFF*np.ones((self.windows[pgo.wid].dy, self.windows[pgo.wid].dx), np.uint8)
                window_bitmap[pgo.box.slice] = wd_bitmap

                bitmap = window_bitmap[oyl:oyl+node.slots[oix].h, oxl:oxl+node.slots[oix].w]
                pals.append(wd_pal)

                #On normal case, we generate one chain of palette update and
                #add in a screen wipe if necessary. This is not used if the object is changed.
                if normal_case_refresh and len(pals[-1]) < k-i:
                    mibm, mabm = min(wd_pal[0].palette), max(wd_pal[0].palette)
                    pals[-1].append(Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(mibm, mabm+1)}))
                    pals[-1].extend([Palette() for _ in range((k-i)-len(pals[-1]))])

                cobjs.append((pgo.wid, cpx, cpy, False))
                cobjs_refs.append((pgo, cobjs[-1]))
                bitmaps.append(bitmap)

            if id_skipped is not None:
                assert isinstance(normal_case_refresh, list)
                #The END segment tells the decoder to use whatever it has in the buffer
                # for the composition without an ODS in the current display set.
                #The refreshed object has to come first in the composition list.
                cobjs = cobjs[::-1]

        cumulated_palette = pals[0][0].copy()
        if has_two_objs:
            cumulated_palette |= pals[1][0]
        else:
            pals.append([Palette() for _ in range(len(pals[0]))])
        return cobjs, pals, bitmaps, cobjs_refs
####

class PGAssetEncoderTasks:
    @staticmethod
    def initialize_encoder(ctx: Any, param: tuple[...]) -> PGAssetEncoder:
        if ctx is not None:
            del ctx
        
        nodes, ectx, bd_video, kwargs = param
        return PGAssetEncoder(nodes, ectx.windows, ectx.events, bd_video, kwargs), True

    @staticmethod
    def encode_chain(ctx: PGAssetEncoder, dc: DisplayChain) -> PGAsset:
        r = ctx.encode_composition_objects(dc.start, dc.stop, dc.pgobjs, dc.has_two_objs, dc.normal_case_refresh)
        return ctx, PGAsset(dc.start, dc.stop, *r)
    
    @classmethod
    def get_tasks(cls):
        return (cls.initialize_encoder, cls.encode_chain)
####
