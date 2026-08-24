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

from dataclasses import dataclass, field
from typing import Sequence
from itertools import chain, repeat
from brule import Brule

from ..display.bdvideo import BDVideo
from ..display.palette import PaletteEntry, Palette
from ..geometry import Shape, Box
from ..bytestream import PCS, ODS, PDS, WDS, END, CompositionObject, DisplaySet

@dataclass
class _AllocatedVersionedResource:
    pts: int = -np.inf
    _version: int = -1

    def reserve(self, pts: int) -> int:
        assert self.pts < pts
        self.pts = pts
        self._version += 1
        return self.version

    @property
    def version(self) -> int:
        return self._version

    def get_cast_version(self) -> int:
        assert self._version >= 0
        return self._version & 0xFF

@dataclass
class DecoderPalette(_AllocatedVersionedResource):
    palette: dict[int, PaletteEntry] = field(default_factory=Palette)

    def __post_init__(self) -> None:
        if not isinstance(self.palette, Palette):
            self.palette = Palette(self.palette)
        assert len(self.palette) < 256

    def get_difference(self, new_palette: dict[int, PaletteEntry]):
        difference = {}

        for entry_ix, entry in new_palette.items():
            stored_entry = self.palette.get(entry_ix, None)
            if stored_entry is None or stored_entry != entry:
                difference[entry_ix] = entry
        return difference

    def clear(self) -> None:
        self.palette.clear()

@dataclass
class ObjectSlot(_AllocatedVersionedResource):
    shape: Shape | None = None # due to inheritance with default args

    def __post_init__(self) -> None:
        assert self.shape.width >= 8 and self.shape.height >= 8

    def size(self) -> int:
        return self.shape.area

class PGStreamCtx:
    __slots__ = ('_composition_number', 'bd_video')
    def __init__(self, bd_video: BDVideo) -> None:
        self.bd_video = bd_video
        self._composition_number = 0

    def get_composition_number(self) -> int:
        return self._composition_number

    def get_cast_composition_number(self) -> int:
        return self._composition_number & 0xFFFF

    def register_new_composition(self) -> None:
        self._composition_number += 1

class PGObjectBuffer:
    __MAX_SIZE = 4 << 20
    __MAX_SLOTS = 64
    def __init__(self):
        self._slots = {}

    def get(self, shape: Shape, dts: int | None) -> tuple[int, ObjectSlot] | None:
        """
        Get a slot of matching shape that can be used to decode an object.
        """
        for slot_id, slot in filter(lambda x: x[1].shape == shape, self._slots.items()):
            if dts is None or slot.pts is None or dts > slot.pts:
                return slot_id, slot

    def allocate(self, shape: Shape) -> tuple[int, ObjectSlot] | None:
        if sum(map(lambda s: s.size(), self._slots.values())) + shape.area > self.__class__.__MAX_SIZE:
            return None
        if len(self._slots) >= self.__class__.__MAX_SLOTS:
            return None
        slot = ObjectSlot(shape=shape)
        for k in range(self.__class__.__MAX_SLOTS):
            if self._slots.get(k, None) is None:
                self._slots[k] = slot
                return k, slot
        assert 0

    def get_indexed(self, slot_id: int) -> ObjectSlot | None:
        return self._slots.get(slot_id, None)

    def allocate_indexed(self, shape: Shape, slot_id: int) -> bool:
        if self._slots.get(slot_id, None) is not None:
            return False
        if sum(map(lambda s: s.size(), self._slots.values())) + shape.area > self.__class__.__MAX_SIZE:
            return False
        self._slots[slot_id] = ObjectSlot(shape=shape)
        return True

class PGEpochContext:
    def __init__(self, stream_ctx: PGStreamCtx, windows: list[Box], differentiate_palette: bool = False) -> None:
        self._windows = windows
        self._stream_ctx = stream_ctx
        self._palettes = [DecoderPalette() for _ in range(8)]
        self.buffer = PGObjectBuffer()
        self.differentiate_palette = differentiate_palette

    def flush(self) -> None:
        for palette in self._palettes:
            palette.clear()

    @property
    def bd_video(self) -> BDVideo:
        return self._stream_ctx.bd_video

    def get_palette_at(self, dts: int) -> tuple[int, DecoderPalette] | None:
        """
        Request a writable palette at DTS. The Decoding TS must be larger than
        the last PTS the palette was used.
        """
        # sort palettes to minimize wrap arounds (proprietary encoders seem to do that...)
        for palette_id, palette in sorted(zip(range(len(self._palettes)), self._palettes),
                                          key=lambda p: (p[1].version+1)//256):
            if palette.pts < dts:
                return palette_id, palette
        return None

    def register_composition(self,
         pts: int, dts: int, composition_state: PCS.CompositionState,
         palette_id: int, palette_update: bool,
         composition_objects: Sequence[CompositionObject]
    ) -> tuple[PCS, END]:
        """
        Register the display set whose decoding time is initiated at DTS, and
        whose to be displayed to the end-user at PTS.
        """
        assert self._palettes[palette_id].pts == pts
        pcs = PCS(
            pts=pts, dts=dts, width=self._stream_ctx.bd_video.fmt.width,
            height=self._stream_ctx.bd_video.fmt.height,
            composition_number=self._stream_ctx.get_cast_composition_number(),
            composition_state=composition_state,
            framerate_value=self.bd_video.fps.to_coded_frame_rate(),
            palette_id=palette_id, palette_update=palette_update,
            composition_objects=composition_objects
        )
        self._stream_ctx.register_new_composition()
        return pcs

    def register_object(self, pts: int, dts: int, bitmap: np.ndarray[tuple[int, int], np.uint8]) -> tuple[ODS, ...]:
        """
        Register an object to decode from DTS onwards, and that will be displayed at PTS.
        - The PTS is the composition presentation timestamp.
        - The DTS is either the DTS of the PCS or the DTS for this very object.

        Return:
            A sequence of ODS (one or more)
        """
        data = Brule.encode(bitmap)
        shp = Shape(*bitmap.shape)
        slot_data = self.buffer.get(shp, dts)
        if slot_data is None:
            slot_data = self.buffer.allocate(shp)
            assert slot_data is not None
        slot_id, slot = slot_data
        slot.reserve(pts)

        ods_list = []
        flag = ODS.DataFlag.FIRST
        len_rle_data_total = 4 + len(data)
        for chunk_size in chain([0xFFE4], repeat(0xFFEB)):
            rle_data = data[:chunk_size]
            if len(rle_data) == 0:
                break
            data = data[chunk_size:]
            if len(data) == 0:
                flag |= ODS.DataFlag.LAST
            ods_list.append(ODS(pts, dts, object_id=slot_id, object_version=slot.get_cast_version(),
                                flag=flag, width=shp.width, height=shp.height,
                                data=rle_data, data_len=len_rle_data_total))
            flag = 0
        return ods_list

    def register_palette(self, pts: int, dts: int, palette: Palette, force: bool = False) -> PDS | None:
        """
        Return a palette definition segment based on the hypothetical decoder state
        """
        palette_id, decoder_palette = self.get_palette_at(dts)
        if self.differentiate_palette:
            palette_diff = decoder_palette.get_difference(palette)
        else:
            palette_diff = palette.copy()
        if len(palette_diff) > 0 or force:
            decoder_palette.reserve(pts)
            decoder_palette.palette |= palette_diff
            return PDS(pts=pts, dts=dts, palette_id=palette_id,
                       palette_version=decoder_palette.get_cast_version(),
                       palette=palette_diff)
        return None

    def get_window_definition_segment(self, pts: int, dts: int) -> WDS:
        """
        Return the WDS segment for this epoch, at the given PTS, DTS.
        """
        definitions = list(map(lambda w: WDS.WindowDefinition(w[0], w[1].x, w[1].y, w[1].dx, w[1].dy), zip(range(len(self._windows)), self._windows)))
        return WDS(pts, dts, definitions)

    def update_object_reservation(self, object_id: int, pts: int, dts: int | None = None) -> bool:
        slot = self.buffer.get_indexed(object_id)
        assert slot.pts < pts
        is_safe = slot.pts < dts if dts is not None else False
        slot.pts = pts
        return is_safe

    def update_palette_reservation(self, palette_id: int, pts: int, dts: int) -> bool:
        palette = self._palettes[palette_id]
        assert palette.pts < pts
        is_safe = palette.pts < dts
        palette.pts = pts
        return is_safe

    def get_undisplay_wds_ds(self, c_pts: int, dts: int, palette_id: int) -> DisplaySet:
        self.update_palette_reservation(palette_id, c_pts, dts)
        pcs = self.register_composition(c_pts, dts, PCS.CompositionState.NORMAL_CASE, palette_id, False, [])
        wds = self.get_window_definition_segment(c_pts, c_pts)
        uds = DisplaySet([pcs, wds, END(pts=c_pts, dts=c_pts)])
        return uds

    def get_undisplay_pds_ds(self, c_pts: int, dts: int, cobjs: list[CompositionObject], n_colors: int) -> DisplaySet:
        palette = Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(n_colors)})
        pds = self.register_palette(c_pts, dts, palette)
        pcs = self.register_composition(c_pts, dts, PCS.CompositionState.NORMAL_CASE, pds.palette_id, True, cobjs)
        uds = DisplaySet([pcs, pds, END(pts=c_pts, dts=c_pts)])
        for cobj in cobjs:
            self.update_object_reservation(cobj.object_id, c_pts)
        return uds
