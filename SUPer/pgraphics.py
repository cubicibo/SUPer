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

from numpy import typing as npt
import numpy as np
from typing import Union, Optional, Type

from .palette import Palette, PaletteEntry
from .segments import ODS, PDS
from .utils import Shape, Box, MPEGTS_FREQ

from dataclasses import dataclass
#%%
try:
    #If numba is available, provide compiled functions for the encoder/decoder
    from numba import njit

    @njit(fastmath=True)
    def njit_encode_rle(bitmap: npt.NDArray[np.uint8]) -> list[np.uint8]:
        """
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param bitmap:    Palette mapped image to encode (2d array)
        :return:          Encoded data (vector)
        """
        i, j = 0, 0
        rle_data = [np.uint8(x) for x in range(0)]

        height, width = bitmap.shape
        assert width <= 16383, "Bitmap too large."

        while i < height:
            color = bitmap[i, j]
            prev_j = j
            while (j := j+1) < width and bitmap[i, j] == color: pass

            dist = j - prev_j
            if color == 0:
                if dist > 63:
                    rle_data += [0x00, 0x40 | ((dist >> 8) & 0x3F), dist & 0xFF]
                else:
                    rle_data += [0x00, dist & 0x3F]
            else:
                if dist > 63:
                    rle_data += [0x00, 0xC0 | ((dist >> 8) & 0x3F), dist & 0xFF, color]
                elif dist > 2:
                    rle_data += [0x00, 0x80 | (dist & 0x3F), color]
                else:
                    rle_data += [color] * dist
            if j == width:
                j = 0
                i += 1
                rle_data += [0x00, 0x00]
        return rle_data

    @njit(fastmath=True)
    def njit_decode_rle(rle_data: Union[list[np.uint8], npt.NDArray[np.uint8], bytes], width: int, height: int, check_rle: bool = False) -> npt.NDArray[np.uint8]:
        i, j, k = 0, -1, -2
        # RLE is terminated by new line command ([0x00, 0x00]), we can ignore it.
        len_rle = len(rle_data) - 2
        bitmap = np.zeros((height, width), np.uint8)
        line_length = 0

        while k < len_rle:
            if i % width == 0:
                if line_length > 0:
                    assert 0 == rle_data[k] == rle_data[k+1], "RLE line terminator not found at given width."
                    assert not check_rle or line_length <= width, "Illegal RLE line width."
                    line_length = 0
                i = 0
                j += 1
                k += 2
            k_start = k
            byte = rle_data[k]
            if byte == 0:
                byte = rle_data[(k:=k+1)]
                if byte & 0x40:
                    length = ((byte & 0x3F) << 8) | rle_data[(k:=k+1)]
                else:
                    length = byte & 0x3F
                if byte & 0x80:
                    bitmap[j, i:(i:=i+length)] = rle_data[(k:=k+1)]
                else:
                    bitmap[j, i:(i:=i+length)] = 0
            else:
                bitmap[j, i:(i:=i+1)] = byte
            k+=1
            line_length += (k - k_start)
        return bitmap

except ModuleNotFoundError:
    njit_decode_rle = None
    njit_encode_rle = None
####
#%%
class PGraphics:
    @classmethod
    def bitmap_to_ods(cls, bitmap: npt.NDArray[np.uint8], o_id: int, **kwargs) -> list[ODS]:
        assert bitmap.dtype == np.uint8
        height, width = bitmap.shape
        o_vn = kwargs.pop('o_vn', 0)
        data = cls.encode_rle(bitmap)

        return ODS.from_scratch(o_id, o_vn, width, height, data, **kwargs)

    @staticmethod
    def encode_rle(bitmap: npt.NDArray[np.uint8]) -> bytes:
        """
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param bitmap:    Palette mapped image to encode (2d array)
        :return:          Encoded data (vector)
        """
        if njit_encode_rle is not None:
            return njit_encode_rle(bitmap)

        rle_data = []
        i, j = 0, 0

        height, width = bitmap.shape
        assert width <= 16383, "Bitmap too large."

        while i < height:
            color = bitmap[i, j]
            prev_j = j
            while (j := j+1) < width and bitmap[i, j] == color: pass

            dist = j - prev_j
            if color == 0:
                if dist > 63:
                    rle_data += [0x00, 0x40 | ((dist >> 8) & 0x3F), dist & 0xFF]
                else:
                    rle_data += [0x00, dist & 0x3F]
            else:
                if dist > 63:
                    rle_data += [0x00, 0xC0 | ((dist >> 8) & 0x3F), dist & 0xFF, color]
                elif dist > 2:
                    rle_data += [0x00, 0x80 | (dist & 0x3F), color]
                else:
                    rle_data += [color] * dist
            if j == width:
                j = 0
                i += 1
                rle_data += [0x00, 0x00]
        return bytes(rle_data)

    @staticmethod
    def decode_rle(data: Union[bytes, bytearray, ODS, list[ODS]],
           o_id: Optional[int] = None,
           width: Optional[int] = None,
           height: Optional[int] = None,
           check_rle: bool = False,
        ) -> npt.NDArray[np.uint8]:
        """
        Decode a RLE object, as defined in 'US 7912305 B1' patent.
        :param data:  Data to decode
        :param o_id:  Optional object ID to display, if rle_data is packed ODSes from a DS
        :param width: Expected width
        :param height:Expected height
        :param check_rle: flag to enforce strict RLE line width compliancy.
        :return:      2D map to associate with the proper palette
        """
        if isinstance(data, ODS):
            data = [data]

        if isinstance(data, list):
            if isinstance(data[0], ODS):
                if o_id is not None:
                    data = b''.join(map(lambda x: x.data, filter(lambda x: o_id == x.o_id, data)))
                else:
                    data = b''.join(map(lambda x: x.data, data))
            else:
                data = bytearray(data)

        if njit_decode_rle and width and height:
            return njit_decode_rle(np.asarray(data, dtype=np.uint8), width, height, check_rle)

        k = 0
        len_data = len(data)
        bitmap = []
        line = []
        rle_line_length = 0

        while k < len_data:
            k_start = k
            byte = data[k]
            if byte == 0:
                byte = data[(k:=k+1)]
                if byte == 0:
                    assert not check_rle or width and rle_line_length <= width
                    bitmap.append(line)
                    line = []
                    rle_line_length = 0
                    k += 1
                    continue
                if byte & 0x40:
                    length = ((byte & 0x3F) << 8) | data[(k:=k+1)]
                else:
                    length = byte & 0x3F
                if byte & 0x80:
                    line += [data[(k:=k+1)]]*length
                else:
                    line += [0]*length
            else:
                line.append(byte)
            k+=1
            rle_line_length += (k - k_start)
        return np.asarray(bitmap, np.uint8)

    @staticmethod
    def show(l_ods: Union[ODS, list[ODS]],
             palette: Optional[Union[npt.NDArray[np.uint8], PDS, list[PDS], dict[int, PaletteEntry]]] = None) -> None:
        """
        Show the ODS with or without a provided palette. If no palette are provided,
        one is generated that illustrates the encoded animation in the bitmap.
        :l_ods:   ODS or list of ODS segment (object to decode)
        :palette: Palette to use. If none, a evenly distributed palette is
                  generated on the fly that illustrates the encoded animation
                  If list[PDS] is provided, the PDS are OR'd together.
        """
        bitmap = __class__.decode_rle(l_ods)

        # Create a evenly distributed palette using YUV wiht constant luma.
        if palette is None:
            mpe, Mpe = np.min(bitmap), np.max(bitmap)
            n_cols = (Mpe-mpe+1)
            luma = 0.5
            palette = np.zeros((Mpe+1, 3), float)
            angles = np.random.permutation(np.arange(0, (Mpe-mpe)/n_cols, 1/n_cols))

            for angle, k in zip(angles, range(mpe+1, Mpe)):
                angle *= 2*np.pi
                palette[k, 0] = luma + np.cos(angle)/0.88
                palette[k, 1] = luma - np.sin(angle)*0.38 - np.cos(angle)*0.58
                palette[k, 2] = luma + np.sin(angle)/0.49
            palette -= np.min(palette)
            palette /= (np.max(palette)/255)
            palette = np.uint8(np.round(palette))
        else:
            if isinstance(palette, list):
                assert isinstance(palette[0], PDS)
                pal = [pds.to_palette().palette for pds in palette]
                palette = {}
                for p in pal:
                    palette |= p
            if isinstance(palette, PDS):
                palette = palette.to_palette()
            elif isinstance(palette, dict):
                palette = Palette(palette)

            if isinstance(palette, Palette):
                if palette.get(255, None) is None:
                    palette[255] = PaletteEntry(16, 128, 128, 0)
                palette = palette.get_rgba_array(keep_indexes=True)
        try:
            from matplotlib import pyplot as plt
            plt.imshow(palette[bitmap])
        except ModuleNotFoundError:
            from PIL import Image
            Image.fromarray(palette[bitmap], 'RGB').show()
####
#%%
class PGDecoder:
    RX =  2e6
    RD = 16e6
    RC = 32e6
    FREQ = MPEGTS_FREQ
    DECODED_BUF_SIZE = 4*(1024**2)
    CODED_BUF_SIZE   = 1*(1024**2)

    @classmethod
    def decode_obj_duration(cls, area: int) -> float:
        return np.ceil(cls.FREQ*area/cls.RD)/cls.FREQ

    @classmethod
    def copy_gp_duration(cls, area: int) -> float:
        return np.ceil(cls.FREQ*area/cls.RC)/cls.FREQ
####

#%%
@dataclass
class PGObject:
    gfx: npt.NDArray[np.uint8]
    box: Box
    mask: list[bool]
    f:  int

    def __post_init__(self) -> None:
        assert len(self.mask) == len(self.gfx)
        assert self.box.area > 0

    @property
    def area(self) -> int:
        return self.gfx.shape[1]*self.gfx.shape[2]

    def get_bbox_at(self, frame: int) -> Optional[Box]:
        if self.is_active(frame):
            return self.__class__._bbox(self.gfx[frame-self.f])
        return None

    def is_active(self, frame) -> bool:
        return frame in range(self.f, self.f+len(self.mask))

    def is_visible(self, frame: int) -> bool:
        if self.is_active(frame):
            return self.mask[frame-self.f]
        return False

    def pad_left(self, padding: int) -> None:
        assert padding > 0
        self.f -= padding
        self.mask[0:0] = [False] * padding
        self.gfx = np.concatenate((np.zeros((padding, *self.gfx.shape[1:]), np.uint8), self.gfx), axis=0, dtype=np.uint8)

    @staticmethod
    def _bbox(img: npt.NDArray[np.uint8]) -> Box:
        rmin, rmax = np.where(np.any(img, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(img, axis=0))[0][[0, -1]]
        return Box.from_coords(cmin, rmin, cmax+1, rmax+1)
####

#%%
@dataclass
class BufferSlot:
    _width: int
    _height: int

    def __post_init__(self) -> None:
        assert 8 <= self._width  <= 4096, f"Illegal PG object width: {self._width}."
        assert 8 <= self._height <= 4096, f"Illegal PG object height: {self._height}."
        self._pts = -np.inf
        self._version = -1

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def size(self) -> int:
        return self.width*self.height

    def version_as_byte(self) -> int:
        assert self._version >= 0
        return self._version & 0xFF

    def writable_at(self, dts: float) -> bool:
        """
        Tells the caller if the slot can be written to starting dts
        :param dts: timestamp when the slot starts to be written to.
        :return: Writability status at dts
        """
        return self._pts <= dts

    def lock_until(self, pts: float) -> None:
        self._pts = pts
        self._version += 1

    @property
    def shape(self) -> Shape:
        return Shape(self.width, self.height)


class PGObjectBuffer:
    """
    This class represents a PG Object Buffer with some functions to interact with it.
    The buffer allocates BufferSlots that are acquired at specified times.
    """
    _MAX_OBJECTS = 64
    def __init__(self, /, *, _max_size: Optional[int] = None, _margin: int = 0) -> None:
        self._max_size = (PGDecoder.DECODED_BUF_SIZE if _max_size is None else _max_size)
        self._max_size -= _margin
        self._slots = {}

    def get_free_size(self) -> int:
        """
        Get the remaining bytes available in the buffer.
        """
        diff = self._max_size - sum(map(lambda x: x.size, self._slots.values()))
        return max(diff, 0)

    def reset(self) -> None:
        """
        Reset the buffer allocations (i.e. epoch start)
        """
        self._slots = {}

    def _find_free_id(self) -> Optional[int]:
        """
        Find the first available object ID.
        """
        for k in range(__class__._MAX_OBJECTS):
            if self._slots.get(k, None) is None:
                return k
        return None

    def request_slot(self, width: int, height: int, dts: float) -> tuple[Optional[int], Optional[BufferSlot]]:
        """
        Request a buffer slot of size (width, height) writable at the specified DTS.

        :param width: width of the slot
        :param height: height of the slot
        :param dts: timestamp at which the slot shall be available for writing.
        :return: the slot id and the buffer slot itself, if one is available.
        """
        for k, slot in self._slots.items():
            if slot.width == width and slot.height == height and slot.writable_at(dts):
                return (k, slot)

        slot_id = self._find_free_id()
        if slot_id is not None and self.get_free_size() - (bs := BufferSlot(width, height)).size >= 0:
            self._slots[slot_id] = bs
            return (slot_id, bs)
        return (None, None)

    def get(self, slot_id: int) -> Optional[BufferSlot]:
        """
        Get a slot if it exists.
        :param slot_id: id of the slot to get
        :return: the buffer slot, if it exists
        """
        return self._slots.get(slot_id, None)

    def get_slot_version(self, slot_id: int) -> Optional[int]:
        if slot_id in self._slots:
            return self._slots[slot_id].version_as_byte()
        return None

    def allocate_id(self, slot_id: int, width: int, height: int) -> bool:
        """
        Allocate a specific buffer slot with given dimensions.
        :param slot_id: desired slot id
        :param height: Object height
        :param width: Object width
        :return: success of the operation
        """
        assert 0 <= slot_id < 64
        bs = BufferSlot(width, height)

        if self.get(slot_id) is None and self.get_free_size() - bs.size >= 0:
            self._slots[slot_id] = bs
            return True
        return False

    def allocate(self, width: int, height: int) -> Optional[int]:
        """
        Allocate a buffer slot (any id) with given dimensions.
        :param height: Object height
        :param width: Object width
        :return: the slot ID, if a slot could be allocated.
        """
        new_id = self._find_free_id()
        if new_id is not None and self.get_free_size() - (bs := BufferSlot(width, height)).size >= 0:
            self._slots[new_id] = bs
            return new_id
        return None
####

#%%
@dataclass
class PGPalette(Palette):
    version: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self._pts = -1

    def lock_until(self, pts: float) -> None:
        self._pts = pts
        self.version += 1

    def diff(self, other: Palette) -> Palette:
        return Palette({k: pe for k, pe in other.palette.items() if self.palette.get(k, None) != pe})

    def writable_at(self, dts: float) -> bool:
        return self._pts < dts

    @property
    def pts(self) -> float:
        return self._pts

    def version_as_byte(self) -> int:
        return (self.version-1) & 0xFF

    def store(self, palette: Union[dict[int, ...], Type['PGPalette'], Palette], pts: float) -> None:
        self.palette |= palette if isinstance(palette, dict) else palette.palette
        self.sort()
        self.version += 1
        self._pts = pts

class PaletteManager:
    def __init__(self, n_palettes: int = 8) -> None:
        self._palettes = [PGPalette() for k in range(n_palettes)]

    def get_palette(self, dts: float) -> int:
        """
        Find a palette in the decoder that can be written to at a given dts.

        :param dts: decoding timestamp as a float (seconds)
        """
        versions = list(map(lambda palette: (palette.version) >> 8, self._palettes))

        best_k = None
        for k, p in enumerate(self._palettes):
            if p.writable_at(dts) and (best_k is None or versions[best_k] > versions[k]):
                best_k = k
        assert best_k is not None, f"No palette available at dts {dts} [s]"
        return best_k

    def get_palette_version(self, palette_id: int) -> int:
        assert palette_id < len(self._palettes), "Not an internal palette of the decoder!"
        assert self._palettes[palette_id].version > 0, "Getting version of unused palette."
        return self._palettes[palette_id].version_as_byte()

    def lock_palette(self, palette_id, pts: float, dts: float, force: bool = False) -> bool:
        assert palette_id < len(self._palettes), "Not an internal palette of the decoder!"
        pgpal = self._palettes[palette_id]
        if pgpal.writable_at(dts) or force:
            pgpal.lock_until(pts)
            return True
        return False

    def assign_palette(self, palette_id: int, palette: Palette, pts: float, dts: float) -> list[PDS]:
        """
        Generate the PDSegment to assign a given PGDecoder palette. The palette
        is then not writable until the PTS has passed.

        :param palette_id: ID of the palette to use (previously obtained via get_palette())
        :param palette: The (full) palette to be displayed
        :param pts: Presentation timestamp of the palette
        :param dts: Decoding timestamp of the palette
        :param only_diff: True if the palette assignement should only contain the difference.
        :return: a list with the PDS.
        """
        assert palette_id < len(self._palettes), "Not an internal palette of the decoder!"
        pgpal = self._palettes[palette_id]
        assert pgpal.writable_at(dts) is True, f"Using locked palette at {dts} [s]."
        pds_fn = lambda pal: PDS.from_scratch(pal, p_vn=pgpal.version_as_byte(), p_id=palette_id, pts=pts, dts=dts)
        if len(palette) == 0:
            assert len(pgpal) > 0, "Attempting to generate an empty PDS"
            # write the entire palette, not too sure of the effect of [PCS(palette_update_flag), END] on a decoder...
            pds = pds_fn(pgpal)
        else:
            pgpal.store(palette, pts)
            pds = pds_fn(palette)
        return [pds]
