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
from enum import IntEnum
from itertools import starmap

from .palette import Palette, PaletteEntry
from .segments import WDS, ODS, DisplaySet, PDS
from .utils import Shape, Box

from dataclasses import dataclass
#%%
try:
    #If numba is available, provide compiled functions for the encoder/decoder 10x gain
    from numba import njit
    from numba.typed import List

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
    def njit_get_rle_lines(rle_data: Union[list[np.uint8], npt.NDArray[np.uint8], bytes], width: int) -> list[list[np.uint8]]:
        """
        Get the RLE lines within the raw RLE data.
        :param rle_data:  The RLE bitmap
        :param width:     The bitmap width when decoded
        :return:          list of RLE lines, N(lines) == bitmap height
        """
        k, j = -2, 0
        prev_k = 0
        rle_lines = List()
        len_rle = len(rle_data) - 2

        while k < len_rle:
            if j % width == 0:
                if k > 0:
                    rle_lines.append(rle_data[prev_k:k])
                j = 0
                k += 2
                prev_k = k
            if rle_data[k] == 0:
                byte = rle_data[(k:=k+1)]
                if byte & 0x40:
                    j += ((byte & 0x3F) << 8) | rle_data[(k:=k+1)]
                else:
                    j += byte & 0x3F
                if byte & 0x80:
                    k += 1
            else:
                j += 1
            k += 1
        rle_lines.append(rle_data[prev_k:-2])
        return rle_lines

    @njit(fastmath=True)
    def njit_decode_rle(rle_data: Union[list[np.uint8], npt.NDArray[np.uint8], bytes], width: int, height: int) -> npt.NDArray[np.uint8]:
        i, j, k = 0, -1, -2
        # RLE is terminated by new line command ([0x00, 0x00]), we can ignore it.
        len_rle = len(rle_data) - 2
        bitmap = np.zeros((height, width), np.uint8)

        while k < len_rle:
            if i % width == 0:
                i = 0
                j += 1
                k += 2
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
        return bitmap

except ModuleNotFoundError:
    njit_decode_rle = None
    njit_encode_rle = None
    njit_get_rle_lines = None
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
           height: Optional[int] = None
        ) -> npt.NDArray[np.uint8]:
        """
        Decode a RLE object, as defined in 'US 7912305 B1' patent.
        :param data:  Data to decode
        :param o_id:  Optional object ID to display, if rle_data is packed ODSes from a DS
        :param width: Expected width
        :param height:Expected height
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
            return njit_decode_rle(np.asarray(data, dtype=np.uint8), width, height)

        k = 0
        len_data = len(data)
        bitmap = []
        line = []

        while k < len_data:
            byte = data[k]
            if byte == 0:
                byte = data[(k:=k+1)]
                if byte == 0:
                    bitmap.append(line)
                    line = []
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
        return np.asarray(bitmap, np.uint8)

    @staticmethod
    def get_rle_lines(rle_data: Union[list[np.uint8], bytes], width: int) -> list[bytes]:
        """
        From raw RLE data, find the RLE code encoding each line separately.
        :param rle_data:  RLE bitmap
        :param width:     bitmap width when decoded
        :return:          list of RLE encoded lines, N(lines) == bitmap height
        """
        if njit_get_rle_lines is not None:
            return [line.tobytes() for line in njit_get_rle_lines(np.frombuffer(rle_data, np.uint8), width)]
        k, j = -2, 0
        prev_k = 0
        rle_lines = []
        len_rle = len(rle_data) - 2

        while k < len_rle:
            if j % width == 0:
                if k > 0:
                    rle_lines.append(rle_data[prev_k:k])
                j = 0
                k += 2
                prev_k = k
            if rle_data[k] == 0:
                byte = rle_data[(k:=k+1)]
                if byte & 0x40:
                    j += ((byte & 0x3F) << 8) | rle_data[(k:=k+1)]
                else:
                    j += byte & 0x3F
                if byte & 0x80:
                    k += 1
            else:
                j += 1
            k += 1
        rle_lines.append(rle_data[prev_k:-2])
        return rle_lines

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
    FREQ = 90e3
    DECODED_BUF_SIZE = 4*(1024**2)
    CODED_BUF_SIZE   = 1*(1024**2)

    @classmethod
    def gplane_write_time(cls, *shape, coeff: int = 1) -> int:
        return  int(np.ceil((cls.FREQ*coeff*shape[0]*shape[1])/cls.RC))

    @classmethod
    def plane_initilaization_time(cls, ds: DisplaySet) -> int:
        init_d = 0
        if ds.pcs.CompositionState.EPOCH_START & ds.pcs.composition_state:
            #The patent gives the coeff 8 but does not explain where it comes from
            # and the statements in the documentation says it is just the size of the
            # graphic plane. However there are two graphic plane so coeff could be
            # equal to 2 but who knows.
            init_d = cls.gplane_write_time(ds.pcs.width, ds.pcs.height, coeff=1)
        else:
            for window in ds.wds.windows:
                init_d += cls.gplane_write_time(window.width, window.height)
        return init_d

    @classmethod
    def wait(cls, ds: DisplaySet, obj_id: int, current_duration: int) -> int:
        wait_duration = 0
        for object_def in ds.ods:
            if object_def.o_id == obj_id:
                c_time = ds.pcs.dts + current_duration
                if c_time < object_def.pts:
                    wait_duration += object_def.pts - c_time
                return int(np.ceil(wait_duration*cls.FREQ))
        #Stream is either corrupted or object already in buffer
        return wait_duration
    ####
    @staticmethod
    def size(ds: DisplaySet, window_id: int) -> Shape:
        for wd in ds.wds.windows:
            if wd.window_id == window_id:
                return Shape(wd.width, wd.height)
        assert False, "Did not find window definition."

    @staticmethod
    def object_areas(ods: list[ODS]) -> int:
        return sum(map(lambda o: o.width*o.height if o.flags & o.ODSFlags.SEQUENCE_FIRST else 0, ods))

    @staticmethod
    def window_areas(wds: WDS) -> int:
        return sum(map(lambda window: window.width * window.height, wds.windows))

    @classmethod
    def rc_coeff(cls, ods: list[ODS], wds: WDS) -> int:
        return cls.object_areas(ods) + cls.window_areas(wds)

    @classmethod
    def decode_duration(cls, ds: DisplaySet) -> int:
        decode_duration = cls.plane_initilaization_time(ds)
        if ds.pcs.n_objects == 2:
            if ds.pcs.cobjects[0].window_id == ds.pcs.cobjects[1].window_id:
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
            else:
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[1].window_id))

        elif ds.pcs.n_objects == 1:
            decode_duration += cls.wait(ds, ds.pcs.cobjects[0].o_id, decode_duration)
            decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
        return decode_duration

    @classmethod
    def decode_obj_duration(cls, area: int) -> float:
        return np.ceil(cls.FREQ*area/cls.RD)/cls.FREQ

    @classmethod
    def copy_gp_duration(cls, area: int) -> float:
        return np.ceil(cls.FREQ*area/cls.RC)/cls.FREQ

    @classmethod
    def decode_display_duration(cls, gp_clear_dur: float, areas: list[int], gp_areas: list[int]) -> float:
        decode_duration = 0
        gp_duration = gp_clear_dur
        for d_area, c_area in zip(areas, gp_areas):
            decode_duration += cls.decode_obj_duration(d_area)
            gp_duration += (decode_duration-gp_duration) * (decode_duration > gp_duration)
            gp_duration += cls.copy_gp_duration(c_area)
        return gp_duration
####

#%%
@dataclass
class PGObject:
    gfx: npt.NDArray[np.uint8]
    box: Box
    mask: list[bool]
    f:  int

    def __post_init__(self) -> None:
        #gfx may have empty bitmaps trailing
        assert len(self.mask) <= len(self.gfx)
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
class PGObjectBuffer:
    """
    This class represents a PG Buffer with some functions to interact with it.
    The PG Buffer is entirely cleared on EPOCH START.
    On ACQUISITION or NORMAL case, we can only use existing buffer slots or define
    new ones, within the limited buffer memory size.
    """
    _MAX_OBJECTS = 64
    _MAX_SIZE = 4 << 20
    def __init__(self, /, *, _max_size: Optional[int] = None, _margin: int = 0) -> None:
        self._max_size = (__class__._MAX_SIZE if _max_size is None else _max_size)
        self._max_size -= _margin

        self._slots = {}
        self._loaded = []

    def get_free_size(self) -> int:
        """
        Get the remaining bytes available in the buffer.
        """
        diff = self._max_size - sum(starmap(lambda h, w: h*w, self._slots.values()))
        return diff if diff >= 0 else 0

    def get_n_free_slots(self) -> int:
        """
        Get the number of free slots left.
        """
        return __class__.MAX_OBJECTS - len(self.slots)

    def _find_free_id(self) -> Optional[int]:
        """
        Find the first available object ID.
        """
        for k in range(__class__.MAX_OBJECTS):
            if self._slots.get(k, None) is None:
                return k
        return None

    def free(self, obj_id: int) -> bool:
        """
        Free a slot in the buffer (this is not realistic HW-wise but
                                   provided out of programming courtesy)
        """
        if obj_id in self.slots:
            self._slots.pop(obj_id)
            return True
        return False

    def get_slot(self, height: int, width: int) -> Optional[int]:
        """
        Find the most suited existing free slot for a graphic of given dimensions.
        :param height: Graphic height to fit
        :param width:  Graphic width to fit
        :return: the slot ID that is large enough to fit the graphic, if any.
        """
        best_fit = np.inf
        best_fit_id = None
        for obj_id in self._slots:
            if obj_id in self._loaded:
                continue
            h, w = self._slots[obj_id]
            area_diff =  w*h-width*height
            if h >= height and w >= width and area_diff < best_fit:
                best_fit = area_diff
                best_fit_id = obj_id
        return best_fit_id

    def get_all_slots(self, only_free: bool = False) -> dict[int, tuple[int, int]]:
        """
        Return all allocated slots.
        :param only_free: Flag to request only the unloaded slots.
        """
        if only_free:
            return dict(filter(lambda x: x[0] in self._loaded, self._slots.items()))
        return self._slots

    def flush_objects(self) -> None:
        """
        Flush all loaded data on acquisition (the slots still exist but empty).
        """
        self._loaded.clear()

    def load(self, obj_id: int) -> None:
        """
        Mark a given slot ID as occupied in the buffer.
        :param obj_id: Slot ID to flag as occupied.
        """
        assert not self.is_loaded(obj_id), "Object already loaded in the buffer."
        self._loaded.append(obj_id)

    def is_loaded(self, obj_id: int) -> bool:
        """
        Check if a slot is already occupied by an object.
        :param obj_id: slot ID
        :return: true if the slot is occupied
        """
        return obj_id in self._loaded

    def check(self, obj_id: int, height: int, width: int) -> None:
        """
        Check that the provided object is suited for the given slot.
        :param obj_id: Slot ID
        :param height: Object height
        :param width:  Object width
        """
        hw = self._slots.get(obj_id, None)
        assert hw is not None, f"No slot allocated for {obj_id}"
        assert hw == (height, width), "Dimensions mismatch."

    @classmethod
    def get_capacity(cls) -> int:
        return cls._MAX_SIZE

    def get(self, slot_id: int) -> Optional[tuple[int, int]]:
        """
        Get a slot if it exists.
        :param slot_id: id of the slot to get
        :return: the slot size, if it exist
        """
        return self._slots.get(slot_id, None)

    def allocate_id(self, slot_id: int, height: int, width: int) -> bool:
        """
        Allocate a buffer slot with given dimensions.
        :param slot_id: desired slot id
        :param height: Object height
        :param width: Object width
        :return: success of the operation
        """
        assert 0 <= slot_id < 64

        desired_id = self._slots.get(slot_id, None)
        if desired_id is None and self.get_free_size() - width*height >= 0:
            self._slots[slot_id] = (height, width)
            return True
        return False

    def allocate(self, height: int, width: int) -> Optional[int]:
        """
        Allocate a buffer slot with given dimensions.
        :param height: Object height
        :param width: Object width
        :return: the object ID, if a slot could be allocated.
        """
        assert 8 <= width <= 4096 and 8 <= height <= 4096, "Incorrect dimensions for PG buffer."

        new_id = self._find_free_id()
        if new_id is not None and self.get_free_size() - width*height >= 0:
            self._slots[new_id] = (height, width)
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
