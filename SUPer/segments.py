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

import struct

from abc import ABC, abstractmethod
from enum import IntEnum, IntFlag
from typing import Self

from .palette import Palette, PaletteEntry
from .internals import _Masks, _classproperty

class PGSegmentType(IntEnum):
    PDS = 0x14
    ODS = 0x15
    PCS = 0x16
    WDS = 0x17
    END = 0x80

class GraphicSegment(ABC):
    __slots__ = 'pts', 'dts'
    
    def __init__(self, pts: int, dts: int):
        self.pts = pts
        self.dts = dts
    
    @_classproperty
    @abstractmethod
    def type(cls) -> PGSegmentType:
        ...

    @property
    def length(self) -> int:
        return len(self.get_payload())

    @abstractmethod
    def get_payload(self) -> bytes:
        ...
    
    def __bytes__(self) -> bytes:
        payload = self.get_payload()
        return struct.pack(">BH",self.type, len(payload) & _Masks.W16) + payload 
    
class CompositionObject:
    __slots__ = 'object_id', 'window_id', 'cropped_flag', 'forced_flag', 'h_pos', 'v_pos', 'crop_obj_x', 'crop_obj_y', 'crop_obj_w', 'crop_obj_h'
    
    def __init__(self, object_id: int, window_id: int, h_pos: int, v_pos: int,
                 cropped_flag: bool = False, forced_flag: bool = False,
                 crop_obj_x: int | None = None, crop_obj_y: int | None = None,
                 crop_obj_w: int | None = None, crop_obj_h: int | None = None) -> None:
        self.object_id = object_id
        self.window_id = window_id
        self.cropped_flag = cropped_flag
        self.forced_flag = forced_flag
        self.h_pos = h_pos
        self.v_pos = v_pos
        self.crop_obj_x = crop_obj_x
        self.crop_obj_y = crop_obj_y
        self.crop_obj_w = crop_obj_w
        self.crop_obj_h = crop_obj_h
        
        list_crop = [self.crop_obj_x, self.crop_obj_y, self.crop_obj_w, self.crop_obj_h]
        if self.cropped_flag:
            assert all(map(lambda x: x is not None, list_crop))
        else:
            assert all(map(lambda x: x is None, list_crop))
            
    def __bytes__(self) -> bytes:
        bs = struct.pack(">HBBHH", self.object_id, self.window_id, (self.cropped_flag << 7) | (self.forced_flag << 6),
                                   self.h_pos, self.v_pos)
        if self.cropped_flag:
            bs += struct.pack(">HHHH", self.crop_obj_x, self.crop_obj_y, self.crop_obj_w, self.crop_obj_h)
        return bs

    def __len__(self) -> int:
        return len(bytes(self))

    @classmethod
    def decode(cls, bs: bytes):
        object_id, window_id, flags, h_pos, v_pos = struct.unpack(">HBBHH", bs[:8])
        cropped_flag = bool(flags >> 7)
        forced_flag = bool((flags >> 6) & 0b1)
        if cropped_flag:
            crop_obj_x, crop_obj_y, crop_obj_w, crop_obj_h = struct.unpack(">HHHH", bs[8:16])
            cls(object_id, window_id, h_pos, v_pos, cropped_flag, forced_flag,
                crop_obj_x, crop_obj_y, crop_obj_w, crop_obj_h)
        return cls(object_id, window_id, h_pos, v_pos, forced_flag=forced_flag)

class PCS(GraphicSegment):
    __slots__ = ('width', 'height', 'framerate_value', 'composition_number',
                 'composition_state', 'palette_update', 'palette_id', 'composition_objects')
        
    class CompositionState(IntEnum):
        NORMAL_CASE = 0b00
        ACQUISITION = 0b01
        EPOCH_START = 0b10
        EPOCH_CONTINUE = 0b11
        
    class FramerateID(IntEnum):
        FPS_23976 = 1
        FPS_24    = 2
        FPS_25    = 3
        PPS_2997  = 4
        FPS_50    = 6
        FPS_5994  = 7
        FPS_60    = 8
    
    @_classproperty
    def type(cls) -> PGSegmentType:
        return PGSegmentType.PCS
    
    def __init__(self, pts: int, dts: int, width: int, height: int, framerate_value: FramerateID,
                 composition_number: int, composition_state: CompositionState,
                 palette_update: bool, palette_id: int,
                 composition_objects: list[CompositionObject]) -> None:
        super().__init__(pts, dts)
        
        self.width = width
        self.height = height
        self.framerate_value = __class__.FramerateID(framerate_value)
        self.composition_number = composition_number
        self.composition_state = __class__.CompositionState(composition_state)
        self.palette_update = bool(palette_update)
        self.palette_id = palette_id
        self.composition_objects = composition_objects
    
    @property
    def num_composition_objects(self) -> int:
        return len(self.composition_objects)
    
    def get_payload(self) -> bytes:
        return struct.pack(">HHBHBBBB", self.width, self.height,
                           self.framerate_value << 4, self.composition_number & _Masks.W16,
                           self.composition_state << 6, self.palette_update << 7,
                           self.palette_id, len(self.composition_objects)) + b''.join(map(bytes, self.composition_objects))

    @classmethod
    def decode(cls, bs: bytes, pts: int = 0, dts: int = 0) -> Self:
        (width, height, framerate_value, composition_number, composition_state,
        palette_update, palette_id, n_objects) = struct.unpack(">HHBHBBBB", bs[:11])
        
        framerate_value = cls.FramerateID(framerate_value >> 4)
        composition_state = cls.CompositionState(composition_state >> 6)
        
        assert 0 <= n_objects <= 2, f"illegal: {n_objects} compositions."
        
        composition_objects = []
        offset = 11
        for composition_id in range(n_objects):
            composition_objects += [CompositionObject.decode(bs[offset:])]
            offset += len(composition_objects[-1])
            
        return cls(pts, dts, width, height, framerate_value, composition_number,
                   composition_state, palette_update, palette_id, composition_objects)

class PDS(GraphicSegment):
    __slots__ = ('palette_id', 'palette_version', 'palette')
    
    @_classproperty
    def type(cls) -> PGSegmentType:
        return PGSegmentType.PDS

    def __init__(self, pts: int, dts: int, palette_id: int, palette_version: int, palette: Palette) -> None:
        #PDS only has a PTS
        super().__init__(pts, pts)
        self.palette_id = palette_id
        self.palette_version = palette_version
        self.palette = palette
        assert len(self.palette) <= 256
        
    def get_payload(self) -> bytes:
        return bytes([self.palette_id, self.palette_version]) + b''.join(map(lambda pe: bytes([pe[0]]) + bytes(pe[1]), self.palette.items()))
    
    @classmethod
    def decode(cls, bs: bytes, pts: int = 0, dts: int = 0) -> Self:
        palette_id, palette_version = bs[:2]
        assert (len(bs) - 2) % 5 == 0
        
        palette = Palette({bs[k]: PaletteEntry(*bs[k+1:k+5]) for k in range(2, len(bs), 5)})
        return cls(pts, dts, palette_id, palette_version, palette)
    
    @property
    def n_entries(self) -> int:
        return len(self.palette)
    
class ODS(GraphicSegment):
    __slots__ = ('object_id', 'object_version', 'flag', 'width', 'height', 'data_len', 'data')
    
    class DataFlag(IntFlag):
        FIRST = 0b01
        LAST  = 0b10

    @_classproperty
    def type(cls) -> PGSegmentType:
        return PGSegmentType.ODS

    def __init__(self, pts: int, dts: int, object_id: int, object_version: int,
                 flag: DataFlag, data: bytes, width: int | None = None,
                 height: int | None = None, data_len: int | None = None) -> None:
        super().__init__(pts, dts)
        self.object_id = object_id
        self.object_version = object_version
        self.flag = __class__.DataFlag(flag)
        if self.flag & __class__.DataFlag.FIRST:
            self.data_len = data_len
            self.width = width
            self.height = height
        else:
            self.width = self.height = self.data_len = None
        self.data = data
        
    def get_payload(self) -> bytes:
        bs = struct.pack(">HBB", self.object_id, self.object_version, self.flag << 6)
        if self.flag & __class__.DataFlag.FIRST:
            length = (4 + len(self.data))
            assert length < _Masks.W24
            bs += bytes([(length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF])
            bs += struct.pack(">HH", self.width, self.height)
        return bs + self.data
    
    @classmethod
    def decode(cls, bs: bytes, pts: int = 0, dts: int = 0) -> Self:
        object_id, object_version, flag = struct.unpack(">HBB", bs[:4])
        flag = __class__.DataFlag(flag >> 6)
        data_len = None
        if flag & __class__.DataFlag.FIRST:
            width, height = struct.unpack(">HH", bs[7:11])
            data = bs[11:]
            if flag & __class__.DataFlag.LAST:
                data_len = 4 + len(data)
                assert data_len == (bs[4] << 16) | (bs[5] << 8) | bs[6]
        else:
            width = height = None
            data = bs[4:]
        
        return cls(pts, dts, object_id, object_version, flag, data, width, height, data_len)
        
class WDS(GraphicSegment):
    __slots__ = 'windows'
    
    class WindowDefinition:
        __slots__ = 'window_id', 'h_pos', 'v_pos', 'width', 'height'
        
        def __init__(self, window_id: int, h_pos: int, v_pos: int, width: int, height: int) -> None:
            self.window_id = window_id
            self.h_pos = h_pos
            self.v_pos = v_pos
            self.width = width
            self.height = height
        
        def __bytes__(self) -> bytes:
            return struct.pack(">BHHHH", self.window_id, self.h_pos, self.v_pos, self.width, self.height)
        
        @classmethod
        def decode(cls, bs: bytes) -> Self:
            return cls(*struct.unpack(">BHHHH", bs))
    
    @_classproperty
    def type(cls) -> PGSegmentType:
        return PGSegmentType.WDS
    
    def __init__(self, pts: int, dts: int, windows: list[WindowDefinition]) -> None:
        super().__init__(pts, dts)
        assert len(windows) in range(1, 3)
        self.windows = windows
        
    def get_payload(self) -> bytes:
        return bytes([len(self.windows)]) + b''.join(map(bytes, self.windows))
    
    @classmethod
    def decode(cls, bs: bytes = b'', pts: int = 0, dts: int = 0) -> Self:
        n_windows = bs[0]
        assert 0 < n_windows <= 2
        windows = []
        for k in range(1, len(bs), 9):
            windows += [__class__.WindowDefinition.decode(bs[k:k+9])]
        return cls(pts, dts, windows)

class END(GraphicSegment):
    def __init__(self, pts: int, dts: int):
        #END segment only has a PTS
        super().__init__(pts, pts)

    @_classproperty
    def type(cls) -> PGSegmentType:
        return PGSegmentType.END

    def get_payload(self) -> bytes:
        return b''
    
    @classmethod
    def decode(cls, bs: bytes, pts: int = 0, dts: int = 0) -> Self:
        return cls(pts, dts)
    
class SegmentParser:
    @classmethod
    def from_pg_segment(cls, bs: bytes) -> GraphicSegment:
        assert len(bs) >= 13
        assert b'PG' == bs[:2]
        
        pts, dts, type_, length = struct.unpack(">IIBH", bs[2:13])
        assert length + 13 >= len(bs)
        # caller can deduce
        bs = bs[:length+13]
        return cls._from_payload_and_ts(type_, pts, dts, bs[13:])
        
    def _from_payload_and_ts(type_: PGSegmentType, pts: int, dts: int, bs: bytes) -> GraphicSegment:
        match PGSegmentType(type_):
            case PGSegmentType.PCS: return PCS.decode(bs, pts, dts)
            case PGSegmentType.WDS: return WDS.decode(bs, pts, dts)
            case PGSegmentType.PDS: return PDS.decode(bs, pts, dts)
            case PGSegmentType.ODS: return ODS.decode(bs, pts, dts)
            case PGSegmentType.END: return END(pts, dts)

    @classmethod
    def from_pesmui_segment(cls, mui_data: bytes, pes_data: bytes) -> GraphicSegment:
        segment_type, segment_size = struct.unpack(">BI", mui_data[:5])
        segment_type = PGSegmentType(segment_type)

        pes_segment_type, pes_segment_size = struct.unpack(">BH", pes_data[:3])
        assert pes_segment_type == segment_type
        assert pes_segment_size == segment_size - 3
        
        dts = struct.unpack(">I", mui_data[5:9])[0] << 1
        dts += (mui_data[9] >> 7)
        
        pts = (struct.unpack(">I", mui_data[10:14])[0]) + ((mui_data[9] & 0x7F) << 32)
        #pts >>= 6
        
        return cls._from_payload_and_ts(segment_type, pts, dts, pes_data[3:])
####
                