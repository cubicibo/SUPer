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

from enum import Enum, IntEnum
from flags import Flags
from struct import unpack, pack
from typing import Union, Optional, Any, Type
from dataclasses import dataclass, field

from .utils import get_super_logger
from .palette import PaletteEntry, Palette

#%%
logging = get_super_logger('SUPer')

class PGSegment:
    class PGSOff(Enum):
        MAGIC_HEADER = slice(0, 2)
        PRES_TS      = slice(2, 6)
        DECODE_TS    = slice(6, 10)
        SEG_TYPE     = 10
        SEG_LENGTH   = slice(11,13)

    FREQ_PGS                = 90e3
    MAGIC: bytes            = b"PG"
    HEADER_LEN: int         = 13
    SEGMENT: dict[int, str] = {0x14: 'PDS', 0x15: 'ODS', 0x16: 'PCS',
                               0x17: 'WDS', 0x80: 'END'}
    I_SEGMENT: dict[str,int]= { v: k for k, v in SEGMENT.items() }

    def __init__(self, data: bytes, /, *, _bypass_checks = False) -> None:
        if __class__.MAGIC != data[__class__.PGSOff.MAGIC_HEADER.value]:
            logging.warning("Incoming data is not a PGS segment, discarding.")
            raise ValueError("Expected a PG segment, got " + \
                             f"{data[__class__.PGSOff.MAGIC_HEADER.value]}.")

        if len(data) < __class__.HEADER_LEN:
            logging.debug("Got too few bytes to decode header. Buffer needs to be refreshed.")
            raise BufferError("Got too few bytes to analyse stream.")

        if _bypass_checks:
            length = len(data[__class__.HEADER_LEN:])
        else:
            length = unpack(">H", data[__class__.PGSOff.SEG_LENGTH.value])[0]

        seg_data = data[__class__.HEADER_LEN:length+__class__.HEADER_LEN]
        if len(seg_data) < length and not _bypass_checks:
            logging.debug("Got too few bytes to decode payload. Buffer needs to be refreshed.")
            raise EOFError(f"Payload is missing {length-len(seg_data)} bytes.")

        self._bytes = bytearray(data[:length+__class__.HEADER_LEN])

    def __bytes__(self) -> bytes:
        return bytes(self._bytes)

    @property
    def pts(self) -> float:
        return unpack(">I", self._bytes[__class__.PGSOff.PRES_TS.value])[0] / __class__.FREQ_PGS

    @pts.setter
    def pts(self, n_pts: float) -> None:
        self._bytes[__class__.PGSOff.PRES_TS.value] = pack(">I", round(n_pts*__class__.FREQ_PGS))

    @property
    def dts(self) -> float:
        return unpack(">I", self._bytes[__class__.PGSOff.DECODE_TS.value])[0] / __class__.FREQ_PGS

    @dts.setter
    def dts(self, n_dts: float) -> None:
        self._bytes[__class__.PGSOff.DECODE_TS.value] = pack(">I", round(n_dts*__class__.FREQ_PGS))

    @property
    def type(self) -> str:
        return __class__.SEGMENT[self._bytes[__class__.PGSOff.SEG_TYPE.value]]

    @property
    def size(self):
        return unpack(">H", self._bytes[__class__.PGSOff.SEG_LENGTH.value])[0]

    def update(self):
        self._bytes[__class__.PGSOff.SEG_LENGTH.value] = \
            pack(">H", (len(self._bytes) - __class__.HEADER_LEN))

    @property
    def payload(self):
        return self._bytes[__class__.HEADER_LEN:]

    @payload.setter
    def payload(self, nf: tuple[Union[slice, int], bytes]) -> None:
        """
        Because payload is already an indirection to a subset of _bytes,
        we have to cheat our way with the setter. Thus it takes (offset, value)
        and under the hood, assign straight _bytes, as assigning payload would
        not work.
        """
        if type(nf[0]) is slice:
            start = __class__.HEADER_LEN + (nf[0].start if nf[0].start else 0)
            stop = None if nf[0].stop is None else nf[0].stop + __class__.HEADER_LEN
            offset = slice(start, stop)
        elif type(nf[0]) is int:
            offset = nf[0] + __class__.HEADER_LEN
        else:
            raise TypeError("Expected a slice or an integer for the index.")

        self._bytes[offset] = nf[1]
        self.update()

    def __str__(self):
        return f"{self.type} at {self.pts}[s], {self.size} bytes."

    def __len__(self):
        return len(self._bytes)

    def __equ__(self, other):
        return self._bytes == other._bytes

    def __hash__(self):
        return hash(self._bytes)

    def __lt__(self, other):
        return self.pts_ts < other.pts_ts

    def __gt__(self, other):
        return self.pts_ts > other.pts_ts

    @property
    def __dict__(self) -> dict[str, Any]:
        return {'pts': self.pts, 'dts': self.dts, 's_type': self.type,
                'size': self.size, 'bytes': self._bytes}

    def specialise(self):
        """
        PGSegment that can be instanciated are valid, this function allows to buid the child
        from the parent.
        """
        seg = { 'PDS': PDS, 'ODS': ODS, 'PCS': PCS, 'WDS': WDS, 'END': ENDS }
        return seg[self.type](self._bytes)

    @staticmethod
    def get_base_header() -> bytes:
        return b'PG' + bytearray([0] * 11)

    @classmethod
    def add_header(cls, data: bytes, s_type: 'str', pts: Optional[float] = None,
                   dts: Optional[float] = None) -> bytes:

        # This is the perfect example why __class__ is useful.
        packet = __class__(cls.get_base_header() + data, _bypass_checks=True)
        packet.update()
        if dts is not None:
            packet.dts = dts
        if pts is not None:
            packet.pts = pts
        else:
            logging.warning(f"{s_type} does not have a PTS, setting to zero...")
        packet._bytes[__class__.PGSOff.SEG_TYPE.value] = cls.I_SEGMENT[s_type]
        return packet._bytes

class CObject:
    class COOff(Enum):
        ODS_ID = slice(0 , 2)
        WIN_ID = 2
        FLAGS  = 3
        H_POS  = slice(4 , 6)
        V_POS  = slice(6 , 8)
        CO_NCL = 8
        HC_POS = slice(8 , 10)
        VC_POS = slice(10, 12)
        C_W    = slice(12, 14)
        C_H    = slice(14, 16)
        CO_COL = 16

    class COFlags(IntEnum):
        CROPPED = 0x80
        FORCED = 0x40
        STANDARD = 0x00

        @classmethod
        def _missing_(cls, value):
            return cls.STANDARD

    def __init__(self, data: bytes, *, _cropped = False) -> None:
        self._data = data

        #Autodetermine length w.r.t cropped flag
        # NOTE: the patent claims there can be N>1 cropping windows… on the same window??

        if len(self._data) > __class__.COOff.CO_COL.value:
            logging.error("SUPer does not know how to parse numerous cropping windows."\
                          " Send this sample to the author please!")

        if self.cropped or _cropped:
            self._data = self._data[:__class__.COOff.CO_COL.value]
        else:
            self._data = self._data[:__class__.COOff.CO_NCL.value]

    @property
    def o_id(self) -> int:
        return unpack(">H", self._data[__class__.COOff.ODS_ID.value])[0]

    @o_id.setter
    def o_id(self, no_id: int) -> None:
        self._data[__class__.COOff.ODS_ID.value] = pack(">H", no_id)

    @property
    def window_id(self) -> int:
        return self._data[__class__.COOff.WIN_ID.value]

    @window_id.setter
    def window_id(self, nw_id: int) -> None:
        self._data[__class__.COOff.WIN_ID.value] = nw_id & 0xFF

    @property
    def cropped(self) -> int:
        return bool(self._data[__class__.COOff.FLAGS.value] & __class__.COFlags.CROPPED.value)

    # #If cropping is changed, then the _bytes needs to be modified, so let's assume not.
    # @cropped.setter
    # def cropped(self, is_cropped: bool) -> None:
    #     self._data[__class__.COOff.FLAGS.value] = self._data[__class__.COOff.FLAGS.value] & (~__class__.COFlags.CROPPED.value)
    #     self._data[__class__.COOff.FLAGS.value] |= __class__.COFlags.CROPPED.value*is_cropped

    @property
    def forced(self) -> bool:
        return bool(self._data[__class__.COOff.FLAGS.value] & __class__.COFlags.FORCED.value)

    @forced.setter
    def forced(self, is_forced: bool) -> None:
        if self.cropped and is_forced:
            logging.warning("Undefined/Untested behaviour: Cropping + Forced.")
        self._data[__class__.COOff.FLAGS.value] = self._data[__class__.COOff.FLAGS.value] & (~__class__.COFlags.FORCED.value)
        self._data[__class__.COOff.FLAGS.value] |= __class__.COFlags.FORCED.value*is_forced

    @property
    def h_pos(self) -> int:
        return unpack(">H", self._data[__class__.COOff.H_POS.value])[0]

    @h_pos.setter
    def h_pos(self, nh_pos: int) -> None:
        self._data[__class__.COOff.H_POS.value] = pack(">H", nh_pos)

    @property
    def v_pos(self) -> int:
        return unpack(">H", self._data[__class__.COOff.V_POS.value])[0]

    @v_pos.setter
    def v_pos(self, nv_pos: int) -> None:
        self._data[__class__.COOff.V_POS.value] = pack(">H", nv_pos)

    @property
    def hc_pos(self) -> int:
        if self.cropped:
            return unpack(">H", self._data[__class__.COOff.HC_POS.value])[0]
        return 0

    @hc_pos.setter
    def hc_pos(self, nhc_pos: int) -> None:
        if self.cropped:
            self._data[__class__.COOff.HC_POS.value] = pack(">H", nhc_pos)
        else:
            raise NotImplementedError("Object needs to be created with cropping flag")

    @property
    def vc_pos(self) -> int:
        if self.cropped:
            return unpack(">H", self._data[__class__.COOff.VC_POS.value])[0]
        return 0

    @vc_pos.setter
    def vc_pos(self, nvc_pos: int) -> None:
        if self.cropped:
            self._data[__class__.COOff.VC_POS.value] = pack(">H", nvc_pos)
        else:
            raise NotImplementedError("Object needs to be created with cropping flag")

    @property
    def c_w(self) -> int:
        if self.cropped:
            return unpack(">H", self._data[__class__.COOff.C_W.value])[0]
        return 0

    @c_w.setter
    def c_w(self, nc_w: int) -> None:
        if self.cropped:
            self._data[__class__.COOff.C_W.value] = pack(">H", nc_w)
        else:
            raise NotImplementedError("Object needs to be created with cropping flag")

    @property
    def c_h(self) -> int:
        if self.cropped:
            return unpack(">H", self._data[__class__.COOff.C_H.value])[0]
        return 0

    @c_h.setter
    def c_h(self, nc_h: int) -> None:
        if self.cropped:
            self._data[__class__.COOff.C_H.value] = pack(">H", nc_h)
        else:
            raise NotImplementedError("Object needs to be created with cropping flag")

    @property
    def payload(self):
        return self._data

    @property
    def __dict__(self) -> dict[str, Any]:
        d1 = { 'o_id': self.o_id, 'window_id': self.window_id, 'forced': self.forced,
              'cropped': self.cropped, 'h_pos': self.h_pos, 'v_pos': self.v_pos }
        if self.cropped:
            d1.update({ 'hc_pos': self.hc_pos, 'vc_pos': self.vc_pos,
                        'c_w': self.c_w, 'c_h': self.c_h })
        return d1

    def __len__(self):
        return len(self._data)

    @classmethod
    def from_scratch(cls, o_id: int, window_id: int, h_pos: int, v_pos: int,
                     forced: bool, **kwargs):
        base = bytearray([0] * cls.COOff.CO_COL.value)
        if kwargs.get('cropped', False):
            base[cls.COOff.FLAGS.value] = cls.COFlags.CROPPED.value
        cobj = cls(base)
        cobj.o_id, cobj.window_id, cobj.forced = o_id, window_id, bool(forced)
        cobj.h_pos, cobj.v_pos = h_pos, v_pos
        if cobj.cropped:
            cobj.hc_pos, cobj.vc_pos = kwargs['hc_pos'], kwargs['vc_pos']
            cobj.c_w, cobj.c_h = kwargs['c_w'], kwargs['c_h']
        return cobj

class PCS(PGSegment):
    _NAME = 'PCS'
    class PCSOff(Enum):
        WIDTH      = slice(0, 2)
        HEIGHT     = slice(2, 4)
        STREAM_FPS = 4
        COMP_NB    = slice(5, 7)
        COMP_STATE = 7
        PAL_FLAG   = 8
        PAL_ID     = 9
        N_OBJ_DEFS = 10
        LENGTH_SEG = N_OBJ_DEFS # must be last

    class CompositionState(Flags):
        #NORMAL = 0x00     #Update
        ACQUISITION = 0x40 #Update current composition with new objects
        EPOCH_START = 0x80 #Display update (new "group of DSs")

    class PCSFPS(IntEnum):
        FILM_NTSC_P = 0x10
        FILM_24P    = 0x20
        PAL_P       = 0x30
        NTSC_P      = 0x40
        PAL_I       = 0x60
        NTSC_I      = 0x70

    _py2pg_pal_update_flag = {False: 0x00, True: 0x80}
    _pg2py_pal_update_flag = {v: k for k, v in _py2pg_pal_update_flag.items()}

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.cobjects: list[CObject] = []
        offset = 1
        for k in range(0, self.n_objects):
            sl = slice(__class__.PCSOff.LENGTH_SEG.value+offset, None)
            cobj = CObject(self.payload[sl])
            self.cobjects.append(cobj)
            offset += len(cobj)

    @property
    def width(self) -> int:
        return unpack(">H", self.payload[__class__.PCSOff.WIDTH.value])[0]

    @width.setter
    def width(self, nw: int) -> None:
        self.payload = (__class__.PCSOff.WIDTH.value, pack(">H", nw))

    @property
    def height(self) -> int:
        return unpack(">H", self.payload[__class__.PCSOff.HEIGHT.value])[0]

    @height.setter
    def height(self, nh: int) -> None:
        self.payload = (__class__.PCSOff.HEIGHT.value, pack(">H", nh))

    @property
    def fps(self) -> PCSFPS:
        return __class__.PCSFPS(self.payload[__class__.PCSOff.STREAM_FPS.value])

    @fps.setter
    def fps(self, nfps: PCSFPS) -> None:
        self.payload = (__class__.PCSOff.STREAM_FPS.value, int(nfps))

    @property
    def composition_n(self) -> int:
        return unpack(">H", self.payload[__class__.PCSOff.COMP_NB.value])[0]

    @composition_n.setter
    def composition_n(self, nc_n: int) -> None:
        self.payload = (__class__.PCSOff.COMP_NB.value, pack(">H", nc_n))

    @property
    def composition_state(self) -> Flags:
        return __class__.CompositionState(self.payload[__class__.PCSOff.COMP_STATE.value])

    @composition_state.setter
    def composition_state(self, cs: Flags) -> None:
        for flag in __class__.CompositionState:
            self.payload = (__class__.PCSOff.COMP_STATE.value,
                            self.payload[__class__.PCSOff.COMP_STATE.value] & (~int(flag)))
            self.payload = (__class__.PCSOff.COMP_STATE.value,
                            self.payload[__class__.PCSOff.COMP_STATE.value] | (int(flag) & int(cs)))

    @property
    def pal_flag(self) -> int:
        return __class__._pg2py_pal_update_flag[self.payload[__class__.PCSOff.PAL_FLAG.value]]

    @pal_flag.setter
    def pal_flag(self, n_flag: bool) -> None:
        self.payload = (__class__.PCSOff.PAL_FLAG.value, __class__._py2pg_pal_update_flag[n_flag])

    @property
    def pal_id(self) -> int:
        return self.payload[__class__.PCSOff.PAL_ID.value]

    @pal_id.setter
    def pal_id(self, np_id: int) -> None:
        self.payload = (__class__.PCSOff.PAL_ID.value, np_id)

    @property
    def n_objects(self) -> int:
        return self.payload[__class__.PCSOff.N_OBJ_DEFS.value]

    def update(self) -> None:
        self.payload = (__class__.PCSOff.N_OBJ_DEFS.value, len(self.cobjects))

        newp = bytearray()
        for cobj in self.cobjects:
            newp += cobj.payload
        self.payload = (slice(__class__.PCSOff.N_OBJ_DEFS.value+1, None), newp)
        super().update()

    @property
    def __dict__(self) -> dict[str, Any]:
        return dict({ 'width': self.width, 'height': self.height, 'fps': self.fps,
                    'composition_n': self.composition_n, 'pal_id': self.pal_id,
                    'composition_state': self.composition_state,
                    'pal_flag': self.pal_flag, 'cobjects': self.cobjects,
                    'n_objects': self.n_objects }, **super().__dict__)

    @classmethod
    def from_scratch(cls, width: int, height: int, fps: PCSFPS, composition_n: int,
                     composition_state: CompositionState, pal_flag: bool, pal_id: bool,
                     cobjects: list[CObject], pts: Optional[float] = None,
                     dts: Optional[float] = None, **kwargs):

        #Create dummy header, then fill it
        base = bytearray([0] * cls.PCSOff.LENGTH_SEG.value + [len(cobjects)])
        for cobj in cobjects:
            base = base + cobj.payload
        seg = cls(cls.add_header(base, cls._NAME, pts, dts))
        seg.pal_flag,  seg.pal_id  = pal_flag, pal_id
        seg.width,     seg.height  = width,    height
        seg.fps, seg.composition_n = fps,      composition_n
        seg.composition_state      = composition_state
        seg.update()
        return seg

class WindowDefinition:
    class WDOff(Enum):
        WINDOW_ID = 0
        H_POS     = slice(1, 3)
        V_POS     = slice(3, 5)
        WIDTH     = slice(5, 7)
        HEIGHT    = slice(7, 9)
        LENGTH    = HEIGHT.stop

    def __init__(self, data: bytes) -> None:
        self._bytes = data

    @property
    def window_id(self) -> int:
        return self._bytes[__class__.WDOff.WINDOW_ID.value]

    @window_id.setter
    def window_id(self, n_wid: int) -> None:
        self._bytes[__class__.WDOff.WINDOW_ID.value] = n_wid

    @property
    def h_pos(self) -> int:
        return unpack(">H", self._bytes[__class__.WDOff.H_POS.value])[0]

    @h_pos.setter
    def h_pos(self, n_hp: int) -> None:
        self._bytes[__class__.WDOff.H_POS.value] = pack(">H", n_hp)

    @property
    def v_pos(self) -> int:
        return unpack(">H", self._bytes[__class__.WDOff.V_POS.value])[0]

    @v_pos.setter
    def v_pos(self, n_vp: int) -> None:
        self._bytes[__class__.WDOff.V_POS.value] = pack(">H", n_vp)

    @property
    def width(self) -> int:
        return unpack(">H", self._bytes[__class__.WDOff.WIDTH.value])[0]

    @width.setter
    def width(self, n_w: int) -> None:
        self._bytes[__class__.WDOff.WIDTH.value] = pack(">H", n_w)

    @property
    def height(self) -> int:
        return unpack(">H", self._bytes[__class__.WDOff.HEIGHT.value])[0]

    @height.setter
    def height(self, n_h: int) -> None:
        self._bytes[__class__.WDOff.HEIGHT.value] = pack(">H", n_h)

    @property
    def payload(self):
        return self._bytes

    @property
    def __dict__(self) -> dict[str, Any]:
        return {'window_id': self.window_id, 'h_pos': self.h_pos, 'v_pos': self.v_pos,
                'width': self.width, 'height': self.height}

    @classmethod
    def from_scratch(cls, window_id: int, h_pos: int, v_pos: int, width: int,
                     height: int, **kwargs):
        wd = cls(bytearray([0] * cls.WDOff.LENGTH.value))
        wd.window_id = window_id
        wd.height, wd.width = height, width
        wd.h_pos,  wd.v_pos = h_pos,  v_pos
        return wd

class WDS(PGSegment):
    _NAME = 'WDS'
    _LENGTH_WINDOW_DEF = 9

    def __init__(self, data: bytes) -> None:
        super().__init__(data)

        # -1 because payload[0] is consumed
        if self.n_windows != (len(self.payload)-1)/__class__._LENGTH_WINDOW_DEF:
            raise Exception("Payload of WDS contains garbage.")

        lwd = __class__._LENGTH_WINDOW_DEF
        self.windows = [WindowDefinition(self.payload[1+i*lwd:1+(i+1)*lwd]) for i in range(self.n_windows)]

    @property
    def n_windows(self) -> int:
        return self.payload[0]

    def update(self) -> None:
        self.payload = (0, len(self.windows))
        newp = bytearray()
        for wd in self.windows:
            newp += wd.payload
        self.payload = (slice(1, None), newp)
        super().update()

    def __getitem__(self, id: int) -> WindowDefinition:
        return self.windows[id]

    def __setitem__(self, idx: int, wd: WindowDefinition) -> None:
        self.windows[idx] = wd

    @property
    def __dict__(self) -> dict[str, Any]:
        return dict({ 'n_windows': self.n_windows,
                      'windows': self.windows }, **super().__dict__)

    @classmethod
    def from_scratch(cls, windows: list[WindowDefinition], pts: Optional[float] = None,
                     dts: Optional[float] = None, **kwargs):
        b = bytearray()
        k = 0
        for wd in windows:
            b += wd.payload
            k += 1
        return cls(cls.add_header(bytearray([k]) + b, s_type=cls._NAME, pts=pts, dts=dts))

class PDS(PGSegment):
    _NAME = 'PDS'
    __STEP = 5
    class PDSOff(Enum):
        PAL_ID      = 0
        PAL_VERS_N  = 1
        PAL_ENTRIES = slice(2, None)

    """
    A PaletteDefinitionSegment updates or defines the entries of a given palette.
    For a given DS, this class allows to visualize or set updates.
    """
    _ids: list[int] = [] # Store the palette IDs used throughout a stream

    def __init__(self, data: bytes) -> None:
        super().__init__(data)

        if len(self.payload[__class__.PDSOff.PAL_ENTRIES.value]) % __class__.__STEP != 0:
            raise Exception("PDS payload appears to be incorrect.")

        if len(self.payload[__class__.PDSOff.PAL_ENTRIES.value])//__class__.__STEP > 256:
            logging.warning("More than 256 PAL entries defined in a PDS."
                            "Consider re-exporting or re-evaluating this segment.")

    @property
    def p_id(self) -> int:
        return self.payload[__class__.PDSOff.PAL_ID.value]

    @p_id.setter
    def p_id(self, p_id: int) -> None:
        self.payload = (__class__.PDSOff.PAL_ID.value, p_id & 0xFF)

    @property
    def p_vn(self) -> int:
        return self.payload[__class__.PDSOff.PAL_VERS_N.value]

    @p_vn.setter
    def p_vn(self, p_vn: int) -> int:
        self.payload = (__class__.PDSOff.PAL_VERS_N.value, p_vn & 0xFF)

    @property
    def n_entries(self) -> int:
        return int((len(self.payload) - __class__.PDSOff.PAL_ENTRIES.value.start)/5)

    @property
    def __dict__(self) -> dict[str, Any]:
        return dict({ 'p_id': self.p_id, 'p_vn': self.p_vn, 'n_entries': self.n_entries,
                      'palette': self.to_palette() }, **super().__dict__)

    def to_palette(self) -> Palette:
        p_data = self.payload[__class__.PDSOff.PAL_ENTRIES.value]

        entries = {}
        for i in range(0, len(p_data), __class__.__STEP):
            entries[p_data[i]] = PaletteEntry(*p_data[i+1:i+__class__.__STEP])
        return Palette(entries)

    def set_palette(self, palette: Palette) -> None:
        assert len(palette) <= 256
        self.payload = (__class__.PDSOff.PAL_ENTRIES.value, bytes(palette))

    @classmethod
    def from_scratch(cls, palette: Palette, p_vn: Optional[int] = None, p_id: int = 0,
                     pts: Optional[float] = None, dts: Optional[float] = None, **kwargs):
        p_vn = p_vn & 0xFF
        p_id = p_id & 0xFF
        if (offval := kwargs.get('offset', 0)) > 0:
            palette.offset(offval)
        assert p_id < 8, "Attempting to use palette ID >= 8 (undefined behaviour)"
        return cls(cls.add_header(bytearray([p_id, p_vn]) + bytes(palette), cls._NAME, pts, dts))

class ODS(PGSegment):
    _NAME = 'ODS'
    class ODSOff(Enum):
        OBJ_ID   = slice(0, 2)
        OBJ_VN   = 2
        SEQ_FLAG = 3
        DATA_LEN = slice(4, 7)
        WIDTH    = slice(7, 9)
        HEIGHT   = slice(9, 11)
        OBJ_DATA_FIRST = slice(11,None)
        OBJ_DATA_OTHERS= slice(4, None)

    class ODSFlags(Flags):
        SEQUENCE_FIRST = 0x80
        SEQUENCE_LAST  = 0x40

    class RLEMaxLength(IntEnum):
        FIRST  = 0xFFE4
        OTHERS = 0xFFEB

    def __init__(self, data: bytes) -> None:
        super().__init__(data)

        if self.flags == __class__.ODSFlags.SEQUENCE_FIRST | __class__.ODSFlags.SEQUENCE_LAST:
            #4 more bytes because width and length are part of the RLE data
            assert self.rle_len == len(self.data) + 4, "ODS length does not match payload."

    @property
    def o_id(self) -> int:
        return unpack(">H", self.payload[__class__.ODSOff.OBJ_ID.value])[0]

    @o_id.setter
    def o_id(self, n_oid: int) -> None:
        self.payload = (__class__.ODSOff.OBJ_ID.value, pack(">H", n_oid))

    @property
    def o_vn(self) -> int:
        return self.payload[__class__.ODSOff.OBJ_VN.value]

    @o_vn.setter
    def o_vn(self, no_vn: int) -> None:
        self.payload = (__class__.ODSOff.OBJ_VN.value, no_vn & 0xFF)

    @property
    def flags(self) -> Flags:
        return __class__.ODSFlags(self.payload[__class__.ODSOff.SEQ_FLAG.value])

    @flags.setter
    def flags(self, n_flags: Flags) -> None:
        for flag in __class__.ODSFlags:
            self.payload = (__class__.ODSOff.SEQ_FLAG.value,
                            self.payload[__class__.ODSOff.SEQ_FLAG.value] & (~int(flag)))
            self.payload = (__class__.ODSOff.SEQ_FLAG.value,
                            self.payload[__class__.ODSOff.SEQ_FLAG.value] | (int(flag) & int(n_flags)))

    @property
    def rle_len(self) -> int:
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            return unpack(">I", bytearray([0]) + self.payload[__class__.ODSOff.DATA_LEN.value])[0]
        raise AttributeError("ODS is not first in sequence.")

    @rle_len.setter
    def rle_len(self, n_len: int) -> None:
        n_len += 4 #Width and Height (2x2 bytes) are part of the RLE length
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            self.payload = (__class__.ODSOff.DATA_LEN.value, pack(">I", n_len)[1:])
        else:
            raise AttributeError("ODS is not first in sequence.")

    @property
    def width(self) -> int:
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            return unpack(">H", self.payload[__class__.ODSOff.WIDTH.value])[0]
        raise AttributeError("ODS is not first in sequence.")

    @width.setter
    def width(self, n_width: int) -> None:
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            self.payload = (__class__.ODSOff.WIDTH.value, pack(">H", n_width))
        else:
            raise AttributeError("ODS is not first in sequence.")

    @property
    def height(self) -> int:
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            return unpack(">H", self.payload[__class__.ODSOff.HEIGHT.value])[0]
        raise AttributeError("ODS is not first in sequence.")

    @height.setter
    def height(self, n_height: int) -> None:
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            self.payload = (__class__.ODSOff.HEIGHT.value, pack(">H", n_height))
        else:
            raise AttributeError("ODS is not first in sequence.")

    @property
    def data(self) -> bytes:
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            return self.payload[__class__.ODSOff.OBJ_DATA_FIRST.value]
        return self.payload[__class__.ODSOff.OBJ_DATA_OTHERS.value]

    @data.setter
    def data(self, n_data: bytes) -> None:
        assert len(n_data) > 0, "Got zero length RLE data for ODS."
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            assert len(n_data) <= __class__.RLEMaxLength.FIRST
            self.payload = (__class__.ODSOff.OBJ_DATA_FIRST.value, n_data)
        else:
            assert len(n_data) <= __class__.RLEMaxLength.OTHERS
            self.payload = (__class__.ODSOff.OBJ_DATA_OTHERS.value, n_data)
        self.update()

    @property
    def __dict__(self) -> dict[str, Any]:
        return dict({'o_id': self.o_id, 'o_vn': self.o_vn, 'flags': self.flags,
                     'rle_len': self.rle_len, 'width': self.width,
                     'height': self.height, 'data': self.data}, **super().__dict__)

    def update(self, total_rle_len: Optional[int] = None) -> None:
        """
        Set the encoded bitmap RLE length for a first in sequence ODS.
        if the bitmap is split across numerous ODS, provide the total RLE length
        +4 will be added internally by the code to account for the width and height.
        """
        if __class__.ODSFlags.SEQUENCE_FIRST in self.flags:
            if __class__.ODSFlags.SEQUENCE_LAST in self.flags:
                self.rle_len = len(self.data)
            elif total_rle_len is not None:
                self.rle_len = total_rle_len
        super().update()

    @classmethod
    def from_scratch(cls, o_id: int, o_vn: int, width: int, height: int, data: bytes,
                     pts: Optional[float] = None, dts: Optional[float] = None, **kwargs):
        """
        Generate from 1 to N ODS to fit the provided data.
        """

        seg = cls(cls.add_header(bytearray([0, 0, o_vn & 0xFF] + [0]*8), cls._NAME, pts, dts))
        seg.o_id = o_id & 0xFFFF
        seg.flags = __class__.ODSFlags.SEQUENCE_FIRST
        seg.width, seg.height = width, height
        seg.rle_len = len(data)

        lseg = [seg]
        MAXLEN_FIRST, MAXLEN_OTHERS = __class__.RLEMaxLength.FIRST, __class__.RLEMaxLength.OTHERS

        if len(data) <= MAXLEN_FIRST:
            seg.flags |= __class__.ODSFlags.SEQUENCE_LAST
            seg.data = data
        else:
            seg.data = data[:MAXLEN_FIRST]

            for k in range(0, len(data[MAXLEN_FIRST:]), MAXLEN_OTHERS):
                iseg = cls(cls.add_header(bytearray([0, 0, o_vn & 0xFF, 0]), cls._NAME, pts, dts))
                iseg.o_id = o_id & 0xFFFF
                iseg.data = data[MAXLEN_FIRST+k:MAXLEN_FIRST+(k+MAXLEN_OTHERS)]
                lseg.append(iseg)
            iseg.flags = __class__.ODSFlags.SEQUENCE_LAST
        return lseg


class ENDS(PGSegment):
    _NAME = 'END'
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        assert len(self.payload) == 0,"Got non-zero payload length END segment."

    @classmethod
    def from_scratch(cls, pts: Optional[float] = None,
                     dts: Optional[float] = None, **kwargs):
        return cls(cls.add_header(data=b'', s_type=cls._NAME, pts=pts, dts=dts))

#%%
class DisplaySet:
    def __init__(self, segments: list[Type[PGSegment]]) -> None:
        self.segments = segments
        self.pds = [s for s in self.segments if isinstance(s, PDS)]
        self.ods = [s for s in self.segments if isinstance(s, ODS)]
        if not isinstance(self.segments[0], PCS):
            raise ValueError("First segment is not a PCS.")
        self.wds = [s for s in self.segments if isinstance(s, WDS)]
        if isinstance(segments[-1], ENDS):
            self.end = segments[-1]
        else:
            self.end = ENDS.from_scratch(self._pcs.pts, self._pcs.dts)
            self.segments.append(self.end)
        self.has_image = bool(self.ods)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if len(self.segments) > self.n:
            self.n += 1
            return self.segments[self.n-1]
        else:
            raise StopIteration

    def __getitem__(self, n: int) -> Type[PGSegment]:
        return self.segments[n]

    def __setitem__(self, n: int, seg: Type[PGSegment]) -> None:
        if 0 <= n < len(self.segments):
            if seg:
                self.segments[n] = seg
            else:
                self.segments.pop(n)
        elif n > 0 and seg:
            self.append(seg)
        elif seg:
            self.segments[0:0] = [seg]
        else:
            raise Exception("Unknown setitem operation.")

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def t_in(self) -> float:
        return self.segments[0].pts

    @property
    def t_out(self) -> float:
        return self.segments[-1].pts

    @property
    def pcs(self) -> PCS:
        return self.segments[0]

    @pcs.setter
    def pcs(self, new_pcs: PCS) -> None:
        if isinstance(new_pcs, PCS):
            self.segments[0] = new_pcs
        else:
            raise TypeError("Not a PCS.")

    @property
    def pts(self) -> float:
        return self.segments[0].pts

    @pts.setter
    def pts(self, new_pts: float) -> None:
        for seg in self.segments:
            seg.pts = new_pts

    def __bytes__(self) -> bytes:
        return bytes(b''.join([bytes(seg) for seg in self.segments]))

    def is_palette_update(self) -> bool:
        try:
            return self.segments[0].pal_flag and len(self.ods) == 0
        except KeyError:
            return False

    def is_crop_update(self) -> bool:
        raise NotImplementedError

@dataclass
class Epoch:
    ds: list[DisplaySet] = field(default_factory=lambda: list())

    @property
    def t_in(self) -> float:
        try:
            return self.ds[0].t_in
        except:
            raise IndexError("Empty Epoch.")

    @property
    def t_out(self) -> float:
        try:
            return self.ds[-1].t_out
        except:
            raise IndexError("Empty Epoch.")

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if len(self.ds) > self.n:
            self.n += 1
            return self.ds[self.n-1]
        else:
            raise StopIteration

    def __getitem__(self, n: int) -> DisplaySet:
        return self.ds[n]

    def __setitem__(self, n: int, ds: DisplaySet) -> None:
        if 0 <= n < len(self.ds):
            if ds:
                self.ds[n] = ds
            else:
                self.ds.pop(n)
        elif n > 0 and ds:
            self.ds.append(ds)
        elif ds:
            self.ds[0:0] = [ds]
        else:
            raise Exception("Unknown setitem operation.")

    def __len__(self) -> int:
        return len(self.ds)

    def __bytes__(self) -> bytes:
        return bytes(b''.join([bytes(ds) for ds in self.ds]))

    def fetch(self) -> DisplaySet:
        for ds in self.ds:
            yield ds

    def inject(self, ds: DisplaySet) -> None:
        for k, ids in enumerate(self.ds):
            if ids.t_in > ds.t_in:
                self.ds.insert(k, ds)
                return
        self.ds.append(ds)

    def remove(self, pts: int, /, *, _tol=1e-4) -> DisplaySet:
        for k, ds in enumerate(self.ds.copy()):
            if abs(ds.t_in-pts) < _tol:
                return self.pop(k)

    def pop(self, idx) -> DisplaySet:
        return self.ds.pop(idx)

    def append(self, ds: DisplaySet) -> None:
        self.ds.append(ds)

    def __lt__(self, other):
        self.t_out < other.t_in

    def __gt__(self, other):
        self.t_in > other.t_out
