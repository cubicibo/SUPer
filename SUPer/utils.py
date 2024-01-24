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

import logging
import numpy as np

from typing import Optional, Callable, TypeVar, Union
from collections import namedtuple
from enum import Enum, IntEnum
from numpy import (typing as npt)
from timecode import Timecode
from dataclasses import dataclass
from contextlib import nullcontext

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    from contextlib import nullcontext as tqdm

MPEGTS_FREQ = int(90e3)

RegionType = TypeVar('Region')
_BaseEvent = TypeVar('BaseEvent')

Shape = namedtuple("Shape", "width height")
Dim = namedtuple("Dim", "w h")
Pos = namedtuple("Pos", "x y")

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
    def coords(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)

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
    def from_region(cls, region: RegionType) -> 'Box':
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
    def from_events(cls, events: list[_BaseEvent]) -> 'Box':
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

    def __eq__(self, other: 'Box') -> bool:
        if isinstance(other, __class__):
            return self.coords == other.coords
        return NotImplemented

####

#%%
@dataclass(frozen=True)
class ScreenRegion(Box):
    t:  int
    dt: int
    region: RegionType

    @classmethod
    def from_slices(cls, slices: tuple[slice], region: Optional[RegionType] = None) -> 'ScreenRegion':
        f_ZWz = lambda slz : (int(slz.start), int(slz.stop-slz.start))
        X, Y, T = f_ZWz(slices[2]), f_ZWz(slices[1]), f_ZWz(slices[0])

        if len(slices) != 3:
            raise ValueError("Expected 3 slices (t, y, x).")
        return cls(*Y, *X, *T, region)

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
    def from_region(cls, region: RegionType) -> 'ScreenRegion':
        return cls.from_slices(region.slice, region)

    @classmethod
    def from_coords(cls, x1: int, y1: int, t1: int, x2: int, y2: int, t2: int, region: RegionType) -> 'ScreenRegion':
        return cls(min(y1, y2), abs(y2-y1), min(x1, x2), abs(x2-x1), min(t1, t2), abs(t2-t1), region=region)
####

class WindowOnBuffer:
    def __init__(self, screen_regions: list[ScreenRegion]) -> None:
        self.srs = screen_regions

    def get_window(self) -> Box:
        mxy = np.asarray([np.inf, np.inf])
        Mxy = np.asarray([-1, -1])
        for sr in self.srs:
            mxy[:] = np.min([np.asarray((sr.y,  sr.x)),  mxy], axis=0)
            Mxy[:] = np.max([np.asarray((sr.y2, sr.x2)), Mxy], axis=0)
        mxy, Mxy = np.int32((mxy, Mxy))
        return Box(mxy[0], max(Mxy[0]-mxy[0], 8), mxy[1], max(Mxy[1]-mxy[1], 8))

    def area(self) -> int:
        return self.get_window().area
####

#%%
class BDVideo:
    class FPS(Enum):
        HFR_60 = 60
        NTSCi = 59.94
        PALi  = 50
        NTSCp = 29.97
        PALp  = 25
        FILM  = 24
        FILM_NTSC = 23.976

        @classmethod
        def from_pcsfps(self, pcsfps: int) -> 'BDVideo.FPS':
            return next((k for k in BDVideo.LUT_PCS_FPS if BDVideo.LUT_PCS_FPS[k] == pcsfps), None)

        @property
        def exact_value(self):
            if int(self.value) != self.value:
                return (np.ceil(self.value)*1e3)/1001
            return self.value

        @classmethod
        def _missing_(cls, value: Union[float, int]) -> 'BDVideo.FPS':
            """
            Find the closest framerate with < 0.1 tolerance (60 -> 59.94...)
            If the user writes 23.988=(24+23.976)/2, which could be both 24 or 23.976,
            the final value is rounded up (24 is chosen).
            """
            candidates = [fps.value for fps in __class__]
            best_fit = list(map(lambda x: abs(x-value), candidates))
            best_idx = best_fit.index(min(best_fit))
            if best_fit[best_idx] < 0.07:
                return cls(candidates[best_idx])
            raise ValueError("Framerate is not BD compliant.")

        def __truediv__(self, other) -> float:
            value = self.exact_value
            return value/other

        def __rtruediv__(self, other) -> float:
            value = self.exact_value
            return other/value

        def __mul__(self, other) -> float:
            value = self.exact_value
            return other*value

        def __rmul__(self, other) -> float:
            return self.__mul__(other)

        def __float__(self) -> float:
            return float(self.value)

        def __eq__(self, other) -> bool:
            if isinstance(other, (float, int)):
                try:
                    return __class__(other).value == self.value
                except ValueError:
                    return False
            elif isinstance(other, __class__):
                return other.value == self.value
            else:
                return NotImplemented

    class VideoFormat(Enum):
        HD1080    = (1920, 1080)
        HD720     = (1280, 720)
        SD576_43  = (720,  576)
        SD480_43  = (720,  480)

        @classmethod
        def from_height(cls, height: int) -> 'BDVideo.VideoFormat':
            for fmt in cls:
                if fmt[1] == height:
                    return cls(*fmt)
            raise ValueError(f"Unknown video format with height '{height}'.")

    class PCSFPS(IntEnum):
        FILM_NTSC_P = 0x10
        FILM_24P    = 0x20
        PAL_P       = 0x30
        NTSC_P      = 0x40
        PAL_I       = 0x60
        NTSC_I      = 0x70
        HFR_60      = 0x80

        @classmethod
        def from_fps(cls, other: float):
            return cls(BDVideo.LUT_PCS_FPS[np.round(other, 3)])

    LUT_PCS_FPS = {
        23.976:0x10,
        24:    0x20,
        25:    0x30,
        29.97: 0x40,
        50:    0x60,
        59.94: 0x70,
        60:    0x80,
    }
    LUT_FPS_PCSFPS = {v: k for k,v in LUT_PCS_FPS.items()}

    def __init__(self, fps: float, height: int, width: Optional[int] = None) -> None:
        self.fps = __class__.FPS(fps)
        self.pcsfps = __class__.PCSFPS.from_fps(self.fps.value)
        if width is None:
            self.format = None
            for vf in __class__.VideoFormat:
                if vf.value[1] == height:
                    self.format = vf
                    break
            assert self.format is not None
        else:
            self.format = __class__.VideoFormat((width, height))


class TimeConv:
    CLS = Timecode
    FORCE_NDF = True
    @staticmethod
    def s2f(s: float, fps: float, *, round_f: Optional[Callable[[float], float]] = round) -> float:
        """
        Convert a timestamp (seconds) to a number of frames
        :param s:           Seconds timestamp
        :param fps:         Framerate (Frames/s)
        :return:            Frame count
        """
        if round_f is None:
            round_f = lambda a : a # passthrough
        return int(round_f(s*fps))

    @classmethod
    def s2tc(cls, s: float, fps: float) -> str:
        #Add 1e-8 to avoid wrong rounding
        return str(Timecode(round(fps, 2), start_seconds=s+1/fps+1e-8, force_non_drop_frame=cls.FORCE_NDF))

    @classmethod
    def tc2s(cls, tc: str, fps: float, *, ndigits: int = 6) -> float:
        fps = round(fps, 2)
        return round(Timecode(fps, tc, force_non_drop_frame=cls.FORCE_NDF).float -\
                     Timecode(fps, '00:00:00:00', force_non_drop_frame=cls.FORCE_NDF).float, ndigits)

    @classmethod
    def ms2tc(cls, ms: int, fps: float) -> str:
        return cls.s2tc(ms/1000, fps)

    @classmethod
    def tc2ms(cls, tc: str, fps: float) -> int:
        return int(cls.tc2s(tc, fps)*1000)

    @classmethod
    def tc2f(cls, tc: str, fps: float, *, add_one: bool = False) -> int:
        return Timecode(round(fps, 2), tc, force_non_drop_frame=cls.FORCE_NDF).frame_number

    @classmethod
    def f2tc(cls, f: int, fps: float, *, add_one: bool = False) -> str:
        return str(Timecode(round(fps, 2), frames=f+1, force_non_drop_frame=cls.FORCE_NDF))

    @classmethod
    def add_framestc(cls, tc: str, fps: float, nf: int) -> str:
        fps = round(fps, 2)
        tc_udf = Timecode(fps, tc, force_non_drop_frame=cls.FORCE_NDF) + nf
        if cls.FORCE_NDF:
            tc_udf.drop_frame = False
        return str(tc_udf)

    @classmethod
    def add_frames(cls, tc: str, fps: float, nf: int) -> float:
        fps = round(fps, 2)
        return cls.tc2s(cls.add_framestc(tc, fps, nf), fps)
    
    @classmethod
    def tc2pts(cls, tc: str, fps: float) -> float:
        return max(0, (cls.tc2s(tc, fps) - (1/3)/MPEGTS_FREQ)) * (1 if float(fps).is_integer() else 1.001)

def get_matrix(matrix: str, to_rgba: bool, range: str) -> npt.NDArray[np.uint8]:
    """
    Getter of colorspace conversion matrix, BT ITU, limited or full
    :param matrix:       Conversion (BTxxx)
    :param range:        'limited' or 'full'
    :return:             Matrix
    """

    cc_matrix = {
        'bt601': {'y2r_l': np.array([[1.164,       0,  1.596, 0],
                                     [1.164,  -0.392, -0.813, 0],
                                     [1.164,   2.017,      0, 0],
                                     [    0,       0,      0, 1]]),
                  'r2y_l': np.array([[ 0.257,  0.504,  0.098, 0],
                                     [-0.148, -0.291,  0.439, 0],
                                     [ 0.439, -0.368, -0.071, 0],
                                     [     0,      0,      0, 1]]),
        },
        'bt709': {'y2r_l': np.array([[1.164,      0,   1.793, 0],
                                     [1.164, -0.213,  -0.533, 0],
                                     [1.164,  2.112,       0, 0],
                                     [    0,      0,       0, 1]]),
                  'r2y_l': np.array([[ 0.183,  0.614,  0.062, 0],
                                     [-0.101, -0.339,  0.439, 0],
                                     [ 0.439, -0.399, -0.040, 0],
                                     [     0,      0,      0, 1]]),
        },
        'bt2020': {'y2r_l':np.array([[1.1644,      0, 1.6787, 0],
                                     [1.1644, -.1873, -.6504, 0],
                                     [1.1644, 2.1418, -1e-04, 0],
                                     [     0,      0,      0, 1]]),
                   'r2y_l':np.array([[0.2256, 0.5823, 0.0509, 0],
                                     [-.1226, -.3166, 0.4392, 0],
                                     [0.4392, -.4039, -.0353, 0],
                                     [     0,      0,      0, 1]]),
        },
    }
    if to_rgba:
        mat = cc_matrix.get(matrix, {}).get(f"y2r_{range[0]}", None)
    else:
        mat = cc_matrix.get(matrix, {}).get(f"r2y_{range[0]}", None)

    if mat is None:
        raise NotImplementedError("Unknown/Not implemented conversion standard.")
    return mat

class LogFacility:
    _logger = dict()
    _logpbar = dict()

    @classmethod
    def set_file_log(cls, logger: logging.Logger, fp: str, level: Optional[int] = None) -> None:
        lfh = logging.FileHandler(fp, mode='w')
        formatter = logging.Formatter('%(levelname).8s: %(message)s')
        lfh.setFormatter(formatter)
        if logger.getEffectiveLevel() > level:
            cls.set_logger_level(logger.name, level)
        lfh.setLevel(logging.WARNING if level is None else level)
        logger.addHandler(lfh)

    @classmethod
    def _init_logger(cls, name: str) -> None:
        cls._extend_logger()
        logger = cls._logger[name] = logging.getLogger(name)

        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(' %(name)s: %(levelname).4s : %(message)s'.format(name))
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    @classmethod
    def set_logger_level(cls, name: str, level: int) -> None:
        assert cls._logger.get(name, None) is not None
        cls._logger[name].setLevel(level)
        cls._logger[name].handlers[0].setLevel(level)

    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        """
        This function takes in two parameters: name and level and logs to console.
        The place to log in this case is defined by the handler which we set
        to logging.StreamHandler().

        Args:
          name: Name for the logger.
          level: Minimum level for messages to be logged
        """
        if cls._logger.get(name, None) is None:
            cls._init_logger(name)
            cls.set_logger_level(name, level)

        return cls._logger[name]

    @staticmethod
    def _extend_logger() -> None:
        INFO_OUT = logging.INFO + 5
        logging.addLevelName(INFO_OUT, "IINFO")
        def info_out(self, message, *args, **kws):
            self._log(INFO_OUT, message, args, **kws)
        logging.Logger.iinfo = info_out

        LOW_DEBUG = logging.DEBUG - 5
        logging.addLevelName(LOW_DEBUG, "LDEBUG")
        def low_debug(self, message, *args, **kws):
            self._log(LOW_DEBUG, message, args, **kws)
        logging.Logger.ldebug = low_debug

        HIGH_DEBUG = logging.DEBUG - 2
        logging.addLevelName(HIGH_DEBUG, "HDEBUG")
        def high_debug(self, message, *args, **kws):
            self._log(HIGH_DEBUG, message, args, **kws)
        logging.Logger.hdebug = high_debug

    @classmethod
    def close_progress_bar(cls, logger: logging.Logger):
        if cls._logger.get(logger.name, None) != None and cls._logpbar.get(logger.name, None) is not None:
            cls._logpbar[logger.name].close()
            cls._logpbar[logger.name] = None

    @classmethod
    def get_progress_bar(cls, logger: logging.Logger, tot: ...) -> Optional[tqdm]:
        if cls._logger.get(logger.name, None) is None:
            return None
        if cls._logpbar.get(logger.name, None) is not None:
            return cls._logpbar[logger.name]
        if logger.getEffectiveLevel() >= logging.INFO:
            pbar = tqdm(tot)
        else:
            pbar = nullcontext()
            pbar.n = 0
        if getattr(pbar, 'update', None) is None:
            pbar.update = pbar.close = pbar.set_description = pbar.reset = pbar.refresh = pbar.clear = lambda *args, **kwargs: None
        cls._logpbar[logger.name] = pbar
        return pbar
    ####
####
