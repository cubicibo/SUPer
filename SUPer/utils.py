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

from typing import Optional, TypeVar, Union
from logging.handlers import BufferingHandler
from enum import Enum, IntEnum
from numpy import typing as npt
from fractions import Fraction
from timecode import Timecode
from dataclasses import dataclass
from contextlib import nullcontext
from functools import lru_cache
from SSIM_PIL import compare_ssim

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = nullcontext

MPEGTS_FREQ = np.uint64(90e3)

_BaseEvent = TypeVar('BaseEvent')

@dataclass
class Pos:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))

@dataclass
class Shape:
    w: int
    h: int

    def __post_init__(self) -> None:
        assert self.w >= 0 and self.h >= 0

    @classmethod
    def from_box(cls, box: 'Box') -> 'Shape':
        return cls(box.dx, box.dy)

    @classmethod
    def union(cls, *shapes) -> 'Shape':
        w = max(map(lambda dim: dim.w, shapes))
        h = max(map(lambda dim: dim.h, shapes))
        return cls(w, h)

    @property
    def area(self) -> int:
        return self.w*self.h

    @property
    def width(self) -> int:
        return self.w

    @property
    def height(self) -> int:
        return self.h

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.w == other.w and self.h == other.h
        elif isinstance(other, (tuple, list)) and len(other) == 2:
            return self.w == other[0] and self.h == other[1]
        return NotImplemented

    def __ne__(self, other):
        test_eq = self.__eq__(other)
        if isinstance(test_eq, bool):
            return not test_eq
        return NotImplemented

    def __iter__(self):
        return iter((self.w, self.h))

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
    def dims(self) -> Shape:
        return Shape(self.dx, self.dy)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the numpy-like shape (col, row)
        """
        return (self.dy, self.dx)

    @property
    def pos_shape(self) -> tuple[Pos, Shape]:
        return Pos(self.x, self.y), Shape(self.dx, self.dy)

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
    def from_slices(cls, slices: tuple[slice]) -> 'Box':
        if len(slices) == 3:
            slyx = slices[1:]
        else:
            slyx = slices
        f_ZWz = lambda slz : (int(slz.start), int(slz.stop-slz.start))
        return cls(*f_ZWz(slyx[0]), *f_ZWz(slyx[1]))

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
class BDVideo:
    _LUT_PCS_FPS = {
        23.976:0x10,
        24:    0x20,
        25:    0x30,
        29.97: 0x40,
        50:    0x60,
        59.94: 0x70,
        60:    0x80,
    }

    class FPS(Enum):
        HFR_60 = Fraction(60, 1)
        NTSCi  = Fraction(60000, 1001)
        PALi   = Fraction(50, 1)
        NTSCp  = Fraction(30000, 1001)
        PALp   = Fraction(25, 1)
        FILM   = Fraction(24, 1)
        FILM_NTSC = Fraction(24000, 1001)

        @classmethod
        def from_pcsfps(cls, pcsfps: int) -> 'BDVideo.FPS':
            return cls(next(filter(lambda v: v[1] == pcsfps, BDVideo._LUT_PCS_FPS.items()))[0])

        def to_pcsfps(self) -> 'BDVideo.PCSFPS':
            rfps = round(float(self), 2)
            return BDVideo.PCSFPS(next(filter(lambda v: rfps == round(v[0],2), BDVideo._LUT_PCS_FPS.items()))[1])

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

        def __float__(self) -> float:
            return float(self.value)

        def __int__(self) -> int:
            return int(self.value)

        def __round__(self, ndigits: int = 0):
            return round(self.value, ndigits)

        def __truediv__(self, other: Union[Fraction, float, int]) -> Union[Fraction, float]:
            return self.value/other

        def __rtruediv__(self, other: Union[Fraction, float, int]) -> Union[Fraction, float]:
            return other/self.value

        def __mul__(self, other: Union[Fraction, float, int]) -> Union[Fraction, float]:
            return self.value*other

        def __rmul__(self, other: Union[Fraction, float, int]) -> Union[Fraction, float]:
            return self.__mul__(other)

        def __float__(self) -> float:
            return float(self.value)

        def __gt__(self, other):
            if isinstance(other, __class__):
                return self.value > other.value
            elif isinstance(other, (int, float, Fraction)):
                return self.value > other
            return NotImplemented

        def __lt__(self, other):
            if isinstance(other, __class__):
                return self.value < other.value
            elif isinstance(other, (int, float, Fraction)):
                return self.value < other
            return NotImplemented

        def __ne__(self, other) -> bool:
            test_eq = self.__eq__(other)
            if test_eq == NotImplemented:
                return test_eq
            return not test_eq

        def __eq__(self, other) -> bool:
            if isinstance(other, (float, int, Fraction)):
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

        @property
        def area(self) -> int:
            return self.value[0]*self.value[1]

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

    def __init__(self, fps: float, height: int, width: Optional[int] = None) -> None:
        self.fps = __class__.FPS(fps)
        self.pcsfps = self.fps.to_pcsfps()
        if width is None:
            self.format = None
            for vf in __class__.VideoFormat:
                if vf.value[1] == height:
                    self.format = vf
                    break
            assert self.format is not None
        else:
            self.format = __class__.VideoFormat((width, height))

    @classmethod
    def check_format_fps(cls, _format: 'BDVideo.VideoFormat', fps: Union[float, 'BDVideo.FPS', Fraction]) -> bool:
        valid = True
        fps = cls.FPS(fps)
        expected = [_fps for _fps in cls.FPS]
        if _format == cls.VideoFormat.HD720:
            expected = [cls.FPS.FILM_NTSC, cls.FPS.FILM, cls.FPS.PALi, cls.FPS.NTSCi]
            valid &= fps in expected
        elif _format == cls.VideoFormat.SD576_43:
            expected = [cls.FPS.PALp]
            valid &= fps in expected
        elif _format == cls.VideoFormat.SD480_43:
            expected = [cls.FPS.NTSCp]
            valid &= fps in expected
        return valid, list(map(lambda x: round((float if x.value.denominator == 1001 else int)(x), 3), expected))

#%%
class TC(Timecode):
    def __init__(self, fps, *args, **kwargs) -> None:
        if not isinstance(fps, BDVideo.FPS):
            fps = BDVideo.FPS(fps)
        super().__init__(fps.value, *args, **kwargs)
        self.fractional_fps = fps

    @classmethod
    def s2tc(cls, s: float, fps: float, drop_frame: bool = False) -> 'TC':
        #Add 1e-8 to avoid wrong rounding
        s = s/(1 if float(fps).is_integer() else 1.001)
        r_tc = cls(round(fps, 2), start_seconds=s+1/fps+1e-8, force_non_drop_frame=True)
        r_tc.drop_frame = drop_frame
        return r_tc

    def to_pts(self) -> float:
        tpts = ((self.frames - 1)/self.fractional_fps.value)*MPEGTS_FREQ
        return (tpts.numerator//tpts.denominator)/MPEGTS_FREQ

    def __add__(self, other: Union['TC', int]) -> 'TC':
        # duplicate current one
        tc = __class__(self.fractional_fps, frames=self.frames)
        tc.drop_frame = self.drop_frame

        if isinstance(other, __class__):
            assert other.fractional_fps == self.fractional_fps
            assert self.drop_frame == other.drop_frame == False
            tc.add_frames(other.frames)
        else:
            assert isinstance(other, int)
            tc.add_frames(other)
        return tc

class SSIMPW:
    use_gpu = True

    @classmethod
    def compare(cls, img1, img2) -> float:
        return compare_ssim(img1, img2, GPU=cls.use_gpu)

@lru_cache(maxsize=6)
def get_matrix(matrix: str, to_rgba: bool) -> npt.NDArray[np.uint8]:
    """
    Getter of colorspace conversion matrix, BT ITU, limited or full
    :param matrix:       Conversion (BTxxx)
    :param range:        'limited' or 'full'
    :return:             Matrix
    """

    cc_matrix = {
        'bt601': {'y2r':   np.array([[1.164,       0,  1.596, 0],
                                     [1.164,  -0.392, -0.813, 0],
                                     [1.164,   2.017,      0, 0],
                                     [    0,       0,      0, 1]]),
                  'r2y':   np.array([[ 0.257,  0.504,  0.098, 0],
                                     [-0.148, -0.291,  0.439, 0],
                                     [ 0.439, -0.368, -0.071, 0],
                                     [     0,      0,      0, 1]]),
        },
        'bt709': {'y2r':   np.array([[1.164,      0,   1.793, 0],
                                     [1.164, -0.213,  -0.533, 0],
                                     [1.164,  2.112,       0, 0],
                                     [    0,      0,       0, 1]]),
                  'r2y':   np.array([[ 0.183,  0.614,  0.062, 0],
                                     [-0.101, -0.339,  0.439, 0],
                                     [ 0.439, -0.399, -0.040, 0],
                                     [     0,      0,      0, 1]]),
        },
        'bt2020': {'y2r':  np.array([[1.16439,      0,1.67867,0],
                                     [1.16439,-.18734,-.65042,0],
                                     [1.16439,2.14175,      0,0],
                                     [     0,      0,       0,1]]),
                   'r2y':  np.array([[0.22561,0.58228,0.05093,0],
                                     [-.12266,-.31656,0.43922,0],
                                     [0.43922,-.40389,-.03533,0],
                                     [      0,      0,      0,1]]),
        },
    }
    mat = cc_matrix.get(matrix, None)
    if mat is None:
        raise NotImplementedError("Unknown/Not implemented conversion standard.")
    return mat["y2r" if to_rgba else "r2y"]

class LogFacility:
    _logger = dict()
    _logpbar = dict()
    _tqdm_off = False

    @classmethod
    def set_file_log(cls, logger: logging.Logger, fp: str, level: Optional[int] = None, simple_format: bool = False) -> None:
        if level is None:
            level = logger.level
        lfh = logging.FileHandler(fp, mode='w')
        formatter = logging.Formatter('%(message)s' if simple_format else '%(levelname).8s: %(message)s')
        lfh.setFormatter(formatter)
        if logger.getEffectiveLevel() > level:
            cls.set_logger_level(logger.name, level)
        lfh.setLevel(level)
        logger.addHandler(lfh)

    @classmethod
    def _init_logger(cls, name: str, with_handler: bool = True) -> None:
        cls._extend_logger()
        logger = cls._logger[name] = logging.getLogger(name)

        if not logger.hasHandlers() and with_handler:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(' %(name)s %(levelname).4s : %(message)s'.format(name))
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    @classmethod
    def set_logger_level(cls, name: str, level: int) -> None:
        assert cls._logger.get(name, None) is not None
        cls._logger[name].setLevel(level)
        if len(cls._logger[name].handlers):
            cls._logger[name].handlers[0].setLevel(level)

    @classmethod
    def get_buffered_msgs(cls, logger: logging.Logger) -> Optional[list[str]]:
        for hdl in logger.handlers:
            if isinstance(hdl, BufferingHandler):
                fmsgs = [(rec.levelno, rec.getMessage()) for rec in hdl.buffer]
                hdl.flush()
                return fmsgs
        return None

    @classmethod
    def exit_on_error(cls, logger: logging.Logger) -> None:
        class ErrorExit:
            def __init__(self, log_error_f) -> None:
                self.f_log_error = log_error_f
            def __call__(self, *args, **kwargs) -> None:
                self.f_log_error(*args, **kwargs)
                self.f_log_error("Error occured in strict mode. Terminating.")
                import sys
                sys.exit(1)

        #isinstance on classes generated inside a function could be brittle?
        if getattr(logger.error.__class__, "__name__", None) != 'ErrorExit':
            logger.error = ErrorExit(logger.error)

    @classmethod
    def set_logger_buffer(cls, logger: logging.Logger) -> None:
        hdl = BufferingHandler(float('inf'))
        hdl.setLevel(logging.INFO)
        logger.addHandler(hdl)

    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO, with_handler: bool = True) -> logging.Logger:
        """
        This function takes in two parameters: name and level and logs to console.
        The place to log in this case is defined by the handler which we set
        to logging.StreamHandler().

        Args:
          name: Name for the logger.
          level: Minimum level for messages to be logged
        """
        if cls._logger.get(name, None) is None:
            cls._init_logger(name, with_handler)
            cls.set_logger_level(name, level)
        return cls._logger[name]

    @staticmethod
    def _extend_logger() -> None:
        if getattr(logging.Logger, 'iinfo', None) is not None:
            return
        INFO_OUT = logging.INFO + 5
        logging.addLevelName(INFO_OUT, "IINFO")
        def info_out(self, message, *args, **kws):
            self._log(INFO_OUT, message, args, **kws)
        logging.Logger.iinfo = info_out

        INFO_EXT = logging.INFO + 1
        logging.addLevelName(INFO_EXT, "INFO")
        def einfo_out(self, message, *args, **kws):
            self._log(INFO_EXT, message, args, **kws)
        logging.Logger.einfo = einfo_out

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
    def disable_tqdm(cls) -> None:
        cls._tqdm_off = True

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
        if logger.getEffectiveLevel() >= logging.INFO and not cls._tqdm_off:
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
