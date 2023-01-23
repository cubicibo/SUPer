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

from typing import Optional, Callable, TypeVar, Union

import logging
import numpy as np
from numpy import (typing as npt)
from datetime import datetime, timezone
from collections import namedtuple
from enum import Enum, IntEnum
from PIL import Image

#ImageEvent is the common container for the Optimiser module.
ImageEvent = namedtuple("ImageEvent", "img event")
Shape = namedtuple("Shape", "width height")
Dim = namedtuple("Dim", "w h")
Pos = namedtuple("Pos", "x y")
_BaseEvent = TypeVar('BaseEvent')

# Elementary plane initialisation time function
_pinit_fn = lambda shape: np.ceil(90e3*(shape.width*shape.height/(32*1e6)))


def min_enclosing_cube(group: list[_BaseEvent], *, _retwh=True) -> npt.NDArray[np.uint8]:
    pxtl, pytl = np.inf, np.inf
    pxbr, pybr = 0, 0
    for event in group:
        if event.x < pxtl:
            pxtl = event.x
        if event.y < pytl:
            pytl = event.y
        if pxbr < event.x + event.width:
            pxbr = event.x + event.width
        if pybr < event.y + event.height:
            pybr = event.y + event.height
    return Pos(pxtl, pytl), Dim(pxbr-pxtl, pybr-pytl)

def merge_events(group: list[_BaseEvent], pos: Pos, dim: Dim) -> Image.Image:
    img_plane = np.zeros((dim.h, dim.w, 4), dtype=np.uint8)
    for k, event in enumerate(group):
        slice_x = slice(event.x-pos.x, event.x-pos.x+event.width)
        slice_y = slice(event.y-pos.y, event.y-pos.y+event.height)
        img_plane[slice_y, slice_x, :] = np.asarray(event.img).astype(np.uint8)
    return Image.fromarray(img_plane).convert('RGBA')


class BDVideo:
    class FPS(Enum):
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
            the final value is rounded up (so 24 is chosen).
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
        HD1080_43 = (1440, 1080)
        HD720     = (1280, 720)
        SD576_169 = (1024, 576) #Probably illegal
        SD480_169 = (856,  480) #Probably illegal
        SD576_43  = (720,  576)
        SD480_43  = (720,  480)

    class PCSFPS(IntEnum):
        FILM_NTSC_P = 0x10
        FILM_24P    = 0x20
        PAL_P       = 0x30
        NTSC_P      = 0x40
        PAL_I       = 0x60
        NTSC_I      = 0x70

        @classmethod
        def from_fps(cls, other: float):
            return BDVideo.LUT_PCS_FPS[np.round(other, 3)]

    LUT_PCS_FPS = {
        23.976:0x10,
        24:    0x20,
        25:    0x30,
        29.97: 0x40,
        50:    0x60,
        59.94: 0x70,
    }

class TimeConv:
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

    @staticmethod
    def f2s(f: int, fps: float, ndigits: int = 6) -> float:
        return round(f/fps, ndigits=ndigits)

    @staticmethod
    def s2tc(s: float, fps: float) -> str:
        h = int(s//3600)
        m = int((s % 3600)//60)
        sec = int((s % 60))
        fc = round((s-int(s))*fps)
        return f"{h:02}:{m:02}:{sec:02}:{fc:02}"

    @classmethod
    def tc2s(cls, tc: str, fps: float, *, ndigits: int = 6) -> float:
        dt =  tc[:(fpos := tc.rfind(':'))]
        dtts = datetime.strptime(dt, '%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        dtts += cls.f2s(int(tc[fpos+1:]), fps, ndigits=ndigits)
        return round(dtts-datetime.strptime('0:0:0', '%H:%M:%S').replace(tzinfo=timezone.utc).timestamp(), ndigits=ndigits)

    @classmethod
    def ms2tc(cls, ms: int, fps: float) -> str:
        return cls.s2tc(ms/1000, fps)

    @classmethod
    def tc2ms(cls, tc: str, fps: float) -> int:
        return int(cls.tc2s(tc, fps)*1000)

    @classmethod
    def tc2f(cls, tc: str, fps: float, *, add_one: bool = False) -> int:
        return cls.s2f(cls.tc2s(tc, fps), fps)

    @classmethod
    def f2tc(cls, f: int, fps: float, *, add_one: bool = False) -> str:
        return cls.s2tc(cls.f2s(f, fps), fps)

    @classmethod
    def tc_addf(cls, tc: str, f: int, fps: float) -> str:
        return cls.f2tc(cls.tc2f(tc, fps) + f, fps)

    @classmethod
    def tc_adds(cls, tc: str, s: float, fps: float) -> str:
        return cls.s2tc(cls.tc2s(tc, fps) + s, fps)

    @classmethod
    def tc_addtc(cls, tc1: str, tc2: str, fps: float) -> str:
        return cls.f2tc(cls.tc2f(tc1, fps)+cls.tc2f(tc2, fps), fps)

    @classmethod
    def tc_addms(cls, tc: str, ms: int, fps: float) -> str:
        return __class__.ms2tc(__class__.tc2ms(tc, fps) + ms, fps)

    @staticmethod
    def pgs2ms(pgts: int, *, _round = lambda x: int(round(x))) -> int:
        return int(_round(pgts/90))

    @staticmethod
    def ms2pgs(ms: float, *, _round = lambda x: int(round(x))) -> int:
        return int(_round(ms*90))

    @classmethod
    def tc2pgs(cls, tc: str, fps: float) -> int:
        return cls.ms2pgs(cls.tc2ms(tc, fps))

    @classmethod
    def pgs2tc(cls, pgts: int, fps: float) -> str:
        return cls.ms2tc(cls.pgs2ms(pgts, _round=lambda a: a), fps)

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
                  'y2r_f': np.array([[1,           0,  1.400, 0],
                                     [1,      -0.343, -0.711, 0],
                                     [1,       1.765,      0, 0],
                                     [0,           0,      0, 1]]),
                  'r2y_l': np.array([[ 0.257,  0.504,  0.098, 0],
                                     [-0.148, -0.291,  0.439, 0],
                                     [ 0.439, -0.368, -0.071, 0],
                                     [     0,      0,      0, 1]]),
                  'r2y_f': np.array([[ 0.299,  0.587,  0.114, 0],
                                     [-0.169, -0.331,  0.500, 0],
                                     [ 0.500, -0.419, -0.081, 0],
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

def get_super_logger(name: str, level: int = logging.INFO):
  """ Example of a custom logger.

    This function takes in two parameters: name and level and logs to console.
    The place to log in this case is defined by the handler which we set
    to logging.StreamHandler().

    Args:
      name: Name for the logger.
      level: Minimum level for messages to be logged
  """
  logger = logging.getLogger(name)
  logger.setLevel(level)

  if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(' %(name)s: %(levelname).4s : %(message)s'.format(name))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  return logger
