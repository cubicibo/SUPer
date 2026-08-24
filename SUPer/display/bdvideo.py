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

from typing import TypeVar, Sequence, Self
from enum import Enum
from fractions import Fraction
from dataclasses import dataclass

from .palette import Matrix

FormatInputT = TypeVar("FormatInputT", Sequence[int], int, str)

FramerateInputT = TypeVar('FramerateInputT', str, Fraction, int, float)

class Framerate(Enum):
    FPS_23976 = Fraction(24000, 1001)
    FPS_24    = Fraction(24, 1)
    FPS_25    = Fraction(25, 1)
    FPS_2997  = Fraction(30000, 1001)
    FPS_50    = Fraction(50, 1)
    FPS_5994  = Fraction(60000, 1001)
    FPS_60    = Fraction(60, 1)

    @classmethod
    def from_fps(cls, fps: FramerateInputT) -> Self:
        if isinstance(fps, (str, int, float)):
            fps = Fraction(fps)
            if round(Fraction(1001, 1000) * fps, 2) == round(fps):
                fps = Fraction(round(fps)*1000, 1001)
        if fps in cls._value2member_map_:
            return cls(fps)
        return None

    @classmethod
    def _missing_(cls, fps: FramerateInputT) -> Self:
        return cls.from_fps(fps)

    @classmethod
    def from_coded_frame_rate(cls, v: int) -> Self:
        if v not in range(1, 9) or v == 4:
            raise ValueError("Unknown coded frame rate.")
        v -= 2 if v >= 4 else 1
        return list(cls)[v]

    def to_coded_frame_rate(self) -> int:
        cls = __class__
        return {cls.FPS_23976: 1, cls.FPS_24: 2, cls.FPS_25: 3, cls.FPS_2997: 4,
                cls.FPS_50: 6, cls.FPS_5994: 7, cls.FPS_60: 8}[self]

    def __bytes__(self) -> bytes:
        return bytes(self.to_coded_frame_rate())

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        # the integer framerate is never truncated, always rounded to closest
        return round(self.value)

    def __round__(self, ndigits: int = 0):
        return round(self.value, ndigits)

    def __truediv__(self, other: FramerateInputT | Self) -> FramerateInputT:
        return self.value/other

    def __rtruediv__(self, other: FramerateInputT | Self) -> FramerateInputT:
        return other/self.value

    def __mul__(self, other: FramerateInputT | Self) -> FramerateInputT:
        return self.value*other

    def __rmul__(self, other: FramerateInputT | Self) -> FramerateInputT:
        return self.__mul__(other)

    def __gt__(self, other: FramerateInputT | Self) -> bool:
        if isinstance(other, __class__):   return self.value > other.value
        elif isinstance(other, str):       return self.value < __class__(other)
        elif isinstance(other, (int, float, Fraction)): return self.value > other
        return NotImplemented

    def __lt__(self, other: FramerateInputT | Self) -> bool:
        if isinstance(other, __class__):   return self.value < other.value
        elif isinstance(other, str):       return self.value < __class__(other)
        elif isinstance(other, (int, float, Fraction)): return self.value < other
        return NotImplemented

#%%
class Format(Enum):
    VIDEO_480 = 720, 480
    VIDEO_576 = 720, 576
    VIDEO_720 = 1280, 720
    VIDEO_1080 = 1920, 1080

    @classmethod
    def from_height(cls, height: int) -> Self:
        for dim, enfo in cls._value2member_map_.items():
            if dim[1] == height:
                return cls(enfo)
        assert ValueError("Illegal BD video format.")

    @classmethod
    def from_string(cls, fmt: str) -> 'Format':
        """
        from "1080" or "720p" or "480i", figure out the format
        """
        fmt = fmt.strip()
        idx = len(fmt)
        while (idx := idx - 1) > 0 and fmt[idx].isalpha():
            pass
        if idx == 0:
            raise ValueError("Incorrect video format.")
        return cls.from_height(int(fmt[:idx+1]))

    @property
    def width(self) -> int:
        return self.value[0]

    @property
    def height(self) -> int:
        return self.value[1]

    @property
    def area(self) -> int:
        return self.width*self.height

    @classmethod
    def _missing_(cls, v: FormatInputT) -> Self:
        if isinstance(v, list):
            return cls(tuple(v))
        if isinstance(v, int):
            return cls.from_height(v)
        if isinstance(v, str):
            return cls.from_string(v)
        return None

    def __eq__(self, o: ...) -> bool:
        if isinstance(o, self.__class__):
            return o.value == self.value
        elif isinstance(o, (tuple, list)):
            return o[0] == self.value[0] and o[1] == self.value[1]
        return NotImplemented

    def __ne__(self, o: ...) -> bool:
        if isinstance((is_eq := self.__eq__(o)), bool):
            return not is_eq
        return NotImplemented
####

@dataclass(frozen=True)
class BDVideo:
    fmt: Format | FormatInputT
    fps: Framerate | FramerateInputT
    uhd_bd: bool = False
    matrix: Matrix | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fmt, Format):    object.__setattr__(self, 'fmt', Format(self.fmt))
        if not isinstance(self.fps, Framerate): object.__setattr__(self, 'fps', Framerate(self.fps))
        if self.fmt.height < 1080 and self.uhd_bd:
            raise ValueError("UHD BD requires a 1920x1080 video format.")
        if self.matrix is None:
            if self.uhd_bd is True:
                raise ValueError("UHD BD requires a colour-space conversion matrix.")
            object.__setattr__(self, 'matrix', Matrix('BT709') if self.fmt.height >= 720 else Matrix('BT601'))

    def validate(self) -> bool:
        match self.fmt.height:
            case 1080:
                return self.fps.value < 30 or self.uhd_bd
            case 720:
                return self.fps in [Framerate.FPS_23976, Framerate.FPS_24, Framerate.FPS_50, Framerate.FPS_5994]
            case 576:
                return self.fps == Framerate.FPS_25
            case 480:
                return self.fps == Framerate.FPS_2997
            case _:
                raise NotImplementedError("Unknown video format.")
