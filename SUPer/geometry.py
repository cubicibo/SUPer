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

from dataclasses import dataclass
from typing import Self, Sequence

@dataclass(frozen=True)
class Point:
    y: int
    x: int

    def __post_init__(self) -> None:
        assert self.y >= 0 and self.x >= 0

    def __add__(self, other: Self | Sequence[int]) -> Self:
        if isinstance(other, __class__):
            return Point(self.y + other.y, self.x + other.x)
        elif isinstance(other, (tuple, list)) and len(other) == 2:
            return Point(self.y + other[0], self.x + other[1])
        return NotImplemented

    def __sub__(self, other: Self | Sequence[int]) -> Self:
        if isinstance(other, __class__):
            return Point(self.y - other.y, self.x - other.x)
        elif isinstance(other, (tuple, list)) and len(other) == 2:
            return Point(self.y - other[0], self.x - other[1])
        return NotImplemented

@dataclass(frozen=True)
class Shape:
    h: int
    w: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'h', int(self.h))
        object.__setattr__(self, 'w', int(self.w))
        assert self.h >= 0 and self.w >= 0

    @classmethod
    def union(cls, *rects) -> Self:
        w = max(map(lambda dim: dim.w, rects))
        h = max(map(lambda dim: dim.h, rects))
        return cls(h, w)

    @property
    def area(self) -> int:
        return self.w*self.h

    @property
    def width(self) -> int:
        return self.w

    @property
    def height(self) -> int:
        return self.h

    def __or__(self, other) -> Self:
        if isinstance(other, __class__):
            return __class__(max(self.h, other.h), max(self.w, other.w))
        if isinstance(other, tuple) and len(other) == 2:
            return __class__(max(self.h, other[0]), max(self.w, other[1]))
        return NotImplemented

    def __and__(self, other) -> Self:
        if isinstance(other, __class__):
            return __class__(min(self.h, other.h), min(self.w, other.w))
        if isinstance(other, tuple) and len(other) == 2:
            return __class__(min(self.h, other[0]), min(self.w, other[1]))
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.w == other.w and self.h == other.h
        elif isinstance(other, (tuple, list)) and len(other) == 2:
            return self.h == other[0] and self.w == other[1]
        return NotImplemented

    def __ne__(self, other):
        if isinstance((test_eq := self.__eq__(other)), bool):
            return not test_eq
        return NotImplemented

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
        return (self.y, self.y2, self.x, self.x2)

    @property
    def shape(self) -> Shape:
        return Shape(self.dy, self.dx)

    @property
    def anchors(self) -> Point:
        return Point(self.y, self.x), Point(self.y2, self.x2)

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
    def from_slices(cls, slices: tuple[slice]) -> 'Box':
        coords = [0]*4
        f_extract_z_z1 = lambda slz : (int(slz.start), int(slz.stop))
        coords[::2] = f_extract_z_z1(slices[1])
        coords[1::2] = f_extract_z_z1(slices[0])
        return cls.from_coords(*coords)

    def intersect(self, *boxes) -> 'Box':
        boxes = [self] + [*boxes] * bool(isinstance(self, __class__) and len(boxes))
        x2 = min(map(lambda b: b.x2, boxes))
        y2 = min(map(lambda b: b.y2, boxes))
        x1 = max(map(lambda b: b.x, boxes))
        y1 = max(map(lambda b: b.y, boxes))
        return __class__(y1, max((y2-y1), 0), x1, max((x2-x1), 0))

    def union(self, *boxes) -> 'Box':
        boxes = [self] + [*boxes] * bool(isinstance(self, __class__) and len(boxes))
        x2 = max(map(lambda b: b.x2, boxes))
        y2 = max(map(lambda b: b.y2, boxes))
        x1 = min(map(lambda b: b.x, boxes))
        y1 = min(map(lambda b: b.y, boxes))
        return __class__(y1, y2-y1, x1, x2-x1)

    @classmethod
    def from_coords(cls, y: int, y2: int, x : int, x2: int) -> Self:
        return cls(min(y, y2), abs(y2-y), min(x, x2), abs(x2-x))

    @classmethod
    def from_layout(cls, coords) -> Self:
        return cls.from_coords(coords[1], coords[3], coords[0], coords[2])

    def to_absolute(self, parent: Self) -> Self:
        return self.__class__(parent.y + self.y, self.dy, parent.x + self.x, self.dx)

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, __class__):
            return self.coords == other.coords
        return NotImplemented

    def __and__(self, other: Self) -> Self:
        if isinstance(other, __class__):
            return self.intersect(other)
        return NotImplemented

    def __or__(self, other: Self) -> Self:
        if isinstance(other, __class__):
            return self.union(other)
        return NotImplemented

    def __neq__(self, other: Self) -> bool:
        if isinstance((is_equal := (self == other)), bool):
            return not is_equal
        return NotImplemented
####
