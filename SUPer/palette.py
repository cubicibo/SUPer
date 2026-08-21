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

import numpy as np

from dataclasses import dataclass
from enum import Enum
from itertools import starmap
from typing import Self, TypeAlias, TypeVar

ColorMatrixT: TypeAlias = np.ndarray[tuple[int, int], np.dtype[float]]
ColorVectorT = TypeVar("ColourVectorT", np.ndarray[tuple[int, int], np.dtype[np.uint8]],
                                         np.ndarray[tuple[int, int, int], np.dtype[np.uint8]])

class _MatrixMeta(type):
    def __new__(cls, name: str, bases: tuple, dct: dict) -> Self:
        dct |= {'_name': name, '_imatrix': None}
        return type.__new__(cls, name, bases, dct)

    def convert(self, cv: ColorVectorT, to_ycc: bool = True) -> ColorVectorT:
        if to_ycc:
            return np.matmul(self.matrix, cv)
        return np.matmul(self._imatrix, cv)

    @property
    def name(self) -> str:
        return self._name

class Matrix(Enum):
    BT601 = _MatrixMeta('BT601', (), {'matrix': np.array([[ 0.257,  0.504,  0.098, 0],[-0.148, -0.291,  0.439, 0],
                                                          [ 0.439, -0.368, -0.071, 0],[     0,      0,      0, 1]])})
    BT709 = _MatrixMeta('BT709', (), {'matrix': np.array([[ 0.183,  0.614,  0.062, 0],[-0.101, -0.339,  0.439, 0],
                                                          [ 0.439, -0.399, -0.040, 0],[     0,      0,      0, 1]])})
    BT2020 =_MatrixMeta('BT2020',(), {'matrix': np.array([[0.22561,0.58228,0.05093,0],[-.12266,-.31656,0.43922,0],
                                                          [0.43922,-.40389,-.03533,0],[      0,      0,      0,1]])})
    
    def __call__(self, cv: ColorVectorT, to_ycc: bool = True) -> ColorVectorT:
        return self.value.convert(cv, to_ycc)
    
    def forward(self) -> ColorMatrixT:
        return self.value.matrix
    
    def inverse(self) -> ColorMatrixT:
        if self.value._imatrix is None:
            self.value._imatrix = np.linalg.inv(self.value.matrix)
        return self.value._imatrix

    @classmethod
    def _missing_(cls, v: ...) -> Self:
        if isinstance(v, (int, str)):
            return cls.from_string(v)
        return None

    @classmethod
    def from_string(cls, name: str | int) -> Self:
        if isinstance(name, int) or isinstance(name, str) and name.isnumeric():
            name = 'BT' + str(name)
        elif isinstance(name, str):
            name = name.strip().replace('.', '').upper()
        return next(filter(lambda v: v.name == name, cls), None)
####

@dataclass
class PaletteEntry:
    Y: int
    Cr: int
    Cb: int
    A: int
    
    def __post_init__(self) -> int:
        self.Y = int(self.Y)
        self.Cr = int(self.Cr)
        self.Cb = int(self.Cb)
        self.A = int(self.A)
    
    def __bytes__(self) -> bytes:
        return bytes([self.Y, self.Cr, self.Cb, self.A])

class Palette(dict):
    def sort(self) -> None:
        self = __class__(sorted(self.items(), key=lambda x: x[0]))
        
    def __bytes__(self) -> bytes:
        bs = bytearray()
        for k in range(256):
            if (entry := self.get(k, None)) is not None:
                bs += bytes([k]) + bytes(entry)
    
    def offset(self, offset: int) -> Self:
        assert min(self) + offset >= 0 and max(self) + offset < 256
        return __class__((k+offset, v) for k, v in self.items())
    
    def to_rgba_array(self, matrix: Matrix) -> np.ndarray[tuple[int, int], np.uint8]:
        ycbcra = np.zeros((256, 4), np.int32)
        for k, v in self.items():
            ycbcra[k, :] = (v.Y,v.Cb,v.Cr,v.A)
        ycbcra -= np.asarray([[16, 128, 128, 0]])
        rgba = np.round(np.matmul(ycbcra.reshape((-1, 4)), matrix.inverse()))
        clip_vals = (np.array([[0, 0, 0, 0]]), np.asarray([[255, 255, 255, 255]]))
        return np.clip(rgba, *clip_vals).astype(np.uint8)

    @classmethod
    def from_ycrcba_array(cls, ycrcba: np.ndarray[tuple[int, int], np.uint8]) -> Self:
        return __class__(zip(range(ycrcba.shape[0]), starmap(PaletteEntry, ycrcba)))
    
    @classmethod
    def from_rgba_array(cls, rgba_array: np.ndarray[tuple[int, int], np.uint8], matrix) -> Self:
        return cls.from_stacked_rgba(np.expand_dims(rgba_array, -1), matrix)
    
    @classmethod
    def from_stacked_rgba(cls, cluts: np.ndarray[tuple[int, int, int], np.uint8], matrix: Matrix | str | int) -> list[Self]:
        matrix = Matrix(matrix)
        stacked_cluts = np.swapaxes(cluts, 1, 0).astype(np.int32)

        shape = stacked_cluts.shape
        stacked_cluts = np.round(np.matmul(stacked_cluts.reshape((-1, 4)), matrix.forward().T))
        stacked_cluts += np.asarray([[16, 128, 128, 0]])
        clip_vals = (np.array([[16, 16, 16, 0]]), np.asarray([[235, 240, 240, 255]]))
        stacked_cluts = np.clip(stacked_cluts, *clip_vals).astype(np.uint8).reshape(shape)
        #YCbCrA -> YCrCbA
        stacked_cluts = stacked_cluts[:, :, [0, 2, 1, 3]]
        
        return [Palette.from_ycrcba_array(clut) for clut in stacked_cluts]
####
