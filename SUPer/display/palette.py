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


from dataclasses import dataclass
from itertools import starmap
from typing import Self, TypeVar

import numpy as np

ColorMatrixT: type = np.ndarray[tuple[int, int], np.dtype[float]]
ColorVectorT = TypeVar("ColorVectorT", np.ndarray[tuple[int, int], np.dtype[np.uint8]],
                                       np.ndarray[tuple[int, int, int], np.dtype[np.uint8]])

_csp_matrices = {
    'BT601':
        np.array([[ 0.257,  0.504,  0.098, 0],[-0.148, -0.291,  0.439, 0],
                  [ 0.439, -0.368, -0.071, 0],[     0,      0,      0, 1]]),
    'BT709':
        np.array([[ 0.183,  0.614,  0.062, 0],[-0.101, -0.339,  0.439, 0],
                  [ 0.439, -0.399, -0.040, 0],[     0,      0,      0, 1]]),
    'BT2020':
        np.array([[0.22561,0.58228,0.05093,0],[-.12266,-.31656,0.43922,0],
                  [0.43922,-.40389,-.03533,0],[      0,      0,      0,1]]),
}

class Matrix:
    def __init__(self, name: str | Self | int):
        if isinstance(name, (int, str)):
            self._cspm, self._name = self.__class__.from_string(name)
        else:
            self._cspm, self._name = name._cspm, name.name
        self._icspm = None

    def __call__(self, cv: ColorVectorT, to_ycc: bool = True) -> ColorVectorT:
        if to_ycc:
            return np.matmul(self._cspm, cv)
        return np.matmul(self.inverse(), cv)

    def forward(self) -> ColorMatrixT:
        return self._cspm

    def inverse(self) -> ColorMatrixT:
        if self._icspm is None:
            self._icspm = np.linalg.inv(self._cspm)
        return self._icspm

    @classmethod
    def from_string(cls, name: str | int) -> Self:
        if isinstance(name, int) or isinstance(name, str) and name.isnumeric():
            name = 'BT' + str(name)
        elif isinstance(name, str):
            name = name.strip().replace('.', '').upper()
        return _csp_matrices.get(name), name

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Matrix('{self.name}')"
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
    def __bytes__(self) -> bytes:
        bs = bytearray()
        for k in range(256):
            if (entry := self.get(k, None)) is not None:
                bs += bytes([k]) + bytes(entry)
        return bytes(bs)

    def offset(self, offset: int) -> Self:
        if len(self):
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
