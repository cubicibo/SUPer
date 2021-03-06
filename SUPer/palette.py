#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""

from dataclasses import dataclass
from collections import namedtuple
from typing import Optional, Union
import logging

import numpy as np
from numpy import (typing as npt)

from .utils import get_matrix

RGBA = namedtuple('RGBA', ['r', 'g', 'b', 'a'])
FpPal = namedtuple('FpPal', 'y cb cr alpha')

def clip_ycbcr(ycbcra: npt.NDArray, s_range: str) -> npt.NDArray[np.uint8]:
    """
    Clip an array of YCxCyA values to s_range either {'limited', 'full'}

    :param ycbcra: Values to clip. Ideally as (N, 4) with entries stacked vertically.
    :param s_range: YUV range. PGS uses limited so should you.
    
    :return: Clipped values according to range. If 'full' generally input=output.
    """
    squeeze = False
    if ycbcra.ndim == 1:
        ycbcra = np.expand_dims(ycbcra, 0)
        squeeze = True
    
    if ycbcra.shape[1] != 4 and ycbcra.shape[0] == 4:
        ycbcra = ycbcra.T
    
    if 'full' not in s_range:
        logging.info("Clipping values to limited range.")
        ycbcra[:, :3][ycbcra[:, :3] <  16] = 16
        ycbcra[:,  0][ycbcra[:,  0] > 235] = 235
        ycbcra[:,1:3][ycbcra[:,1:3] > 240] = 240            
    ycbcra[ycbcra > 255] = 255
    ycbcra[ycbcra < 0] = 0
    
    if squeeze:
        ycbcra = ycbcra.squeeze()
    
    return ycbcra.astype(np.uint8)


def clip_rgba(rgba: npt.NDArray) -> npt.NDArray[np.uint8]:
    """
    Clip RGBA values to uint8 range before casting the array.
    :param rgba: array of RGBA values, whatever the shape.
    :return: array of rgba values clipped and casted.
    """
    rgba[rgba < 0] = 0
    rgba[rgba > 255] = 255
    return rgba.astype(np.uint8)


@dataclass
class PaletteEntry:
    y : int
    cr: int
    cb: int
    alpha: int
    
    def to_rgba(self, matrix: str ='bt709', /, *,
                s_range: str = 'limited'):
        """
        :param matrix: BT ITU conversion to use
        :param s_range: YUV space
        :return: RGBA equivalent.
        """
        corr = 0 if 'full' in s_range else 16
        pe = FpPal(self.y-corr, self.cb-128, self.cr-128, self.alpha)
        
        rgba_v = np.matmul(get_matrix(matrix, True, s_range), np.asarray([[*pe]]).T)
        return RGBA(*clip_rgba(np.round(rgba_v)).reshape(4,))

    
    def __iter__(self):
        self.n = 0
        return self
    
    
    def __next__(self):
        if self.n < 4:
            self.n += 1
            return self[self.n-1]
        else:
            raise StopIteration
    
    
    def __getitem__(self, n: Union[int, slice]):
        return list([self.y, self.cr, self.cb, self.alpha])[n]
    
    
    def __copy__(self):
        return PaletteEntry(*self)
    
    
    def __bytes__(self):
        return bytes([self.y, self.cr, self.cb, self.alpha])
     
        
    def swap_cbcr(self) -> None:
        """
        Swap CbCr because I am a fool and thought PGS used CbCr rather than CrCb.
        """
        self.cb, self.cr = self.cr, self.cb
    
    
    @classmethod
    def from_rgba(cls, rgba: Union[RGBA, tuple[int]], /, *, matrix: str = 'bt709',
                  s_range: str = 'limited'):
        """
        Construct a PaletteEntry from a RGBA value.
        :param rgba: rgba iterable
        :param matrix: BT ITU conversion to use.
        :param s_range: YUV range.
        """
        mat = get_matrix(matrix, False, s_range)
        pe = np.round(np.matmul(mat, np.asarray(rgba).T)).T
        pe = pe + np.asarray([0 if 'full' in s_range else 16, 128, 128, 0])
        
        ret = cls(*clip_ycbcr(pe.astype(np.int32), s_range))
        ret.swap_cbcr()
        
        return ret

@dataclass
class Palette:
    id : int
    v_num : int
    palette : dict[int, PaletteEntry]   
    
    def __len__(self):
        return len(self.palette)
    

    def __getitem__(self, id: int) -> PaletteEntry:
        if id not in self.palette or 0 < id > 255:
            raise KeyError(f"Palette entry {id} is incorrect or does not exist.")
        return self.palette[id]
      
                  
    def __iter__(self):
        self.n = 0
        return self
    

    def __next__(self):
        if self.n < len(self.palette.keys()):
            pe = self[list(self.palette.keys())[self.n]]
            self.n += 1
            return pe
        raise StopIteration
       
         
    def __bytes__(self):
        bpal = bytearray()
        for idx, entry in self.palette.items():
            bpal += bytes([idx]) + bytes(entry)
        return bytes(bpal)
      
      
    def __setitem__(self, id: int, entry: PaletteEntry) -> None:
        if 0 <= id <= 255:
            if id == 0:
                logging.info("Changing default transparent color to something else.")
            if min(*entry[:3]) < 16 or entry.y > 235 or max(*entry[1:3]) > 240:
                logging.warning("Palette clamps outside limited YCbCr range.")
            if isinstance(entry, PaletteEntry):
                self.palette[id] = entry
            else:
                self.palette[id] = PaletteEntry(*entry)
        else:
            raise KeyError(f"Tried to set {id} entry, outside of [0;255].")
    
    
    def get(self, idx: int, default = None):
        try:
            return self[idx]
        except KeyError:
            return default
    
    
    def pop(self, idx: int) -> PaletteEntry:
        return self.palette.pop(idx)


    def get_ycbcr(self, /, *, _no_key = False) -> npt.NDArray[np.uint8]:
        """
        Get palette as an array of YCbCr values.
        :param _no_key: remove the entry index from the array.
        :return: (len(pal),3+1) shape array
        """
        if _no_key:
         return np.array([(p.y, p.cb, p.cr) for p in self.palette.values()])
        return np.array([(k, p.y, p.cb, p.cr) for k, p in self.palette.items()])

    
    def get_ycbcra(self, /, *, _no_key = False) -> npt.NDArray[np.uint8]:
        """
        Get palette as an array of YCbCrA values.
        :param _no_key: remove the entry index from the array.
        :return: (len(pal),4+1) shape array
        """
        if _no_key:
         return np.array([(p.y, p.cb, p.cr, p.alpha) for p in self.palette.values()])
        return np.array([(k, p.y, p.cb, p.cr) for k, p in self.palette.items()])


    def get_alpha(self, /, *, _no_key = False) -> npt.NDArray[np.uint8]:
        """
        Get the alpha values of the entire palette.
        :param _no_key: remove the entry index from the array.
        :return: (len(pal),1+1) shape array
        """
        if _no_key:
            return np.array([a.alpha for a in self.palette.values()])
        return np.array([(k, a.alpha) for k, a in self.palette.items()])
    
    
    @classmethod
    def from_rgba(cls,
                  rgba: Union[list, bytes, bytearray,
                              dict[int, Union[npt.NDArray[np.uint8], tuple]]], /, *,
                  prev_pal: Optional['Palette'] = None, matrix: str = 'bt709',
                  s_range: str= 'limited', **kwargs):
        """
        Construct a Palette from a mapping or an iterable, like Pillow ImagePalette.
        :param rgba: Iterable list or dictionary with RGBA entries.
        :param prev_pal: Previous palette, used as a base to build the new one.
          This is handy to deal when rgba defines just updated entries.
        :param matrix: BT ITU conversion
        :param s_range: YUV range
        :param kwargs: Additional parameters for palette version and number.
        :return: Palette object
        """
        
        cmat = get_matrix(matrix, False, s_range)
        if prev_pal is not None:
            kwargs['id'] = prev_pal.id
            kwargs['v_num'] = (prev_pal.v_num + 1) % 256
        
        new_dc = {} if not prev_pal else prev_pal.palette
        new_pal = cls(kwargs.get('id', -1), kwargs.get('v_num', -1), new_dc)
        
        offset = np.asarray([0 if 'full' in s_range else 16, 128, 128, 0]).T
        
        if type(rgba) is dict:
            for k, v in rgba.items():
                tmp = clip_ycbcr(np.matmul(cmat, np.asarray(v).T) + offset, s_range)
                new_pal[k] = PaletteEntry(*tmp)
                
        elif type(rgba) in [list, bytes, bytearray]:
            assert len(rgba) % 5 == 0,"Expected [Id1 Y Cb Cr A Id2 Y ...] structure."
            for k in range(0, len(rgba), 5):
                tmp = np.matmul(cmat, np.asarray(rgba[k+1:k+5]).T) + offset
                new_pal[rgba[k]] = PaletteEntry(*clip_ycbcr(tmp, s_range))
                new_pal[rgba[k]].swap_cbcr()
        else:
            raise NotImplementedError("Unknown rgba variable type.")
        return new_pal


    def to_rgb(self, matrix: str = 'bt709',
               s_range: str = 'limited') -> dict[int, tuple[int]]:
        """
        Construct a Palette from a mapping or an iterable, like Pillow ImagePalette.
        :param rgba: Iterable list or dictionary with RGBA entries.
        :param prev_pal: Previous palette, used as a base to build the new one.
          This is handy to deal when rgba defines just updated entries.
        :param matrix: BT ITU conversion
        :param s_range: YUV range
        :param kwargs: Additional parameters for palette version and number.
        :return: Palette object
        """
        cmat = get_matrix(matrix, True, s_range)[:3,:3]
        
        ycbcr = self.get_ycbcr(_no_key=True).astype(float).reshape((-1, 3)).T
        ycbcr[[1,2],:] -= 128
        if 'full' not in s_range: ycbcr[[0],:] -= 16
        t = np.round(np.dot(cmat, ycbcr)).T
        t = clip_rgba(t)
        
        return dict(map(lambda it: (it[0], tuple(it[1])), zip(self.palette.keys(), t)))


    def to_rgba(self, matrix: str = 'bt709',
                s_range: str = 'limited') -> dict[int, tuple[int]]:
        """
        Export a palette to a RGBA mapping
        :param matrix: BT ITU conversion
        :param s_range: YUV range
        :return: Mapping with, as key the palette entry ID and value: RGBA tuple.
        """
        cmat = get_matrix(matrix, True, s_range)
        
        ycbcra = self.get_ycbcra(_no_key=True).astype(float).reshape((-1, 4)).T
        ycbcra[[1,2],:] -= 128
        if 'full' not in s_range: ycbcra[[0],:] -= 16
        t = np.round(np.dot(cmat, ycbcra)).T
        t = clip_rgba(t)
        
        return dict(map(lambda it: (it[0], tuple(it[1])), zip(self.palette.keys(), t)))