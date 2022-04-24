#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""

from numpy import typing as npt
import numpy as np
from typing import Union
from enum import Enum

# from PIL import Image

# from segments import CObject
# from .palette import Palette

class PGraphics:          
    def encode_rle(bitmap: npt.NDArray[np.uint8]) -> bytes:
        """
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param bitmap:    Palette mapped image to encode (2d array)
        :return:          Encoded data (vector)
        """
        rle_data = bytearray()
                
        _bitmap = np.squeeze(bitmap)
        _fp = np.ravel(_bitmap)
        i = 0
        insert_line_end = False
                
        while i < _fp.size:
            for k in range(1, 16384):
                if (i % _bitmap.shape[1]) + k >= _bitmap.shape[1]:
                    insert_line_end = True
                    break
                
                if _fp[i+k] != _fp[i]:
                    break
    
            if _fp[i] != 0: #color
                if k < 3:
                    rle_data += bytearray([_fp[i]]*k)
                elif k <= 63:
                    rle_data += bytearray([0, 0x80 | k, _fp[i]])
                else:
                    rle_data += bytearray([0, 0xC0 | (k >> 8), k&0xFF, _fp[i]])
            else: #transparent
                if k <= 63:
                    rle_data += bytearray([0, k])
                else:
                    rle_data += bytearray([0, 0x40 | (k >> 8), k&0xFF])
                                        
            if insert_line_end:
                rle_data += bytearray([0, 0])
                insert_line_end = False
            i += k
        return bytes(rle_data)
    
    def decode_rle(rle_data: Union[bytes, bytearray]) -> npt.NDArray[np.uint8]:   
        """
        Decode a RLE object, as defined in 'US 7912305 B1' patent.
        :param rle_data:  Data to decode
        :return:          2D map to associate with the proper palette
        """
        class RLEDecoderState(Enum):
            NEED_MORE = -2
            NEW_CODE  = -1
            SMALL_TSP = 0
            LARGE_TSP = 1
            SMALL_CCO = 2
            LARGE_CCO = 3
        
        plane2d, line_l = [], []
        decoder_state = RLEDecoderState.NEW_CODE
        tmp = 0
    
        # Always use a state machine, even in place where you totally don't need it.
        for byte in rle_data:      
            if decoder_state == RLEDecoderState.NEW_CODE:
                if byte > 0:
                    line_l.append(byte)
                else:
                    decoder_state = RLEDecoderState.NEED_MORE
                
            elif decoder_state == RLEDecoderState.NEED_MORE:
                if byte == 0:
                    plane2d.append(line_l)
                    line_l = []
                    decoder_state = RLEDecoderState.NEW_CODE
                else: 
                    decoder_state = RLEDecoderState(byte >> 6)
                    tmp = byte & 0x3F
    
                    if decoder_state == RLEDecoderState.SMALL_TSP:
                        line_l.extend([0] * tmp)
                        decoder_state = RLEDecoderState.NEW_CODE
    
            elif decoder_state == RLEDecoderState.LARGE_TSP:
                tmp = (tmp << 8) + byte
                line_l.extend([0]*tmp)
                decoder_state = RLEDecoderState.NEW_CODE
            
            elif decoder_state == RLEDecoderState.SMALL_CCO:
                line_l.extend([byte]*tmp)
                decoder_state = RLEDecoderState.NEW_CODE
            
            elif decoder_state == RLEDecoderState.LARGE_CCO:
                #first pass (some RLE encoders use long code for small distances
                # hence we must check for equal zero...)
                if tmp >= 0:
                    tmp = ((tmp << 8) + byte)*-1
                else: #second pass
                    line_l.extend([byte]*(-1*tmp))
                    decoder_state = RLEDecoderState.NEW_CODE
        return np.asarray(plane2d)


# class GraphicsPlane:
#     def __init__(self, shape: tuple[int, int]) -> None:
#         self.h = shape[0]
#         self.w = shape[1]
        
#         self._plane = np.zeros(shape, dtype=np.uint8)
#         self.palette: Palette = None
        
#     def put(self, data, v_pos, h_pos) -> None:
#         sl = lambda idx, length : slice(idx, idx+length)
#         self._plane[sl(h_pos, data.shape[0]), sl(v_pos, data.shape[1])] = data
                
#     def to_rgba(self, matrix: str = 'bt709') -> Image:
#         if self.palette:
#             return cmap_to_img(self._plane, self.palette, matrix)
#         raise AttributeError("Palette not set.")
        
#     def flush(self) -> None:
#         self._plane[:] = 0
#         self.palette = None
    
# class ObjectBuffer:
#     def __init__(self):
#         # roughly 4 MiB plane
#         self.plane = np.zeros((2, 1920, 1080), dtype=np.uint8)
    
#     def put(self, obj: CObject, plane: int) -> None:
#         self.plane[plane,:,:] = self.decode_rle(obj)
      


# def rgba_to_cmap(img: Image, palette: Optional[Palette] = None, colors: Optional[np.uint8] = 250) -> Image:
#     """
#     Converts a RGBA image to a color map usable in a PGS stream
#     :param img:      RGBA PIL Image object
#     :param palette:  RGBA palette to use to quantize the RGBA image.
#                         If not provided PIL will find the palette itself.
#     :param colors:   Number of colors to use.
#                         PGS supports 255 + 1 but you may constrain it to 250.
                     
#     :return:          "P" Image with palette array in 'palette' attribute.
#     """
    
#     if 2 <= colors > 255:
#         raise ValueError("Too few/many colors to quantize to. Expected value to lie within [2;255].")
    
#     pal = bytes(palette) if palette is not None else palette
#     return img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE, palette=pal, dither=Image.Dither.NONE)

# def cmap_to_img(cmap: npt.NDArray[np.uint8], palette: Palette, matrix: str = 'bt709') -> Image:
#     alpha = Image.fromarray(palette.get_alpha(), mode='L')
#     img = Image.fromarray(cmap, mode='P')
#     img.putpalette(palette.to_rgb(matrix))
#     img.putalpha(alpha)
#     return img

