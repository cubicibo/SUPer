#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""

from typing import Optional, Callable

import numpy as np
from numpy import (typing as npt)
from dataclasses import dataclass
from datetime import datetime
from collections import namedtuple
from enum import Enum, IntEnum

ImageEvent = namedtuple("ImageEvent", "img event")

class BDVideo:
    class FPS(Enum):
        NTSCi = 59.94
        PALi  = 50
        NTSCp = 29.97
        PALp  = 25
        FILM  = 24
        FILM_NTSC = 23.976
        
    class VideoFormat(Enum):
        HD1080    = (1920, 1080)
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
        
    LUT_PCS_FPS = {
        23.976:0x10,
        24:    0x20, 
        25:    0x30,
        29.97: 0x40,
        50:    0x60,
        59.94: 0x70,
        }

@dataclass
class PGSTarget:
    dim:    BDVideo.VideoFormat
    fps:    BDVideo.FPS
    h_pos:  int  = -1
    v_pos:  int  = -1
    forced: bool = False
    pal_id: int  = 0
    voffset:int  = 100

    @property
    def pos(self) -> tuple[int]:
        return (self.h_pos, self.v_pos)
    
    @property
    def width(self):
        return self.dim.value[0]
    
    @property
    def height(self):
        return self.dim.value[1]
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def readable_fps(self):
        return self._fps_true.value
        
    @fps.setter
    def fps(self, nfps: float) -> None:
        if getattr(nfps, 'value'):
            self._fps = BDVideo.PCSFPS(BDVideo.LUT_PCS_FPS.get(nfps.value))
        else:
            self._fps = BDVideo.PCSFPS(BDVideo.LUT_PCS_FPS.get(nfps))
        self._fps_true = BDVideo.FPS(nfps)


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
    def f2s(f: int, fps: float, ndigits: int = 3) -> float:
        return round(f/fps, ndigits=ndigits)
                    
    @staticmethod
    def s2tc(s: float, fps: float) -> str:
        h = int(s//3600)
        m = int((s % 3600)//60)
        sec = int((s % 60))
        fc = int((s-int(s))*fps)
        return f"{h:02}:{m:02}:{sec:02}:{fc:02}"
 
    @classmethod
    def tc2s(cls, tc: str, fps: float, *, ndigits: int = 3) -> float:
        dt =  tc[:(fpos := tc.rfind(':'))]
        dtts = datetime.strptime(dt, '%H:%M:%S').timestamp()
        dtts += cls.f2s(int(tc[fpos+1:]), fps, ndigits=ndigits)
        return round(dtts-datetime.strptime('0:0:0', '%H:%M:%S').timestamp(), ndigits=ndigits)
        
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
        return int(round(pgts/90))
    
    @staticmethod
    def ms2pgs(ms: float, *, _round = lambda x: int(round(x))) -> int:
        return int(round(ms*90))

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
    }
    if to_rgba:
        mat = cc_matrix.get(matrix, {}).get(f"y2r_{range[0]}", None)
    else:
        mat = cc_matrix.get(matrix, {}).get(f"r2y_{range[0]}", None)

    if mat is None:
        raise NotImplementedError("Unknown/Not implemented conversion standard.")
    return mat

