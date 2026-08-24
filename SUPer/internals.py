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
import logging
import numpy as np

from abc import ABC
from enum import IntEnum
from PIL import Image
from contextlib import nullcontext
from timecode import Timecode
from typing import TypeAlias, TypeVar

from .geometry import Box
from .display.bdvideo import Framerate, FramerateInputT

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = nullcontext

#%%
class GraphicsDecoder:
    RD = 16000000
    RC = 32000000
    FREQ = 90000

    @classmethod
    def get_object_duration(cls, object_area: int) -> int:
        return int(np.ceil(object_area * cls.FREQ / cls.RD))

    @classmethod
    def get_composition_duration(cls, window_area: int) -> int:
        return int(np.ceil(window_area * cls.FREQ / cls.RC))

#%%
TimecodeInputT: TypeAlias = TypeVar('TimecodeT', 'TC', str, int)
MPEGTick: TypeAlias = int

class TC(Timecode):
    def __init__(self, fps, *args, **kwargs) -> None:
        if not isinstance(fps, Framerate):
            fps = Framerate(fps)
        super().__init__(fps.value, *args, **kwargs)
        self.fractional_fps = fps

    @classmethod
    def s2tc(cls, s: float, fps: FramerateInputT, drop_frame: bool = False) -> 'TC':
        #Add 1e-8 to avoid wrong rounding
        s = s/(1 if float(fps).is_integer() else 1.001)
        if isinstance(fps, float):
            fps = round(fps, 2)
        r_tc = cls(fps, start_seconds=s+1/fps+1e-8, force_non_drop_frame=True)
        r_tc.drop_frame = drop_frame
        return r_tc

    def to_pts(self) -> MPEGTick:
        tpts = ((self.frames - 1)/self.fractional_fps.value)*GraphicsDecoder.FREQ
        return int(tpts)

    def to_seconds(self) -> float:
        return self.to_pts()/GraphicsDecoder.FREQ

    def __add__(self, other: TimecodeInputT) -> 'TC':
        if isinstance(other, __class__):
            assert other.fractional_fps == self.fractional_fps and self.drop_frame == other.drop_frame == False
            frames = other.frames
        else:
            assert isinstance(other, int)
            frames = other
        tc = __class__(self.fractional_fps, frames=1)
        tc.drop_frame = self.drop_frame
        tc.frames = self.frames + frames
        return tc

class _Masks(IntEnum):
    W39 = 0x7FFFFFFFFF
    W33 = 0x1FFFFFFFF
    W32 = 0xFFFFFFFF
    W24 = 0xFFFFFF
    W16 = 0xFFFF
    W8  = 0xFF

class _classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
    def __set__(self, cls, value):
        return classmethod(self.fset).__set__(None, value)()

class GfxCompositor(ABC):
    def get_bbox(self) -> Box:
        return Box.union(*list(map(lambda e: e.box, self.graphics)))

    def _combine_graphics(self) -> Image.Image:
        container = self.get_bbox()
        frame = Image.new('RGBA', (container.dx, container.dy), (0,0,0,0))

        for gfx in self.graphics:
            if (img := Image.open(gfx.filepath)).mode != 'RGBA':
                img = img.convert('RGBA')
            frame.paste(img, (gfx.box.x - container.x, gfx.box.y - container.y))
        return frame

    @property
    def image(self) -> Image.Image:
        if len(self.graphics) > 1:
            return self._combine_graphics()
        if (img := Image.open(self.graphics[0].filepath)).mode != 'RGBA':
            img = img.convert('RGBA')
        return img

class LogFacility:
    _logger = dict()
    _logpbar = dict()
    _logrep = None
    _tqdm_off = False

    @classmethod
    def set_file_log(cls, logger: logging.Logger, fp: str, level: int | None = None, simple_format: bool = False) -> None:
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
    def get_progress_bar(cls, logger: logging.Logger, tot: ...) -> tqdm | None:
        if cls._logger.get(logger.name, None) is None:
            return None
        if cls._logpbar.get(logger.name, None) is not None:
            return cls._logpbar[logger.name]
        if logger.getEffectiveLevel() >= logging.INFO and not cls._tqdm_off:
            pbar = tqdm(tot)
        else:
            pbar = nullcontext()
            pbar.n = 0
        # amend the null context with all of the functions we may access
        if getattr(pbar, 'update', None) is None:
            pbar.update = pbar.close = pbar.set_description = pbar.reset = pbar.refresh = pbar.clear = lambda *args, **kwargs: None
        cls._logpbar[logger.name] = pbar
        return pbar

    @classmethod
    def add_file_report(cls, logger, report_filename) -> None:
        cls._logrep = cls.get_logger('event_report', with_handler=False)
        cls.set_file_log(cls._logrep, report_filename, simple_format=True)
        cls.set_logger_level('event_report', cls._logrep.level)
        logger.addHandler(cls._logrep)

    @classmethod
    def dissociate_log_and_report(cls, logger) -> None:
        if cls._logrep is None:
            return
        for ih, hdl in enumerate(logger.handlers):
            if hdl == cls._logrep.handlers[0]:
                logger.handlers.pop(ih)
                break

    @classmethod
    def associate_log_and_report(cls, logger) -> None:
        if cls._logrep is None:
            return
        logger.addHandler(cls._logrep)