#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 cibo
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

import os
import re
import xml.etree.ElementTree as ET
import numpy as np

from numpy import typing as npt
from PIL import Image
from pathlib import Path
from io import BytesIO

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Union, Optional, Type, Any, Callable

from .segments import PGSegment, PCS, WDS, PDS, ODS, ENDS, DisplaySet, Epoch
from .utils import (BDVideo, TimeConv as TC, get_super_logger,
                    min_enclosing_cube, merge_events, Shape, Pos, Dim)

logging = get_super_logger('SUPer')

#%%
class SUPFile:
    """
    Represents a .SUP file that contains a (valid) PGS stream.
    """
    def __init__(self, fp: Union[Path, str], **kwargs) -> None:
        self.file = fp
        self.bytes_per_read = int(kwargs.pop('bytes_per_read', 1*1024**2))
        assert self.bytes_per_read > 0


    @property
    def file(self) -> str:
        return str(self._file)


    @file.setter
    def file(self, file: Union[Path, str]) -> None:
        if (file := Path(file)).exists():
            self._file = file
        else:
            raise OSError("File does not exist.")


    def gen_segments(self) -> Generator[Type[PGSegment], None, None]:
        """
        Returns a generator of PG segments. Stops when all segments in the
        file have been consumed. This is the main parsing function.

        :yield: Every segment, in order, as they appear in the SUP file.
        """
        with open(self.file, 'rb') as f:
            buff = f.read(self.bytes_per_read)
            while buff:
                renew = False
                try:
                    yield (pseg := PGSegment(buff).specialise())
                    buff = buff[len(pseg):]
                except (BufferError, EOFError):
                    renew = True
                except ValueError:
                    buff = buff[buff.find(PGSegment.MAGIC):]
                    #if b'P' is not at pos 0, then we can discard the last byte.
                    if buff[0] != PGSegment.MAGIC[0]:
                        buff = bytearray()
                if renew or not buff:
                    if not (new_data := f.read(self.bytes_per_read)):
                        break
                    buff = buff + new_data
            ####while
        ####with
        return


    def get_fps(self) -> BDVideo.FPS:
        pcs = next(self.gen_segments())
        assert isinstance(pcs, PCS)
        return BDVideo.FPS(BDVideo.FPS.from_pcsfps(pcs.fps))


    def get_video_format(self) -> BDVideo.VideoFormat:
        pcs = next(self.gen_segments())
        assert isinstance(pcs, PCS)
        return BDVideo.VideoFormat((pcs.width, pcs.height))


    def gen_displaysets(self) -> Generator[DisplaySet, None, None]:
        """
        Returns a generator of DisplaySets. Stops when all DisplaySets in the
        file have been consumed.

        :yield: DisplaySet, in order, as they appear in the SUP file.
        """
        condition = lambda seg: isinstance(seg, PCS)
        yield from __class__._gen_group(self.gen_segments(), condition, DisplaySet)


    def gen_epochs(self) -> Generator[Epoch, None, None]:
        condition = lambda ds: ds.pcs.composition_state & ds.pcs.CompositionState.EPOCH_START
        yield from __class__._gen_group(self.gen_displaysets(), condition, Epoch)


    def check_infos(self) -> tuple[BDVideo.VideoFormat, BDVideo.FPS]:
        """
        Verify FPS and Video Format consistancy across the entire file.
        """
        displaysets = self.gen_displaysets()
        ds0 = next(displaysets)

        width, height = ds0.pcs.width, ds0.pcs.height
        fps = ds0.pcs.fps

        for ds in displaysets:
            assert width == ds.pcs.width, "Width is not constant in SUP."
            assert height == ds.pcs.height, "Height is not constant in SUP."
            assert fps == ds.pcs.fps, "FPS is not constant in SUP."

        return BDVideo.VideoFormat((width, height)), BDVideo.FPS(BDVideo.FPS.from_pcsfps(fps))


    def segments(self) -> list[Type[PGSegment]]:
        """
        Get all PG segments contained in the file.
        """
        return list(self.gen_segments())


    def displaysets(self) -> list[DisplaySet]:
        """
        Get all displaysets in the given file.
        """
        return list(self.gen_displaysets())


    def epochs(self) -> list[Epoch]:
        """
        Get all epochs in the given file.
        """
        return list(self.gen_epochs())


    @staticmethod
    def _gen_group(elements: Generator[..., None, None],
                   condition: Callable[[...], bool],
                   group_class: Type[object]) -> Generator[..., None, None]:
        """
        Generate groups (of type group_class) from elements w.r.t. condition.

        :param elements:  Iterable containing elements that must be grouped.
        :param condition: Callable that returns true when a new group should be
                          started with the analyzed element as its first entry.
        :param group_class: A Callable that instanciate the group (from a list)
                            passed as the sole argument.
        :yield:           Group of type group_class
        """
        group = [next(elements)]
        while True:
            try:
                elem = next(elements)
            except StopIteration:
                if group:
                    yield group_class(group)
                return
            else:
                if condition(elem):
                    yield group_class(group)
                    group = []
                group.append(elem)
        ####while True
####SUPFile

#%%
class BaseEvent:
    """
    Container event for any graphic object displayed on screen for a given time duration.
    """
    def __init__(self, in_tc, out_tc, file, x, y) -> None:
        self._intc = in_tc
        self._outtc = out_tc
        self.gfxfile = file
        self.x, self.y = x, y
        self._img, self._width, self._height = None, None, None
        self._custom = False

    @property
    def width(self) -> int:
        if self._width == -1:
            self.load()
            self.unload()
        return self._width

    @property
    def height(self) -> int:
        if self._height == -1:
            self.load()
            self.unload()
        return self._height

    @property
    def pos(self) -> tuple[int]:
        return (self.x, self.y)

    @property
    def shape(self) -> tuple[int]:
        return Shape(self.width, self.height)

    @property
    def image(self) -> Image.Image:
        if self._img is None:
            self.load()
        return self._img

    @property
    def img(self) -> Image.Image:
        return self.image

    def set_custom_image(self, img: npt.NDArray[np.uint8]) -> None:
        self._img = img.convert('RGBA')
        self._custom = True

    def load(self, fp: Union[str, Path] = None) -> None:
        if self._custom:
            return

        self._open = True
        if fp is None:
            self._img = Image.open(self.gfxfile).convert('RGBA')
        else:
            self._img = Image.open(os.path.join(fp, self.gfxfile)).convert('RGBA')
        # Update wh
        self._width = self._img.width
        self._height = self._img.height

    def unload(self) -> None:
        if not self._custom:
            self._open = False
            if self._img is not None:
                self._img.close()
                self._img = None

    @property
    def tc_in(self) -> str:
        return self._intc

    @property
    def tc_out(self) -> str:
        return self._outtc


class BDNXMLEvent(BaseEvent):
    """
    A BDNXML event can have numerous child elements such as fade timing and >1
    graphics (files) shown at once.
    """
    def __init__(self, te: dict[str, int], ie: dict[str, Any], others: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        te : dict[str, int]
            Temporal informations related to the event.
        ie : dict[str, Any]
            Spatial informations related to the event (incl. file name).
        others : dict[str, Any]
            Other elements related to the event such as fades.
        """
        super().__init__(te.get('InTC'), te.get('OutTC'), ie.get('fp'),
                         int(ie.get('X')), int(ie.get('Y')))
        self.forced = (te.get('Forced', 'False')).lower() == 'true'
        self._width = int(ie.get('Width'))
        self._height = int(ie.get('Height'))

        self.fade_in = dict()
        self.fade_out = dict()
        self._custom = False

        # Internal raw data
        self.__te = te
        self.__ie = ie
        self.__others = others

        #Apparently there's "Crop", "Position" and "Color" but god knows how these are even structured and
        # no commonly used program appears to generate any of those tags.
        for e in others:
            if e.get('Type', None) == 'Fade':
                if e.find('Fade').attrib['FadeType'] == 'FadeIn':
                    self.fade_in = e.attrib
                elif e.find('Fade').attrib['FadeType'] == 'FadeOut':
                    self.fade_out = e.attrib
                else:
                    raise ValueError(f"Unknown fade type {e.attrib['FadeType']}")
            # Do you notice how the implementers of BDNXML thought that people would
            #  consider to anchor fade-in at the end????

    @classmethod
    def copy_custom(cls, other: Type['BDNXMLEvent'], image: Optional[Image.Image] = None,
                    props: Optional[tuple[Pos, Dim]] = None) -> Type['BDNXMLEvent']:
        """
        Create an event alike "other" but with new image (& spatial properties)
        """
        new = cls(other.__te, other.__ie, other.__others)
        new.gfxfile = None
        if image:
            if props is None and other.img._size != image._size:
                raise ValueError("New image does not have the same size and no new dims or pos given.")
            new.set_custom_image(image)
            if props:
                new.x, new.y = props[0]
                new._width, new._height = props[1]
        return new
####BDNXMLEvent

class SeqIO(ABC):
    """
    Base class to describe a sequence of events and the common properties
    """
    def __init__(self, file: Union[str, Path], folder: Optional[Union[str, Path]] = None) -> None:
        self._file = file
        self.events = []

        if folder is None:
            self.folder = self.file[:self.file.rfind('/')]
        else:
            self.folder = folder

    @abstractmethod
    def parse(self) -> None:
        raise NotImplementedError

    def get(self, tc_in: str, default = None) -> Optional[Type[BaseEvent]]:
        for e in self.events:
            if e.intc == tc_in:
                return e
            elif TC.tc2f(e.intc, self.fps) > TC.tc2f(tc_in, self.fps):
                break
        return default

    # Very roughly, if we have to set up two 1920x1080 compositon objects with two
    #  windows of the same size, we need to initialise 4 planes -> about 6 frames at 24p.
    def groups(self, nf_split: Optional[float] = 0.26, tc_in: Optional[str] = None,
               tc_out: Optional[str] = None, /, *, _hard: bool = True) -> list[Type[BaseEvent]]:
        le = []
        nf_split *= 1e3

        for event in self.fetch(tc_in, tc_out):
            if le == []:
                le = [event]
                continue
            td = TC.tc2ms(event.tc_in, self.fps) - TC.tc2ms(le[-1].tc_out, self.fps)

            if _hard and td < 0:
                raise Exception(f"Events are not ordered in time: {event.tc_in}, "
                                f"{event.gfxfile.split(os.path.sep)[-1]} predates previous event.")
            if le == [] or abs(td) < nf_split:
                le.append(event)
            else:
                yield le
                le = [event]
        if le != []:
            yield le
        return

    def fetch(self, tc_in: Optional[str] = None, tc_out: Optional[str] = None):
        for e in self.events:
         if tc_in is None or TC.tc2ms(e.tc_in, self.fps) >= TC.tc2ms(tc_in, self.fps):
          if tc_out is None or TC.tc2ms(e.tc_out, self.fps) <= TC.tc2ms(tc_out, self.fps):
           yield e
          else: #Past tc out point
           return

    def __len__(self):
        return len(self.events)

    @property
    def format(self) -> BDVideo.VideoFormat:
        return self._format

    @format.setter
    def format(self, nf: Union[str, tuple[int, int], int, BDVideo.VideoFormat]) -> None:
        if type(nf) is tuple or type(nf) is BDVideo.VideoFormat:
            self._format = BDVideo.VideoFormat(nf)
        elif type(nf) is str:
            # reversed to alleviate the potential illegal entries key overwrites
            dc = {vf.value[1]: vf.value[0] for vf in reversed(BDVideo.VideoFormat)}
            try:
                # Quick and dirty 16/9 look-up with appended scan format
                if nf[-1].lower() not in ['i', 'p']:
                    raise TypeError
                nf_rs = int(nf[:-1])
            except TypeError:
                try:
                    nf_rs = int(nf)
                except ValueError:
                    raise TypeError("Don't know how to parse format string.")
            self._format = BDVideo.VideoFormat((dc[nf_rs], nf_rs))

    @property
    def fps(self) -> float:
        return self._fps.exact_value

    @fps.setter
    def fps(self, nfps: float) -> None:
        self._fps = BDVideo.FPS(nfps)

    @property
    def file(self) -> Union[str, Path]:
        return self._file

    @file.setter
    def file(self, newf: Union[str, Path]) -> None:
        if not os.path.exists(newf):
            raise OSError("File not found.")
        self._file = newf
        self.parse()

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @folder.setter
    def folder(self, newf: Union[str, Path]) -> None:
        if not os.path.exists(newf):
            raise OSError("Folder not found.")
        self._folder = newf


class BDNXML(SeqIO):
    def __init__(self,
            file: Union[str, Path],
            folder: Optional[Union[str, Path]] = None,
            skip_zero_dur: bool = True,
        ) -> None:
        """
        BDNXML handler object/parser
        """
        super().__init__(file, folder)

        self._skip_zero_duration = skip_zero_dur
        self.events: list[BDNXMLEvent] = []
        self.parse()

    def parse(self) -> None:
        self.parse_header()
        self._parse_events()

    def parse_header(self) -> None:
        escape_xml = lambda c: re.sub(u'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '', c)

        content = None
        with open(self._file, 'r') as f:
            content = ET.fromstring(escape_xml(f.read()))
        assert content is not None, "Failed to parse file."
        header, self._raw_events = content[0:2]

        hformat = header.find('Format')
        self.fps = float(hformat.attrib['FrameRate'])
        self.dropframe = bool(1 if hformat.attrib['DropFrame'].lower() == 'true' else 0)
        self.format = hformat.attrib['VideoFormat']

    def _parse_events(self) -> None:
        """
        BDNXML repesents events with PNG images. But the way those PNG images
        are generated differs vastly. Some have one image for overlapping
        events [in time] while others will generate two images with different
        spatial properties. This is a problem for consistency because the two
        are entirely different in term.
        SUPer assumes the worst case and always assumes there's a single bitmap
        per BDNXMLEvent. (2+ images are merged to have one image).
        """
        # Parse global effects here then LTU wwhile cycling the events
        #  https://forum.doom9.org/showthread.php?t=146493&page=9

        #BDNXML have n>=1 graphical object in each event but we don't want to
        # have subgroup for a given timestamp to not break the SeqIO class
        # so, we merge sub-evnets on the same plane.
        prev_f_out = -1
        self.events = []

        for event in self._raw_events:
            cnt = 0
            while event[cnt:]:
                assert event[cnt].tag == 'Graphic', "Expected a 'Graphic' first."
                effects, gevents, k = [], [], 0
                for k, subevent in enumerate(event[cnt+1:]):
                    if subevent.tag == 'Graphic':
                        gevents.append(subevent)
                    else:
                        effects.append(subevent)
                # Event.attrib contains the <Event> tag params
                # Event[cnt] features the internal content of the <event> tag.
                # i.e <Graphic>, <Fade ...>
                if gevents != []:
                    gevents[0:0] = [event[cnt]]
                    group2merge = [BDNXMLEvent(event.attrib, dict(gevent.attrib, fp=os.path.join(self.folder, gevent.text)), []) for gevent in gevents]
                    pos, dim = min_enclosing_cube(group2merge)
                    image_info = dict(Width=dim.w, Height=dim.h, X=pos.x, Y=pos.y, fp=None)
                    image = merge_events(group2merge, dim=dim, pos=pos)
                    image_info['fp'] = os.path.join(self.folder, 'temp', event[cnt].text)
                    ea = BDNXMLEvent(event.attrib, image_info, others=effects)
                    ea.set_custom_image(image)
                else:
                    ea = BDNXMLEvent(event.attrib, dict(event[cnt].attrib, fp=os.path.join(self.folder, event[cnt].text)), effects)
                if not (ea.tc_in == ea.tc_out and self._skip_zero_duration):
                    self.events.append(ea)
                else:
                    logging.warning(f"Ignored zero-duration graphic: '{ea.gfxfile.split(os.path.sep)[-1]}' @ '{ea.tc_in}'.")
                new_out = TC.tc2f(ea.tc_out, self.fps)
                assert prev_f_out <= new_out, "Event ahead finish before last event!"
                prev_f_out = new_out
                cnt += k+2
        # for event

    @property
    def dropframe(self) -> bool:
        return self._dropframe

    @dropframe.setter
    def dropframe(self, dropframe: bool) -> None:
        self._dropframe = dropframe
        TC.FORCE_NDF = not dropframe
        if dropframe:
            logging.warning("WARNING: Detected drop frame flag in BDNXML! SUPer is untested with drop frame timecodes!")
