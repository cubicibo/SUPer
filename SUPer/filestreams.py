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

import xml.etree.ElementTree as ET
import os

from numpy import typing as npt
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Union, Optional, Type, Any

from .segments import PGSegment, PCS, WDS, PDS, ODS, ENDS, DisplaySet, Epoch
from .utils import (BDVideo, PGSTarget, TimeConv as TC, get_super_logger,
                    min_enclosing_cube, merge_events, Shape, Pos, Dim)

logging = get_super_logger('SUPer')

class SupStream:
    BUFFER_N_BYTES = 1048576
    SEGMENTS = { 'PCS': PCS, 'WDS': WDS, 'PDS': PDS, 'ODS': ODS, 'END': ENDS }

    def __init__(self, data: Union[str, Path, BytesIO, bytes], auto_close: Optional[bool] = True) -> None:
        """
        Manage a Sup Stream from a file, bytestring or an actual data stream.
         Stop iterating once the buffer is consumed.
         If the
        :param auto_close: If the stream is a file, autoclose it when done reading.

        :return:          PGSegment (that is, any child class)
        """
        self._is_file = type(data) in [Path, str]
        self.stream = data if not self._is_file else open(data, 'rb')
        self.s_index = 0
        self._data = bytearray()
        self.auto_close = auto_close
        self._pending_segs = []

    def renew(self) -> None:
        len_before = len(self._data)
        self._data += self.stream.read(__class__.BUFFER_N_BYTES)
        read_back = len(self._data) - len_before
        self.s_index += read_back
        return read_back

    def epochs(self, epoch: Optional[Epoch] = Epoch()) -> Epoch:
        """
        Generator of Epoch for the given stream.
         This function is stupid and can fail.
        :param: epoch_ds : DSs to add to the first Epoch (catching behaviour)
                            if dealing with a live bytestream

        :yields: Epoch containing a list of DS.
        :return: An incomplete Epoch.
        """

        for ds in self.fetch_displayset():
            if PCS.CompositionState.EPOCH_START == ds.pcs.composition_state:
                if epoch.ds:
                    yield epoch
                    epoch = Epoch()
            epoch.append(ds)
        yield epoch #Return a probably incpmplete Epoch (EOS reached)

    def fetch_displayset(self) -> DisplaySet:
        for segment in self.fetch_segment(_watch_for_pending=False):
            self._pending_segs.append(segment)
            if segment.type == 'END':
                yield DisplaySet(self._pending_segs)
                self._pending_segs = []
        return # Ran out of segments to generate a DS.

    def fetch_segment(self, *, _watch_for_pending: bool = True) -> PGSegment:
        """
        Generator of PGS segment in the specified stream.
         Stop iterating once the buffer is consumed.
         Can be used after fetch_displayset() to retrieve orphaned segments.

        :return:          PGSegment (that is, any child class)
        """
        if self.s_index == -1 or self.stream.closed:
            raise Exception("Attempting to use a closed datastream.")

        while True:
            if self._pending_segs != [] and _watch_for_pending:
                yield self._pending_segs.pop(0) # yield oldest segment pending
                continue

            if len(self._data) == 0 and not self.renew():
                return

            try:
                seg = __class__.SEGMENTS[PGSegment(self._data).type](self._data)
                self._data = self._data[len(seg):]
                yield seg
            except EOFError:
                if self.renew() == 0:
                    if self.auto_close and self._is_file:
                        self.close()
                    return
            except (ValueError, BufferError) as e:
                pg_id = self._data.find(PGSegment.MAGIC)
                if pg_id == -1:
                    self._data = bytearray(b'P') if self._data[-1] == b'P' else bytearray()
                else:
                    self._data = self._data[pg_id:]
                if e == ValueError:
                    logging.warning("Garbage in PGStream encountered.")

    def close(self) -> None:
        self.stream.close()
        self.s_index = -1

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
        # provided for courtesy & compatbility reasons
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

        return self._img

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

        for event in self.fetch(tc_in, tc_out):
            if le == []:
                le = [event]
                continue
            td = TC.tc2ms(event.tc_in, self.fps) - TC.tc2ms(le[-1].tc_out, self.fps)

            if _hard and td < 0:
                raise Exception("Events are not ordered in time: {event.tc_in},"
                                "{event.gfxfile.split(os.path.sep)[-1]} predates previous event.")
            if le == [] or abs(td) < nf_split*1e3:
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

    def __len__(self):
        return len(self.events)

    @property
    def format(self) -> BDVideo.VideoFormat:
        return self._format

    @format.setter
    def format(self, nf: str) -> None:
        if type(nf) is tuple or type(nf) is BDVideo.VideoFormat:
            self._format = BDVideo.VideoFormat(nf)
        elif type(nf) is str:
            dc = {480: 720, 576: 720, 720: 1280, 1080: 1920}
            try:
                # Quick and dirty 16/9 look-up table for BDNXML format
                ord(nf[-1])
                nf_rs = int(nf[:-1])
                self._format = BDVideo.VideoFormat((dc[nf_rs], nf_rs))
            except TypeError:
                try:
                    nf_rs = int(nf)
                    self._format = BDVideo.VideoFormat((dc[nf_rs], nf_rs))
                except ValueError:
                    raise TypeError("Don't know how to parse format string.")

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

    def get_target(self) -> PGSTarget:
        return PGSTarget(BDVideo.VideoFormat(self.format), BDVideo.FPS(self.fps))


class BDNXML(SeqIO):
    def __init__(self, file: Union[str, Path], folder: Optional[Union[str, Path]] = None) -> None:
        super().__init__(file, folder)

        self.events: list[BDNXMLEvent] = []
        self.parse()

    def parse(self) -> None:
        """
        BDNXML just repesents events with PNG images. But the way those PNG images are
        generated differs vastly depending of the program. Some will just generate one PNG
        images for overlapping events while others will generate two images with different
        properties. THis is a problem for consistency because the two are entirely different
        SUPer choose to just stack events one after the other, as such as:
            [E1, ... En, .... EN] where E(n).tc_in <= E(n+1).tc_in
        Fortunately if the above condition catch a ==, we also have E(n+1).tc_out == E(n).tc_out
        thnaks to BDNXML properties.
        """
        with open(self._file, 'r') as f:
            content = ET.parse(f).getroot()
            header, events = content[0:2]

            hformat = header.find('Format')
            self.fps = float(hformat.attrib['FrameRate'])
            self.dropframe = bool(1 if hformat.attrib['DropFrame'].lower() == 'true' else 0)
            self.format = hformat.attrib['VideoFormat']

            # Parse global effects here then LTU wwhile cycling the events
            #  https://forum.doom9.org/showthread.php?t=146493&page=9

            #BDNXML have n>=1 graphical object in each event but we don't want to
            # have subgroup for a given timestamp to not break the SeqIO class
            # so, we merge sub-evnets on the same plane.
            prev_f_out = -1

            for event in events:
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
                    self.events.append(ea)
                    assert prev_f_out <= TC.tc2f(ea.tc_out, self.fps), "Event ahead finish before last event!"
                    prev_f_out = TC.tc2f(ea.tc_out, self.fps)
                    cnt += k+2
            # for event

    @property
    def dropframe(self) -> bool:
        return self._dropframe

    @dropframe.setter
    def dropframe(self, dropframe: bool) -> None:
        if dropframe:
            logging.warning("Drop frame timecodes are not implemented properly.")
        self._dropframe = dropframe

class ImgSequence(SeqIO):
    """
    Used to define a sequence of image with very basic timing provided in a csv file.
    First line of the csv needs to define: FPS,ts_type,start_ts,dt_type
    -fps: fps of stream
    -ts_type: type of start_ts ('f': frame count, 'ms': milliseconds)
    -start_ts: timestamp at which the effect appears/start, w/ aforementionned format
    -dt_type: format for on-screen time of each image, specified on the next lines.
        can be either 'ms' or 'f'. One number per line (associated to each image)
    """

    # :)
    EXTS = ['.png', '.gif', '.jpg', 'tiff', '.tif', 'jpeg', 'webp']

    def __init__(self, file: Union[str, Path],
                 folder: Optional[Union[str, Path]] = None, delimiter: str = ','):

        super().__init__(file, folder)
        self.delimiter = delimiter

        self.parse(False)

    def parse(self, skip_header: bool = True):
        import csv

        dc, rows = {}, []
        for fn in sorted(os.listdir(self.folder)):
            if fn.lower()[-4:] in __class__.EXTS:
                dc[int(fn.split('.')[0])] = fn

        if os.path.exists(self.file):
            with open(os.path.join(self.file), 'r') as csvfile:
                csvre = csv.reader(csvfile, delimiter=self.delimiter, quotechar='|')
                rows = [row for row in csvre]
        else:
            raise OSError("Cannot find CSV timing file.")

        if not skip_header:
            if rows == []:
                logging.warning("No timing file provided: assuming:"
                                "1920x1080p23.976, 1 fpi, 0 pts.")
                self.fps, self._type_ts, self.t_start, self.t_sep, x, y, vw, vh =\
                    (23.976, 'f', 0, 'f', -1, -1, 1920, 1080)
            else:
                temp_fps, self._type_ts, self.t_start, self.t_sep = rows[0][:4]
                if len(rows[0]) == 8:
                    x, y, vw, vh = rows[0][4:6]
                else:
                    x, y, vw, vh = -1, -1, 1920, 1080
            self.fps = float(temp_fps)
            self.format = (vw, vh)

        if rows == []:
            rows = [1] * (len(dc)+1)

        dcf = {
            'f': lambda tc, f : TC.tc_addf(tc, int(f), self.fps),
            'ms': lambda tc, ms : TC.tc_addms(tc, int(ms), self.fps),
            's': lambda tc, s : TC.tc_adds(tc, float(s), self.fps),
            'tc': lambda tc1, tc2 : TC.tc_addtc(tc1, tc2, self.fps),
        }

        if self._type_ts == 'ms':
            self.t_start = TC.ms2tc(int(self.t_start), self.fps)
        elif self._type_ts == 'f':
            self.t_start = TC.f2tc(int(self.t_start), self.fps)
        else:
            raise NotImplementedError(f"Unknown timestamp format {self._type_ts}.")

        offset_fn = dcf[self.t_sep]

        t_in = self.t_start
        self.events = []
        for key, event in zip(sorted(list(dc.keys())), rows[1:]):
            t_out = offset_fn(t_in, event[0])
            self.events.append(BaseEvent(t_in, t_out, os.path.join(self.folder, dc[key]), x, y))
            t_in = t_out

    @property
    def type_ts(self) -> str:
        return self._type_ts

    @property
    def t_out(self) -> str:
        return self.events[-1].event.outtc

    @property
    def t_in(self) -> float:
        if self._type_ts == 'ms':
            return TC.tc2ms(self.t_start, self.fps)
        return self.t_start

    @t_in.setter
    def t_in(self, ts):
        """
        Timestamp when effect starts. Either NFrames or seconds, depending of how the
         stream was initialised. Internally, the ts is always converted to Nframes
        """
        if self._type_ts == 'ms':
            self.t_start = TC.ms2tc(ts, self.fps)
        else:
            self.t_start = TC.f2tc(ts, self.fps)
