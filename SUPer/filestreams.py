#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""

import xml.etree.ElementTree as ET
import logging
import os

from PIL import Image
from pathlib import Path
#from flags import Flags
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Union, Optional, Type, Any

from .segments import PGSegment, PCS, WDS, PDS, ODS, ENDS, DisplaySet, Epoch
from .utils import BDVideo, ImageEvent, TimeConv as TC
    
# class InjectFlags(Flags):
#     OVERWRITE_EPOCH = () # Overwrite the whole Epoch
#     OVERWRITE_DISPLAYSET = () # Overwrite the whole displayset with another
#     OVERWRITE_SEGMENT = () # Overwrite the segment of a given type in a given DS
#     APPEND_TO_EPOCH = () # Append to the local epoch (Adds a DS to the Epoch)
#     APPEND_TO_DISPLAYSET = () # Append to the closest displayset (in the Epoch)

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
    
    def fetch_epoch(self, epoch: Optional[Epoch] = Epoch()) -> Epoch:
        """
        Generator of Epoch for the given stream.
         This function is stupid and can fail.
        :param: epoch_ds : DSs to add to the first Epoch (catching behaviour)
                            if dealing with a live bytestream
            
        :yields: Epoch containing a list of DS.
        :return: An incomplete Epoch.
        """
        
        for ds in self.fetch_displayset():
            if ds.pcs and 'epoch' in ds.pcs[0].composition_state:
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
        return (self.height, self.width)

    @property
    def image(self):
        if self._img is None:
            self.load()
        return self._img
    
    @property
    def img(self):
        # provided for courtesy & compatbility reasons
        return self.image
    
    def load(self, fp: Union[str, Path] = None):
        if self._img is not None:
            self.unload()
        self._open = True
        if fp is None:
            self._img = Image.open(self.gfxfile)
        else:
            self._img = Image.open(os.path.join(fp, self.gfxfile))
        # Update wh
        self._width = self._img.width
        self._height = self._img.height
        
        return self.image
        
    def unload(self):
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
    A BDNXML event can have numerous child elements such as fade timing and >1 gfx file.
    """
    def __init__(self, te: dict[str, int], ie: dict[str, Any], others: dict[str, Any]) -> None:
        super().__init__(te.get('InTC'), te.get('OutTC'), ie.get('fp'),
                         int(ie.get('X')), int(ie.get('Y')))
        self.forced = (te.get('Forced', 'False')).lower() == 'true'
        self._width = int(ie.get('Width'))
        self._height = int(ie.get('Height'))
        
        self.fade_in = dict()
        self.fade_out = dict()
        
        #Apparently there's "Crop", "Position" and "Color" but only god knows how these are even structured and
        # no commonly used program appears to generate any of those tags.
        for e in others:
            for inline_effect in e.get('InlineEffect', []):
                if inline_effect.get('Type', None) == 'Fade':
                    if inline_effect[0].attrib['FadeType'] == 'FadeIn':
                        self.fade_in = inline_effect.attrib
                    elif inline_effect[0].attrib['FadeType'] == 'FadeOut':
                        self.fade_out = inline_effect.attrib
                    else:
                        raise ValueError(f"Unknown fade type {inline_effect[0].attrib['FadeType']}")
        

class SeqIO(ABC):
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
    
    def fetch(self, tc_in: Optional[str] = None, tc_out: Optional[str] = None):
        for e in self.events:
         if tc_in is None or TC.tc2ms(e.intc, self.fps) >= TC.tc2ms(tc_in, self.fps):
          if tc_out is None or TC.tc2ms(e.outtc, self.fps) <= TC.tc2ms(tc_out, self.fps):
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
        return self._fps.value
    
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

    def load_images(self, tc_start = None, tc_end = None, *,
                    _group: bool = True) -> list[list[Type[BaseEvent]]]:
        """
        Load a set of subtitles from a sequence of event object w.r.t. timecodes.
               
        :param tc_start: Starting timecode 'HH:MM:SS:FF' By default, up to the end of the file.
        :param tc_end:   Last timestamp to consider. By default, up to the end of the file.
        :param _group:   Split by groups. I.e returned list looks like: [G1: list[BaseEvent], G2: list[BaseEvent], ...]
                So one can optimize per effects within a given subtitle. (i.e: fade followed by karaoke)
        :return:  Image events grouped together so they can be optimised by sets.
        """

        imlist = []
        
        for e in self.fetch(tc_start, tc_end):
            ime = ImageEvent(Image.open(os.path.join(self.folder, e.gfxfile)).convert('RGBA'), e)
            imlist.append(ime)
        
        if not _group:
            return [imlist]
        
        #Group related events (i.e one karaoke line for the whole karaoke song)
        # the pairing relies on consecutive timestamps.
        g_imlist = []
        group = []
        group.append(imlist[0])
        for k in range(0, len(imlist)-1, 1):
            if abs(TC.tc2ms(imlist[k].event.outtc, self.fps) - \
                TC.tc2ms(imlist[k+1].event.intc, self.fps)) < 1001/self.fps:
                group.append(imlist[k+1])
            elif group != []:
                g_imlist.append(group)
                group = [imlist[k+1]]
        if group != []:
            g_imlist.append(group)
        return g_imlist

class BDNXML(SeqIO):
    def __init__(self, file: Union[str, Path], folder: Optional[Union[str, Path]] = None) -> None:
        super().__init__(file, folder)
        
        self.events: list[BDNXMLEvent] = []
        self.parse()
        
    def parse(self) -> None:
        with open(self._file, 'r') as f:
            content = ET.parse(f).getroot()
            header, events = content[0:2]
            
            hformat = header.find('Format')
            self.fps = float(hformat.attrib['FrameRate'])
            self.dropframe = hformat.attrib['DropFrame']
            self.format = hformat.attrib['VideoFormat']
            
            # Parse global effects here then LTU wwhile cycling the events
            #  https://forum.doom9.org/showthread.php?t=146493&page=9
            
            for event in events:
                cnt = 0
                while event[cnt:]:
                    assert event[cnt].tag == 'Graphic',"Expected a 'Graphic' first."
                    isolated_event = []
                    for k, subevent in enumerate(event[cnt+1:]):
                        if subevent.tag == 'Graphic':
                            break
                        isolated_event.append(subevent)                    
                    self.events.append(BDNXMLEvent(event.attrib,
                                       dict(event[cnt].attrib, fp=os.path.join(self.folder, event[cnt].text)),
                                       isolated_event))
                    cnt += k+2

                    
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
                                "1080p23.976, 1 fpi, 0 pts, 16/9.")
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