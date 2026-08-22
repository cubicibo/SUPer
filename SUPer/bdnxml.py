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

import os
import xml.etree.ElementTree as ET

from typing import Self
from dataclasses import dataclass
from pathlib import Path

from .bdvideo import Format, Framerate
from .internals import GfxCompositor, TC
from .geometry import Box

@dataclass(frozen=True)
class BDNVideoFormat:
    fmt: Format
    fps: Framerate
    drop_frame: bool = False

@dataclass(frozen=True)
class BDNGraphic:
    x: int
    y: int
    width: int
    height: int
    filepath: Path
    
    def __post_init__(self) -> None:
        assert self.box.area > 0
        assert self.filepath.exists()
    
    @classmethod
    def from_tag(cls, graphic: ET, vf: BDNVideoFormat, parent_folder: Path) -> Self:
        def extract_gfxat(Width = vf.fmt.width, Height = vf.fmt.height, X = 0, Y = 0) -> tuple[int, int, int, int]:
            return int(Width), int(Height), int(X), int(Y)
        dx, dy, x, y = extract_gfxat(**graphic.attrib)
        return cls(x, y, dx, dy, parent_folder.joinpath(graphic.text))
    
    @property
    def box(self) -> Box:
        return Box(self.y, self.height, self.x, self.width)

@dataclass(frozen=True)
class BDNEvent(GfxCompositor):
    inTC: TC
    outTC: TC
    graphics: tuple[BDNGraphic]
    forced: bool = False
    
    def __post_init__(self) -> None:
        assert isinstance(self.inTC, TC) and isinstance(self.outTC, TC)
        assert self.inTC < self.outTC
        assert isinstance(self.graphics, (tuple, list)) and len(self.graphics) in range(1, 3)
        if self.forced is True:
            raise RuntimeError("Forced flag is not supported.")
        
        #cache?
        object.__setattr__(self, '_bbox', Box.union(*[gfx.box for gfx in self.graphics]))

    @classmethod
    def from_element(cls, ev_elem: ET, vfmt: BDNVideoFormat, parent_folder: Path) -> Self:
        gfx = tuple([BDNGraphic.from_tag(g, vfmt, parent_folder) for g in filter(lambda x: x.tag == 'Graphic', ev_elem)])
        inTC = TC(vfmt.fps, ev_elem.attrib['InTC'], force_non_drop_frame=True)
        outTC = TC(vfmt.fps, ev_elem.attrib['OutTC'], force_non_drop_frame=True)
        forced = ev_elem.attrib.get('Forced', 'False').lower() == 'true'
        return cls(inTC, outTC, gfx, forced)
    
    def get_bbox(self) -> Box:
        return self._bbox

    def set_outTC(self, outTC: TC) -> None:
        object.__setattr__(self, 'outTC', outTC)

class BDNXML:
    """
    Class to parse in-place of a BDN XML file.
    """
    def __init__(self, xml_filepath: Path | str) -> None:
        if not (xml_filepath := Path(xml_filepath)).exists():
            raise OSError("Provided BDN XML does not exist.")
        self._fp = xml_filepath
        self._description, self._events = None, None
    
    def parse(self, *, skip_size_check: bool = False) -> None:
        if not skip_size_check and os.stat(self._fp).st_size > (100 << 20):
            raise OSError("XML file too large.")
        
        content = None
        with open(self._fp, 'r', encoding="utf-8-sig") as f:
            content = ET.fromstring(f.read())
        assert content is not None, "Failed to parse file."
        if content.tag.lower() != 'bdn':
            raise RuntimeError("Incorrect file format, expected BDN.")
        
        descr = content.find('Description')
        events = content.find('Events')
        if descr is None or events is None:
            raise RuntimeError("Malformed BDN file: <Description> or <Events> section missing.")

        bdn_vf = self._parse_header(descr)
        events = self._parse_events(bdn_vf, events)
        
        self._description = bdn_vf
        self._events = events
    ####
            
    @property
    def description(self) -> BDNVideoFormat:
        """
        Parse the description section
        So far only the video format attributes are handled.
        """
        if self._description is None:
            self.parse()
        return self._description
    
    @property
    def events(self) -> list[BDNEvent]:
        if self._events is None:
            self.parse()
        return self._events
    
    def _parse_header(self, header: ET) -> BDNVideoFormat:
        hcontent = header.find('Events')
        assert hcontent.attrib['Type'].lower() == 'graphic', "Text BDN not supported."

        hformat = header.find('Format')
        fps = Framerate(hformat.attrib['FrameRate'])
        vfmt = Format(hformat.attrib['VideoFormat'])
        dropframe = hformat.attrib['DropFrame'].lower() == 'true'
        if dropframe is True:
            raise NotImplementedError("Drop frame BDN not supported.")
        return BDNVideoFormat(vfmt, fps, dropframe)

    def _parse_events(self, vfmt: BDNVideoFormat, events_tree: ET) -> list[BDNEvent]:
        bdn_ref_directory = self._fp.parent
        events = []
        for ev in filter(lambda x: x.tag == 'Event', events_tree):
            events.append(BDNEvent.from_element(ev, vfmt, bdn_ref_directory))
        
        events = sorted(events, key=lambda e: e.inTC.frames)
        for ev0, ev1 in zip(events, events[1:]):
            if ev0.outTC > ev1.inTC:
                raise RuntimeError("BDN has overlapped events. SUPer only supports monotonic timelines.")
        return events