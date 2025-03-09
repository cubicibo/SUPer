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

import os
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image
from pathlib import Path

from collections.abc import Generator
from typing import Union, Optional, Type, Any, Callable

from .segments import PGSegment, PCS, DisplaySet, Epoch
from .utils import BDVideo, TC, LogFacility, Shape, Box

logger = LogFacility.get_logger('SUPer')

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
                    next_seg_pos = buff.find(PGSegment.MAGIC)
                    if next_seg_pos >= 0:
                        buff = buff[next_seg_pos:]
                    #if b'P' is not at pos 0, then we can discard the last byte.
                    elif buff[0] != PGSegment.MAGIC[0]:
                        buff = bytearray()
                if renew or len(buff) < 2:
                    if not (new_data := f.read(self.bytes_per_read)):
                        break
                    buff = buff + new_data
            ####while
        ####with
        return


    def get_fps(self) -> BDVideo.FPS:
        pcs = next(self.gen_segments())
        assert isinstance(pcs, PCS)
        return BDVideo.FPS.from_pcsfps(pcs.fps)


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
            assert width == ds.pcs.width, "Width is not constant."
            assert height == ds.pcs.height, "Height is not constant."
            assert fps == ds.pcs.fps, "FPS is not constant."

        return BDVideo.VideoFormat((width, height)), BDVideo.FPS.from_pcsfps(fps)


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
class BDNXMLEvent:
    """
    A BDNXML event can have numerous child elements such as fade timing and >1
    graphics (files) shown at once.
    """
    can_warn_palettized = False
    def __init__(self, te: dict[str, int], ie: dict[str, Any], others: dict[str, Any] = {}) -> None:
        """
        Parameters
        ----------
        te : dict[str, int]
            Temporal informations related to the event.
        ie : dict[str, Any]
            Spatial informations related to the event (incl. base path).
        others : dict[str, Any]
            Other elements related to the event such as fades.
        """
        _fps = te.get('fps')
        if te.get('dropframe'):
            # Parse correctly as DF, then change to NDF
            self._intc = TC(_fps, te.get('InTC'))
            self._outtc = TC(_fps, te.get('OutTC'))
            self._intc.drop_frame = self._outtc.drop_frame = False
        else:
            self._intc = TC(_fps, te.get('InTC'), force_non_drop_frame=True)
            self._outtc = TC(_fps, te.get('OutTC'), force_non_drop_frame=True)

        self._img, self._width, self._height = None, None, None

        base_folder = ie.get('fp')
        assert len(ie['graphics']) > 0

        self.gfxfile = os.path.join(base_folder, ie['graphics'][0].text)

        f_box = lambda gfx: Box(int(gfx.get('Y')), int(gfx.get('Height')),
                                int(gfx.get('X')), int(gfx.get('Width')))
        box = f_box(ie['graphics'][0])
        for gfx_ev in ie['graphics'][1:]:
            box = Box.union(box, f_box(gfx_ev))
        self.x, self.y = box.x, box.y

        self._custom = False
        if len(ie['graphics']) > 1:
            self._custom = True
            self._gfx = ie['graphics']
            self._bf = base_folder

        self.forced = (te.get('Forced', 'False')).lower() == 'true'
        self._width = box.dx
        self._height = box.dy

        # backup parameters
        self._te = te
        self._ie = ie
        #Apparently there's "Crop", "Position" and "Color" but god knows how these are even structured and
        # no commonly used program appears to generate any of those tags.

    def copy(self) -> 'BDNXMLEvent':
        return __class__(self._te, self._ie)

    @property
    def width(self) -> int:
        if self._width is None:
            self.load()
            self.unload()
        return self._width

    @property
    def height(self) -> int:
        if self._height is None:
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

    def unload(self) -> None:
        if self._img is not None:
            self._img.close()
            self._img = None

    @property
    def tc_in(self) -> str:
        return self._intc

    @property
    def tc_out(self) -> str:
        return self._outtc

    def set_tc_in(self, tc_in: Union[str, TC]) -> None:
        if isinstance(tc_in, str):
            self._intc = TC(self._intc.fractional_fps, tc_in, force_non_drop_frame=True)
        else:
            self._intc = tc_in
        assert self._intc.drop_frame == self._outtc.drop_frame
        assert self._outtc > self._intc


    def set_tc_out(self, tc_out: Union[str, TC]) -> None:
        if isinstance(tc_out, str):
            self._outtc = TC(self._outtc.fractional_fps, tc_out, force_non_drop_frame=True)
        else:
            self._outtc = tc_out
        assert self._intc.drop_frame == self._outtc.drop_frame
        assert self._outtc > self._intc

    def load(self) -> None:
        should_warn = False
        if self._custom:
            self._img = Image.new('RGBA', (self._width, self._height), (0,0,0,0))
            boxes = []
            for gfx in self._gfx:
                gfxp = Image.open(os.path.join(self._bf, gfx.text))
                should_warn = gfxp.mode == 'P'
                box = Box(int(gfx.get('Y')), gfxp.height, int(gfx.get('X')), gfxp.width)
                self._img.paste(gfxp.convert('RGBA'), (box.x - self.x, box.y - self.y))
                boxes.append(box)
            assert Box.intersect(*boxes).area == 0, f"Overlapping <Graphic>s at {self._intc}"
        else:
            gfximg = Image.open(self.gfxfile)
            should_warn = gfximg.mode == 'P'
            self._img = gfximg.convert('RGBA')
        if should_warn and __class__.can_warn_palettized:
            __class__.can_warn_palettized = False
            logger.warning("Some PNGs are already palettized. Prefer 32-bit RGBA images, else quality may be subpar.")
        assert self._img.width == self._width
        assert self._img.height == self._height
####BDNXMLEvent

def remove_dupes(events: list[BDNXMLEvent]) -> list[BDNXMLEvent]:
    output_events = [events[0]]
    for i in range(0, len(events)-1):
        is_diff = events[i+1].pos != events[i].pos
        is_diff = is_diff or events[i+1].shape != events[i].shape
        #only diff the images if they have the same size and position.
        is_diff = is_diff or (not np.array_equal(np.asarray(events[i+1].image), np.asarray(events[i].image)))

        events[i].unload()
        if is_diff or output_events[-1].tc_out != events[i+1].tc_in:
            output_events.append(events[i+1])
        else:
            output_events[-1].set_tc_out(events[i+1].tc_out)
    events[-1].unload()
    assert output_events[0].tc_in == events[0].tc_in and output_events[-1].tc_out == events[-1].tc_out
    logger.debug(f"Removed {len(events) - len(output_events)} duplicate event(s).")
    return output_events
####

def add_periodic_refreshes(events: list[BDNXMLEvent], fps: float, period: float) -> list[BDNXMLEvent]:
    if period < 1:
        return

    frame_period = int(round(period*fps))

    new_events = []
    for event in events:
        frames_duration = (event.tc_out - event.tc_in).frames
        new_events.append(event)

        count = (frames_duration//frame_period - 1)
        if count >= 1:
            original_tc_in = event.tc_in
            final_tc_out = event.tc_out
            start_idx = len(new_events)
            prev_tc_out = event.tc_in + frame_period
            event.set_tc_out(prev_tc_out)
            for _ in range(count):
                event = event.copy()
                event.set_tc_in(prev_tc_out)
                prev_tc_out = prev_tc_out + frame_period
                event.set_tc_out(prev_tc_out)
                new_events.append(event)
            event.set_tc_out(final_tc_out)
            assert len(new_events) == start_idx + count
            # validate
            for k in range(start_idx, len(new_events)):
                assert new_events[k - 1].tc_out == new_events[k].tc_in
        ####
    ####
    return new_events
####

#%%
class BDNXML:
    def __init__(self,
            file: Union[str, Path],
            folder: Optional[Union[str, Path]] = None,
        ) -> None:
        """
        BDNXML handler object/parser
        """
        self._file = Path(file)
        if folder is None:
            self._folder = self.file.parent
        else:
            self._folder = Path(folder)
            assert self._folder.exists()

        self.split_seen = False
        self.events: list[BDNXMLEvent] = []
        self._parse()

    def _parse(self) -> None:
        self._parse_header()
        self._parse_events()

    def _parse_header(self) -> None:
        content = None
        with open(self._file, 'r', encoding="utf-8-sig") as f:
            content = ET.fromstring(f.read())
        assert content is not None, "Failed to parse file."
        header, self._raw_events = content[0:2]

        hformat = header.find('Format')
        self._fps = BDVideo.FPS(float(hformat.attrib['FrameRate']))
        self._dropframe = bool(hformat.attrib['DropFrame'].lower() == 'true')
        self._set_format(hformat.attrib['VideoFormat'])

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
        # TODO: Parse global effects here then LTU while cycling the events
        #  https://forum.doom9.org/showthread.php?t=146493&page=9

        #BDNXML have 2>=n>=1 graphical object in each event but we don't want to
        # have subgroup for a given timestamp to not break the SeqIO class
        # so, we merge sub-evnets on the same plane.
        self.events.clear()
        self.split_seen = False

        for event in self._raw_events:
            assert event.tag == 'Event'
            effects, gevents, k = [], [], 0
            for k, subevent in enumerate(event):
                assert k > 0 or subevent.tag == 'Graphic', "Expected a 'Graphic' first."
                if subevent.tag == 'Graphic':
                    gevents.append(subevent)
                else:
                    effects.append(subevent)
            # Event.attrib contains the <Event> tag params
            # Event[cnt] features the internal content of the <event> tag.
            # i.e <Graphic>, <Fade ...>
            self.split_seen = self.split_seen or len(gevents) > 1
            ev = BDNXMLEvent(event.attrib | {'fps': self._fps, 'dropframe': self._dropframe},
                             dict(graphics=gevents, fp=os.path.join(self.folder)), effects)
            if ev.tc_in != ev.tc_out:
                self.events.append(ev)
            else:
                logger.warning(f"Ignored zero-duration graphic: '{ev.gfxfile.split(os.path.sep)[-1]}' @ '{ev.tc_in}'.")
        # for event
        self.events.sort(key=lambda e: e.tc_in.frames)
        self.assert_event_list()

    def assert_event_list(self) -> None:
        for k, ev in enumerate(self.events):
            assert ev.tc_in < ev.tc_out, f"Illegal event duration at InTC={ev.tc_in}."
            assert 0 == k or self.events[k-1].tc_out <= ev.tc_in, f"Two events overlap in time around InTC={ev.tc_in}."

    def groups(self, dt_split: float) -> list[BDNXMLEvent]:
        le = []

        for event in self.events:
            if 0 == len(le):
                le = [event]
                continue
            td = event.tc_in.to_pts() - le[-1].tc_out.to_pts()
            assert td >= 0, f"Events are not ordered in time: {event.tc_in}, {event.gfxfile.split(os.path.sep)[-1]} predates previous event."
            if td < dt_split:
                le.append(event)
            else:
                yield le
                le = [event]
        if len(le):
            yield le
        return

    def __len__(self):
        return len(self.events)

    @property
    def format(self) -> BDVideo.VideoFormat:
        return self._format

    def _set_format(self, nf: Union[str, tuple[int, int], int, BDVideo.VideoFormat]) -> None:
        if type(nf) is tuple or type(nf) is BDVideo.VideoFormat:
            self._format = BDVideo.VideoFormat(nf)
        elif type(nf) is str:
            # First try to parse WIDTHxHEIGHT format string
            try:
                self._format = BDVideo.VideoFormat(tuple(map(int, nf.split("x", 1))))
            except ValueError:
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
        valid_fmt, valid_fps = BDVideo.check_format_fps(self._format, self.fps)
        str_fps = int(self._fps) if float(self._fps).is_integer() else round(float(self._fps), 3)
        if not valid_fmt:
            logger.error(f"Non standard VideoFormat-FPS combination for Blu-ray ({self._format.value[1]}@{str_fps})!")
            logger.warning(f"Expected one of these framerates: {valid_fps} for format {'x'.join(map(str, self._format.value))}.")
        elif self._format == BDVideo.VideoFormat.HD1080:
            if self._fps > BDVideo.FPS.NTSCp:
                logger.warning(f"UHD BD VideoFormat-FPS combination: 1080p@{str_fps} only exists with a HEVC video stream.")

    @property
    def fps(self) -> BDVideo.FPS:
        return self._fps

    @property
    def file(self) -> Union[str, Path]:
        return self._file

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def dropframe(self) -> bool:
        return self._dropframe
