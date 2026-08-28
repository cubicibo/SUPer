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

import struct
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

from ..internals import _Masks
from .graphicstream import DisplaySet, Epoch
from .segments import PCS, GraphicSegment, SegmentParser

_FileStreamT: type = str | Path | BytesIO

class SUPReader:
    def __init__(self, fp: _FileStreamT):
        self.fp = Path(fp) if isinstance(fp, (str, Path)) else fp

    def read_segment(self) -> Generator[GraphicSegment, None, None]:
        if isinstance(self.fp, Path):
            open_stream = lambda: open(self.fp, 'rb')
        else:
            open_stream = lambda: nullcontext(self.fp)
        with open_stream() as f:
            buff = bytearray()
            can_parse = True
            while True:
                if len(buff) < 13 or not can_parse:
                    # no new data, nothing we can do
                    buff += (f.read(1 << 20))
                    if len(buff) < 13:
                        break
                    can_parse = True
                assert b'PG' == buff[:2]
                segment_length = (buff[12] | (buff[11] << 8)) + 13
                if len(buff) >= segment_length:
                    segment = SegmentParser.from_pg_segment(buff[:segment_length])
                    #handle wrap around
                    if segment.dts > segment.pts * 4e4:
                        segment.dts -= _Masks.W32
                    yield segment
                    buff = buff[segment_length:]
                else:
                    can_parse = False
            ####while

    @staticmethod
    def _gen_group(elements: Generator[..., None, None],
                   condition: Callable[[...], bool],
                   group_class: type[object]) -> Generator[..., None, None]:
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
    ####

    def read_displayset(self) -> Generator[DisplaySet, None, None]:
        """
        Returns a generator of DisplaySets. Stops when all DisplaySets in the
        file have been consumed.

        :yield: DisplaySet, in order, as they appear in the SUP file.
        """
        condition = lambda seg: isinstance(seg, PCS)
        yield from __class__._gen_group(self.read_segment(), condition, DisplaySet)

    def read_epoch(self) -> Generator[Epoch, None, None]:
        condition = lambda ds: ds.pcs.composition_state & ds.pcs.CompositionState.EPOCH_START
        yield from __class__._gen_group(self.read_displayset(), condition, Epoch)

    def read_epochs(self) -> list[Epoch]:
        return list(self.read_epoch())

    def read_displaysets(self) -> list[DisplaySet]:
        return list(self.read_displayset())

class GraphicWriter(ABC):
    def __init__(self, mux_offset: int = 0, ts_mask: int = _Masks.W33):
        self.mux_offset = int(mux_offset)
        self._ts_mask = int(ts_mask)

    @abstractmethod
    def _setup_io(self) -> tuple[BytesIO, ...]: ...

    def _write_header(self, st: tuple[BytesIO, ...]) -> None: pass

    def _write_tail(self, st: tuple[BytesIO, ...]) -> None: pass

    def get_mux_timestamps(self, segment: GraphicSegment) -> tuple[int, int]:
        assert isinstance(segment.pts, int) and isinstance(segment.dts, int)
        mux_pts = (self.mux_offset + segment.pts) & self._ts_mask
        mux_dts = (self.mux_offset + segment.dts) & self._ts_mask
        return mux_pts, mux_dts

    @abstractmethod
    def _write_segment(self, st: tuple[BytesIO, ...], mux_pts: int, mux_dts: int, segment: GraphicSegment):
        ...

    @staticmethod
    def _close_io(st: tuple[BytesIO, ...], use_file_io: bool):
        if use_file_io:
            for stream in st:
                stream.close()


    def writer(self) -> Generator[None, GraphicSegment, None]:
        st, use_file_io = self._setup_io()

        self._write_header(st)

        while True:
            segment = yield
            if segment is None:
                break

            self._write_segment(st, *self.get_mux_timestamps(segment), segment)

        self._write_tail(st)
        self.__class__._close_io(st, use_file_io)
        return

    def write_segments(self, segments: list[GraphicSegment]) -> None:
        writer = self.writer()
        next(writer)
        for segment in segments:
            writer.send(segment)
        try: writer.send(None)
        except StopIteration: ...

    def write_displaysets(self, displaysets: list[DisplaySet]) -> None:
        writer = self.writer()
        next(writer)
        for ds in displaysets:
            for segment in ds:
                writer.send(segment)
        try: writer.send(None)
        except StopIteration: ...


    def write_epochs(self, epochs: list[Epoch]) -> None:
        writer = self.writer()
        next(writer)
        for epoch in epochs:
            for ds in epoch:
                for segment in ds:
                    writer.send(segment)
        try: writer.send(None)
        except StopIteration: ...

class PesMuiWriter(GraphicWriter):
    def __init__(self, pes_fp: _FileStreamT, mui_fp: _FileStreamT | None = None, mux_offset: int | None = None) -> None:
        if mui_fp is not None and type(pes_fp) != type(mui_fp):
            raise ValueError("PES and MUI output aren't the same type.")
        if isinstance(pes_fp, (str, Path)):
            self.pes_stream = Path(pes_fp)
            self.mui_stream = Path(mui_fp) if mui_fp is not None else (Path.joinpath(self.pes_stream.parent, Path(self.pes_stream.stem + '.pes.mui')))
        else:
            if mui_fp is None:
                raise ValueError("No IO stream for MUI output")
            self.pes_stream = pes_fp
            self.mui_stream = mui_fp

        super().__init__(mux_offset or 54000000, _Masks.W33)

    def _write_segment(self, st: tuple[BytesIO, ...], mux_pts: int, mux_dts: int, segment: GraphicSegment):
        mui_segment_meta = struct.pack(">BI", segment.type, segment.length + 3)
        payload = bytearray(b'\x00'*9)
        # encode DTS MSBs.LSB
        payload[:4] = struct.pack(">I", (mux_dts >> 1) & _Masks.W32)

        # encode PTS as 39 bits (easier than 33 bits in the middle of two bytes)
        payload[4:9] = struct.pack(">Q", (mux_pts << 6) & _Masks.W39)[3:]
        payload[4] |= ((mux_dts & 0x1) << 7)

        st[0].write(mui_segment_meta + payload)
        st[1].write(bytes(segment))

    def _write_header(self, st: tuple[BytesIO, ...]) -> None:
        st[0].write(b'\x00\x00\x00\x03')

    def _write_tail(self, st: tuple[BytesIO, ...]) -> None:
        st[0].write(b'\xFF' + bytes(13))

    def _setup_io(self) -> tuple[tuple[BytesIO, ...], bool]:
        use_file_io = isinstance(self.pes_stream, Path)
        pes_stream = open(self.pes_stream, 'wb') if use_file_io else self.pes_stream
        mui_stream = open(self.mui_stream, 'wb') if use_file_io else self.mui_stream
        return (mui_stream, pes_stream), use_file_io

class SUPWriter(GraphicWriter):
    def __init__(self, sup_fp: _FileStreamT, mux_offset: int = 0):
        self.sup_stream = Path(sup_fp) if isinstance(sup_fp, (Path, str)) else sup_fp
        super().__init__(mux_offset, _Masks.W32)

    def _write_segment(self, st: tuple[BytesIO, ...], mux_pts: int, mux_dts: int, segment: GraphicSegment):
        raw = b'PG' + struct.pack(">II", mux_pts & _Masks.W32, mux_dts & _Masks.W32) + bytes(segment)
        st[0].write(raw)

    def _setup_io(self) -> tuple[tuple[BytesIO, ], bool]:
        use_file_io = isinstance(self.sup_stream, Path)
        sup_stream = open(self.sup_stream, 'wb') if use_file_io else self.sup_stream
        return (sup_stream,), use_file_io

# class BufferedGraphicEpochWriter:
#     """
#     Buffer incoming epoch and write them linearly whenever possible.
#     """
#     def __init__(self, write_callback: Callable[[Any], None | int]) -> None:
#         self._write_cb = write_callback
#         self._pending = dict()
#         self._next_write_idx = 0
#         self._next_promise_idx = 0
#         self._pcs_id = 0

#     def register_promise(self) -> Promise:
#         promise = Promise(self._next_promise_idx)
#         self._next_promise_idx += 1
#         return promise

#     def write(self, data: Any, promise: Promise) -> None:
#         assert promise.uuid < self._next_promise_idx
#         self._pending[promise.uuid] = data
#         self._flush()

#     def _flush(self) -> None:
#         k = 0
#         while (epoch := self._pending.get(self._next_write_idx + k, None)) is not None:
#             for ds in epoch:
#                 ds.pcs.composition_number = (self._pcs_id & 0xFFFF)
#                 self._pcs_id += 1
#             self._write_cb(epoch)
#             k += 1
#         self._next_write_idx += k

