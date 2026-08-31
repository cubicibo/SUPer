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

from collections.abc import Sequence

from .segments import END, ODS, PCS, PDS, WDS, GraphicSegment, PGSegmentType


class DisplaySet:
    def __init__(self, segments: Sequence[GraphicSegment]):
        assert isinstance(segments[0], PCS)
        assert isinstance(segments[-1], END)

        self._pcs = segments[0]
        self._wds = next(filter(lambda s: s.type == PGSegmentType.WDS, segments), None)
        self._pds = list(filter(lambda s: s.type == PGSegmentType.PDS, segments))
        self._ods = list(filter(lambda s: s.type == PGSegmentType.ODS, segments))
        self._end = segments[-1]

    def __len__(self) -> int:
        return len(self.segments)

    def copy(self, new_pcs_pts: int | None):
        segments = [self._pcs.copy()]
        if self._wds:
            segments.append(self._wds.copy())
        segments.extend(p.copy() for p in self._pds)
        segments.extend(o.copy() for o in self._ods)
        segments.append(self._end.copy())
        if new_pcs_pts is not None:
            delta = new_pcs_pts - segments[0].pts
            for segment in segments:
                segment.pts += delta
                segment.dts += delta
        return self.__class__(segments)

    @property
    def pcs(self) -> PCS:
        return self._pcs

    @pcs.setter
    def pcs(self, pcs: PCS) -> None:
        assert isinstance(pcs, PCS)
        self._pcs = pcs

    @property
    def wds(self) -> WDS:
        return self._wds

    @wds.setter
    def wds(self, wds: WDS | None) -> None:
        assert isinstance(wds, WDS) or wds is None
        self._wds = wds

    @property
    def pds(self) -> list[PDS]:
        return self._pds

    @pds.setter
    def pds(self, pds: PDS | Sequence[PDS]) -> None:
        if isinstance(pds, PDS):
            pds = [pds]
        else:
            assert all(isinstance(p, PDS) for p in pds)
        self._pds = pds

    @property
    def ods(self) -> list[ODS]:
        return self._ods

    @ods.setter
    def ods(self, ods: ODS | Sequence[ODS] | None) -> None:
        if isinstance(ods, ODS):
            ods = [ods]
        else:
            assert all(isinstance(o, ODS) for o in ods)
        self._ods = ods

    @property
    def end(self) -> END:
        return self._end

    @end.setter
    def end(self, end: END) -> None:
        assert isinstance(end, END)
        self._end = end

    @property
    def segments(self) -> tuple[GraphicSegment]:
        if self.wds is not None:
            return [self.pcs, self.wds] + self.pds + self.ods + [self.end]
        return [self.pcs] + self.pds + self.ods + [self.end]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if len(self.segments) > self.n:
            self.n += 1
            return self.segments[self.n-1]
        else:
            raise StopIteration

    def __getitem__(self, n: int) -> GraphicSegment:
        return self.segments[n]

class Epoch:
    def __init__(self, display_sets: Sequence[DisplaySet]):
        self.ds = display_sets

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if len(self.ds) > self.n:
            self.n += 1
            return self.ds[self.n-1]
        else:
            raise StopIteration

    def __getitem__(self, n: int) -> DisplaySet:
        return self.ds[n]

    def __len__(self) -> int:
        return len(self.ds)
