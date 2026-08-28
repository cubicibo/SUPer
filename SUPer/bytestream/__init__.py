"""
Copyright (C) 2023-26 cibo
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

from .graphicstream import DisplaySet, Epoch
from .pgstreams import PesMuiWriter, SUPReader, SUPWriter
from .segments import END, ODS, PCS, PDS, WDS, CompositionObject, PGSegmentType, SegmentParser
from .verifier import check_pts_dts_sanity, debug_stats, is_compliant, test_diplayset, test_rx_bitrate
