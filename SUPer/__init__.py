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

from .__metadata__ import __name__, __version__

from .palette import Palette, PaletteEntry
from .segments import PCS, WDS, ODS, PDS, ENDS, WindowDefinition, CObject, PGSegment, DisplaySet, Epoch
from .filestreams import SUPFile, BDNXML
from .optim import Preprocess, Optimise
from .utils import BDVideo, TC, LogFacility
from .pgraphics import PGraphics
from .interface import BDNRender
from .pgstream import is_compliant, test_diplayset, check_pts_dts_sanity, test_rx_bitrate
