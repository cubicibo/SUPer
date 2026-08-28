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

from .__metadata__ import __name__, __version__
from .bdnxml import BDNXML
from .bytestream.graphicstream import DisplaySet, Epoch
from .bytestream.pgstreams import PesMuiWriter, SUPReader, SUPWriter
from .bytestream.segments import END, ODS, PCS, PDS, WDS, CompositionObject, SegmentParser
from .display import BDVideo, Format, Framerate, Matrix, Palette, PaletteEntry
from .encoder import EpochFinder, EventsPreprocessor, LayoutMode, QuantizerWrap
from .interface import BDNEncoder
