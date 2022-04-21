#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of SUPer

(c) 2022 cubicibo@doom9 forums

This file is under GPLv2. You MUST read the licence before using this software.
"""

from .__metadata__ import __name__, __version__

from .palette import Palette, PaletteEntry
from .segments import PCS, WDS, ODS, PDS, ENDS, WindowDefinition, CObject, PGSegment, DisplaySet, Epoch, Soup
from .filestreams import SupStream, BDNXML, ImgSequence
from .optim import Preprocess, Optimise, PalettizeMode, FadeCurve
from .utils import BDVideo, PGSTarget, TimeConv
from .pgraphics import PGraphics