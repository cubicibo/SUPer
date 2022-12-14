#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 cibo 
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

import logging
logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
logger.info("Loading SUPer...")

import numpy as np
from os import path

from SUPer.render import group_event_alikes, to_epoch2, is_compliant
from SUPer.utils import _pinit_fn, Shape, TimeConv as TC
from SUPer import BDNXML

#Leave norm threshold to zero, it can generate unexpected behaviours.
#Colors should be 256. Anything above is illegal, anything below results in a
# loss of quality.
kwargs = {'norm_thresh': 0, 'colors': 256}

def write(filep, epochs):
    from os import path
    with open(path.expanduser(filep), 'wb') as f:
        for mepoch in epochs:
            for ds in mepoch.ds:
                for seg in ds.segments:
                    f.write(seg._bytes)
    logger.info("Wrote file, exiting...")

#%%
if __name__ == '__main__':
    #####
    OUTPUT_FILE = './tetsujin.sup'
    bdn = BDNXML(path.expanduser('~/examples/tetsujin/bdn.xml'))
    #####
    out = []
    logger.info("Starting...")
    for group in bdn.groups():
        offset = len(group)-1
        subgroups = []
        last_split = len(group)
        largest_shape = Shape(0, 0)

        #Backward pass for fine epochs definition
        for k, event in enumerate(reversed(group[1:])):
            offset -= 1
            if np.multiply(*group[offset].shape) > np.multiply(*largest_shape):
                largest_shape = event.shape
            nf = TC.tc2f(event.tc_in, bdn.fps) - TC.tc2f(group[offset].tc_out, bdn.fps)
            
            if nf > 0 and nf/bdn.fps > 3*_pinit_fn(largest_shape)/90e3:
                subgroups.append(group[offset+1:last_split])
                last_split = offset + 1
        if group[offset+1:last_split] != []:
            subgroups.append(group[offset+1:last_split])
        if subgroups:
            subgroups[-1].insert(0, group[0])
        else:
            subgroups = [[group[0]]]
            
        #Epoch generation (each subgroup will be its own epoch)
        for subgroup in reversed(subgroups):
            logger.info(f"Generating epoch {subgroup[0].tc_in}->{subgroup[-1].tc_out}...")
            regions_ods_mapping, box = group_event_alikes(subgroup)
            outm, _ = to_epoch2(bdn, subgroup, regions_ods_mapping, box, **kwargs)
            out.append(outm)
            logger.info(f" => optimised as {len(outm.ds)} display sets.")
    is_compliant(out, bdn.fps)
    write(path.expanduser(OUTPUT_FILE), out)
    ####
