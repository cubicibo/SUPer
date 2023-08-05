#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 cibo
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

from SUPer import BDNRender, get_super_logger
from SUPer.__metadata__ import __author__, __version__ as LIB_VERSION

import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from typing import NoReturn, Union

#%% Main code
if __name__ == '__main__':
    print()
    logger = get_super_logger('SUPer')

    def exit_msg(msg: str, is_error: bool = True) -> NoReturn:
        if msg != '':
            if is_error:
                logger.error(msg)
            else:
                logger.info(msg)
        sys.exit(is_error)
    ####exit_msg

    def check_output(fp: Union[Path, str], overwrite: bool) -> None:
        ext = ''
        fp =  Path(fp)
        if fp.exists() and not overwrite:
            exit_msg("Output file already exist, not overwriting.")
        if fp.name.find('.') == -1:
            logger.warning("No extension provided, assuming .SUP.")
            fp = str(fp) + '.sup'
            ext = 'sup'
        elif (ext := fp.name.split('.')[-1].lower()) not in ['pes', 'sup']:
            exit_msg("Not a known PG stream extension, aborting.")
        return str(os.path.expandvars(os.path.expanduser(fp))), ext

    def check_ext(fp: Union[Path, str]) -> None:
        fp = Path(fp)

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input BDNXML file.", default='', required=True)
    parser.add_argument('-c', '--compression', help="Time threshold for acquisitions. [int, 0-100]", type=int, default=85, required=False)
    parser.add_argument('-r', '--comprate', help="Decay rate to attain time threshold. [int, 0-100]", type=int, default=100, required=False)
    parser.add_argument('-q', '--qmode', help="Image quantization mode. [0: PIL, 1: PIL+K-Means, 2: K-Means]", type=int, default=0, required=False)
    parser.add_argument('-b', '--bt', help="Target BT matrix [601, 709, 2020]", type=int, default=709, required=False)
    parser.add_argument('-n', '--ntsc', help="Scale all timestamps by a 1.001.", action='store_true', default=False, required=False)
    parser.add_argument('-s', '--subsampled', help="Flag to indicate BDNXML is subsampled", action='store_true', default=False, required=False)
    parser.add_argument('-f', '--softcomp', help="Use compatibility mode for software decoder", action='store_true', default=False, required=False)
    parser.add_argument('-d', '--nodts', help="Don't compute DTS in stream", action='store_true', default=False, required=False)
    parser.add_argument('-a', '--aheadoftime', help="Allow ahead of time decoding.", action='store_true', default=False, required=False)
    parser.add_argument('-p', '--palette', help="Always write the full palette.", action='store_true', default=False, required=False)
    parser.add_argument('-y', '--yes', help="Overwrite output file", action='store_true', default=False, required=False)
    parser.add_argument('-w', '--withsup', help="Write SUP aside of PES file", action='store_true', default=False, required=False)


    parser.add_argument('-v', '--version', action='version', version=f"(c) {__author__}, v{LIB_VERSION}")
    parser.add_argument("output", type=str)

    args = parser.parse_args()

    #### Sanity checks and conversion
    args.output, ext = check_output(args.output, args.yes)

    assert 0 <= args.compression <= 100
    assert 0 <= args.comprate <= 100
    if args.qmode not in range(0, 3):
        logger.warning("Unknown quantization mode, using PIL (0).")
    if args.bt not in [601, 709, 2020]:
        logger.warning("Unknown BT ITU target, using bt709.")
        args.bt = 709

    if (args.nodts or args.aheadoftime) and ext == 'pes':
        exit_msg("PES output without DTS or with ahead-of-time decoding is not allowed, aborting.")
    if ext == 'pes' and not args.palette:
        logger.warning("PES output: generating with full palette flag.")
        args.palette = True
    if ext != 'pes' and args.withsup:
        args.withsup = False
        logger.warning("Ignoring withsup flag, output is not PES.")

    print("\n @@@@@@@   &@@@  @@@@   @@@@@@@\n"\
          "@@@B &@@@  @@@@  @@@@  @@@@  @@@\n"\
          "@@@@       @@@@  @@@@  @@@@  @@@\n"\
          "J&@@@@&G   @@@@  @@@@  @@@@&@@@ \n"\
          "    &@@@@  @@@@  @@@@  @@@@\n"\
          "@@@P B@@@  @@@@  @@@&  &@@@\n"\
          "@@@&!&@@@  B@@@G#&YY5  YJ5#\n"\
          " G&&@&&B    5#&@B  @@PB@&    @@&@\n"\
          "                  @@@ ,@@@  @@@&G5\n"\
          "                  @@@BP€    @@@\n"\
          "                  @@@       @@@\n"\
          "                   @@YY@@   @@@\n")

    ##
    parameters = {
        'quality_factor': int(args.compression)/100,
        'refresh_rate': int(args.comprate)/100,
        'adjust_dropframe': args.ntsc,
        'scale_fps': args.subsampled,
        'kmeans_fade': args.qmode == 1,
        'kmeans_quant': args.qmode == 2,
        'bt_colorspace': f"bt{args.bt}",
        'pgs_compatibility': args.softcomp,
        'enforce_dts': not args.nodts,
        'no_overlap': not args.aheadoftime,
        'full_palette': args.palette,
        'output_all_formats': args.withsup,
    }

    bdnr = BDNRender(args.input, parameters)
    bdnr.optimise()
    bdnr.write_output(args.output)
    exit_msg("Success.", False)
####
