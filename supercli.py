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
from SUPer.optim import Quantizer

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
    parser.add_argument("-i", "--input", type=str, help="Set input BDNXML file.", default='', required=True)
    parser.add_argument('-c', '--compression', help="Set compression rate [int, 0-100] (def:  %(default)s)", type=int, default=65, required=False)
    parser.add_argument('-a', '--acqrate', help="Set acquisition rate [int, 0-100] (def:  %(default)s)", type=int, default=100, required=False)
    parser.add_argument('-q', '--qmode', help="Set image quantization mode. [1: PIL+K-Means, 2: K-Means, 3: PNGQ/LIQ]  (def:  %(default)s)", type=int, default=1, required=False)
    parser.add_argument('-n', '--allow-normal', help="Flag to allow normal case object redefinition.", action='store_true', default=False, required=False)
    parser.add_argument('-b', '--bt', help="Set target BT matrix [601, 709, 2020]  (def:  %(default)s)", type=int, default=709, required=False)
    parser.add_argument('-s', '--subsampled', help="Flag to indicate BDNXML is subsampled", action='store_true', default=False, required=False)
    parser.add_argument('-p', '--palette', help="Flag to always write the full palette.", action='store_true', default=False, required=False)
    parser.add_argument('-y', '--yes', help="Flag to overwrite output file", action='store_true', default=False, required=False)
    parser.add_argument('-w', '--withsup', help="Flag to write both SUP and PES+MUI files.", action='store_true', default=False, required=False)
    parser.add_argument('-t', '--tslong', help="Flag to use PTS/DTS strategy with additional margins.", action='store_true', default=False, required=False)

    parser.add_argument('--nodts', help="Flag to not set DTS in stream (NOT COMPLIANT)", action='store_true', default=False, required=False)
    #parser.add_argument('--aheadoftime', help="Flag to allow ahead of time decoding. (NOT COMPLIANT)", action='store_true', default=False, required=False)

    parser.add_argument('-v', '--version', action='version', version=f"(c) {__author__}, v{LIB_VERSION}")
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    args.aheadoftime = False

    #### Sanity checks and conversion
    args.output, ext = check_output(args.output, args.yes)

    assert 0 <= args.compression <= 100
    assert 0 <= args.acqrate <= 100
    if args.qmode not in range(1, 4):
        logger.warning("Unknown quantization mode, using PIL+K-Means (1).")
        args.qmode = 1
    if args.bt not in [601, 709, 2020]:
        logger.warning("Unknown BT ITU target, using bt709.")
        args.bt = 709

    if (args.nodts or args.aheadoftime) and ext == 'pes':
        exit_msg("PES output without DTS or with ahead-of-time decoding is not allowed, aborting.")
    if ext == 'pes' and not args.palette:
        logger.warning("PES output: generating with full palette flag.")
        args.palette = True

    print("\n @@@@@@@   &@@@  @@@@   @@@@@@@\n"\
          "@@@B &@@@  @@@@  @@@@  @@@@  @@@\n"\
          "@@@@       @@@@  @@@@  @@@@  @@@      Special Thanks to:\n"\
          "J&@@@@&G   @@@@  @@@@  @@@@&@@@               Masstock\n"\
          "    &@@@@  @@@@  @@@@  @@@@                   NLScavenger\n"\
          "@@@P B@@@  @@@@  @@@&  &@@@                   Prince 7\n"\
          "@@@&!&@@@  B@@@G#&YY5  YJ5#                   Emulgator\n"\
          " G&&@&&B    5#&@B  @@PB@&    @@&@\n"\
          "                  @@@ ,@@@  @@@&G5\n"\
          "                  @@@BP€    @@@\n"\
          "                  @@@       @@@\n"\
          "                   @@YY@@   @@@\n")
    
    if args.qmode == 3:
        config_file = Path('config.ini')
        exepath = None
        if config_file.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(config_file)
                exepath = get_value_key(config['PILIQ'], 'quantizer')
                if not os.path.isabs(exepath):
                    exepath = str(CWD.joinpath(exepath))
            except:
                ...
        if Quantizer.init_piliq(exepath):
            logger.info(f"Advanced image quantizer armed: {Quantizer.get_piliq().lib_name}")
        else:
            exit_msg("Could not initialise advanced image quantizer, aborting.", True)
    ###
    parameters = {
        'quality_factor': int(args.compression)/100,
        'refresh_rate': int(args.acqrate)/100,
        'scale_fps': args.subsampled,
        'quantize_lib': args.qmode,
        'bt_colorspace': f"bt{args.bt}",
        'enforce_dts': not args.nodts,
        'no_overlap': not args.aheadoftime,
        'full_palette': args.palette,
        'output_all_formats': args.withsup,
        'normal_case_ok': args.allow_normal,
        'ts_long': args.tslong,
    }

    bdnr = BDNRender(args.input, parameters)
    bdnr.optimise()
    bdnr.write_output(args.output)
    exit_msg("Success.", False)
####
