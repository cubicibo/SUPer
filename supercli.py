#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2023-2025 cibo
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

from warnings import filterwarnings
filterwarnings("ignore", message=r"Non-empty compiler", module="pyopencl")

import multiprocessing as mp
if __name__ == '__main__':
    mp.freeze_support()

from SUPer import BDNRender, LogFacility
from SUPer.__metadata__ import __author__, __version__ as LIB_VERSION

import os
import sys
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction
from typing import NoReturn, Union
import time
from datetime import timedelta

#%% Main code
if __name__ == '__main__':
    logger = LogFacility.get_logger('SUPer')

    def exit_msg(msg: str, is_error: bool = True) -> NoReturn:
        if msg != '':
            if is_error:
                logger.critical(msg)
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

    class BruleCapAction(BooleanOptionalAction):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values = None, option_string=None):
            from brule import LayoutEngine, Brule, HexTree, KDMeans
            f_strcap = lambda caps: ', '.join(caps)

            print(f"LayoutEngine: {f_strcap(LayoutEngine.get_capabilities())}")
            print(f"   RLE codec: {f_strcap(Brule.get_capabilities())}")
            print(f"    kd-Means: {f_strcap(KDMeans.get_capabilities())}")
            print(f"     HexTree: {f_strcap(HexTree.get_capabilities())}")
            exit_msg('', is_error=False)

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Set input BDNXML file.", default='', required=True)
    parser.add_argument('-c', '--compression', help="Set compression rate [int, 0-100] (def:  %(default)s)", type=int, default=80, required=False)
    parser.add_argument('-a', '--acqrate', help="Set acquisition rate [int, 0-100] (def:  %(default)s)", type=int, default=100, required=False)
    parser.add_argument('-q', '--qmode', help="Set image quantization mode. [0: KD-Means, 1: Pillow, 2: HexTree, 3: PNGQ/LIQ] (def:  %(default)s)", type=int, default=3, required=False)
    parser.add_argument('-k', '--prefer-normal', help="Flag to prefer normal case over acquisitions.", action='store_true', default=False, required=False)
    parser.add_argument('-n', '--allow-normal', help="Flag to allow normal case object refreshes.", action='store_true', default=False, required=False)
    parser.add_argument('-b', '--bt', help="Set target Rec. BT matrix [601, 709, 2020] (def:  %(default)s)", type=int, default=709, required=False)
    parser.add_argument('-p', '--palette', help="Flag to always write the full palette.", action='store_true', default=False, required=False)
    parser.add_argument('-d', '--ahead', help="Flag to enable flexible palette update buffering.", action='store_true', default=False, required=False)
    parser.add_argument('-y', '--yes', help="Flag to overwrite an existing file with the same name.", action='store_true', default=False, required=False)
    parser.add_argument('-w', '--withsup', help="Flag to write both SUP and PES+MUI files.", action='store_true', default=False, required=False)
    parser.add_argument('-e', '--extra-acq', help="Set min count of palette updates needed to add an acquisition. [0: off] (def:  %(default)s)", type=int, default=2, required=False)
    parser.add_argument('-m', '--max-kbps', help="Set a max bitrate to validate the output against.", type=int, default=0, required=False)
    parser.add_argument('-l', '--log-to-file', help="Enable logging to file and specify the minimum logging level. [10: debug, 20: normal, 30: warn/errors]", type=int, default=0, required=False)
    parser.add_argument('-t', '--threads', help="Set number of concurrent threads, up to 10 supported. [0: auto] (def:  %(default)s)", type=int, default=0, required=False)
    parser.add_argument('--layout', help="Set window layout mode. [0: safe, 1: normal, 2: aggressive] (def: config.ini)", type=int, default=-1, required=False)
    parser.add_argument('--capabilities', help="Display Brule library capabilities and exit.", action=BruleCapAction)

    parser.add_argument('--ssim-tol', help="Set a SSIM analysis offset (positive: higher sensitivity) [int, -100-100] (def:  %(default)s)", type=int, default=0, required=False)

    parser.add_argument('-v', '--version', action='version', version=f"(c) {__author__}, v{LIB_VERSION}")
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    print(f"SUPer version {LIB_VERSION} - (c) 2025 cubicibo")
    print("HDMV PGS encoder, with support from Masstock, Alllen and Emulgator.")

    #### Sanity checks and conversion
    args.output, ext = check_output(args.output, args.yes)

    assert abs(args.ssim_tol) <= 100
    assert 0 <= args.compression <= 100
    assert 0 <= args.acqrate <= 100
    if args.qmode not in range(0, 5):
        logger.warning("Unknown quantization mode, attempting to use pngquant/libimagequant.")
        args.qmode = 3
    if args.bt not in [601, 709, 2020]:
        logger.warning("Unknown transfer matrix, using bt709.")
        args.bt = 709

    if not (2 >= args.layout >= -1):
        logger.warning("Invalid layout mode specified, falling back to ini config (or default, 2).")
        args.layout = -1

    if args.extra_acq < 0:
        logger.warning("Got invalid extra-acq, disabling option.")
        args.extra_acq = 0

    if args.max_kbps > 48000:
        logger.warning("Max bitrate is beyond BDAV limit.")
    elif 10 < args.max_kbps < 500:
        logger.warning("Max bitrate is low. Buffer underflow errors will be spammed.")
    elif args.max_kbps < 10 and args.max_kbps != 0:
        exit_msg("Meaningless max bitrate, aborting.")

    if args.log_to_file > 40:
        logger.warning("Meaningless logging level, disabling file logging.")
        args.log_to_file = False

    if args.threads < 0 or args.threads > 10:
        exit_msg("Incorrect number of threads, aborting.")

    if args.prefer_normal and not args.allow_normal:
        logger.warning("--prefer-normal requires --allow-normal, forcefully enabling this flag.")
        args.allow_normal = True

    if ext == 'pes' or args.withsup:
        if args.ahead:
            logger.warning("PES output + buffering: PES shall NOT be Built or Rebuilt at authoring!")
        if args.allow_normal:
            logger.warning("PES output + Normal Case: PES shall NOT be Built or Rebuilt at authoring!")
        if not args.palette:
            logger.warning("PES output requires --palette, forcefully enabling this flag.")
            args.palette = True
    parameters = {'ini_opts': {'super_cfg': {}}}

    try:
        application_path = Path(sys.argv[0]).resolve().parent
    except:
        application_path = Path(sys.argv[0]).absolute().parent

    config_file = application_path.joinpath('config.ini')

    if config_file.exists():
        ini_opts = {'super_cfg': {}}
        import configparser
        def get_value_key(config, key: str):
            try: return config[key]
            except KeyError: return None
        config = configparser.ConfigParser()
        config.read(config_file)
        if (super_cfg := get_value_key(config, 'SUPer')) is not None:
            ini_opts['super_cfg'] |= dict(super_cfg)
            if int(ini_opts['super_cfg'].pop('abort_on_error', 0)):
                LogFacility.exit_on_error(logger)

        if args.qmode >= 3:
            exepath = None
            piq_values = {}
            if (piq_sect := get_value_key(config, 'PILIQ')) is not None:
                if (exepath := piq_sect.pop('quantizer', None)) is not None and not os.path.isabs(exepath):
                    exepath = str(application_path.joinpath(exepath))
                piq_values |= {k: int(v) for k, v in piq_sect.items()}
            ini_opts['quant'] = {'qpath': exepath} | piq_values
        if len(ini_opts):
            parameters['ini_opts'] |= ini_opts
    else:
        logger.error("config.ini not found!")

    if args.layout >= 0:
        parameters['ini_opts']['super_cfg']['layout_mode'] = args.layout
    if parameters['ini_opts']['super_cfg'].get('layout_mode', None) is None:
        parameters['ini_opts']['super_cfg']['layout_mode'] = 2
    ###
    parameters |= {
        'quality_factor': int(args.compression)/100,
        'refresh_rate': int(args.acqrate)/100,
        'quantize_lib': args.qmode,
        'bt_colorspace': f"bt{args.bt}",
        'allow_overlaps': args.ahead,
        'full_palette': args.palette,
        'output_all_formats': args.withsup,
        'allow_normal_case': args.allow_normal,
        'prefer_normal_case': args.prefer_normal,
        'max_kbps': args.max_kbps,
        'log_to_file': args.log_to_file,
        'insert_acquisitions': args.extra_acq,
        'ssim_tol': args.ssim_tol/100,
        'threads': 'auto' if args.threads == 0 else args.threads,
    }
    ts_start = time.monotonic()
    bdnr = BDNRender(args.input, parameters, args.output)
    bdnr.encode_input()
    bdnr.write_output()
    exit_msg(f"Success. Duration: {timedelta(seconds=round(time.monotonic() - ts_start, 3))}", False)
####
