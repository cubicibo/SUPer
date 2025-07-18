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

import os
import sys
import time
import signal
from typing import Optional, Any, Callable

### CONSTS
SUPER_STRING = "Make it SUPer!"

def from_bdnxml(queue: ...) -> None:
    from SUPer import BDNRender, LogFacility
    import time
    from datetime import timedelta

    #### This function runs in MP context, not main.
    logger = LogFacility.get_logger('SUPer')
    kwargs = queue.get()
    bdnf = queue.get()
    supo = queue.get()

    if int(kwargs.get('ini_opts', {}).get('super_cfg', {}).pop('abort_on_error', 0)):
        LogFacility.exit_on_error(logger)

    ts_start = time.monotonic()
    logger.info(f"Loading input BDN: {bdnf}")
    sup_obj = BDNRender(bdnf, kwargs, supo)
    sup_obj.encode_input()
    sup_obj.write_output()
    logger.info(f"Finished in {timedelta(seconds=round(time.monotonic() - ts_start, 3))}, exiting...")
####

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    print("Loading...")

from pathlib import Path
from guizero import App, PushButton, Text, CheckBox, Combo, Box, TextBox
from idlelib.tooltip import Hovertip

from SUPer import LogFacility
from SUPer.optim import Quantizer
from SUPer.__metadata__ import __version__ as SUPVERS, __author__

#### Functions, main at the end of the file
def get_kwargs() -> dict[str, Any]:
    return {
        'quality_factor': int(compression_txt.value)/100,
        'refresh_rate': int(refresh_txt.value)/100,
        'quantize_lib': Quantizer.get_option_id(quantcombo.value),
        'bt_colorspace': colorspace.value,
        'allow_overlaps': bool(allow_overlaps.value),
        'full_palette': bool(fullpalette.value),
        'output_all_formats': bool(all_formats.value),
        'allow_normal_case': bool(normal_case_ok.value),
        'prefer_normal_case': bool(prefer_normal_case.value),
        'insert_acquisitions': int(biacqs_val.value),
        'ini_opts': init_extra_libs(application_path, verbose=False),
        'max_kbps': int(max_kbps.value),
        'log_to_file': opts_log[logcombo.value],
        'ssim_tol': int(ssim_tolb.value)/100,
        'redraw_period': float(acqinttb.value),
        'threads': int(threadscombo.value) if threadscombo.value.lower() != 'auto' else 'auto',
        'daemonize': False,
    }

def wrapper_mp() -> None:
    if supout.value == '' or bdnname.value == '':
        return
    try:
        kwargs = get_kwargs()
    except ValueError as e:
        logger.error(f"Aborting, incorrect parameter found: {e}.")
        return
    else:
        invalid = False
        invalid = invalid or not (abs(kwargs[evkey := 'ssim_tol']) <= 1)
        invalid = invalid or not (0 <= kwargs[evkey := 'insert_acquisitions'])
        invalid = invalid or not (0 <= kwargs[evkey := 'quality_factor'] <= 1)
        invalid = invalid or not (0 <= kwargs[evkey := 'refresh_rate'] <= 1)
        invalid = invalid or not (kwargs[evkey := 'threads'] == 'auto' or 1 <= kwargs['threads'] <= 8)
        invalid = invalid or not (0 <= kwargs[evkey := 'max_kbps'] <= 48000)
        invalid = invalid or not (kwargs[evkey := 'redraw_period'] == 0 or (kwargs['redraw_period'] >= 1.0 and kwargs['redraw_period'] <= 3600.0))
        if invalid:
            logger.error(f"Invalid parameter range found for '{evkey}', aborting.")
            return

    do_super.enabled = False
    do_abort.enabled = True #and (1 == kwargs['threads'])
    logger.debug("Starting encoder process.")
    do_super.text = "Encoding (check console)..."

    while True:
        try:
            do_super.queue.get_nowait()
        except:
            break
    do_super.proc = mp.Process(target=from_bdnxml, args=(do_super.queue,), daemon=(1 == kwargs['threads']), name="SUPinternal")
    do_super.proc.start()
    do_super.queue.put(kwargs)
    do_super.queue.put(bdnname.value)
    do_super.queue.put(supout.value)
    do_super.ts = time.time()

def _tryfunc(f: Callable[[Any], None]) -> None:
    try: f()
    except: pass

def _win_nt_abort(proc) -> None:
    import psutil
    procs = psutil.Process().children(recursive=True)
    for child in procs:
        child.terminate()
    alive = psutil.wait_procs(procs, timeout=0.2)[1]
    for child in alive:
        child.kill()
    alive = psutil.wait_procs(procs, timeout=0.2)[1]
    #do a hard taskkill
    if alive:
        from subprocess import call as scall
        for child in alive:
            logger.info(f"Using OS to terminate {child.pid}.")
            _tryfunc(lambda: scall(f"taskkill /f /PID {child.pid}", creationflags=0x08000000))
            _tryfunc(lambda: child.wait(0.1))
####

def _posix_abort(proc, hard: bool = True) -> None:
    _tryfunc(proc.terminate)
    if hard:
        time.sleep(0.5)
        _tryfunc(proc.kill)
    _tryfunc(lambda: proc.join(0.2))

def abort(proc: Optional['mp.Process'] = None, hard: bool = True) -> None:
    try:
        do_abort.enabled = False
    except:
        pass
    if proc is None:
        proc = do_super.proc
    if proc is not None:
        if 'nt' == os.name and hard and not proc.daemon:
            _win_nt_abort(proc)
        else:
            _posix_abort(proc, hard)

def monitor_mp() -> None:
    do_reset = False
    if time.time()-do_super.ts < 2:
        return
    if do_super.proc and do_super.proc.pid:
        if not do_super.proc.is_alive():
            while True:
                try:
                    do_super.queue.get_nowait()
                except:
                    break
            abort(do_super.proc, False)
            do_super.proc = None
            logger.info("Closed gracefully encoder process.")
            do_super.ts = time.time()
            do_reset = True
            do_abort.enabled = False
    if do_reset and bdnname.value and supout.value:
        do_super.enabled = True
        do_super.text = SUPER_STRING

def hide_chkbox() -> None:
    if all_formats.value:
        fullpalette.value = True
        fullpalette.enabled = False
    elif not supout.value.lower().endswith('pes'):
        fullpalette.enabled = True
    if prefer_normal_case.value:
        normal_case_ok.value = True
        normal_case_ok.enabled = False
    else:
        normal_case_ok.enabled = True

def get_bdnxml() -> None:
    bdn_xml_types = ("*.xml", "*.XML")
    bdnname.value = app.select_file(filetypes=[["BDNXML", bdn_xml_types], ["All files", "*"]])
    if supout.value != '' and do_super.text == SUPER_STRING:
        do_super.enabled = True

def set_outputsup() -> None:
    pg_sup_types = ('*.sup', '*.SUP')
    pg_pes_types = ('*.pes', '*.PES')
    supout.value = app.select_file(filetypes=[["SUP", pg_sup_types], ['PES', pg_pes_types], ["All files", "*"]], save=True)
    if supout.value == '':
        return

    if len(Path(supout.value).name.split('.')) == 1:
        logger.info("No extension given, assuming SUP.")
        supout.value += '.sup'

    if supout.value.lower().endswith('pes'):
        fullpalette.value = True
        fullpalette.enabled = False
    else:
        if not all_formats.value:
            fullpalette.enabled = True

    if bdnname.value != '' and do_super.text == SUPER_STRING:
        do_super.enabled = True
####

def terminate(frame = None, sig = None):
    global app
    global do_super
    proc, do_super.proc = do_super.proc, None

    app.cancel(monitor_mp)
    app.destroy()
    abort(proc)

def init_extra_libs(CWD: Path, verbose: bool = True):
    def get_value_key(config, key: str) -> Optional[Any]:
        try: return config[key]
        except KeyError: return None
    ####
    params = {}
    ini_file = CWD.joinpath('config.ini')

    exepath = None
    piq_values = {}
    if ini_file.exists():
        exepath, piq_quality = None, None
        import configparser
        config = configparser.ConfigParser()
        config.read(ini_file)
        if (piq_params := get_value_key(config, 'PILIQ')) is not None:
            if (exepath := piq_params.pop('quantizer', None)) is not None and not os.path.isabs(exepath):
                exepath = str(CWD.joinpath(exepath))
            piq_values = {k: int(v) for k, v in piq_params.items()}
        if (sup_params := get_value_key(config, 'SUPer')) is not None:
            params['super_cfg'] = dict(sup_params)
    elif verbose:
        logger.error("config.ini not found!")
    if Quantizer.init_piliq(exepath):
        if verbose:
            logger.info(f"Advanced image quantizer armed: {Quantizer.get_piliq().lib_name}")
        params['quant'] = {'qpath': exepath} | piq_values
    else:
        logger.warning("No good image quantizer found. Falling back to low quality embedded one.")
    return params

if __name__ == '__main__':
    is_win32 = sys.platform == 'win32'
    try:
        application_path = Path(sys.argv[0]).resolve().parent
    except:
        application_path = Path(sys.argv[0]).absolute().parent

    logger = LogFacility.get_logger('SUPui')
    logger.info(f"SUPer v{SUPVERS}, (c) {__author__}")

    #Do not keep returned params, we just want to initialize PILIQ
    init_extra_libs(application_path)
    opts_quant = Quantizer.get_options()
    opts_log = {'Disabled':  0, 'Succint': -10, 'Standard': 20, 'Minimalist': 25, 'Warnings/errors': 30, 'Debug': 10, 'Max debug': 5}

    pos_v = 0

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    if is_win32:
        signal.signal(signal.SIGBREAK, terminate)
    else:
        signal.signal(signal.SIGQUIT, terminate)        

    app = App(title=f"SUPer {SUPVERS}", layout='grid')
    meipass = getattr(sys, '_MEIPASS', None)
    ico_paths = Path(Path.cwd() if meipass is None else meipass)
    ico_paths = next(filter(lambda x: x.exists(), map(lambda fl: Path.joinpath(ico_paths, fl, 'icon.ico'), ['misc', 'lib', '.'])), None)
    if ico_paths is not None:#and not (is_win32 and meipass is not None):
        from PIL import Image
        app.icon = Image.open(ico_paths)

    PushButton(app, command=get_bdnxml, text="Select bdn.xml file", grid=[0,pos_v:=pos_v+1],align='left', width=15)
    bdnname = Text(app, grid=[1,pos_v], align='left', size=10)

    PushButton(app, command=set_outputsup, text="Set output", grid=[0,pos_v:=pos_v+1], align='left', width=15)
    supout = Text(app, grid=[1,pos_v], align='left', size=10)

    do_super = PushButton(app, command=wrapper_mp, text=SUPER_STRING, grid=[0,pos_v:=pos_v+1,2,1], align='left', enabled=False)
    do_abort = PushButton(app, command=abort, text="Abort", grid=[1,pos_v], align='left', enabled=False)

    do_super.queue = mp.Queue(10)
    do_super.proc = None
    do_super.ts = time.time()
    do_super.text_color = 'red'
    do_super.sup_kwargs = {}

    bcompre = Box(app, layout="grid", grid=[0,pos_v:=pos_v+1])
    compression_txtstr = Text(bcompre, "Compression [int]%: ", grid=[0,0], align='left', size=11)
    compression_txt = TextBox(bcompre, width=4, height=1, grid=[1,0], text="85")
    Hovertip(bcompre.tk, "Defined as the minimum percentage of time to have between two events to perform an acquisition (object refresh).\n"\
                         "-> 0: update as often as possible, -> 100 update as few times as possible.")

    brate = Box(app, layout="grid", grid=[1,pos_v], align='left')
    brate_txtstr = Text(brate, "Acquisition rate [int]%: ", grid=[0,0], align='left', size=11)
    refresh_txt = TextBox(brate, width=4, height=1, grid=[1,0], text="100")
    Hovertip(brate.tk, "Affect the decay ratio that determines the compression factor and thus, PG acquisitions (object refreshes).\n"\
                              "Low values: slow decay -> fewer acquisitions. High values: more often (always within PG decoders limits).\n"\
                              "A value of zero results in the strict minimum number of refreshes and may show artifacts.")

    bspace = Box(app, layout="grid", grid=[0,pos_v:=pos_v+1])
    Text(bspace, "Color space: ", grid=[0,0], align='right', size=11)
    colorspace = Combo(bspace, options=["bt709", "bt601", "bt2020"], grid=[1,0], align='left')

    bquant = Box(app, layout="grid", grid=[1, pos_v], align='left')
    Text(bquant, "Quantizer: ", grid=[0,0], align='left', size=11)
    quantcombo = Combo(bquant, options=list(map(lambda x: ' '.join(x), opts_quant.values())), grid=[1,0], align='left')
    Hovertip(bquant.tk, "Image quantizer to use (Quality, Speed).\n")

    bthread = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1])
    Text(bthread, "Threads: ", grid=[0,0], align='left', size=11)
    threadscombo = Combo(bthread, options=['auto'] + list(range(1, 9)), grid=[1,0], align='left', selected='auto')

    normal_case_ok = CheckBox(app, text="Allow normal case object redefinition.", grid=[0,pos_v:=pos_v+1,2,1], align='left', command=hide_chkbox)
    Hovertip(normal_case_ok.tk, "Update only one composition out of the two, whenever updating both is not possible due to time constraints.\n"\
                                "This exploits the PG object buffer capabilities as intended by the format designers.\n"\
                                "Stream shall NOT be Built or Rebuilt at the authoring stage.")

    prefer_normal_case = CheckBox(app, text="Prefer normal case object redefinition.", grid=[0,pos_v:=pos_v+1,2,1], align='left', command=hide_chkbox)
    Hovertip(prefer_normal_case.tk, "Update only one composition out of the two, even when decoding time is sufficient to refresh both (default).\n"\
                                    "It can reduce the bitrate, but the palette is not shared across composition objects whenever it occurs.")

    allow_overlaps = CheckBox(app, text="Allow palette update buffering.", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(allow_overlaps.tk, "Buffer palette updates whenever possible to drop fewer events.\n"\
                                "Stream shall NOT be Built or Rebuilt at the authoring stage.")

    fullpalette = CheckBox(app, text="Write the full palette.", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(fullpalette.tk, "Some authoring suites mendle with the imported files and may mess up the palette assignments.\n"\
                             "Writing the full palette everytime ensures palette data consistency throughout the stream.")

    all_formats = CheckBox(app, text="Generate both SUP and PES+MUI files.", grid=[0,pos_v:=pos_v+1,2,1], align='left', command=hide_chkbox)

    biacqs = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1, 2, 1], align='left')
    biacqs_val = TextBox(biacqs, width=4, height=1, grid=[1,0], text="3")
    Text(biacqs, "Insert acquisition after N palette updates [0: off, 2-5: advised]: ", grid=[0,0], align='left', size=11)
    Hovertip(biacqs.tk, "Long palette effects can alter the bitmap quality and be visible to the viewer if the end\n"\
                        "result remains on screen. To improve psychovisual quality, an acquisition can be added after\n"\
                        "to hide small artifacts originating from the palette animation encoding.")

    bacqint = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1, 2, 1], align='left')
    acqinttb = TextBox(bacqint, width=4, height=1, grid=[1,0], text="0.0")
    Text(bacqint, "Anchor interval [1.0 or above, 0: disabled] (seconds): ", grid=[0,0], align='left', size=11)
    Hovertip(bacqint.tk, "Insert anchors at the specified interval to let decoders catch-up on long-lasting events.\n"\
                         "0: disabled. minimum: 1 second. Small values may increase bitrate significantly.")


    bssimtol = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1, 2, 1], align='left')
    ssim_tolb = TextBox(bssimtol, width=4, height=1, grid=[1,0], text="0")
    Text(bssimtol, "SSIM tolerance offset [-100;100]: ", grid=[0,0], align='left', size=11)
    Hovertip(bssimtol.tk, "Adjust the similarity threshold to classify two images as similar, to potentially encode them as one.\n"\
                        "A positive value may fix the smearing on selected moves, a lower value may reduce the bitrate.\n"\
                        "The set value should rarely differ from zero.")

    bmax_kbps = Box(app, layout="grid", grid=[0,pos_v:=pos_v+1])
    max_kbps = TextBox(bmax_kbps, width=6, height=1, grid=[1,0], text="16000", align='left')
    Text(bmax_kbps, "Max bitrate test [Kbps]: ", grid=[0,0], align='left', size=11)
    Hovertip(bmax_kbps.tk, "Test the stream against the given bitrate. This value does not shape the output.\n"\
                           "Change the quantizer, compression, acquisition, SSIM parameters to lower the bitrate.\n"\
                           "Set to zero to disable the test. Unrealistic values will lead to a spam of messages.")

    blog = Box(app, layout="grid", grid=[1, pos_v], align='left')
    Text(blog, "Log to file: ", grid=[0,0], align='left', size=11)
    logcombo = Combo(blog, options=list(opts_log), grid=[1,0], align='left')

    Text(app, grid=[0,pos_v:=pos_v+2,2,1], align='left', text=" "*11 + "Progress data is displayed in the command line!", size=11)
    app.repeat(1000, monitor_mp)  # Schedule call to monitor_mp() every 1000ms

    app.when_closed = terminate
    app.display()
    sys.exit(0)
