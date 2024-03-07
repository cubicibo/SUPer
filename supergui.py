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
from pathlib import Path
from typing import Optional, Any, Callable

from guizero import App, PushButton, Text, CheckBox, Combo, Box, TextBox
from idlelib.tooltip import Hovertip

from warnings import filterwarnings
filterwarnings("ignore", message=r"Non-empty compiler", module="pyopencl")

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

    ts_start = time.monotonic()
    logger.info(f"Loading input BDN: {bdnf}")
    sup_obj = BDNRender(bdnf, kwargs, supo)
    sup_obj.optimise()
    sup_obj.write_output()
    logger.info(f"Finished in {timedelta(seconds=round(time.monotonic() - ts_start, 3))}, exiting...")
####

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    print("Loading...")

from SUPer import BDNRender, LogFacility
from SUPer.optim import Quantizer
from SUPer.__metadata__ import __version__ as SUPVERS, __author__

#### Functions, main at the end of the file
def get_kwargs() -> dict[str, int]:
    return {
        'quality_factor': int(compression_txt.value)/100,
        'refresh_rate': int(refresh_txt.value)/100,
        'quantize_lib': Quantizer.get_option_id(quantcombo.value),
        'bt_colorspace': colorspace.value,
        'no_overlap': True, #scenarist_checks.value,
        'full_palette': bool(scenarist_fullpal.value),
        'output_all_formats': bool(all_formats.value),
        'normal_case_ok': bool(normal_case_ok.value),
        'insert_acquisitions': int(biacqs_val.value),
        'ini_opts': ini_opts,
        'max_kbps': int(max_kbps.value),
        'log_to_file': opts_log[logcombo.value],
        'ssim_tol': int(ssim_tolb.value)/100,
        'threads': int(threadscombo.value),
        'daemonize': False,
    }

def wrapper_mp() -> None:
    if supout.value == '' or bdnname.value == '':
        return
    try:
        kwargs = get_kwargs()
    except ValueError:
        logger.error("Incorrect parameter(s), aborting.")
        return
    else:
        invalid = False
        invalid |= not (abs(kwargs['ssim_tol']) <= 1)
        invalid |= not (0 <= kwargs['insert_acquisitions'])
        invalid |= not (0 <= kwargs['quality_factor'] <= 1)
        invalid |= not (0 <= kwargs['refresh_rate'] <= 1)
        invalid |= not (1 <= kwargs['threads'] <= 8)
        invalid |= not (0 <= kwargs['max_kbps'] <= 48000)
        if invalid:
            logger.error("Invalid parameter found, aborting.")
            return

    do_super.enabled = False
    do_abort.enabled = True #and (1 == kwargs['threads'])
    logger.info("Starting optimiser process.")
    do_super.text = "Generating (check console)..."
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
            logger.info("Closed gracefully SUPer process.")
            do_super.ts = time.time()
            do_reset = True
            do_abort.enabled = False
    if do_reset and bdnname.value and supout.value:
        do_super.enabled = True
        do_super.text = SUPER_STRING

def hide_chkbox() -> None:
    if all_formats.value:
        scenarist_fullpal.value = True
        #scenarist_checks.value = True
        scenarist_fullpal.enabled = False
        #scenarist_checks.enabled = False
    elif not supout.value.lower().endswith('pes'):
        scenarist_fullpal.enabled = True
        #scenarist_checks.enabled = True

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
        #scenarist_checks.value = True
        #scenarist_checks.enabled = False
        scenarist_fullpal.value = True
        scenarist_fullpal.enabled = False
    else:
        if not all_formats.value:
            #scenarist_checks.enabled = True
            scenarist_fullpal.enabled = True

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

def init_extra_libs():
    def get_value_key(config, key: str) -> Optional[Any]:
        try: return config[key]
        except KeyError: return None
    ####
    params = {}
    CWD = Path(os.path.abspath(Path(sys.argv[0]).parent))
    ini_file = CWD.joinpath('config.ini')

    exepath = None
    piq_quality = 100
    if ini_file.exists():
        exepath, piq_quality = None, None
        import configparser
        config = configparser.ConfigParser()
        config.read(ini_file)
        if (piq_params := get_value_key(config, 'PILIQ')) is not None:
            if (exepath := get_value_key(piq_params, 'quantizer')) is not None and not os.path.isabs(exepath):
                exepath = str(CWD.joinpath(exepath))
            if (piq_quality := get_value_key(piq_params, 'quality')) is not None:
                piq_quality = int(piq_quality)
        if (sup_params := get_value_key(config, 'SUPer')) is not None:
            params['super_cfg'] = dict(sup_params)
    if Quantizer.init_piliq(exepath):
        logger.info(f"Advanced image quantizer armed: {Quantizer.get_piliq().lib_name}")
        params['quant'] = (exepath, piq_quality)
    return params

if __name__ == '__main__':
    logger = LogFacility.get_logger('SUPui')
    logger.info(f"SUPer v{SUPVERS}, (c) {__author__}")

    ini_opts = init_extra_libs()
    opts_quant = Quantizer.get_options()
    opts_log = {'Disabled':  0, 'Standard': 20, 'Minimalist': 25, 'Warnings/errors': 30, 'Debug': 10, 'Max debug': 5}

    pos_v = 0

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    if sys.platform != 'win32':
        signal.signal(signal.SIGQUIT, terminate)
    else:
        signal.signal(signal.SIGBREAK, terminate)

    app = App(title=f"SUPer {SUPVERS}", layout='grid')

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
    Text(bquant, "Quantization: ", grid=[0,0], align='left', size=11)
    quantcombo = Combo(bquant, options=list(map(lambda x: ' '.join(x), opts_quant.values())), grid=[1,0], align='left')

    bthread = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1])
    Text(bthread, "Threads: ", grid=[0,0], align='left', size=11)
    threadscombo = Combo(bthread, options=list(range(1, 9)), grid=[1,0], align='left')

    normal_case_ok = CheckBox(app, text="Allow normal case object redefinition.", grid=[0,pos_v:=pos_v+1,2,1], align='left', command=hide_chkbox)
    Hovertip(normal_case_ok.tk, "This option may reduce the number of dropped events on complicated animations.\n"\
                                "When there are two objects on screen and one must be updated, it may be possible\n"\
                                "to update the given object in a tighter time window than in an acquisition (both objects refreshed).")

    #scenarist_checks = CheckBox(app, text="Apply additional compliancy rules for Scenarist BD", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    #scenarist_checks.value = 1
    #Hovertip(scenarist_checks.tk, "Scenarist BD has additional hard rules. This checkbox enforces them\n"\
    #                              "and the generated stream shall pass all Scenarist checks.")

    scenarist_fullpal = CheckBox(app, text="Always write the full palette", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(scenarist_fullpal.tk, "Scenarist BD mendles with the imported files and may mess up the palette assignments.\n"\
                                   "Writing the full palette everytime ensures palette data consistency throughout the stream.")

    all_formats = CheckBox(app, text="Generate both SUP and PES+MUI files.", grid=[0,pos_v:=pos_v+1,2,1], align='left', command=hide_chkbox)

    biacqs = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1, 2, 1], align='left')
    biacqs_val = TextBox(biacqs, width=2, height=1, grid=[0,0], text="2")
    Text(biacqs, "Insert acquisition after N palette updates. [0: off, 3: recommended].", grid=[1,0], align='left')
    Hovertip(biacqs.tk, "Long palette effects can alter the bitmap quality and be visible to the viewer if the end\n"\
                        "result remains on screen. To improve psychovisual quality, an acquisition can be added after\n"\
                        "to hide small artifacts originating from the palette animation encoding.")

    bssimtol = Box(app, layout="grid", grid=[0, pos_v:=pos_v+1, 2, 1], align='left')
    ssim_tolb = TextBox(bssimtol, width=3, height=1, grid=[0,0], text="0")
    Text(bssimtol, "SSIM tolerance offset [-100;100]", grid=[1,0], align='left')
    Hovertip(bssimtol.tk, "Higher sensitivity increases the needed structural similarity to classify two images as similar.\n"\
                        "similar images can be encoded as palette updates, while dissimilar ones require an acquisition.")

    bmax_kbps = Box(app, layout="grid", grid=[0,pos_v:=pos_v+1])
    max_kbps = TextBox(bmax_kbps, width=6, height=1, grid=[1,0], text="16000", align='left')
    Text(bmax_kbps, "Max bitrate test [Kbps]: ", grid=[0,0], align='left', size=11)
    Hovertip(bmax_kbps.tk, "Test the stream against the given bitrate. This value does not shape the output.\n"\
                           "Change the quantizer, compression and acquisition value to lower the bitrate.\n"\
                           "Set to zero to disable the test. Unrealistic values will lead to a spam of errors.")

    blog = Box(app, layout="grid", grid=[1, pos_v], align='left')
    Text(blog, "Log to file: ", grid=[0,0], align='left', size=11)
    logcombo = Combo(blog, options=list(opts_log), grid=[1,0], align='left')

    Text(app, grid=[0,pos_v:=pos_v+1,2,1], align='left', text="Progress data is displayed in the command line!")
    app.repeat(1000, monitor_mp)  # Schedule call to monitor_mp() every 1000ms

    app.when_closed = terminate
    app.display()
    sys.exit(0)
