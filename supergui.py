#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2023 cibo
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

if __name__ == '__main__':
    print("Loading...")

import sys
import time
import signal

from guizero import App, PushButton, Text, CheckBox, Combo, Box, TextBox
from idlelib.tooltip import Hovertip

from SUPer import BDNRender, get_super_logger
from SUPer.__metadata__ import __version__ as SUPVERS

### CONSTS
SUPER_STRING = "Make it SUPer!"

#### FUnctions, main at the end of the file
def get_kwargs() -> dict[str, int]:
    return {
        'quality_factor': int(compression_txt.value)/100,
        'refresh_rate': int(refresh_txt.value)/100,
        'adjust_dropframe': dropframebox.value,
        'scale_fps': scale_fps.value,
        'kmeans_fade': kmeans_fade.value,
        'kmeans_quant': kmeans_quant.value,
        'bt_colorspace': colorspace.value,
        'pgs_compatibility': compat_mode.value,
        'enforce_dts': set_dts.value,
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
        invalid |= not (0 <= kwargs['quality_factor'] <= 1)
        invalid |= not (0 <= kwargs['refresh_rate'] <= 1)
        if invalid:
            logger.error("Invalid parameter found, aborting.")
            return

    do_super.enabled = False
    logger.info("Starting optimiser process.")
    do_super.text = "Generating (check console)..."
    do_super.proc.start()
    do_super.queue.put(kwargs)
    do_super.queue.put(bdnname.value)
    do_super.queue.put(supout.value)
    do_super.queue.put(supname.value)
    do_super.ts = time.time()


def monitor_mp() -> None:
    from multiprocessing import Process
    do_reset = False
    if time.time()-do_super.ts < 2:
        return
    if do_super.proc and do_super.proc.pid:
        if not do_super.proc.is_alive():
            try:
                do_super.proc.join(0.1)
            except RuntimeError:
                ...
            else:
                logger.info("Closed gracefully SUPer process.")
                do_super.proc = Process(target=from_bdnxml, args=(do_super.queue,), daemon=True, name="SUPinternal")
                do_super.ts = time.time()
                do_reset = True
    if do_reset and bdnname.value and supout.value:
        do_super.enabled = True
        do_super.text = SUPER_STRING

def get_sup() -> None:
    pg_sup_types = ('*.sup', '*.SUP')
    file_returned = app.select_file(filetypes=[["SUP", pg_sup_types], ["All files", "*"]])
    supname.value = file_returned

def get_bdnxml() -> None:
    bdn_xml_types = ("*.xml", "*.XML")
    file_returned = app.select_file(filetypes=[["BDNXML", bdn_xml_types], ["All files", "*"]])
    bdnname.value = file_returned
    if supout.value != '':
        do_super.enabled = True

def set_outputsup() -> None:
    pg_sup_types = ('*.sup', '*.SUP')
    pg_pes_types = ('*.pes', '*.PES')
    file_returned = app.select_file(filetypes=[["SUP", pg_sup_types], ['PES', pg_pes_types], ["All files", "*"]], save=True)
    supout.value = file_returned
    if bdnname.value != '':
        do_super.enabled = True

####
def terminate(frame = None, sig = None):
    global app
    global do_super
    app.cancel(monitor_mp)
    app.destroy()

    if do_super.proc is not None:
        proc, do_super.proc = do_super.proc, None
        try:
            proc.kill()
        except:
            ...
        else:
            proc.join(0.1)

def from_bdnxml(queue: ...) -> None:
    logger = get_super_logger('SUPer')
    kwargs = queue.get()
    bdnf = queue.get()
    supo = queue.get()
    try:
        supi = queue.get(timeout = 1)
    except:
        supi = ''

    #### This function is run in MP context, not main.
    logger.info(f"Loading input BDN: {bdnf}")
    sup_obj = BDNRender(bdnf, kwargs)

    sup_obj.optimise()

    if supi != '':
        logger.info(f"Merging output with {supi}")
        sup_obj.merge(supi)

    logger.info(f"Writing output file {supo}")
    sup_obj.write_output(supo)

    logger.info("Finished, exiting...")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()

    pos_v = 0

    logger = get_super_logger('SUPui')
    logger.info(f"SUPer v{SUPVERS}")

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    if sys.platform != 'win32':
        signal.signal(signal.SIGQUIT, terminate)
    else:
        signal.signal(signal.SIGBREAK, terminate)

    app = App(title=f"SUPer {SUPVERS}", layout='grid')

    PushButton(app, command=get_sup, text="Select SUP to inject (opt.)", grid=[0,pos_v], align='left', width=15)
    supname = Text(app, grid=[1,pos_v], align='left', size=10)

    PushButton(app, command=get_bdnxml, text="Select bdn.xml file", grid=[0,pos_v:=pos_v+1],align='left', width=15)
    bdnname = Text(app, grid=[1,pos_v], align='left', size=10)

    PushButton(app, command=set_outputsup, text="Set output", grid=[0,pos_v:=pos_v+1], align='left', width=15)
    supout = Text(app, grid=[1,pos_v], align='left', size=10)

    do_super = PushButton(app, command=wrapper_mp, text=SUPER_STRING, grid=[0,pos_v:=pos_v+1,2,1], align='left', enabled=False)
    do_super.queue = mp.Queue(10)
    do_super.proc = mp.Process(target=from_bdnxml, args=(do_super.queue,), daemon=True, name="SUPinternal")
    do_super.ts = time.time()
    do_super.text_color = 'red'
    do_super.sup_kwargs = {}

    bcompre = Box(app, layout="grid", grid=[0,pos_v:=pos_v+1])
    brate = Box(app, layout="grid", grid=[1,pos_v])
    compression_txtstr = Text(bcompre, "Compression [int]%: ", grid=[0,0], align='left', size=11)
    Hovertip(compression_txtstr.tk, "Defined as the minimum percentage of time to have between two events to perform an acquisition (object refresh).\n"\
                              "-> 0: update as often as possible, -> 100 update as few times as possible.")

    compression_txt = TextBox(bcompre, width=4, height=1, grid=[1,0], text="80")
    brate_txtstr = Text(brate, "Acquisition rate [int]%: ", grid=[0,0], align='left', size=11)
    refresh_txt = TextBox(brate, width=4, height=1, grid=[1,0], text="100")
    Hovertip(brate_txtstr.tk, "Affect the decay ratio that determines the compression factor and thus, PG acquisitions (object refreshes).\n"\
                              "Low values: slow decay -> fewer acquisitions. High values: more often (always within PG decoders limits).\n"\
                              "A value of zero results in the strict minimum number of refreshes and may show artifacts.")

    kmeans_fade = CheckBox(app, text="Use KMeans quantization on fades", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(kmeans_fade.tk, "Use K-Means to quantize highly transparent image. This is a work-around\n"\
                             "to a known bug in Pillow (PIL) with RGBA images.")

    kmeans_quant = CheckBox(app, text="Use KMeans quantization everywhere (slow)", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(kmeans_quant.tk, "Use K-Means to quantize and palettize images. This offers the highest quality.")

    dropframebox = CheckBox(app, text="Correct NTSC timings (*1.001)", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(dropframebox.tk, "Multiply timestamps by 1.001 to fix NTSC drifts (30/29.97, 24/23.976...).")

    scale_fps = CheckBox(app, text="Subsampled BDNXML (e.g. 29.97 BDNXML for 59.94 SUP, ignored if 24p)", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(scale_fps.tk, "A BDNXML generated at half the framerate will limit the pressure on the PG decoder\n"\
                                   "while ensuring synchronicity with the video footage. This is recommended for 50i/60i content.\n"\
                                   "E.g if the target is 59.94, the BDNXML would be generated at 29.97. SUPer would then write the PGS\n"\
                                   "as if it was 59.94. This flag is ignored for 23.976p or 24p content.")

    compat_mode = CheckBox(app, text="Compatibility mode for software players (see tooltip)", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(compat_mode.tk, "Software players don't decode palette updates with two objects correctly or cropping.\n"\
                                       "If enabled, SUPer insert instructions for the decoder to redraw the graphic plane.\n"\
                                       "I.e, the decoder re-copy existing objects in the buffer to the graphic plane and apply the new palette.\n"\
                                       "However, hardware decoders can only redraw a portion of the graphic plane per frame.\n"\
                                       "Should be unticked for commercial BDs.")

    set_dts = CheckBox(app, text="Set rough DTS in stream (for PES output)", grid=[0,pos_v:=pos_v+1,2,1], align='left')
    Hovertip(set_dts.tk, "PG streams can include a decoding timestamp. This is used by some muxers.\n"\
                         "This should only be required for PES+MUI output (Scenarist BD format).")

    bspace = Box(app, layout="grid", grid=[0,pos_v:=pos_v+1,2,1])
    Text(bspace, "Color space: ", grid=[0,0], align='right')
    colorspace = Combo(bspace, options=["bt709", "bt601", "bt2020"], grid=[1,0], align='left')

    Text(app, grid=[0,pos_v:=pos_v+1,2,1], align='left', text="Progress data is displayed in the command line!")
    app.repeat(1000, monitor_mp)  # Schedule call to monitor_mp() every 1000ms

    app.when_closed = terminate
    app.display()
    sys.exit(0)
