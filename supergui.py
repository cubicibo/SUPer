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


import sys
import multiprocessing as mp
import time
import signal

from guizero import App, PushButton, Text, CheckBox, Combo

from SUPer import BDNRender, get_super_logger
from SUPer.__metadata__ import __version__ as SUPVERS

### CONSTS
SUPER_STRING = "Make it SUPer!"

#### FUnctions, main at the end of the file
def get_kwargs() -> dict[str, int]:
    return {
        'merge_nonoverlap': merge_nonoverlap.value,
        'noblur_grouping': no_blur.value,
        'adjust_dropframe': dropframebox.value,
        'scale_fps': scale_fps.value,
        'kmeans_quant': kmeans_quant.value,
        'bt_colorspace': colorspace.value,
    }
    
def wrapper_mp() -> None:
    if supout.value == '' or bdnname.value == '':
        return
    do_super.enabled = False
    logger.info("Starting optimiser process.")
    do_super.text = "Generating (check console)..."
    do_super.proc.start()
    do_super.queue.put(get_kwargs())
    do_super.queue.put(bdnname.value)
    do_super.queue.put(supout.value)
    do_super.queue.put(supname.value)
    do_super.ts = time.time()


def monitor_mp() -> None:
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
                do_super.proc = mp.Process(target=from_bdnxml, args=(do_super.queue,), daemon=True, name="SUPinternal")
                do_super.ts = time.time()
                do_reset = True
    if do_reset and bdnname.value and supout.value:
        do_super.enabled = True
        do_super.text = SUPER_STRING

def get_sup() -> None:
    file_returned = app.select_file(filetypes=[["All files", "*.sup"], ["PGS", "*.sup"]])
    supname.value = file_returned

def get_bdnxml() -> None:
    file_returned = app.select_file(filetypes=[["All files", "*.sup"], ["BDNXML", "*.xml"]])
    bdnname.value = file_returned
    if supout.value != '':
        do_super.enabled = True

def set_outputsup() -> None:
    file_returned = app.select_file(filetypes=[["All files", "*.sup"], ["PGS", "*.sup"]], save=True)
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

def from_bdnxml(queue: mp.Queue) -> None:
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
    logger = get_super_logger('SUPui')
    logger.info(f"Loading SUPer {SUPVERS}...")
    
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGQUIT, terminate)
    
    app = App(title=f"SUPer {SUPVERS}", layout='grid')    
    
    PushButton(app, command=get_sup, text="Select SUP to inject (opt.)", grid=[0,0], align='left', width=13)
    supname = Text(app, grid=[1,0], align='left')
    
    PushButton(app, command=get_bdnxml, text="Select bdn.xml file", grid=[0,1],align='left', width=13)
    bdnname = Text(app, grid=[1,1], align='left')
    
    PushButton(app, command=set_outputsup, text="Set SUP output", grid=[0,2], align='left', width=13)
    supout = Text(app, grid=[1,2], align='left')
    
    do_super = PushButton(app, command=wrapper_mp, text=SUPER_STRING, grid=[0,3,2,1], align='left', enabled=False)
    do_super.queue = mp.Queue(10)
    do_super.proc = mp.Process(target=from_bdnxml, args=(do_super.queue,), daemon=True, name="SUPinternal")
    do_super.ts = time.time()
    do_super.text_color = 'red'
    do_super.sup_kwargs = {}
    
    merge_nonoverlap = CheckBox(app, text="Merge non-overlapping (can fix buffer issues but damage few animations)", grid=[0,4,2,1], align='left')
    kmeans_quant = CheckBox(app, text="KMeans quantize on fades (good for fades, bad for other animations)", grid=[0,5,2,1], align='left')
    no_blur = CheckBox(app, text="Disable blur grouping (recommended for SD content)", grid=[0,6,2,1], align='left')
    dropframebox = CheckBox(app, text="Force dropframe timing", grid=[0,7,2,1], align='left')
    scale_fps = CheckBox(app, text="Subsampled BDNXML (e.g. 29.97 BDNXML for 59.94 SUP, ignored with 24p)", grid=[0,8,2,1], align='left')
    
    Text(app, "Color space: ", grid=[0,9], align='right')
    colorspace = Combo(app, options=["bt709", "bt601", "bt2020"], grid=[1,9], align='left')

    #flatten = CheckBox(app, text="Reduce palette individually (can improve kara)", grid=[0,7,2,1], align='left')
    #flatten_grp = CheckBox(app, text="Reduce palette at once (can improve gradients)", grid=[0,8,2,1], align='left')

    Text(app, grid=[0,10,2,1], align='left', text="Progress data is displayed in the command line !")
    app.repeat(1000, monitor_mp)  # Schedule call to monitor_mp() every 1000ms

    app.when_closed = terminate
    app.display()
    sys.exit(0)