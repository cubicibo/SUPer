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

from typing import TypeVar, Optional, Type, Union, Callable
from dataclasses import dataclass
from itertools import chain, zip_longest

from PIL import Image
from SSIM_PIL import compare_ssim
from numpy import typing as npt
import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    from contextlib import nullcontext as tqdm

#%%
from .utils import LogFacility, Pos, Dim, BDVideo, TimeConv as TC, Box, ScreenRegion, RegionType, WindowOnBuffer
from .filestreams import BDNXMLEvent, BaseEvent
from .segments import DisplaySet, PCS, WDS, PDS, ODS, ENDS, WindowDefinition, CObject, Epoch
from .optim import Optimise
from .pgraphics import PGraphics, PGDecoder, PGObject, PGObjectBuffer, PaletteManager
from .palette import Palette, PaletteEntry

logger = LogFacility.get_logger('SUPer')

#%%
class GroupingEngine:
    def __init__(self, box: Box, n_groups: int = 2) -> None:
        if n_groups not in range(1, 3):
            raise AssertionError(f"GroupingEngine expects 1 or 2 groups, not '{n_groups}'.")
        self.box = box
        self.n_groups = n_groups

    def coarse_grouping(self, group: list[Type[BaseEvent]], box: Box) -> tuple[npt.NDArray[np.uint8]]:
        (pxtl, pytl), (w, h) = box.posdim
        gs_orig = np.zeros((1, h, w), dtype=np.uint8)

        for k, event in enumerate(group):
            slice_x = slice(event.x-pxtl, event.x-pxtl+event.width)
            slice_y = slice(event.y-pytl, event.y-pytl+event.height)
            alpha = np.array(event.img.getchannel('A'), dtype=np.uint8)
            event.unload()
            
            alpha[alpha > 0] = 1
            gs_orig[0, slice_y, slice_x] |= (alpha > 0)
        return gs_orig
    
    def find_layout(self, gs_origs: npt.NDArray[np.uint8]) -> tuple[WindowOnBuffer]:
        xl, yl, xr, yr = Image.fromarray(gs_origs[0, :, :], 'L').getbbox(alpha_only=False)
        if self.n_groups == 1 or (gs_origs.shape[1] < 8 and gs_origs.shape[2] < 8):
            logger.debug("Single window due to shape or n_groups")
            return (WindowOnBuffer([ScreenRegion(yl, yr-yl, xl, xr-xl, 0, 1, None)]),)
        
        def check_best(best_area: int, wd1: Box, wd2: Box, prev_wob: tuple[WindowOnBuffer]) -> tuple[int, tuple[WindowOnBuffer]]:
            #Clean up this mess. WOB and ScreenRegion are outdated containers.
            wd1 = WindowOnBuffer([ScreenRegion(wd1.y, wd1.dy, wd1.x, wd1.dx, 0, 1, None)])
            wd1b = wd1.get_window()
            wd2 = WindowOnBuffer([ScreenRegion(wd2.y, wd2.dy, wd2.x, wd2.dx, 0, 1, None)])
            wd2b = wd2.get_window()
    
            if Box.intersect(wd1b, wd2b).area > 0:
                wd = WindowOnBuffer(wd1.srs + wd2.srs)
                new_area = wd.get_window().area
                lwobs = (wd,)
            else:
                new_area = wd1b.area + wd2b.area
                lwobs = (wd1, wd2)
    
            if new_area < best_area:
                best_area = new_area
                best_wob = lwobs
            else:
                best_wob = prev_wob
            return best_area, best_wob
    
        best_wob = None
        best_area = np.inf
        for yj in range(yl+8, yr-8):
            top_wd = Box.from_coords(*Image.fromarray(gs_origs[0, :yj, :]).getbbox(alpha_only=False))
            xt0, yt0, xt1, yt1 = Image.fromarray(gs_origs[0, yj:, :]).getbbox(alpha_only=False)
            bottom_wd = Box.from_coords(xt0, yt0+yj, xt1, yt1+yj)
            best_area, best_wob = check_best(best_area, top_wd, bottom_wd, best_wob)
    
        for xj in range(xl+8, xr-8):
            left_wd = Box.from_coords(*Image.fromarray(gs_origs[0, :, :xj]).getbbox(alpha_only=False))
            xt0, yt0, xt1, yt1 = Image.fromarray(gs_origs[0, :, xj:]).getbbox(alpha_only=False)
            right_wd = Box.from_coords(xt0+xj, yt0, xt1+xj, yt1)
            best_area, best_wob = check_best(best_area, left_wd, right_wd, best_wob)
    
        # 356 = 32e6/90e3: number of pixels we can output in a tick. If area diff is smaller,
        # the tick overhead for dual windows/objects may not be worthwile. 
        if best_wob is None or sum(map(lambda x: x.get_window().area, best_wob)) >= (yr - yl)*(xr - xl) - 356:
            logger.debug("No layout found or a single window is as efficient.")
            return (WindowOnBuffer([ScreenRegion(yl, yr-yl, xl, xr-xl, 0, 1, None)]),)
    
        for wd in best_wob:
            wd = wd.get_window()
            assert wd.dx >= 8 and wd.dy >= 8, "Incorrect window or object size."
        assert len(best_wob) == 1 or Box.intersect(*list(map(lambda x: x.get_window(), best_wob))).area == 0
        return best_wob

    def group(self, subgroup: list[Type[BaseEvent]]) -> tuple[WindowOnBuffer]:
        gs_origs = self.coarse_grouping(subgroup, self.box)
        return self.find_layout(gs_origs)
####

#%%
class WOBSAnalyzer:
    def __init__(self, wobs: tuple[WindowOnBuffer], events: list[BDNXMLEvent], box: Box, fps: Union[float, int], bdn: ..., **kwargs):
        self.wobs = wobs
        self.events = events
        self.box = box
        self.target_fps = fps
        self.bdn = bdn
        self.kwargs = kwargs
        self.buffer = PGObjectBuffer()
        self.pcs_id = kwargs.pop('pcs_id', 0)

    def mask_event(self, window, event) -> Optional[npt.NDArray[np.uint8]]:
        if event is not None:
            #+8 for minimum object width and height
            work_plane = np.zeros((self.box.dy+8, self.box.dx+8, 4), dtype=np.uint8)

            hsi = slice(event.x-self.box.x, event.x-self.box.x+event.width)
            vsi = slice(event.y-self.box.y, event.y-self.box.y+event.height)
            work_plane[vsi, hsi, :] = np.array(event.img, dtype=np.uint8)
            event.unload() #Help a bit to save on RAM

            return work_plane[window.y:window.y2, window.x:window.x2, :]
        return None

    def analyze(self):
        allow_normal_case = self.kwargs.get('normal_case_ok', False)
        scale_pts = 1.001 if self.kwargs.get('adjust_dropframe', False) else 1
        dts_strat = self.kwargs.get('ts_long', False)
        enforce_dts = self.kwargs.get('enforce_dts', True)

        DSNode.configure(scale_pts, self.bdn.fps, dts_strat, enforce_dts)

        woba = []
        pm = PaletteManager()

        #Adjust slightly SSIM threshold depending of res
        ssim_score = 0.9608 + self.bdn.format.value[1]*(0.986-0.972)/(1080-480)

        #Init
        gens, windows = [], []
        for k, swob in enumerate(self.wobs):
            woba.append(WOBAnalyzer(swob, ssim_threshold=ssim_score))
            windows.append(swob.get_window())
            gens.append(woba[k].analyze())
            next(gens[-1])

        #get all windowed bitmaps
        pgobjs = [[] for k in range(len(windows))]
        for event in chain(self.events, [None]*2):
            if event is not None:
                logger.hdebug(f"Event TCin={event.tc_in}")
            for wid, (window, gen) in enumerate(zip(windows, gens)):
                try:
                    pgobj = gen.send(self.mask_event(window,  event))
                except StopIteration:
                    pgobj = None
                if pgobj is not None:
                    logger.debug(f"Window={wid} has new PGObject: f={pgobj.f}, S(mask)={len(pgobj.mask)}, mask={pgobj.mask}")
                    pgobjs[wid].append(pgobj)
        pgobjs_proc = [objs.copy() for objs in pgobjs]

        acqs, absolutes, margins, durs, nodes, flags, bslots, cboxes = self.find_acqs(pgobjs_proc, windows)
        allow_normal_case = self.kwargs.get('normal_case_ok', False)
        allow_overlaps = self.kwargs.get('allow_overlaps', False)

        states = [PCS.CompositionState.NORMAL] * len(acqs)
        states[0] = PCS.CompositionState.EPOCH_START
        drought = 0

        thresh = self.kwargs.get('quality_factor', 0.75)
        dthresh = self.kwargs.get('dquality_factor', 0.035)
        refresh_rate = max(0, min(self.kwargs.get('refresh_rate', 1.0), 1.0))

        positions = cboxes[0].copy()
        last_acq = 0
        for k, (acq, forced, margin, node) in enumerate(zip(acqs[1:], absolutes[1:], margins[1:], nodes[1:]), 1):
            if not node.nc_refresh:
                for wid in range(len(windows)):
                    box_assets = list(filter(lambda x: x != None, [positions[wid], cboxes[k][wid]]))
                    if len(box_assets) > 0:
                        cont = Box.union(*box_assets)
                        
                        if cont.dx > bslots[wid][1] or cont.dy > bslots[wid][0]:
                            assert cboxes[k][wid] is not None
                            states[k] = PCS.CompositionState.ACQUISITION
                            absolutes[k] = True
                            node.new_mask[wid] = True #For possible Normal case update
                            drought = 0
                        else:
                            positions[wid] = cont                    
                #### for wid
            #### if not nc
            if thresh == 0 and not node.nc_refresh:
                states[k] = PCS.CompositionState.ACQUISITION
                absolutes[k] = True
            if states[k] != PCS.CompositionState.ACQUISITION:
                if (forced or (acq and margin > max(thresh-dthresh*drought, 0))) and not node.nc_refresh:
                    states[k] = PCS.CompositionState.ACQUISITION
                    drought = 0
                else:
                    #try to not do too many acquisitions, as we want to compress the stream.
                    drought += 1*refresh_rate
                if states[k] == PCS.CompositionState.NORMAL:
                    nodes[k].nc_refresh = True
            if states[k] > 0:
                for zk in range(last_acq, k):
                    nodes[zk].pos = positions
                positions = cboxes[k].copy()
                last_acq = k
        for zk in range(last_acq, k+1):
            nodes[zk].pos = positions

        #At this point, we have the stream acquisitions. Some may be impossible,
        # so we have to filter out some less relevant events.
        pts_delta = nodes[0].write_duration()/PGDecoder.FREQ

        logger.debug("Backtracking to filter acquisitions and events.")
        k = len(states)-1
        while k > 0:
            #If the acquisition is not a mandatory one or was already discarded
            if not absolutes[k] or acqs[k] or flags[k] == -1:
                if flags[k] == -1:
                    logger.debug(f"Not analyzing event at {nodes[k].tc_pts} due to filtering (f={absolutes[k]}, a={flags[k]}).")
                k -= 1
                continue

            assert states[k] == PCS.CompositionState.ACQUISITION
            dts_start_nc = dts_start = nodes[k].dts()
            dropped_nc = dropped = 0
            j = j_nc = k - 1
            while j > 0 and (nodes[j].dts_end() >= dts_start or nodes[j].pts() + pts_delta >= nodes[k].pts()):
                if absolutes[j]:
                    dropped += 1
                j -= 1

            #Normal case is only possible if we discard past acquisitions that redefined the same object
            normal_case_possible = sum(nodes[k].new_mask) == 1 and sum(map(lambda x: x is not None, nodes[k].objects)) == 2
            normal_case_possible &= allow_normal_case
            if normal_case_possible:
                mask = nodes[k].new_mask.copy()
                nodes[k].partial = True
                dts_start_nc = nodes[k].dts()

                while j_nc > 0 and (nodes[j_nc].dts_end() >= dts_start_nc or nodes[j_nc].pts() + pts_delta >= nodes[k].pts()):
                    if absolutes[j_nc]:
                        for km, mask_v in enumerate(nodes[j_nc].new_mask):
                            mask[km] |= mask_v
                        dropped_nc += 1
                    j_nc -= 1
                # Normal case
                normal_case_possible &= sum(mask) == 1
                nodes[k].partial = False

            #Normal case is not possible (collision with epoch start)
            nc_not_ok = normal_case_possible and j_nc == 0 and (nodes[j_nc].dts_end() >= dts_start_nc or nodes[j_nc].pts() + pts_delta >= nodes[k].pts())
            #
            if nc_not_ok or (not normal_case_possible and j == 0 and (nodes[j].dts_end() >= dts_start or nodes[j].pts() + pts_delta >= nodes[k].pts())):
                #If this event is long enough, we shift it forward in time.
                wipe_area = nodes[j].wipe_duration()
                worst_dur = (np.ceil(wipe_area*2) + 3)

                #K-node are always a mandatory acquisition
                if durs[k]/self.bdn.fps > np.ceil(worst_dur*2+PGDecoder.FREQ/self.bdn.fps)/PGDecoder.FREQ:
                    nodes[k].tc_shift = int(np.ceil(worst_dur/PGDecoder.FREQ*self.bdn.fps))
                    logger.warning(f"Shifted event at {nodes[k].tc_pts} by +{nodes[k].tc_shift} frames to account for epoch start and compliancy.")
                    #wipe all events in between epoch start and this point
                    for ze in range(j+1, k):
                        logger.warning(f"Discarded event at {nodes[k].tc_pts} to perform a mendatory acquisition right after epoch start.")
                        flags[ze] = -1
                else:
                    # event is short, we can't shift it so we just discard it.
                    logger.warning(f"Discarded event at {nodes[k].tc_pts} colliding with epoch start.")
                    flags[k] = -1

                    ze = k
                    #We may have discarded an acquisition followed by NCs, we must find the new acquisition point.
                    while (ze := ze+1) < len(states) and states[ze] != PCS.CompositionState.ACQUISITION:
                        #Screen wipes do not define any composition
                        if nodes[ze].objects == [] and nodes[ze].nc_refresh:
                            continue
                        nodes[ze].nc_refresh = False
                        if flags[ze] != -1 and (nodes[ze].dts() > dts_start and nodes[ze].pts() - pts_delta > nodes[0].pts()):
                            logger.info(f"Epoch start collision: promoted normal case to acquisition at {nodes[ze].tc_pts}.")
                            states[ze] = PCS.CompositionState.ACQUISITION
                            for zek in range(k+1, ze):
                                if nodes[zek].dts_end() >= nodes[ze].dts() or nodes[zek].pts() + pts_delta >= nodes[ze].pts():
                                    flags[zek] = -1
                                    logger.warning(f"Dropped event at {nodes[zek].tc_pts} as it hinders the promoted acquisition point.")
                            break
                    ###while ze
                ###event shift
            else:
                #Filter the events
                is_normal_case = dts_start_nc > dts_start and normal_case_possible and j_nc > j
                j_iter = j_nc if is_normal_case else j
                dts_iter = dts_start_nc if is_normal_case else dts_start

                num_pcs_buffered = 0
                objs = list(map(lambda x: x is not None, nodes[j_iter].objects))
                for l in range(j_iter+1, k):
                    if allow_overlaps:
                        for ko, (obj, mask) in enumerate(zip(nodes[l].objects, nodes[l].new_mask)):
                            objs[ko] &= (obj is not None) & (not mask)
                    # We ran out of PCS to buffer or the objects are too different or min delta PTS -> drop
                    if not allow_overlaps or sum(objs) == 0 or num_pcs_buffered >= 7 or nodes[l].pts() + pts_delta >= nodes[k].pts():
                        logger.warning(f"Discarded event at {nodes[l].tc_pts} to perform a mendatory acquisition.")
                        flags[l] = -1
                    else:
                        num_pcs_buffered += 1
                        nodes[l].nc_refresh = True
                        if nodes[l].dts() >= dts_iter:
                            nodes[l].set_dts(dts_iter - 1/PGDecoder.FREQ)
                    states[l] = PCS.CompositionState.NORMAL

                states[k] = PCS.CompositionState.NORMAL if is_normal_case else PCS.CompositionState.ACQUISITION
                nodes[k].partial = is_normal_case
                flags[k] = int(is_normal_case)
                if is_normal_case:
                    logger.info(f"Object refreshed with a NORMAL CASE at {nodes[k].tc_pts} (tight timing).")
            k -= 1
        ####while k > 0

        #Allocate palettes as a test, this is essentially doing a final sanity check
        #on the selected display sets. The palette values generated here are not used.
        for k, (node, state, flag) in enumerate(zip(nodes, states, flags)):
            if flag == 0 and state == PCS.CompositionState.NORMAL:
                assert nodes[k].nc_refresh
                assert allow_overlaps or not node.is_custom_dts()
            elif flag == 1:
                assert state == PCS.CompositionState.NORMAL
                assert nodes[k].objects != [] and sum(nodes[k].new_mask) == 1
            #Deleted event are skipped
            if flag == -1:
                continue
            node.palette_id = pm.get_palette(node.dts())
            if not pm.lock_palette(node.palette_id, node.pts(), node.dts()):
                logger.error(f"Cannot acquire palette (rendering error) at {nodes[k].pts()}, discarding.")
                flags[k] = -1
            else:
                node.pal_vn = pm.get_palette_version(node.palette_id)
            logger.debug(f"{state:02X} {flag} - {node.partial} DTS={node.dts():.05f}->{node.dts_end():.05f} PTS={node.pts():.05f} {node.palette_id} {node.pal_vn}")
        ####
        r_states, r_durs, r_nodes, r_flags = self.roll_nodes(nodes, durs, flags, states)
        return self._convert(r_states, pgobjs, windows, r_durs, r_flags, r_nodes)
    ####

    @staticmethod
    def _get_stack_direction(*box) -> tuple[npt.NDArray[np.uint16], tuple[int, int]]:
        widths = list(map(lambda b: b.dx, box))
        heights = list(map(lambda b: b.dy, box))

        if max(heights)*sum(widths) <= max(widths)*sum(heights):
            return np.array([widths[0], 0], np.int32), (sum(widths), max(heights))
        return np.array([0, heights[0]], np.int32), (max(widths), sum(heights))

    def _generate_acquisition_ds(self, i: int, k: int, pgobs_items, windows: list[Box], node: 'DSNode', 
                                 double_buffering: int, has_two_objs: bool, is_compat_mode: bool,
                                 ods_reg: list[int], c_pts: float, normal_case_refresh: bool, flags: list[int]) -> ...:
        box_to_crop = lambda cbox: {'hc_pos': cbox.x, 'vc_pos': cbox.y, 'c_w': cbox.dx, 'c_h': cbox.dy}
        cobjs, cobjs_cropped = [], []
        pals, o_ods = [], []

        #In this mode, we re-combine the two objects in a smaller areas than in the original box
        # and then pass that to the optimiser. Colors are efficiently distributed on the objects.
        # In the future, this will be the default behaviour unless there's a NORMAL CASE to update
        # to redefine an object in the middle.
        if has_two_objs and normal_case_refresh is False:
            compositions = [pgo for _, pgo in pgobs_items if not (pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f]))]

            offset, dims = self.__class__._get_stack_direction(*list(map(lambda x: x.box, compositions)))
            imgs_chain = []

            for j in range(i, k):
                coords = np.zeros((2,), np.int32)
                a_img = Image.new('RGBA', dims, (0, 0, 0, 0))
                multiplier = int(flags[j] >= 0)
                for pgo in compositions:
                    if len(pgo.mask[j-pgo.f:j+1-pgo.f]) == 1:
                        paste_box = (coords[0], coords[1], coords[0]+pgo.box.dx, coords[1]+pgo.box.dy)
                        a_img.paste(Image.fromarray(multiplier*pgo.gfx[j-pgo.f, :, : ,:], 'RGBA').crop(pgo.box.coords), paste_box)
                    coords += offset
                imgs_chain.append(a_img)
            ####
            #We have the "packed" object, the entire palette is usable
            bitmap, palettes = Optimise.solve_and_remap(imgs_chain, 255, 1, **self.kwargs)
            pals.append(palettes)

            coords = np.zeros((2,), np.int32)
            for wid, pgo in pgobs_items:
                if not (pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f])):
                    oid = wid + double_buffering[wid]
                    double_buffering[wid] = abs(2 - double_buffering[wid])
                    #get bitmap
                    window_bitmap = 0xFF*np.ones((windows[wid].dy, windows[wid].dx), np.uint8)
                    nx, ny = coords
                    window_bitmap[pgo.box.slice] = bitmap[ny:ny+pgo.box.dy, nx:nx+pgo.box.dx]

                    #Generate object related segments objects
                    oxl = max(0, node.pos[wid].x2 - node.slots[wid][1])
                    oyl = max(0, node.pos[wid].y2 - node.slots[wid][0])
                    cpx = windows[wid].x + self.box.x + oxl
                    cpy = windows[wid].y + self.box.y + oyl

                    cobjs.append(CObject.from_scratch(oid, wid, cpx, cpy, False))
                    # cparams = box_to_crop(pgo.box)
                    # cobjs_cropped.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x+cparams['hc_pos'], windows[wid].y+self.box.y+cparams['vc_pos'], False,
                    #                                           cropped=True, **cparams))
                    window_bitmap = window_bitmap[oyl:oyl+node.slots[wid][0], oxl:oxl+node.slots[wid][1]]
                    ods_data = PGraphics.encode_rle(window_bitmap)
                    o_ods += ODS.from_scratch(oid, ods_reg[oid] & 0xFF, window_bitmap.shape[1], window_bitmap.shape[0], ods_data, pts=c_pts)
                    assert window_bitmap.shape == node.slots[wid]
                    ods_reg[oid] += 1
                    coords += offset
            pals.append([Palette()] * len(pals[0]))
            ####for wid, pgo
        else:
            # If in the chain there's a NORMAL CASE redefinition, we
            # must work with separate palette for each object (127 colors per window)
            id_skipped = None
            for wid, pgo in pgobs_items:
                if pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f]):
                    optional_skip = True
                    if normal_case_refresh:
                        #An object may exist but be masked for the whole acquisition, ignore it.
                        optional_skip = pgo is None or (isinstance(normal_case_refresh, bool) and not np.any(pgo.mask[i-pgo.f:k-pgo.f]))
                        if optional_skip:
                            assert isinstance(normal_case_refresh, bool)
                            pals.append([Palette()] * (k-i))
                        else:
                            assert isinstance(normal_case_refresh, list)
                    if optional_skip:
                        continue

                oxl = max(0, node.pos[wid].x2 - node.slots[wid][1])
                oyl = max(0, node.pos[wid].y2 - node.slots[wid][0])
                cpx = windows[wid].x + self.box.x + oxl
                cpy = windows[wid].y + self.box.y + oyl

                if isinstance(normal_case_refresh, list) and not normal_case_refresh[wid]:
                    assert sum(normal_case_refresh) == 1
                    #Take latest used object id
                    oid = wid + abs(2 - double_buffering[wid])
                    cobjs.append(CObject.from_scratch(oid, wid, cpx, cpy, False))
                    # cparams = box_to_crop(pgo.box)
                    # cobjs_cropped.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x+cparams['hc_pos'], windows[wid].y+self.box.y+cparams['vc_pos'],
                    #                                           False, cropped=True, **cparams))
                    pals.append([Palette()] * (k-i))
                    id_skipped = oid
                    continue
                oid = wid + double_buffering[wid]
                double_buffering[wid] = abs(2 - double_buffering[wid])

                assert len(flags[i:k]) >= len(pgo.gfx[i-pgo.f:k-pgo.f])
                imgs_chain = [Image.fromarray(img*int(flag >= 0)) for img, flag in zip(pgo.gfx[i-pgo.f:k-pgo.f], flags[i:k])]
                
                cobjs.append(CObject.from_scratch(oid, wid, cpx, cpy, False))
                # cparams = box_to_crop(pgo.box)
                # cobjs_cropped.append(CObject.from_scratch(oid, wid, windows[wid].x+self.box.x+cparams['hc_pos'], windows[wid].y+self.box.y+cparams['vc_pos'], False,
                #                                           cropped=True, **cparams))

                n_colors = 128 if has_two_objs else 255
                clut_offset = 1 + (127*(wid == 1 and has_two_objs))
                wd_bitmap, wd_pal = Optimise.solve_and_remap(imgs_chain, n_colors, clut_offset, **self.kwargs)
                wd_bitmap = wd_bitmap[oyl:oyl+node.slots[wid][0], oxl:oxl+node.slots[wid][1]]
                pals.append(wd_pal)
                ods_data = PGraphics.encode_rle(wd_bitmap)

                #On normal case, we generate one chain of palette update and
                #add in a screen wipe if necessary. This is not used if the object is changed.
                if normal_case_refresh and len(pals[-1]) < k-i:
                    mibm, mabm = min(wd_pal[0].palette), max(wd_pal[0].palette)
                    pals[-1].append(Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(mibm, mabm+1)}))
                    pals[-1].extend([Palette()] * ((k-i)-len(pals[-1])))
                
                o_ods += ODS.from_scratch(oid, ods_reg[oid] & 0xFF, wd_bitmap.shape[1], wd_bitmap.shape[0], ods_data, pts=c_pts)
                ods_reg[oid] += 1
            if id_skipped is not None:
                assert isinstance(normal_case_refresh, list)
                #The END segment tells the decoder to use whatever it has in the buffer
                # for objects not defined in the current display set.
                f_is_first_cobj = lambda cobj: cobj.o_id == id_skipped
                #So the refreshed object has to come first in the composition list (key eval to zero)
                cobjs = sorted(cobjs, key=f_is_first_cobj)
                # cobjs_cropped = sorted(cobjs_cropped, key=f_is_first_cobj)

        pal = pals[0][0]
        if has_two_objs:
            pal |= pals[1][0]
        else:
            pals.append([Palette()] * len(pals[0]))

        return cobjs if is_compat_mode else cobjs_cropped, pals, o_ods, pal

    def _get_undisplay(self, c_pts: float, pcs_id: int, wds_base: WDS, palette_id: int, pcs_fn: Callable[[...], PCS]) -> tuple[DisplaySet, int]:
        pcs = pcs_fn(pcs_id, PCS.CompositionState.NORMAL, False, palette_id, [], c_pts)
        wds = wds_base.copy(pts=c_pts, in_ticks=False)
        uds = DisplaySet([pcs, wds, ENDS.from_scratch(pts=c_pts)])
        DSNode.apply_pts_dts(uds, DSNode.set_pts_dts_sc(uds, self.buffer, wds))
        return uds, pcs_id+1

    def _get_undisplay_pds(self, c_pts: float, pcs_id: int, node: 'DSNode', cobjs: list[CObject],
                           pcs_fn: Callable[[...], PCS], n_colors: int, wds_base: WDS) -> tuple[DisplaySet, int]:
        pcs = pcs_fn(pcs_id, PCS.CompositionState.NORMAL, True, node.palette_id, cobjs, c_pts)
        tsp_e = PaletteEntry(16, 128, 128, 0)
        pds = PDS.from_scratch(Palette({k: tsp_e for k in range(n_colors)}), p_vn=node.pal_vn, p_id=node.palette_id, pts=c_pts)
        uds = DisplaySet([pcs, pds, ENDS.from_scratch(pts=c_pts)])
        DSNode.apply_pts_dts(uds, DSNode.set_pts_dts_sc(uds, self.buffer, wds_base, node))
        return uds, pcs_id+1

    def _convert(self, states, pgobjs, windows, durs, flags, nodes):
        wd_base = [WindowDefinition.from_scratch(k, w.x+self.box.x, w.y+self.box.y, w.dx, w.dy) for k, w in enumerate(windows)]
        wds_base = WDS.from_scratch(wd_base, pts=0.0)
        n_actions = len(durs)
        is_compat_mode = True #self.kwargs.get('pgs_compatibility', True)
        insert_acqs = self.kwargs.get('insert_acquisitions', 0)
        displaysets = []
        time_scale = 1.001 if self.kwargs.get('adjust_dropframe', False) else 1
        use_full_pal = self.kwargs.get('full_palette', False)
        palette_manager = PaletteManager()

        ## Internal helper function
        def get_obj(frame, pgobjs: dict[int, list[PGObject]]) -> dict[int, Optional[PGObject]]:
            objs = {k: None for k, objs in enumerate(pgobjs)}

            for wid, pgobj in enumerate(pgobjs):
                for obj in pgobj:
                    if obj.is_active(frame):
                        objs[wid] = obj
            return objs

        def get_palette_data(pal_manager: PaletteManager, node: DSNode) -> tuple[int, int]:
            pal_id = pal_manager.get_palette(node.dts())
            assert pal_manager.lock_palette(pal_id, node.pts(), node.dts())
            return pal_id, pal_manager.get_palette_version(pal_id)
        ####

        i = 0
        double_buffering = [0]*2
        ods_reg = [0]*4
        pcs_id = self.pcs_id
        c_pts = 0
        last_cobjs = []
        last_palette_id = -1

        get_pts: Callable[[float], float] = lambda c_pts: max(c_pts - (1/3)/PGDecoder.FREQ, 0) * time_scale
        pcs_fn = lambda pcs_cnt, state, pal_flag, palette_id, cl, pts:\
                    PCS.from_scratch(*self.bdn.format.value, BDVideo.LUT_PCS_FPS[round(self.target_fps, 3)], pcs_cnt & 0xFFFF, state, pal_flag, palette_id, cl, pts=pts)

        final_node = DSNode([], windows, self.events[-1].tc_out, nc_refresh=True)
        #Do we have time to redraw the window (with some margin)?
        perform_wds_end = durs[-1][0] >= np.ceil(((final_node.write_duration() + 10)/PGDecoder.FREQ)*self.target_fps)

        if (use_pbar := logger.getEffectiveLevel() >= 20):
            pbar = tqdm(range(n_actions))
            if getattr(pbar, 'update', None) is None:
                pbar.update = pbar.close = lambda *args, **kwargs: None
        #Main conversion loop, using all assets
        while i < n_actions:
            if durs[i][1] != 0:
                assert i > 0
                assert nodes[i].parent is not None
                w_pts = get_pts(TC.tc2s(self.events[i-1].tc_out, self.bdn.fps))
                wds_doable = (nodes[i].parent.write_duration() + 3)/PGDecoder.FREQ < 1/self.target_fps
                if wds_doable and not nodes[i].parent.is_custom_dts():
                    uds, pcs_id = self._get_undisplay(w_pts, pcs_id, wds_base, last_palette_id, pcs_fn)
                    logger.debug(f"Writing screen clear with WDS at PTS={self.events[i-1].tc_out} before an acquisition.")
                else:
                    p_id, p_vn = get_palette_data(palette_manager, nodes[i].parent)
                    nodes[i].parent.palette_id = p_id
                    nodes[i].parent.pal_vn = p_vn
                    uds, pcs_id = self._get_undisplay_pds(w_pts, pcs_id, nodes[i].parent, last_cobjs, pcs_fn, 255, wds_base)
                    logger.debug(f"Writing screen clear with palette update before an acquisition at PTS={self.events[i-1].tc_out}")
                displaysets.append(uds)

            if flags[i] == -1:
                logger.debug(f"Skipping discarded event at PTS={self.events[i].tc_in}")
                i+=1
                continue

            assert states[i] != PCS.CompositionState.NORMAL
            normal_case_refresh = False
            for k in range(i+1, n_actions+1):
                if k < n_actions:
                    normal_case_refresh |= (flags[k] == 1)
                if k == n_actions or states[k] != PCS.CompositionState.NORMAL:
                    break
            assert k > i

            if nodes[i].tc_shift == 0:
                assert nodes[i].tc_pts == self.events[i].tc_in
                c_pts = get_pts(TC.tc2s(self.events[i].tc_in, self.bdn.fps))
            else:
                nodes[i].tc_pts = TC.add_framestc(self.events[i].tc_in, self.bdn.fps, nodes[i].tc_shift)
                c_pts = get_pts(TC.tc2s(nodes[i].tc_pts, self.bdn.fps))
                logger.debug(f"Shifted event: {self.events[i].tc_in} -> {nodes[i].tc_pts}, {get_pts(TC.tc2s(self.events[i].tc_in, self.bdn.fps))} -> c_pts={c_pts}")

            pgobs_items = get_obj(i, pgobjs).items()
            has_two_objs = 0
            for wid, pgo in pgobs_items:
                if pgo is None or not np.any(pgo.mask[i-pgo.f:k-pgo.f]):
                    continue
                has_two_objs += 1

            #Normal case refresh implies we are refreshing one object out of two displayed.
            has_two_objs = has_two_objs > 1 or normal_case_refresh

            r = self._generate_acquisition_ds(i, k, pgobs_items, windows, nodes[i], double_buffering,
                                              has_two_objs, is_compat_mode, ods_reg, c_pts, normal_case_refresh, flags)
            cobjs, pals, o_ods, pal = r

            wds = wds_base.copy(pts=c_pts, in_ticks=False)
            p_id, p_vn = get_palette_data(palette_manager, nodes[i])
            pds = PDS.from_scratch(pal, p_vn=p_vn, p_id=p_id, pts=c_pts)
            pcs = pcs_fn(pcs_id, states[i], False, p_id, cobjs, c_pts)

            nds = DisplaySet([pcs, wds, pds] + o_ods + [ENDS.from_scratch(pts=c_pts)])
            DSNode.apply_pts_dts(nds, DSNode.set_pts_dts_sc(nds, self.buffer, wds, nodes[i]))
            displaysets.append(nds)

            pcs_id += 1
            last_palette_id = p_id
            logger.debug(f"Acquisition: PTS={nodes[i].tc_pts}={c_pts:.03f}, 2OBJs={has_two_objs}, NC={normal_case_refresh} Npalups={len(pals[0])-1} S(ODS)={sum(map(lambda x: len(bytes(x)), o_ods))}, L(ODS)={len(o_ods)}")

            if len(pals[0]) > 1:
                # Pad palette chains
                if not normal_case_refresh:
                    zip_length = max(map(len, pals))
                    if len(pals[0]) < zip_length:
                        pals[0] += [Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(min(pals[0][0].palette), max(pals[0][0].palette)+1)})]
                    if has_two_objs and len(pals[1]) < zip_length:
                        pals[1] += [Palette({k: PaletteEntry(16, 128, 128, 0) for k in range(min(pals[1][0].palette), max(pals[1][0].palette)+1)})]
                pals[0] += [Palette()] * (k-i - len(pals[0]))
                pals[1] += [Palette()] * (k-i - len(pals[1]))

                for z, (p1, p2) in enumerate(zip_longest(pals[0][1:], pals[1][1:], fillvalue=Palette()), i+1):
                    c_pts = get_pts(TC.tc2s(self.events[z].tc_in, self.bdn.fps))
                    assert states[z] == PCS.CompositionState.NORMAL
                    pal |= pals[0][z-i] | pals[1][z-i]

                    #Is there a know screen clear in the chain? then use palette screen clear here
                    if durs[z][1] != 0:
                        assert nodes[z].parent is not None
                        logger.debug(f"Writing screen wipe in palette update chain at PTS={self.events[z-1].tc_out}")
                        p_id, p_vn = get_palette_data(palette_manager, nodes[z].parent)
                        nodes[z].parent.palette_id = p_id
                        nodes[z].parent.pal_vn = p_vn
                        uds, pcs_id = self._get_undisplay_pds(get_pts(TC.tc2s(self.events[z-1].tc_out, self.bdn.fps)), pcs_id, nodes[z].parent, cobjs, pcs_fn, max(pal.palette)+1, wds_base)
                        displaysets.append(uds)
                        #We just wipped a palette, whatever the next palette id, rewrite it fully
                        last_palette_id = None
                        #Should not be necessary but in any case...
                        durs[z] = (durs[z][0], 0)

                    if flags[z] == 1:
                        normal_case_refresh = nodes[z].new_mask
                        r = self._generate_acquisition_ds(z, k, get_obj(z, pgobjs).items(), windows, nodes[z], double_buffering,
                                                          has_two_objs, is_compat_mode, ods_reg, c_pts, normal_case_refresh, flags)
                        cobjs, n_pals, o_ods, new_pal = r
                        logger.debug(f"NORMAL CASE: PTS={self.events[z].tc_in}={c_pts:.03f}, NM={nodes[z].new_mask} S(ODS)={sum(map(lambda x: len(bytes(x)), o_ods))}")
                        pal |= new_pal
                        for nz, (new_p1, new_p2) in enumerate(zip_longest(n_pals[0], n_pals[1], fillvalue=Palette()), z):
                            pals[0][nz-i] |= new_p1
                            pals[1][nz-i] |= new_p2
                        normal_case_refresh = True
                        last_palette_id = None
                    elif flags[z] == -1:
                        logger.debug(f"Skipped discarded event at PTS={self.events[z].tc_in}.")
                        continue

                    p_write = (pals[0][z-i] | pals[1][z-i])
                    #Skip empty palette updates
                    if len(p_write) == 0:
                        logger.debug(f"Skipped an empty palette at PTS={self.events[z].tc_in}.")
                        continue

                    p_id, p_vn = get_palette_data(palette_manager, nodes[z])
                    #If the palette ID change, we must give the full palette.
                    if last_palette_id != p_id or use_full_pal:
                        p_write = pal

                    pcs = pcs_fn(pcs_id, states[z], flags[z] != 1, p_id, cobjs, c_pts)
                    pds = PDS.from_scratch(p_write, p_vn=p_vn, p_id=p_id, pts=c_pts)
                    wds_upd = [wds_base.copy(pts=c_pts, in_ticks=False)] if flags[z] == 1 else []
                    ods_upd = o_ods if flags[z] == 1 else []

                    nds = DisplaySet([pcs] + wds_upd + [pds] + ods_upd +[ENDS.from_scratch(pts=c_pts)])
                    DSNode.apply_pts_dts(nds, DSNode.set_pts_dts_sc(nds, self.buffer, wds_base, nodes[z]))
                    displaysets.append(nds)

                    pcs_id += 1
                    last_palette_id = p_id
                    if z+1 == k:
                        break
                assert z+1 == k
            
            if insert_acqs > 0 and len(pals[0]) > insert_acqs and flags[k-1] != -1:
                t_diff = TC.tc2s(self.events[k-1].tc_out, self.bdn.fps) - TC.tc2s(self.events[k-1].tc_in, self.bdn.fps)
                if t_diff > 0.5:
                    dts_end = nodes[k-1].dts_end() + 2/PGDecoder.FREQ
                    npts = nodes[k-1].pts() + 2/PGDecoder.FREQ
                    nodes[k-1].nc_refresh = False
                    while nodes[k-1].dts() < dts_end or nodes[k-1].pts() < nodes[k-1].write_duration()/PGDecoder.FREQ + npts:
                        nodes[k-1].tc_pts = TC.add_framestc(nodes[k-1].tc_pts, self.bdn.fps, 1)
                    if nodes[k-1].dts() - dts_end < 0.25:
                        logger.debug(f"INS Acquisition: PTS={nodes[k-1].tc_pts}={c_pts:.03f} from event at {self.events[k-1].tc_in}.")
                        c_pts = get_pts(TC.tc2s(nodes[k-1].tc_pts, self.bdn.fps))
                        
                        r = self._generate_acquisition_ds(k-1, k, pgobs_items, windows, nodes[k-1], double_buffering,
                                                          has_two_objs, is_compat_mode, ods_reg, c_pts, False, flags)
                        cobjs, _, o_ods, pal = r
                        wds = wds_base.copy(pts=c_pts, in_ticks=False)
                        p_id, p_vn = get_palette_data(palette_manager, nodes[k-1])
                        pds = PDS.from_scratch(pal, p_vn=p_vn, p_id=p_id, pts=c_pts)
                        pcs = pcs_fn(pcs_id, PCS.CompositionState.ACQUISITION, False, p_id, cobjs, c_pts)
                        pcs_id += 1
                        nds = DisplaySet([pcs, wds, pds] + o_ods + [ENDS.from_scratch(pts=c_pts)])
                        DSNode.apply_pts_dts(nds, DSNode.set_pts_dts_sc(nds, self.buffer, wds, nodes[k-1]))
                        displaysets.append(nds)
                    else:
                        logger.error(f"Tried to insert an acquisition at {nodes[k-1].tc_pts} but required shift beyond 0.25 s: ACQ dropped.")   
            i = k
            last_cobjs = cobjs
            if use_pbar:
                pbar.n = i
                pbar.update()
        if use_pbar:
            pbar.close()
        ####while

        final_ds = None
        #We can't undraw the screen due to delta PTS constraint, we clear it with a palette update and will undraw optionally at +N frames
        if not perform_wds_end:
            logger.debug("Performing palette wipe (delta PTS too short) at end of epoch.")
            p_id, p_vn = get_palette_data(palette_manager, final_node)
            last_palette_id = final_node.palette_id = p_id
            final_node.pal_vn = p_vn
            uds, pcs_id = self._get_undisplay_pds(get_pts(TC.tc2s(self.events[-1].tc_out, self.bdn.fps)), pcs_id, final_node, last_cobjs, pcs_fn, 255, wds_base)
            displaysets.append(uds)

            #Prepare an additional display set to undraw the screen. Will be added by parent if there's enough time before the next epoch.
            final_pts = TC.add_frames(self.events[-1].tc_out, self.bdn.fps, max(1, int(np.ceil(((final_node.write_duration()+10)*self.target_fps)/PGDecoder.FREQ))))
            logger.debug(f"final screen clear PTS: {final_pts} vs palette wipe: {self.events[-1].tc_out}, sf={get_pts(final_pts)}, so={get_pts(TC.tc2s(self.events[-1].tc_out, self.bdn.fps))}")
        else:
            logger.debug("Performing standard screen clear at end of epoch.")
            final_pts = TC.tc2s(self.events[-1].tc_out, self.bdn.fps)
        final_ds, pcs_id = self._get_undisplay(get_pts(final_pts), pcs_id, wds_base, last_palette_id, pcs_fn)

        if perform_wds_end:
            displaysets.append(final_ds)
            final_ds = None
        return Epoch(displaysets), final_ds, pcs_id
    ####

    def find_acqs(self, pgobjs_proc: dict[..., list[...]], windows: list[Box]):
        #get the frame count between each screen update and find where we can do acqs
        durs, nodes = self.get_durations(windows)

        dtl = np.zeros((len(durs)), dtype=float)
        valid = np.zeros((len(durs),), dtype=np.bool_)
        absolutes = np.zeros_like(valid)
        flags = [0] * len(durs)

        chain_boxes = []
        min_boxes = 8*np.ones((len(windows), 2), np.uint16)

        objs = [None for objs in pgobjs_proc]
        write_duration = nodes[0].write_duration()/PGDecoder.FREQ

        k_offset = 0
        for k, node in enumerate(nodes):
            is_new = [False]*len(windows)
            boxes = [None] * len(windows)
            force_acq = False
            #NC palette updates don't need to know about the objects
            if not node.nc_refresh:
                for wid, wd in enumerate(windows):
                    is_new[wid] = False
                    if objs[wid] and not objs[wid].is_active(k-k_offset):
                        objs[wid] = None
                    if len(pgobjs_proc[wid]):
                        if not objs[wid] and pgobjs_proc[wid][0].is_active(k-k_offset):
                            objs[wid] = pgobjs_proc[wid].pop(0)
                            force_acq = True
                            is_new[wid] = True
                        else:
                            assert not pgobjs_proc[wid][0].is_active(k-k_offset)
                    if objs[wid] is not None and objs[wid].is_visible(k-k_offset):
                        ob = objs[wid].get_bbox_at(k-k_offset)
                        min_boxes[wid] = np.max((min_boxes[wid], (ob.dy, ob.dx)), axis=0)
                        boxes[wid] = ob
                node.objects = objs.copy()
            else:
                k_offset += 1
            ####!nc_refresh
            node.new_mask = is_new
            chain_boxes.append(boxes)
            absolutes[k] = force_acq

        min_boxes = list(map(tuple, min_boxes))
        prev_dt = 6
        for k, (dt, node) in enumerate(zip(durs, nodes)):
            if not node.nc_refresh:
                margin = prev_dt/self.bdn.fps
                node.slots = min_boxes
            if k == 0:
                prev_pts = prev_dts = -np.inf
            else:
                prev_dts = nodes[k-1].dts_end()
                prev_pts = nodes[k-1].pts()
            valid[k] = (node.dts() > prev_dts and node.pts() - prev_pts > write_duration)
            dtl[k] = (node.dts() - prev_dts)/margin if valid[k] and k > 0 else (-1 + 2*(k==0))
            prev_dt = dt
        return valid, absolutes, dtl, durs, nodes, flags, min_boxes, chain_boxes
    ####

    def get_durations(self, windows: list[Box]) -> npt.NDArray[np.uint32]:
        """
        Returns the duration of each event in frames.
        Additionally, the offset from the previous event is also returned. This value
        is zero unless there are no PG objects shown at some point in the epoch.
        """
        top = TC.tc2f(self.events[0].tc_in, self.bdn.fps)
        delays = []
        nodes = []
        for ne, event in enumerate(self.events):
            tic = TC.tc2f(event.tc_in, self.bdn.fps)
            toc = TC.tc2f(event.tc_out,self.bdn.fps)
            clear_duration = tic-top
            delays += [toc-tic]
            if clear_duration > 0:
                delays += [clear_duration]
                nodes.append(DSNode([], windows, self.events[ne-1].tc_out, nc_refresh=True))
            nodes.append(DSNode([], windows, event.tc_in))
            top = toc
        return delays, nodes
    ####

    def roll_nodes(self, nodes, durs, flags, states) -> ...:
        k = 0
        r_nodes = []
        r_durs = []
        r_states = []
        r_flags = []
        for ne, event in enumerate(self.events):
            parent = nodes[k] if nodes[k].objects == [] else None
            valid_parent = parent is not None and flags[k] == 0
            k += parent is not None
            nodes[k].parent = parent

            assert parent is None or self.events[ne-1].tc_out == nodes[k].parent.tc_pts
            assert nodes[k].tc_pts == event.tc_in

            r_durs.append((durs[k], 0 if not valid_parent else durs[k-1]))
            r_nodes.append(nodes[k])
            r_flags.append(flags[k])
            r_states.append(states[k])
            k += 1
        assert k == len(nodes)
        return r_states, r_durs, r_nodes, r_flags
    ####
####
#%%

class WOBAnalyzer:
    def __init__(self, wob, ssim_threshold: float = 0.986, overlap_threshold: float = 0.995) -> None:
        self.wob = wob
        assert ssim_threshold < 1.0, "Not a valid SSIM threshold"
        self.ssim_threshold = ssim_threshold
        assert 0 < overlap_threshold < 1.0, "Not a valid overlap threshold."
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def get_grayscale(rgba: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rgba = rgba.astype(np.uint16)
        img = np.round(0.2989*rgba[:,:,0] + 0.587*rgba[:,:,1] + 0.114*rgba[:,:,2])
        return (img.clip(0, 255) & (255*(rgba[:,:,3] > 0))).astype(np.uint8)

    def compare(self, bitmap: Image.Image, current: Image.Image) -> tuple[float, float]:
        """
        :param bitmap: (cropped or padded) aggregate of the previous bitmaps
        :param current: current bitmap under analysis
        :return: comparison score between the two
        """
        assert bitmap.width == current.width and bitmap.height == current.height, "Different shapes."

        # Intersect alpha planes
        a_bitmap = np.array(bitmap)
        a_current = np.array(current)
        inters_inv = np.logical_and(a_bitmap[:,:,3] == 0, a_current[:,:,3] == 0)
        inters = np.logical_and(a_bitmap[:,:,3] != 0, a_current[:,:,3] != 0)
        inters_area = np.sum(inters)
        #if the images have the exact same alpha channel, this measure is equal to 1
        overlap = (inters_area > 0) * (inters_area + np.sum(inters_inv))/inters.size

        if overlap > 0:
            #score = compare_ssim(bitmap.convert('L'), current.convert('L'))
            #Broadcast transparency mask of current on all channels of ref
            mask = 255*(np.logical_and((a_bitmap[:, :, 3] > 0), (a_current[:, :, 3] > 0)).astype(np.uint8))
            score = compare_ssim(Image.fromarray(a_bitmap & mask[:, :, None]).convert('L'), current.convert('L'))
            cross_percentage = np.sum(mask > 0)/mask.size
        else:
            cross_percentage = 1.0
            score = 1.0
        return score, cross_percentage

    @staticmethod
    def get_patch(image1, image2) -> Image.Image:
        bbox1 = image1.getbbox()
        bbox2 = image2.getbbox()

        #One of the image is fully transparent.
        if bbox1 is None or bbox2 is None:
            return bbox1, bbox2

        f_area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        A1 = f_area(bbox1)
        A2 = f_area(bbox2)

        if A1 >= A2:
            return (image1.crop(bbox1), Pos(*bbox1[:2])), image2
        return image1, (image2.crop(bbox2), Pos(*bbox2[:2]))

    @staticmethod
    def correlate_patch(image, patch) -> tuple[float, Type['Box']]:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

        #pad image with patch dimension, so we can look at the edges.
        image = cv2.copyMakeBorder(image, patch.height-1, patch.height-1,
                                   patch.width-1, patch.width-1,
                                   cv2.BORDER_CONSTANT, value=[0]*4)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        patch_gray = cv2.cvtColor(np.asarray(patch, dtype=np.uint8), cv2.COLOR_RGBA2GRAY)

        #Correlate across entire surface and return the best scores
        res = cv2.matchTemplate(img_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
        score1, score2, p1, p2 = cv2.minMaxLoc(res)
        return (score1, p1), (score2, p2)

    def analyze(self):
        cls = self.__class__
        window = self.wob.get_window()

        bitmaps = []
        alpha_compo = Image.fromarray(np.zeros((window.dy, window.dx, 4), np.uint8))

        mask = []
        event_cnt = 0
        pgo_yield = None
        unseen = 0

        while True:
            rgba = yield pgo_yield
            pgo_yield = None

            if rgba is None:
                if len(bitmaps):
                    bbox = alpha_compo.getbbox()
                    if unseen > 0:
                        mask = mask[:-unseen]
                        bitmaps = bitmaps[:-unseen]
                    pgo_yield = PGObject(np.stack(bitmaps).astype(np.uint8), Box.from_coords(*bbox), mask, f_start)
                    bitmaps, mask = [], []
                    continue
                else:
                    break

            has_content = np.any(rgba)
            if has_content or len(mask):
                if not len(mask):
                    f_start = event_cnt

                rgba_i = Image.fromarray(rgba)
                score, cross_percentage = self.compare(alpha_compo, rgba_i)
                logger.hdebug(f"Image analysis: score={score:.05f} cross={cross_percentage:.05f}, fuse={score >= min(1.0, self.ssim_threshold + (1-self.ssim_threshold)*(1-cross_percentage) - 0.008333)}")
                if score >= min(1.0, self.ssim_threshold + (1-self.ssim_threshold)*(1-cross_percentage) - 0.008333):
                    bitmaps.append(rgba)
                    alpha_compo.alpha_composite(rgba_i)
                    mask.append(has_content)
                else:
                    bbox = alpha_compo.getbbox()
                    if unseen > 0:
                        mask = mask[:-unseen]
                        bitmaps = bitmaps[:-unseen]
                    pgo_yield = PGObject(np.stack(bitmaps).astype(np.uint8), Box.from_coords(*bbox), mask, f_start)

                    #new bitmap
                    mask = [has_content]
                    bitmaps = [rgba]
                    f_start = event_cnt
                    alpha_compo = Image.fromarray(rgba.copy())
                unseen = (not has_content)*(unseen + 1)
            event_cnt += 1
        ####while
        return # StopIteration
#%%
def get_wipe_duration(wds: WDS) -> int:
    return np.ceil(sum(map(lambda w: PGDecoder.FREQ*w.height*w.width/PGDecoder.RC, wds.windows)))

#%%
####
class DSNode:
    scale_pts = 1.0
    bdn_fps = None
    conservative_dts = False

    def __init__(self,
            objects: list[Optional[PGObject]],
            windows: list[Box],
            tc_pts: str,
            nc_refresh: bool = False,
        ) -> None:
        self.objects = objects
        self.windows = windows
        self.slots = [None] * len(self.windows)
        self.pos = [None] * len(self.windows)
        self.tc_pts = tc_pts
        self.nc_refresh = nc_refresh

        self.new_mask = []
        self.partial = False
        self.tc_shift = 0

        self.parent = None
        self.palette_id = None
        self.pal_vn = 0
        self._dts = None

    @classmethod
    def configure(cls, scale_pts: float, fps: BDVideo.FPS, conservative_dts: bool = False, enforce_dts: bool = True) -> None:
        cls.scale_pts = scale_pts
        cls.bdn_fps = fps
        cls.conservative_dts = conservative_dts
        cls.enforce_dts = enforce_dts

    def wipe_duration(self) -> int:
        return np.ceil(sum(map(lambda w: PGDecoder.FREQ*w.dy*w.dx/PGDecoder.RC, self.windows)))

    def write_duration(self) -> int:
        return sum(map(lambda w: np.ceil(PGDecoder.FREQ*w.dy*w.dx/PGDecoder.RC), self.windows))

    def set_dts(self, dts: Optional[float]) -> None:
        assert dts is None or dts <= self.dts()
        self._dts = round(dts*PGDecoder.FREQ) if dts is not None else None

    def dts_end(self) -> float:
        if self._dts is not None:
            return (self.get_dts_markers()[1] + self._dts)/PGDecoder.FREQ
        return sum(self.get_dts_markers())/PGDecoder.FREQ

    def dts(self) -> float:
        if self._dts is not None:
            return self._dts/PGDecoder.FREQ
        return self.get_dts_markers()[0]/PGDecoder.FREQ

    def delta_dts(self) -> float:
        return self.get_dts_markers()[1]/PGDecoder.FREQ

    def pts(self) -> float:
        return max(TC.tc2s(self.tc_pts, self.__class__.bdn_fps) - (1/3)/PGDecoder.FREQ, 0) * self.__class__.scale_pts

    def is_custom_dts(self) -> bool:
        return not (self._dts is None)

    def get_dts_markers(self) -> float:
        t_decoding = 0

        if not self.nc_refresh:
            assigned_wd = list(map(lambda x: x is not None, self.objects))
            decode_duration = sum([np.ceil(self.windows[wid].dy*self.windows[wid].dx*PGDecoder.FREQ/PGDecoder.RC) for wid, flag in enumerate(assigned_wd) if not flag])

            if self.__class__.conservative_dts:
                decode_duration = self.wipe_duration()

            t_other_copy = 0
            for wid, obj in enumerate(self.objects):
                if obj is None:
                    continue

                box = self.windows[wid]
                write = box.dy*box.dx*PGDecoder.FREQ
                if self.slots[wid] is not None:
                    read = int(self.slots[wid][0])*int(self.slots[wid][1])*PGDecoder.FREQ
                else:
                    #no slot -> buffer is sized to the window
                    read = write
                if not self.partial or (self.partial and self.new_mask[wid]):
                    t_decoding += np.ceil(read/PGDecoder.RD)
                elif self.partial and not self.new_mask[wid]:
                    #the other object is copied at the end.
                    assert sum(self.new_mask) == 1 and t_other_copy == 0
                    t_other_copy += np.ceil(write/PGDecoder.RC)
                    continue

                decode_duration = max(decode_duration, t_decoding) + np.ceil(write/PGDecoder.RC)
            ####
            assert t_other_copy == 0 or self.partial
            decode_duration += t_other_copy
        else:
            decode_duration = self.write_duration() + 1
        return (round(self.pts()*PGDecoder.FREQ) - decode_duration, t_decoding)
    ####

    @classmethod
    def set_pts_dts_sc(cls, ds: DisplaySet, buffer: PGObjectBuffer, wds: WDS, node: Optional['DSNode'] = None) -> list[tuple[int, int]]:
        """
        This function generates the timestamps (PTS and DTS) associated to a given DisplaySet.

        :param ds: DisplaySet, PTS of PCS must be set to the right value.
        :param buffer: Object buffer that supports allocation and returning a size of allocated slots.
        :param wds: WDS of the epoch.
        :return: Pairs of timestamps in ticks for each segment in the displayset.
        """
        ddurs = {}
        for ods in ds.ods:
            if ods.flags & ods.ODSFlags.SEQUENCE_FIRST:
                assert ods.o_id not in ddurs, f"Object {ods.o_id} defined twice in DS."
                if (shape := buffer.get(ods.o_id)) is not None:
                    assert (ods.height, ods.width) == shape, "Dimension mismatch, buffer corruption."
                else:
                    # Allocate a buffer slot for this object
                    assert buffer.allocate_id(ods.o_id, ods.height, ods.width) is True, "Slot already allocated or buffer overflow."
                ddurs[ods.o_id] = np.ceil(ods.height*ods.width*PGDecoder.FREQ/PGDecoder.RD)

        t_decoding = 0
        decode_duration = 0
        wipe_duration = get_wipe_duration(wds)

        windows = {wd.window_id: (wd.height, wd.width) for wd in wds.windows}

        if ds.pcs.composition_state == ds.pcs.CompositionState.EPOCH_START:
            decode_duration = np.ceil(ds.pcs.width*ds.pcs.height*PGDecoder.FREQ/PGDecoder.RC)
        else:
            assigned_windows = list(map(lambda x: x.window_id, ds.pcs.cobjects))
            unassigned_windows = [wd for wd in windows if wd not in assigned_windows]
            decode_duration = sum([np.ceil(windows[wid][0]*windows[wid][1]*PGDecoder.FREQ/PGDecoder.RC) for wid in unassigned_windows])

            if cls.conservative_dts:
                decode_duration = wipe_duration

        object_decode_duration = ddurs.copy()

        if not ds.pcs.pal_flag:
            #For every composition object, compute the transfer time
            for k, cobj in enumerate(ds.pcs.cobjects):
                shape = buffer.get(cobj.o_id)
                assert shape is not None, "Object does not exist in buffer."
                w, h = windows[cobj.window_id][0], windows[cobj.window_id][1]

                t_dec_obj = object_decode_duration.pop(cobj.o_id, 0)
                t_decoding += t_dec_obj

                # Same window -> patent claims a window is written only once after the two cobj are processed.
                if k == 0 and ds.pcs.n_objects > 1 and ds.pcs.cobjects[1].window_id == cobj.window_id:
                    continue
                copy_dur = np.ceil(w*h*PGDecoder.FREQ/PGDecoder.RC)
                decode_duration = max(decode_duration, t_decoding) + copy_dur

        #Prevent PTS(WDS) = PTS(PCS)
        decode_duration = max(decode_duration, sum(map(lambda w: np.ceil(PGDecoder.FREQ*w[0]*w[1]/PGDecoder.RC), windows.values())) + 1)

        mask = ((1 << 32) - 1)
        dts = int(ds.pcs.tpts - decode_duration) & mask
        if node is not None and node.is_custom_dts():
            new_dts = round(node.dts()*PGDecoder.FREQ)
            assert new_dts <= dts
            dts = new_dts

        #PCS always exist
        ts_pairs = [(ds.pcs.tpts, dts)]

        if ds.wds:
            ts_pairs.append((int(ds.pcs.tpts - wipe_duration) & mask, dts))
        for pds in ds.pds:
            ts_pairs.append((dts, dts))

        for ods in ds.ods:
            ods_pts = int(dts + ddurs.get(ods.o_id)) & mask
            ts_pairs.append((ods_pts, dts))
            if ods.flags & ods.ODSFlags.SEQUENCE_LAST:
                dts = ods_pts
        ts_pairs.append((dts, dts))
        return ts_pairs
    ####

    @classmethod
    def apply_pts_dts(cls, ds: DisplaySet, ts: tuple[int, int]) -> None:
        nullify_dts = lambda x: x*(1 if cls.enforce_dts else 0)
        select_pts = lambda x: x if cls.enforce_dts else ts[0][0]

        assert len(ds) == len(ts), "Timestamps-DS size mismatch."
        for seg, (pts, dts) in zip(ds, ts):
            seg.tpts, seg.tdts = select_pts(pts), nullify_dts(dts)
    ####
####
