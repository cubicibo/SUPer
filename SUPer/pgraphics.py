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

from numpy import typing as npt
import numpy as np
from typing import Union, Optional
from enum import Enum

from .utils import Shape
from .segments import WDS, ODS, DisplaySet
from .filestreams import BaseEvent

class PGraphics:
    @classmethod
    def bitmap_to_ods(cls, bitmap: npt.NDArray[np.uint8], o_id: int, **kwargs) -> list[ODS]:
        assert bitmap.dtype == np.uint8
        height, width = bitmap.shape
        o_vn = kwargs.pop('o_vn', 0)
        data = cls.encode_rle(bitmap)

        return ODS.from_scratch(o_id, o_vn, width, height, data, **kwargs)

    @staticmethod
    def encode_rle(bitmap: npt.NDArray[np.uint8]) -> bytes:
        """
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param bitmap:    Palette mapped image to encode (2d array)
        :return:          Encoded data (vector)
        """
        rle_data = bytearray()

        _bitmap = np.squeeze(bitmap)
        _fp = np.ravel(_bitmap)
        i = 0
        insert_line_end = False

        while i < _fp.size:
            for k in range(1, 16384):
                if (i % _bitmap.shape[1]) + k >= _bitmap.shape[1]:
                    insert_line_end = True
                    break

                if _fp[i+k] != _fp[i]:
                    break

            if _fp[i] != 0: #color
                if k < 3:
                    rle_data += bytearray([_fp[i]]*k)
                elif k <= 63:
                    rle_data += bytearray([0, 0x80 | k, _fp[i]])
                else:
                    rle_data += bytearray([0, 0xC0 | (k >> 8), k&0xFF, _fp[i]])
            else: #transparent
                if k <= 63:
                    rle_data += bytearray([0, k])
                else:
                    rle_data += bytearray([0, 0x40 | (k >> 8), k&0xFF])

            if insert_line_end:
                rle_data += bytearray([0, 0])
                insert_line_end = False
            i += k
        return bytes(rle_data)

    @staticmethod
    def decode_rle(rle_data: Union[bytes, bytearray, ODS, list[ODS]], o_id: Optional[int] = None) -> npt.NDArray[np.uint8]:
        """
        Decode a RLE object, as defined in 'US 7912305 B1' patent.
        :param rle_data:  Data to decode
        :param o_id:      Optional object ID to display, if rle_data is packed ODSes from a DS
        :return:          2D map to associate with the proper palette
        """
        class RLEDecoderState(Enum):
            NEED_MORE = -2
            NEW_CODE  = -1
            SMALL_TSP = 0
            LARGE_TSP = 1
            SMALL_CCO = 2
            LARGE_CCO = 3

        if isinstance(rle_data, ODS):
            rle_data = [rle_data]

        if isinstance(rle_data, list):
            if o_id is not None:
                rle_data = b''.join(map(lambda x: x.data, filter(lambda x: o_id == x.o_id, rle_data)))
            else:
                rle_data = b''.join(map(lambda x: x.data, rle_data))

        plane2d, line_l = [], []
        decoder_state = RLEDecoderState.NEW_CODE
        tmp = 0

        # Always use a state machine, even in place where you totally don't need it.
        for byte in rle_data:
            if decoder_state == RLEDecoderState.NEW_CODE:
                if byte > 0:
                    line_l.append(byte)
                else:
                    decoder_state = RLEDecoderState.NEED_MORE

            elif decoder_state == RLEDecoderState.NEED_MORE:
                if byte == 0:
                    plane2d.append(line_l)
                    line_l = []
                    decoder_state = RLEDecoderState.NEW_CODE
                else:
                    decoder_state = RLEDecoderState(byte >> 6)
                    tmp = byte & 0x3F

                    if decoder_state == RLEDecoderState.SMALL_TSP:
                        line_l.extend([0] * tmp)
                        decoder_state = RLEDecoderState.NEW_CODE

            elif decoder_state == RLEDecoderState.LARGE_TSP:
                tmp = (tmp << 8) + byte
                line_l.extend([0]*tmp)
                decoder_state = RLEDecoderState.NEW_CODE

            elif decoder_state == RLEDecoderState.SMALL_CCO:
                line_l.extend([byte]*tmp)
                decoder_state = RLEDecoderState.NEW_CODE

            elif decoder_state == RLEDecoderState.LARGE_CCO:
                #first pass (some RLE encoders use long code for small distances
                # hence we must check for equal zero...)
                if tmp >= 0:
                    tmp = ((tmp << 8) + byte)*-1
                else: #second pass
                    line_l.extend([byte]*(-1*tmp))
                    decoder_state = RLEDecoderState.NEW_CODE
        return np.asarray(plane2d)

    @staticmethod
    def show(l_ods: list[ODS], palette: Optional[npt.NDArray[np.uint8]] = None) -> None:
        """
        Show the ODS with or without a provided palette. If no palette are provided,
        one is generated that illustrates the encoded animation in the bitmap.
        """
        bitmap = __class__.decode_rle(l_ods)

        # Create a evenly distributed palette using YUV wiht constant luma.
        if palette is None:
            mpe, Mpe = np.min(bitmap), np.max(bitmap)
            n_cols = (Mpe-mpe+1)
            luma = 0.5
            palette = np.zeros((Mpe+1, 3), float)
            angles = np.random.permutation(np.arange(0, (Mpe-mpe)/n_cols, 1/n_cols))

            for angle, k in zip(angles, range(mpe+1, Mpe)):
                angle *= 2*np.pi
                palette[k, 0] = luma + np.cos(angle)/0.88
                palette[k, 1] = luma - np.sin(angle)*0.38 - np.cos(angle)*0.58
                palette[k, 2] = luma + np.sin(angle)/0.49
            palette -= np.min(palette)
            palette /= (np.max(palette)/255)
            palette = np.uint8(np.round(palette))

        try:
            from matplotlib import pyplot as plt
            plt.imshow(palette[bitmap])
        except ModuleNotFoundError:
            from PIL import Image
            Image.fromarray(palette[bitmap], 'RGB').show()
####
#%%
class PGDecoder:
    RX =  2*(1024**2)
    RD = 16*(1024**2)
    RC = 32*(1024**2)
    FREQ = 90e3
    DECODED_BUF_SIZE = 4*(1024**2)
    CODED_BUF_SIZE   = 1*(1024**2)

    @classmethod
    def gplane_write_time(cls, *shape, coeff: int = 1):
        return cls.FREQ * np.ceil(coeff*shape[0]*shape[1]/cls.RC)

    @classmethod
    def plane_initilaization_time(cls, ds: DisplaySet) -> int:
        init_d = 0
        if PCS.CompositionState.EPOCH_START & ds.pcs.composition_state:
            #The patent gives the coeff 8 but does not explain where it comes from
            # and the statements in the documentation says it is just the size of the
            # graphic plane. But there are two graphics planes (swapped on each
            # composition). So I assume the coefficient of two. Also, it makes no
            # sense for an epoch start to be faster than an acquisition or a normal
            # case and this is not validated with empirical trials.
            init_d = cls.gplane_write_time(ds.pcs.width, ds.pcs.height, coeff=2)
        else:
            for window in ds.wds.windows:
                init_d += cls.gplane_write_time(ds.pcs.width, ds.pcs.height)
        return init_d

    @classmethod
    def wait(cls, ds: DisplaySet, obj_id: int, current_duration: int) -> int:
        wait_duration = 0
        for object_def in ds.ods:
            if object_def.o_id == obj_id:
                c_time = ds.pcs.dts + current_duration
                if c_time < object_def.pts:
                    wait_duration += object_def.pts - c_time
                return np.ceil(wait_duration*cls.FREQ)
        return wait_duration
    ####
    @staticmethod
    def size(ds: DisplaySet, window_id: int) -> Shape:
        window = None
        for wd in ds.pcs.wds.window:
            if ds.pcs.cobjects[0].window_id == wd.window_id:
                window = wd
                break
        assert window is not None, "Did not find window definition."
        return Shape(window.width, window.height)

    @staticmethod
    def object_areas(ods: list[ODS]) -> int:
        return sum(map(lambda o: o.width*o.height if o.flags & o.ODSFlags.SEQUENCE_FIRST else 0, ods))

    @staticmethod
    def window_areas(wds: WDS) -> int:
        return sum(map(lambda window: window.width * window.height, wds.windows))

    @classmethod
    def rc_coeff(cls, ods: list[ODS], wds: WDS) -> int:
        return cls.object_areas(ods) + cls.window_areas(wds)

    @classmethod
    def decode_duration(cls, ds: DisplaySet) -> int:
        decode_duration = cls.plane_initilaization_time(ds)
        if ds.pcs.n_objects == 2:
            if ds.pcs.cobjects[0].window_id == ds.pcs.cobjects[1].window_id:
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
            else:
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
                decode_duration += cls.wait(ds, ds.pcs.cobjects[1].o_id, decode_duration)
                decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[1].window_id))

        elif ds.pcs.n_objects == 1:
            decode_duration += cls.wait(ds, ds.pcs.cobjects[0].o_id, decode_duration)
            decode_duration += cls.gplane_write_time(*cls.size(ds, ds.pcs.cobjects[0].window_id))
        return decode_duration

    @classmethod
    def decode_obj_duration(cls, area: int) -> float:
        return area/cls.RD

    @classmethod
    def copy_gp_duration(cls, area: int) -> float:
        return area/cls.RC
    ####
#%%
from SSIM_PIL import compare_ssim

class WOBSAnalyzer:
    def __init__(self, wobs: ..., events: ..., box: ..., fps: ...):
        self.wobs = wobs
        self.events = events
        self.box = box
        self.fps = fps

    def analyze(self):
        #First we get the active mask
        active_mask = self.wobs[0].event_mask()
        woba = [],[]
        for wob in self.wobs[1:]:
            active_mask += wob.event_mask()
        for k, wob in enumerate(self.wobs):
            woba.append(WOBAnalyzer(wob, box, k))
            gens.append(woba[-1].analyze())
        #Run first level analysis: find effects

        pgobjs = {k: [] for k in self.wobs}
        for event in self.events + [None]:
            for k, obj in enumerate(map(lambda gwob: gwob.send(event), gens)):
                if obj is not None:
                    pgobjs[self.wobs[k]].extend(obj if isinstance(obj, list) else [obj])

        n_obj = sum(map(len, pgobjs.values()))
        area = sum(map(lambda x: x.area, pgobjs.values()))
        self.find_acqs(pgobjs)

    def find_acqs(self, pgobjs: dict[..., list[...]]):
        #get the frame count between each screen update and find where we can do acqs
        occupancy = []
        obj_pipeline = [wl for wl in pgobjs.values()]

        delays = self.get_durations()

        ts_frame = 0
        for delay, offset in delays:
            occupancy.append([])
            ts_frame += offset
            for wob_objs in obj_pipeline:
                for ko, obj in enumerate(wob_objs.copy()):
                    if obj.is_active(ts_frame):
                        occupancy[-1].extend(obj)
            assert len(occupancy[-1]) <= 2, "More than 2 compositions at once."
            ts_frame += delay

        gp_clear_dur = PGDecoder.copy_gp_duration(sum(map(lambda x: x.get_window().area, self.wobs)))

        acqs_possibles = [None] * len(delays)
        acqs_singles = [0] * len(delays)
        acqs_freetime = [0] * len(delays)
        idx = len(delays) - 1

        for (delay, offset), cobjs in reversed(zip(delays, occupancy)):
            spacing = delay + offset
            ts_frame -= delay

            dt = spacing*1./self.fps

            decode_duration = 0
            gp_duration = gp_clear_dur
            acqs_deadlines.append(0)

            for cobj in cobjs:
                decode_duration += PGDecoder.decode_obj_duration(cobj.area)
                if decode_duration > gp_duration:
                    gp_duration += decode_duration-gp_duration
                gp_duration += PGDecoder.copy_gp_duration(cobj.area)
                if cobj.f == ts_frame:
                    acqs_singles[idx] += 1

            #if negative, we lack time!
            acqs_possibles[idx](dt - decode_duration)
            acqs_freetime[idx] = dt
            idx -= 1
            ts_frame -= offset

        return acqs_possibles, acqs_singles, acqs_freetime
    ####

    def get_durations(self, events: ..., fps: ...) -> npt.NDArray[np.uint32]:
        """
        Returns the duration of each event in frames.
        Additionally, the offset from the previous event is also returned. This value
        is zero unless there are no PG objects shown at some point in the epoch.
        """
        top = TC.tc2f(self.events[0].tc_in, self.fps)
        delays = []
        for event in events:
            tic = TC.tc2f(event.tc_in, self.fps)
            toc = TC.tc2f(event.tc_out,self.fps)
            delays.append((toc-tic, top-tic))
            top = toc
        return delays

from PIL import Image
from SUPer.utils import Pos
from SUPer.render2 import Box
from dataclasses import dataclass

@dataclass
class PGObject:
    box: Box
    gfx: npt.NDArray[np.uint8]
    mask: list[bool]
    f:  int

    def __post_init__(self) -> None:
        self.effects = None #Note: effects cannot overlap presently (either color or fade)

    @property
    def still_bitmap(self) -> npt.NDArray[bool]:
        """
        Return a mask that gives timing when a bitmap is still and
        only requires alpha updates (or none at all)
        """
        mask = np.zeros((df,), dtype=np.bool_)

        for k, eff in enumerate(self.effects):
            #Fade effect defines alpha-ONLY effect applied on a given bitmap
            # so if alpha does not vary, it IS classified as a fade effect
            # color effect are special as they are encoded within the bitmap
            if isinstance(eff, FadeEffect):
                mask[eff.t:eff.t+eff.dt] = 1
        return mask

    def is_active(self, frame) -> bool:
        return frame in range(f, f+len(mask))

    def decode_duration(self) -> float:
        return box.area/PGDecoder.RD

    def transfer_duration(self) -> float:
        return box.area/PGDecoder.RC

    def estimate_rle_length(self) -> int:
        return int(box.area*0.275)

    def estimate_arrival_duration(self) -> float:
        return self.estimate_rle_length()/PGDecoder.RX

#%%
from itertools import chain

@dataclass
class FadeEffect:
    ref_img_idx: int
    t: int
    dt: int
    coeffs: npt.NDArray[np.uint8]

    @classmethod
    def get_fade_chain(cls, chain: npt.NDArray[np.uint8]) -> float:
        if len(chain) > 1:
            I = np.zeros((len(chain)))
            Imax = np.zeros((len(chain)))

            for k, img in enumerate(chain):
                I[k:k+1] = np.sum(img[:,:,3])/(img.shape[0]*img.shape[1])
                Imax[k] = np.max(img[:,:,3])
            I /= np.max(I)
            dI = np.diff(I)
            dImax = np.diff(Imax)
            if np.all(dI*dImax >= 0):
                # Get the fade effects (where we don't need to update the image!)
                return cls.check_mse(chain, I)
        return None

    @classmethod
    def check_mse(cls, imgs: npt.NDArray[np.uint8], I: npt.NDArray[float]) -> list['FadeEffect']:
        fade_start = None
        effects = []
        chain32 = imgs.astype(np.int32)
        for k, rgba_img in enumerate(chain(chain32[1:], [None])):
            if rgba_img is not None:
                mse = np.square(np.subtract(chain32[k,:,:,:3], rgba_img[:,:,:3]))
            if rgba_img is not None and 70 > mse.mean() and mse.max() < 1500:
                if fade_start is None:
                    fade_start = k
            elif fade_start is not None or rgba_img is None:
                ref_idx = fade_start + np.argmax(I[fade_start:k+1])
                effects.append(cls(ref_idx, fade_start, k+1-fade_start, I[fade_start:k+1]/I[k+1-fade_start]))
                fade_start = None
        return effects if len(effects) > 0 else None

#%%
class WOBAnalyzer:
    def __init__(self, wob: ..., box: ..., k: int) -> None:
        self.wob = wob
        self.box = box
        self.wob_id = wob_id

    def mask_event(self, window, event) -> npt.NDArray[np.uint8]:
        windowed_event = np.zeros((window.dy, window.dx, 4), dtype=np.uint8)
        work_plane = np.zeros((box.dy, box.dx, 4), dtype=np.uint8)

        hsw = slice(event.x-self.box.x+window.x-1, event.x-self.box.x+window.x+min(window.dx, event.width)-1)
        vsw = slice(event.y-self.box.y+window.y, event.y-self.box.y+window.y+min(window.dy, event.height))
        hsi = slice(event.x-self.box.x, event.x-self.box.x+event.width)
        vsi = slice(event.y-self.box.y, event.y-self.box.y+event.height)
        work_plane[vsi, hsi, :] = np.array(event.img, dtype=np.uint8)
        windowed_event[vsw, hsw, :] = work_plane[vsw, hsw, :]
        return windowed_event

    @staticmethod
    def get_grayscale(rgba: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rgba = rgba.astype(np.uint16)
        img = np.round(0.2989*rgba[:,:,0] + 0.587*rgba[:,:,1] + 0.114*rgba[:,:,2])
        return (img.clip(0, 255) & (255*(rgba[:,:,3] > 0))).astype(np.uint8)

    @staticmethod
    def compare(bitmap, current) -> float:
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
        #if the images are identical, this measure is equal to 1.0
        overlap = (np.sum(inters) + np.sum(inters_inv))/inters.size

        if overlap < 0.99 and overlap > 0:
            #score = compare_ssim(bitmap.convert('L'), current.convert('L'))
            #Broadcast transparency mask of current on all channels of ref
            mask = 255*(np.logical_and((a_bitmap > 0), (a_current[:, :, 3, None] > 0)).astype(np.uint8))
            score = compare_ssim(Image.fromarray(a_bitmap & mask).convert('L'), current.convert('L'))
        else:
            #Perfect overlap or no overlap, the current bitmap fits perfectly on the previous
            score = 1.0
        return score

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

    def correlate_patch(image, patch) -> tuple[float, Box]:
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

        bitmaps = np.zeros((0, window.dy, window.dx, 4))
        alpha_compo = Image.fromarray(np.zeros((window.dy, window.dx, 4), np.uint8))

        mask = []
        event_cnt = 0
        pgo_yield = None
        event = True

        while event is not None:
            event = yield pgo_yield
            pgo_yield = None

            if event is None:
                bbox = alpha_compo.getbbox()
                yield PGObject(bitmaps, Box.from_coords(bbox), mask, event_cnt-len(mask[:-1]))
                continue

            rgba = self.mask_event(window, event)
            mask.append(np.any(rgba))
            if mask[-1]:
                score = cls.compare(alpha_compo, event)

                if score > 0.91:
                    bitmaps = np.stack((bitmaps, np.expand_dims(rgba, 0)))
                    alpha_compo.alpha_composite(Image.fromarray(rgba))
                else:
                    bbox = alpha_compo.getbbox()
                    pgo_yield = PGObject(bitmaps, Box.from_coords(bbox), mask, event_cnt-len(mask[:-1]))

                    #new bitmap
                    mask = mask[-1:]
                    bitmaps = np.expand_dims(rgba, 0)
                    alpha_compo = Image.fromarray(np.zeros((window.dy, window.dx, 4), np.uint8))
                    alpha_compo.alpha_composite(Image.fromarray(rgba))
            event_cnt += 1
        ####while
        return # StopIteration

#%%
class PGObjectBuffer:
    """
    This class represents a PG Buffer with some functions to interact with it.
    The PG Buffer is entirely cleared on EPOCH START.
    On ACQUISITION or NORMAL case, we can only use existing buffer slots or define
    new ones, within the limited buffer memory size.
    """
    _MAX_OBJECTS = 64
    _MAX_SIZE = 4 << 20
    def __init__(self, /, *, _max_size: Optional[int] = None, _margin: int = 0) -> None:
        self._max_size = (__class__._MAX_SIZE if _max_size is None else _max_size)
        self._max_size -= _margin

        self._slots = {}
        self._loaded = []

    def get_free_size(self) -> int:
        """
        Get the remaining bytes available in the buffer.
        """
        diff = self._max_size - sum(starmap(lambda h, w: h*w, self._slots.values()))
        return diff if diff >= 0 else 0

    def get_n_free_slots(self) -> int:
        """
        Get the number of free slots left.
        """
        return __class__.MAX_OBJECTS - len(self.slots)

    def _find_free_id(self) -> Optional[int]:
        """
        Find the first available object ID.
        """
        for k in range(__class__.MAX_OBJECTS):
            if self._slots.get(k, None) is None:
                return k
        return None

    def free(self, obj_id: int) -> bool:
        """
        Free a slot in the buffer (this is not realistic HW-wise but
                                   provided out of programming courtesy)
        """
        if obj_id in self.slots:
            self._slots.pop(obj_id)
            return True
        return False

    def get_slot(self, height: int, width: int) -> Optional[int]:
        """
        Find the most suited existing free slot for a graphic of given dimensions.
        :param height: Graphic height to fit
        :param width:  Graphic width to fit
        :return: the slot ID that is large enough to fit the graphic, if any.
        """
        best_fit = np.inf
        best_fit_id = None
        for obj_id in self._slots:
            if obj_id in self._loaded:
                continue
            h, w = self._slots[obj_id]
            area_diff =  w*h-width*height
            if h >= height and w >= width and area_diff < best_fit:
                best_fit = area_diff
                best_fit_id = obj_id
        return best_fit_id

    def get_all_slots(self, only_free: bool = False) -> dict[int, tuple[int, int]]:
        """
        Return all allocated slots.
        :param only_free: Flag to request only the unloaded slots.
        """
        if only_free:
            return dict(filter(lambda x: x[0] in self._loaded, self._slots.items()))
        return self._slots

    def flush_objects(self) -> None:
        """
        Flush all loaded data on acquisition (the slots still exist but empty).
        """
        self._loaded.clear()

    def load(self, obj_id: int) -> None:
        """
        Mark a given slot ID as occupied in the buffer.
        :param obj_id: Slot ID to flag as occupied.
        """
        assert not self.is_loaded(obj_id), "Object already loaded in the buffer."
        self._loaded.append(obj_id)

    def is_loaded(self, obj_id: int) -> bool:
        """
        Check if a slot is already occupied by an object.
        :param obj_id: slot ID
        :return: true if the slot is occupied
        """
        return obj_id in self._loaded

    def check(self, obj_id: int, height: int, width: int) -> None:
        """
        Check that the provided object is suited for the given slot.
        :param obj_id: Slot ID
        :param height: Object height
        :param width:  Object width
        """
        hw = self._slots.get(obj_id, None)
        assert hw is not None, f"No slot allocated for {obj_id}"
        assert hw == (height, width), "Dimensions mismatch."

    def allocate(self, height: int, width: int) -> Optional[int]:
        """
        Allocate a buffer slot with given dimensions.
        :param height: Object height
        :param width: Object width
        :return: the object ID, if a slot could be allocated.
        """
        assert 8 <= width <= 4096 and 8 <= height <= 4096, "Incorrect dimensions for PG buffer."

        new_id = self._find_free_id()
        if new_id is not None and self.get_free_size() - width*height >= 0:
            self._slots[new_id] = (height, width)
            return new_id
        return None
