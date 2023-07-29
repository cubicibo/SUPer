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

import numpy as np
import numpy.typing as npt

from PIL import Image, ImagePalette
from typing import Optional, Union
from collections.abc import Iterable
from flags import Flags
from enum import IntEnum, auto
import cv2

from .palette import Palette, PaletteEntry
from .utils import ImageEvent, TimeConv as TC, get_super_logger

logging = get_super_logger('SUPer')

class PalettizeMode(Flags):
    MERGE_QUANTIZE = ()
    INDIV_QUANTIZE = ()

class FadeCurve(IntEnum):
    LINEAR = auto()
    QUADRATIC = auto()
    EXPONENTIAL = auto()

class Preprocess:
    @staticmethod
    def quantize(img: Image.Image, colors: int = 256, kmeans_quant: bool = False, kmeans_fade: bool = False, **kwargs) -> Image.Image:
        #use cv2 for high transparency images, pillow has issues

        alpha = np.asarray(img.split()[-1], dtype=np.uint16)
        non_tsp_pix = alpha[alpha > 0]
        if non_tsp_pix.size > 0:
            kmeans_fade = (np.mean(non_tsp_pix) < 38) and kmeans_fade

        if kmeans_quant or kmeans_fade:
            # Use PIL to get approximate number of clusters
            nk = len(img.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
            ocv_img = np.asarray(img)
            flat_img = np.float32(ocv_img.reshape((-1, 4)))

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.33)
            ret, label, center = cv2.kmeans(flat_img, nk, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

            center = np.uint8(np.round(np.clip(center, 0, 255)))

            offset = 1024
            occs = np.argsort(np.bincount(label[:,0]))[::-1]
            for idx in occs:
                label[label == idx] = offset
                offset += 1

            label -= 1024
            center = center[occs]

            pil_pal = ImagePalette.ImagePalette(mode='RGBA', palette=bytes(np.uint8(center)))
            return np.reshape(label.flatten(), ocv_img.shape[:-1]).astype(np.uint8), pil_pal.colors
        else:
            if colors == 256:
                img_out = img.convert('P')
            else:
                img_out = img.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
            return np.asarray(img_out, dtype=np.uint8), img_out.palette.colors

    @staticmethod
    def crop_right(imgs: list[Image.Image], shape: tuple[int]) -> list[Image.Image]:
        return [Image.fromarray(np.asarray(img)[:shape[0],:shape[1],:], 'RGBA') for img in imgs]

    @staticmethod
    def palettize_events(events: list[ImageEvent], flags: PalettizeMode,
                            colors: Union[int, list[int]] = 256) -> list[ImageEvent]:
        """
        Perform basic preprocessing to ease the work of the solver.
         This functions optimises subtitles by groups. Groups are sublists in events
         which typically represents an animation that is to be solved to a colormap.

        :param events:  List of XML Event entries with aossiciated images
        :param flags:   Optimisation(s) to apply, see PalettizeMode enum.
        :param colors:  Number(s) of colors limit to apply to each or all images.
        :return:        The events with optimised images.
        """
        if 2 <= colors > 256:
            raise ValueError("Palettization is always performed on 2< colors <=256.")

        if not PalettizeMode(flags):
            logging.info("No known optimisation selected, skipping.")
            return events

        n_event: list[ImageEvent] = []

        if PalettizeMode.INDIV_QUANTIZE in PalettizeMode(flags):
            itcolors = colors if type(colors) is list else [colors] * len(events)

            for event, i_colors in zip(events, itcolors):
                n_event.append(ImageEvent(event.img.quantize(colors=i_colors,
                                               method=Image.Quantize.FASTOCTREE,
                                               palette=None,
                                               dither=Image.Dither.NONE).convert('RGBA'),
                                          event.event))

        if PalettizeMode.MERGE_QUANTIZE in PalettizeMode(flags):
            #Allow to perform group-wise preprocessing
            itobj = events if type(events[0]) is list else [events]
            itcolors = colors if type(colors) is list else [colors] * len(itobj)

            for i_events, i_colors in zip(itobj, itcolors):
                wt, ht = 0, 0
                for event in i_events:
                    if event.img.width > wt:
                        wt = event.img.width
                    ht += event.img.height

                stack = Image.new('RGBA', (wt,ht), (0, 0, 0, 0))
                heights = []
                for k, event in enumerate(i_events):
                    stack.paste(event.img, (0, sum(heights)))
                    heights.append(event.img.height)

                qtevts = stack.quantize(colors=i_colors,
                                        method=Image.Quantize.FASTOCTREE,
                                        palette=None,
                                        dither=Image.Dither.NONE)

                qtevts = qtevts.convert('RGBA')

                h_prev = 0
                for event in i_events:
                    n_event.append(ImageEvent(Image.fromarray(np.asarray(qtevts)\
                                              [h_prev:event.img.height+h_prev,:,:])),
                                              event.event)
                    h_prev += event.img.height

        return n_event

    @staticmethod
    def find_most_opaque(events: Union[list[ImageEvent], list[Image.Image]]) -> ImageEvent:
        """
        Find out which image is the most opaque of a set. Useful to recalculate fades
         without merging intermediate images to the final bitmap (=lower quality)

        :param events:  List of PNG update events.
        :return:        Event which has the most opaque image.
        """
        # A single PNG image got flagged, just return it.
        if not isinstance(events, Iterable) or isinstance(events, ImageEvent):
            return events

        a_max, idx = 0, -1

        for k, event in enumerate(events):
            if isinstance(events, Image.Image):
                event = ImageEvent(event, '')
            tmp = np.linalg.norm(np.asarray(event.img)[:,:,3], ord=1)
            if tmp > a_max:
                a_max = tmp
                idx = k
        return events[idx]

    @staticmethod
    def merge_captions(imgs: list[Image.Image], *, _mode: str = 'P') -> Image:
        """
        Merge together N subtitles using alpha compositing

        :param imgs:  Image to merge together, they must have the same dimensions.
        :param _mode:  Mode of the return image.

        :return:  Image in _mode, palettized by default.
        """
        logging.info("Merging RGBA images together to find the overall bitmap.")
        for k, img in enumerate(imgs):
            if k == 0:
                overall = img
            else:
                if img.width != overall.width and img.height != overall.height:
                    raise Exception("Set of images have varying dimensions."
                                    "Pad them all to the same shape.")

                overall = Image.alpha_composite(overall, img)
        return overall.convert(_mode)

    @staticmethod
    def palettize_img(img: Image, pal: npt.NDArray[np.uint8], *,
                      _return_mode: str = 'P') -> tuple[Image,npt.NDArray[np.uint8]]:
        """
        Palettize an image without dropping graphics. Can be slow.

        :param img:  Image to palettize.
        :param palette: PIL-like palette to use.
        :return: image, palette used
        """
        imga = np.asarray(img, np.int16)
        if pal is None:
            pal = np.asarray(list(img.convert('P').palette.colors.keys()), np.uint8)
        subs = np.asarray(imga - pal[:, None, None], dtype=np.int64)
        out = pal[np.einsum('ijkl,ijkl->ijk', subs, subs).argmin(0)].astype(np.uint8)
        out = Image.fromarray(out, 'RGBA').convert(_return_mode)
        return out, pal


class Optimise:
    @staticmethod
    def solve_sequence_fast(events, colors: int = 256, **kwargs) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """
        This functions finds a solution for the provided subtitle animation.
        :param events: PIL images, stacked one after the other
        :param colors: max number of sequences usable
        :param kmwans: enable kmeans quantization

        :return: bitmap, sequence of palette update to obtain the said input animation.
        """

        sequences = []
        for event in events:
            img, img_pal = Preprocess.quantize(event, colors, **kwargs)
            clut = np.asarray(list(img_pal.keys()), dtype=np.uint8)
            sequences.append(clut[img])

        sequences = np.stack(sequences, axis=2).astype(np.uint8)
        #catalog the sequences
        seq_occ: dict[int, tuple[int, npt.NDArray[np.uint8]]] = {}
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                seq = sequences[i, j, :, :]
                hsh = hash(seq.tobytes())
                try:
                    seq_occ[hsh][0] += 1
                except KeyError:
                    seq_occ[hsh] = [1, seq]

        #Sort sequences by commonness
        seq_sorted = {k: x[1] for k, x in sorted(seq_occ.items(), key=lambda item: item[1][0], reverse=True)}
        seq_ids = {k: z for z, k in enumerate(seq_sorted.keys())}

        #Fill a new array with kept sequences to perform fast norm calculations
        norm_mat = np.ndarray((colors, *sequences[i,j,:,:].shape[0:2]))

        #Match sequences to the most common ones (N[colors] kept)
        remap = {}
        for cnt, v in enumerate(seq_sorted.values()):
            if cnt < colors:
                norm_mat[cnt, :, :] = v
            else:
                nm = np.linalg.norm(norm_mat - v[None, :], 2, axis=2)

                id1 = np.argsort(np.sum(nm, axis=1))
                id2 = np.argsort(np.sum(nm, axis=1)/np.sum(nm != 0, axis=1))

                best_fit = np.abs(id1 - id2[:, None])
                remap[cnt] = id1[best_fit.argmin() % id1.size]

        bitmap = np.zeros(sequences.shape[0:2], dtype=np.uint8)
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                seq = sequences[i, j, :, :]
                hsh = hash(seq.tobytes())
                if seq_ids[hsh] < colors:
                    bitmap[i, j] = seq_ids[hsh]
                else:
                    bitmap[i, j] = remap[seq_ids[hsh]]
        #retun bitmap and the color sequence.
        return bitmap, np.asarray(list(seq_sorted.values()), dtype=np.uint8)[:colors]


    @staticmethod
    def diff_cluts(cluts: npt.NDArray[np.uint8], to_ycbcr: bool = True, /, *,
                   matrix: str = 'bt709', s_range: str = 'limited') -> list[Palette]:
        """
        This functions finds the chain of palette updates for consecutives cluts.
        :param cluts:  Color look-up tables of the sequence, stacked one after the other.
        :param to_ycbcr: Convert to YCbCr when diffing (for PGS)
        :param matrix:  BT ITU conversion
        :param s_range: YUV range.

        :return: N palette objects defining palette that can be converted to PDSes.
        """
        stacked_cluts =  np.swapaxes(cluts, 1, 0)
        if to_ycbcr:
            PE_fn = lambda rgba: PaletteEntry.from_rgba(rgba, matrix=matrix,
                                                        s_range=s_range)
        else:
            PE_fn = lambda ycbcra: PaletteEntry(*ycbcra)

        l_pal = []

        for j, clut in enumerate(stacked_cluts):
            pal = Palette()
            for k, pal_entry in enumerate(clut):
                n_e = PE_fn(pal_entry)
                if j == 0: # For t0, set the entire palette regardless
                    pal[k] = n_e
                    continue

                # Seek backwards and find last time this entry was set
                for bw in range(j-1, -1, -1):
                    p_e = l_pal[bw].get(k, None)
                    if p_e == n_e:
                        break  #Identical, exit out
                    if p_e is not None and p_e != n_e:
                        pal[k] = n_e
                        break #We found the last time it was set, exit backward loop
            l_pal.append(pal)
        return l_pal

    @staticmethod
    def fade(tc_in: str, fade_out: bool, tc_out: str, fps: float, stride: int=1,
                      palette: Optional[Palette] = None, img: Optional[Image.Image] = None,
                      dynamics: FadeCurve = FadeCurve.LINEAR, flip_dyn: bool = False,
                      **kwargs) -> Union[list[Palette], list[ImagePalette.ImagePalette]]:
        """
        Optimize a fade (in/out) for a given image

        :param palette: Palette to use (if img, palette is ignored.)
        :param img:     P-mode image to fade (if img, palette is ignored.)

        :param tc_in:   Timecode when the effect starts (start appearing/disappear)
        :param tc_out:  Timecode when the effects ends (end appearing/disappearing)
        :param fps:     FPS of the video stream
        :param stride:  FPS prescaler for palette updates (1=every, 2=every other frame)
        :param flip_dyn: Flip the dynamics of the curve (slow2fast becomes fast2slow)
        :param kwargs: Provide additional values to customize the behaviour (Read code)
        :return:        list of palettes (1 entry in list -> 1 DS.PDS).
        """

        if img is not None and palette is not None:
            raise ValueError("Provided both an image and a palette. Pick either!")

        if stride < 1 or type(stride) is not int:
            raise ValueError(f"Incorrect stride parameter, got '{stride}'.")

        if img.mode != 'P' or not isinstance(palette, Palette) :
            raise NotImplementedError("Image must be in P mode with an inner palette.")

        nf = TC.tc2f(tc_out, fps) - TC.tc2f(tc_in, fps)

        if nf < 3:
            raise ValueError("Attempting fade on less than three frames...")

        if FadeCurve(dynamics) == FadeCurve.LINEAR:
            coeffs = np.arange(1, nf)/nf

        elif FadeCurve(dynamics) == FadeCurve.QUADRATIC:
            # User can abuse this to define their own function too
            power = kwargs.get('pow', 2)
            f = kwargs.get('qfunc', lambda x: (x**power)/nf)
            coeffs = f(np.arange(0, nf, stride))

        elif FadeCurve(dynamics) == FadeCurve.EXPONENTIAL:
            # By default the beginning of the animation is abrupt then it smooths out
            sig = kwargs.get('sig', nf/3) # /3 is empirical
            f = lambda x: (np.exp(x/sig)-1)/np.exp(nf/sig)
            coeffs = f(np.arange(1, nf, stride))

        else:
            raise NotImplementedError("Requested fade dynamics is not available.")

        if fade_out:
            coeffs = np.flip(coeffs)

        # Flip the coefficients of a curve, (i.e fast to slow out -> slow to fast out)
        if flip_dyn:
            coeffs = np.flip(1-coeffs)

        #Broadcast coeffs to palette alpha
        if isinstance(palette, Palette):
            pal_tsp = np.asarray([tuple(entry) for entry in palette.palette.values()])
        else:
            pal_tsp = np.array([*img.palette.colors.keys()])

        alphas_per_frame = pal_tsp[:, 3, None]*coeffs
        alphas_per_frame = np.round(alphas_per_frame)
        alphas_per_frame[alphas_per_frame < 0] = 0
        alphas_per_frame[alphas_per_frame > 255] = 255

        pals = []
        for k, alphas in enumerate(alphas_per_frame[:].T):
            pal_tsp[:,3] = alphas
            if isinstance(palette, Palette):
                pals.append(Palette(dict(zip(palette.palette.keys(), pal_tsp))))
            else:
                pals.append(ImagePalette.ImagePalette('RGBA',
                                            pal_tsp.reshape((np.dot(*pal_tsp.shape)))))

        return pals

    @staticmethod
    def eval_animation(cmap: npt.NDArray[np.uint8], sequence: npt.NDArray[np.uint8],
                       ret_array: bool = False) -> Union[Optional[list[Image.Image]],
                                                    Optional[npt.NDArray[np.uint8]]]:
        """
        Evaluate the output of the solver.
        :param cmap: P-Array, data that will be RLE-coded for PGS.
        :param sequence: Sequence of CLUTs for the animation, linked to cmap.

        :return: Either a Nx(RGBA) numpy object that can be iterated through or images objs
        """
        anim = np.moveaxis(sequence[cmap], [2,], [0,]).astype(np.uint8)
        if ret_array:
            return anim
        return [Image.fromarray(anim[k], 'RGBA') for k in range(len(anim))]

    @classmethod
    def show(cls, ri = True, cmap: Optional[npt.NDArray[np.uint8]] = None,
             cluts: Optional[npt.NDArray[np.uint8]] = None,
             imgs: Optional[list[Image.Image]] = None) -> None:
        if cmap is not None or cluts is not None:
            if not(cmap is not None and cluts is not None):
                raise ValueError("Missing color map or CLUT sequence.")
            ret = cls.eval_animation(cmap, cluts)
        elif imgs:
            ret = imgs

        wt, ht = 0, 0
        for event in ret:
            if event.width > wt:
                wt = event.width
            ht += event.height

        stack = Image.new('RGBA', (wt,ht), (0, 0, 0, 0))
        heights = []
        for k, event in enumerate(ret):
            stack.paste(event, (0, sum(heights)))
            heights.append(event.height)
        if ri:
            return stack
        else:
            stack.show()
