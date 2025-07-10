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

import numpy as np
import numpy.typing as npt
import cv2

from PIL import Image
from pathlib import Path
from typing import Optional, Union, Any
from collections.abc import Iterable
from enum import IntEnum, auto
from piliq import PILIQ, PNGQuantWrapper
from brule import HexTree, QtzrUTC

from .palette import Palette, PaletteEntry
from .utils import LogFacility, get_matrix, SSIMPW

logger = LogFacility.get_logger('SUPer')

class FadeCurve(IntEnum):
    LINEAR = auto()
    QUADRATIC = auto()
    EXPONENTIAL = auto()

class Quantizer:
    class Libs(IntEnum):
        QTZR  = 0
        PIL_KM  = 1
        HEXTREE = 2
        PILIQ   = 3
        PNGQNT  = 4

        @classmethod
        def _missing_(cls, v: Any) -> 'Quantizer.Libs':
            if isinstance(v, cls):
                v = v.value
            else:
                try:
                    v = int(v)
                except ValueError:
                    ...
            if v in [ev.value for ev in cls]:
                return cls(v)
            return cls(cls.HEXTREE.value)

    _opts = {}
    _piliq = None
    _alt_piliq = None
    @classmethod
    def get_options(cls) -> dict[int, (str, str)]:
        if cls._opts == {}:
            cls.find_options()
        return cls._opts

    @classmethod
    def get_option_id(cls, option_str: str) -> 'Quantizer.Libs':
        algo = option_str.strip().split(' ')[0]
        for opt_id, opt in cls._opts.items():
            if opt[0] == algo:
                return opt_id
        logger.error("Unknown quantizer library requested, returning default.")
        return 0

    @classmethod
    def find_options(cls) -> None:
        if cls._piliq is not None:
            cls._opts[cls.Libs.PILIQ] = (cls.get_piliq().lib_name,'(best, fast)')
        if cls._alt_piliq is not None:
            cls._opts[cls.Libs.PNGQNT] = (cls._alt_piliq.lib_name,'(best, fast)')
        if 'C' in HexTree.get_capabilities():
            cls._opts[cls.Libs.HEXTREE] = ("HexTree", '(good, very fast)')
        qtzr_info = '(better, fast)' if 'C' in QtzrUTC.get_capabilities() else '(good, slow)'
        cls._opts[cls.Libs.QTZR] = ('Qtzr', qtzr_info)
        cls._opts[cls.Libs.PIL_KM] = ('Pillow', '(average, turbo)')

    @classmethod
    def init_piliq(cls,
        qpath: Optional[Union[str, Path]] = None,
        quality: Optional[int] = 100,
        speed: Optional[int] = 4,
        dither: Optional[int] = 100,
    ) -> bool:
        piliq = None
        try:
            piliq = PILIQ(qpath)
        except (FileNotFoundError, AssertionError):
            logger.debug(f"Failed to load advanced quantizer at '{qpath}'.")
        if piliq is None and qpath is not None:
            #Perform auto-look up, likely to fail but can still find libs
            try:
                piliq = PILIQ()
            except:
                logger.debug("Failed to load advanced quantizer with auto look-up.")
        cls._piliq = piliq
        success = False
        if piliq is not None and piliq.is_ready():
            logger.debug(f"Configuring {piliq.lib_name} with: speed={speed}:quality={quality}:dither={dither/100.0}")
            cls.write_piliq_config(piliq, speed, quality, dither)
            success = True
        if success and piliq.lib_name != 'pngquant':
            if PNGQuantWrapper.is_ready():
                cls._alt_piliq = PILIQ(_wrapper=PNGQuantWrapper())
                cls.write_piliq_config(cls._alt_piliq, speed, quality, dither)
        return success

    @staticmethod
    def write_piliq_config(piq_inst: PILIQ, speed: int, quality: int, dither: float) -> None:
        piq_inst.return_pil = False
        piq_inst.set_speed(speed)
        piq_inst.set_quality(quality)
        piq_inst.set_dithering_level(dither/100.0)

    @classmethod
    def select_quantizer(cls, option_id: int) -> int:
        if option_id > cls.Libs.PNGQNT:
            logger.error("Unknown quantizer ID '{option_id}', attempting to use piliq library.")
            option_id = cls.Libs.PILIQ

        if option_id == cls.Libs.PNGQNT:
            assert cls._piliq is not None
            if cls._piliq.lib_name != 'pngquant':
                if cls._alt_piliq is not None:
                    cls._piliq.destroy()
                    cls._piliq = cls._alt_piliq
                else:
                    logger.error("Requesting specifically pngquant, but executable not found.")
            option_id = cls.Libs.PILIQ
        if option_id == cls.Libs.PILIQ and not cls.get_piliq():
            fallback = cls.get_brule_fallback()
            logger.error(f"Unable to find an advanced quantizer (pngquant, libimagequant). Using lower quality: {fallback.name}.")
            option_id = fallback
        return int(option_id)

    @classmethod
    def get_brule_fallback(cls) -> 'Quantizer.Libs':
        if 'C' in HexTree.get_capabilities():
            return cls.Libs.HEXTREE
        return cls.Libs.QTZR

    @classmethod
    def get_piliq(cls) -> Optional[PILIQ]:
        return cls._piliq

    @classmethod
    def log_selection(cls, idx: Union[int, 'Quantizer.Libs']) -> None:
        idxi = cls.Libs(idx)
        if idxi in [cls.Libs.QTZR, cls.Libs.HEXTREE]:
            cap_string = (', capabilities: ') + (', '.join((QtzrUTC if idxi == cls.Libs.QTZR else HexTree).get_capabilities()))
        else:
            cap_string = ''
        logger.debug(f"RGBA quantizer '{idxi.name}'{cap_string}.")
    ####
####

class Preprocess:
    @classmethod
    def quantize(cls, img: Image.Image, colors: int = 256, **kwargs) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        quant_method = Quantizer.Libs(kwargs.pop('quantize_lib', Quantizer.Libs.HEXTREE))
        single_bitmap = kwargs.get('single_bitmap', False)

        if Quantizer.Libs.PILIQ == quant_method:
            if single_bitmap:
                nc = colors
            else:
                nc = len(img.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)

            lib_piq = Quantizer.get_piliq()
            assert lib_piq is not None

            if not single_bitmap:
                original_quality = lib_piq.get_quality()
                original_dither = lib_piq.get_dithering_level()
                lib_piq.set_quality(max(1, int(np.ceil(original_quality*0.975))))
                lib_piq.set_dithering_level(original_dither*0.9)

            pal, qtz_img = lib_piq.quantize(img, min(colors, int(np.ceil(20+nc*235/255))))
            if not single_bitmap:
                lib_piq.set_dithering_level(original_dither)
                lib_piq.set_quality(original_quality)
            return qtz_img, pal

        elif Quantizer.Libs.QTZR == quant_method:
            # Use PIL to get approximate number of clusters
            nk = len(img.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
            nk = min(colors, int(np.ceil(20+nk*235/255)))
            return QtzrUTC.quantize(np.asarray(img, dtype=np.uint8), nk)

        elif Quantizer.Libs.HEXTREE == quant_method:
            nc = colors if single_bitmap else len(img.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
            npimg = np.asarray(img, dtype=np.uint8)
            npbm, nppal = HexTree.quantize(npimg, max(16, min(colors, int(np.ceil(20+nc*235/255)))))
            return npbm, nppal

        else:
            odim = (img.height, img.width)
            oimg = img
            if min(odim) < 8:
                img_padded = Image.new('RGBA', (max(img.width, 8), max(img.height, 8)), (0, 0, 0, 0))
                img_padded.paste(img, (0, 0))
                img = img_padded
            img_out = img.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
            npimg = np.asarray(img_out, dtype=np.uint8)
            nppal = np.asarray(list(img_out.palette.colors.keys()), dtype=np.uint8)

            #Somehow, pillow may sometimes not return all palette entries?? I've seen a case where one ID was consistently missing.
            pil_failed = len(img_out.palette.colors) != 1+max(img_out.palette.colors.values())

            #When PIL fails to quantize alpha channel, there's a clear discrepancy between original and quantized image.
            pil_failed = pil_failed or SSIMPW.compare(Image.fromarray(nppal[npimg], 'RGBA'), img) < 0.95

            if pil_failed:
                logger.ldebug("Pillow failed to palettize image, falling back to HexTree.")
                return cls.quantize(oimg, colors, quantize_lib=Quantizer.Libs.HEXTREE, **kwargs)

            return npimg[:odim[0], :odim[1]], nppal

    @staticmethod
    def find_most_opaque(events: list[Image.Image]) -> int:
        """
        Find out which image is the most opaque of a set. Useful to recalculate fades
         without merging intermediate images to the final bitmap (=lower quality)

        :param events:  List of PNG update events.
        :return:        Event which has the most opaque image.
        """
        # A single PNG image got flagged, just return it.
        if not isinstance(events, Iterable):
            return events

        a_max, idx = 0, -1

        for k, event in enumerate(events):
            tmp = np.linalg.norm(np.asarray(event)[:,:,3], ord=1)
            if tmp > a_max:
                a_max = tmp
                idx = k
        return idx

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

        :return: bitmap, sequence of palette update to obtain the said input animation.
        """

        if 1 == len(events):
            img, clut = Preprocess.quantize(events[0], colors, single_bitmap=True, **kwargs)
            return img.copy(), np.expand_dims(clut, 1).copy()

        sequences = np.zeros((len(events), *events[0].size[::-1], 4), np.uint8)
        for ke, event in enumerate(events):
            img, clut = Preprocess.quantize(event, colors, single_bitmap=False, **kwargs)
            sequences[ke, :, :, :] = clut[img]
        sequences = np.moveaxis(sequences, 0, 2)

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
        remap: dict[int, int] = {}
        for cnt, v in enumerate(seq_sorted.values()):
            if cnt < colors:
                norm_mat[cnt, :, :] = v
            else:
                nm = np.linalg.norm(norm_mat - v[None, :], 2, axis=2)

                id1 = np.argsort(np.sum(nm, axis=1))
                id2 = np.argsort(np.sum(nm, axis=1)/np.sum(nm != 0, axis=1))

                best_fit = np.abs(id1 - id2[:, None])
                remap[cnt] = id1[best_fit.argmin() % id1.size]
        del norm_mat

        bitmap = np.zeros(sequences.shape[0:2], dtype=np.uint8)
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                seq = sequences[i, j, :, :]
                hsh = hash(seq.tobytes())
                if seq_ids[hsh] < colors:
                    bitmap[i, j] = seq_ids[hsh]
                else:
                    bitmap[i, j] = remap[seq_ids[hsh]]
        #retun bitmap and the color sequence (copy only the kept sequences)
        return bitmap, np.asarray([seq for seq, _ in zip(seq_sorted.values(), range(colors))], dtype=np.uint8)


    @classmethod
    def solve_and_remap(cls, events: list[Image.Image], colors: int = 255, first_index: int = 1, **kwargs) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """
        This function solves the input event sequence and perform ID remapping
        to optimise the distribution of colour indices wrt PGS constraints
        :param events: list of PIL images to optimise.
        :param colors: max number of colours to use.
        :param first_index: CLUT offset for given bitmap, must be a positive number.
        :return: bitmap and chain of palettes.
        """
        assert 0 < first_index + colors <= 256, "8-bit ID out of range."
        assert first_index > 0, "Usage of palette ID zero."

        bitmap, cluts = cls.solve_sequence_fast(events, colors, **kwargs)
        transparent_id = np.nonzero(np.all(cluts[:,:,-1] == 0, axis=1))[0]

        kwargs_diff = {'matrix': kwargs.get('bt_colorspace', 'bt709')}

        #No transparency at all in this bitmap
        if 0 == len(transparent_id):
            #All colours used incl reserved transparent index. This is incorrect, requantize with colors-1
            if np.max(bitmap) == colors - 1:
                logger.ldebug("Too many colours used, lowering count.")
                bitmap, cluts = cls.solve_sequence_fast(events, colors-1, **kwargs)
            palettes = cls.diff_cluts(cluts, **kwargs_diff)
            bitmap += first_index
        else:
            # Transparent ID is the last one and will be mapped to 0xFF by the first_index shift.
            if max(transparent_id) == (0xFF - first_index):
                transparent_id = 0xFF - first_index
                bitmap += first_index
            else:
                #Shift only IDs
                transparent_id = int(transparent_id[0])
                tsp_mask = (bitmap == transparent_id)
                smaller = bitmap < transparent_id
                larger = bitmap > transparent_id
                bitmap[smaller] += first_index
                bitmap[larger] += (first_index - 1)
                bitmap[tsp_mask] = 0xFF
            #logger.ldebug(f"Remapped fully transparent ID {transparent_id:02X} to FF.")
            cluts = np.delete(cluts, [transparent_id], axis=0)
            palettes = cls.diff_cluts(cluts, **kwargs_diff)

        for pal in palettes:
            pal.offset(first_index)
        assert len(palettes[0]) < colors
        return bitmap, palettes
    ####

    @staticmethod
    def diff_cluts(cluts: npt.NDArray[np.uint8], /, *,
                   matrix: str = 'bt709') -> list[Palette]:
        """
        This functions finds the chain of palette updates for consecutives cluts.
        :param cluts:  Color look-up tables of the sequence, stacked one after the other.
        :param matrix:  BT ITU conversion

        :return: N palette objects defining palette that can be converted to PDSes.
        """
        stacked_cluts = np.swapaxes(cluts, 1, 0).astype(np.int32)
        matrix = get_matrix(matrix, False)

        shape = stacked_cluts.shape
        stacked_cluts = np.round(np.matmul(stacked_cluts.reshape((-1, 4)), matrix.T))
        stacked_cluts += np.asarray([[16, 128, 128, 0]])
        clip_vals = (np.array([[16, 16, 16, 0]]), np.asarray([[235, 240, 240, 255]]))
        stacked_cluts = np.clip(stacked_cluts, *clip_vals).astype(np.uint8).reshape(shape)
        #YCbCrA -> YCrCbA
        stacked_cluts = stacked_cluts[:, :, [0, 2, 1, 3]]
        l_pal = []
        for j, clut in enumerate(stacked_cluts):
            pal = Palette()
            for k, pal_entry in enumerate(clut):
                n_e = PaletteEntry(*pal_entry)
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
