#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 cibo
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
filterwarnings("ignore", message=r"Kernel", module="SSIM_PIL")

import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from SSIM_PIL import compare_ssim
from typing import TypeAlias, Any

from brule import HexTree as _HexTree, QtzrUTC as _Qtzr

from ..internals import _classproperty, LogFacility
from ..display.palette import Palette, Matrix, PaletteEntry

logger = LogFacility.get_logger('SUPer')

class SSIMPW:
    use_gpu = True

    @classmethod
    def compare(cls, img1: Image.Image, img2: Image.Image) -> float:
        return compare_ssim(img1, img2, GPU=cls.use_gpu)

_PaletteT: TypeAlias = np.ndarray[tuple[int, int], np.uint8]
_BitmapT: TypeAlias = np.ndarray[tuple[int, int], np.uint8]

class QuantizerWrap(ABC):
    @classmethod
    def quantize(cls, image: Image.Image, colors: int, **kwargs) -> tuple[_PaletteT, _BitmapT]:
        assert 2 <= colors <= 256
        return cls._postprocess(*cls._palettize(*cls._preprocess(image, colors, **kwargs)))

    @classmethod
    def _preprocess(cls, image: Image.Image, colors: int, *args, **kwargs) -> tuple[Image.Image, int, ...]:
        return image, colors, *args, kwargs

    @classmethod
    def _postprocess(cls, palette: _PaletteT, bitmap: _BitmapT, *args, **kwargs) -> tuple[_PaletteT, _BitmapT]:
        return palette, bitmap

    @classmethod
    @abstractmethod
    def _palettize(cls, image: Image.Image, colors: int, *args, **kwargs) -> tuple[_PaletteT, _BitmapT, ...]:
        ...

    @_classproperty
    def name(cls) -> str:
        return cls.__name__.replace("Wrap", "")

    @classmethod
    def is_optimized(cls) -> bool:
        return True

    @classmethod
    def is_ready(cls) -> bool:
        return True

    @classmethod
    def configure(cls, settings: dict[Any, Any]) -> bool:
        return cls.is_ready()

class HexTreeWrap(QuantizerWrap):
    @classmethod
    def _palettize(cls, image: Image.Image, colors: int) -> tuple[_PaletteT, _BitmapT, ...]:
        bitmap, palette = _HexTree.quantize(np.asarray(image, dtype=np.uint8), colors)
        return palette, bitmap

    @classmethod
    def _preprocess(cls, image: Image.Image, colors: int, **kwargs) -> tuple[Image.Image, int, ...]:
        if not kwargs.get('single_bitmap', False):
            nc = len(image.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
        else:
            nc = colors
        nc = max(16, min(colors, int(np.ceil(20 + nc*235/255))))
        return image, nc

    @classmethod
    def is_optimized(cls) -> bool:
        return 'C' in _HexTree.get_capabilities()

class QtzrWrap(QuantizerWrap):
    @classmethod
    def _preprocess(cls, image: Image.Image, colors: int, **kwargs) -> tuple[Image.Image, int, ...]:
        # Use PIL to get approximate number of clusters. never use colours as is because it will overfit
        n_clusters = len(image.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
        n_clusters = min(colors, int(np.ceil(20 + n_clusters*235/255)))
        return image, n_clusters

    @classmethod
    def _palettize(cls, image: Image.Image, colors: int) -> tuple[_PaletteT, _BitmapT]:
        return _Qtzr.quantize(np.asarray(image, dtype=np.uint8), colors)[::-1]

    @classmethod
    def is_optimized(cls) -> bool:
        return 'C' in _Qtzr.get_capabilities()

class PillowWrap(QuantizerWrap):
    @classmethod
    def _preprocess(cls, image: Image.Image, colors: int, **kwargs) -> tuple[Image.Image, int, dict[str, Any]]:
        if min(image.size) < 8:
            img_padded = Image.new('RGBA', (max(image.width, 8), max(image.height, 8)), (0, 0, 0, 0))
            img_padded.paste(image, (0, 0))
        else:
            img_padded = image
        return img_padded, colors, {'_pil_input_img': image, '_colors': colors}

    @classmethod
    def _palettize(cls, image: Image.Image, colors: int, ctx: dict[str, Image.Image]) -> tuple[_PaletteT, _BitmapT, dict[str, Any]]:
        img_out = image.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        bitmap = np.asarray(img_out, dtype=np.uint8)
        palette = np.asarray(list(img_out.palette.colors.keys()), dtype=np.uint8)
        return palette, bitmap, ctx | {'_pil_output_img': img_out}

    @classmethod
    def _postprocess(cls, palette: _PaletteT, bitmap: _BitmapT, ctx: dict[str, Image.Image]) -> tuple[_PaletteT, _BitmapT]:
        img_in, img_out, colors = ctx['_pil_input_img'], ctx['_pil_output_img'], ctx['_colors']
        #bug workaround: sometime pillow may sometimes not return all palette entries
        pil_failed = len(img_out.palette.colors) != 1+max(img_out.palette.colors.values())

        #When PIL fails to quantize alpha channel, there's a clear discrepancy between original and quantized image.
        if pil_failed or SSIMPW.compare(Image.fromarray(palette[bitmap]).convert('RGBA'), img_in) < 0.95:
            # Out of the builtin quantizer, qtzr is always the best fallback.
            return QtzrWrap.quantize(img_in, colors)
        #no-op crop if the image was not padded
        return palette, bitmap[:img_in.size[1], :img_in.size[0]]

class ImageQuantWrap(QuantizerWrap):
    __piq = None
    @classmethod
    def _preprocess(cls, image: Image.Image, colors: int, **kwargs) -> tuple[Image.Image, int, dict[str, Any]]:
        assert cls.__piq is not None, "PIQ wrapper not configured."
        if not kwargs.get('single_bitmap', False):
            nc = len(image.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
        else:
            nc = colors
            kwargs |= {'__orig_dither':  cls.__piq.get_dithering_level(),
                       '__orig_quality': cls.__piq.get_quality()}
            cls.__piq.set_dithering_level(kwargs['__orig_dither']*0.9)
            cls.__piq.set_quality(max(1, int(np.ceil(kwargs['__orig_quality']*0.975))))
        colors = max(2, min(colors, int(np.ceil(20 + nc*235/255))))
        return image, colors, kwargs

    @classmethod
    def _palettize(cls, image: Image.Image, colors: int, settings: dict[str, Any]) -> tuple[_PaletteT, _BitmapT, dict[str, Any]]:
        palette, bitmap = cls.__piq.quantize(image, colors)
        return palette, bitmap, settings

    @classmethod
    def _postprocess(cls, palette: _PaletteT, bitmap: _BitmapT, settings: dict[str, Any]) -> tuple[_PaletteT, _BitmapT]:
        if settings.get('__orig_dither', None) is not None:
            cls.__piq.set_dithering_level(settings['__orig_dither'])
        if settings.get('__orig_quality', None) is not None:
            cls.__piq.set_quality(settings['__orig_quality'])
        return palette, bitmap

    @_classproperty
    def name(cls) -> str:
        return super().name + f"({cls.__piq.lib_name})"

    @classmethod
    def is_ready(cls) -> bool:
        return False if cls.__piq is None else cls.__piq.is_ready()

    @classmethod
    def configure(cls, settings: dict[Any, Any] = {}) -> bool:
        from piliq import PILIQ
        piq = None
        try: piq = PILIQ(settings.get('qpath', None), False)
        except (FileNotFoundError, AssertionError):
            logger.debug(f"Failed to load advanced quantizer at '{settings.get('qpath', None)}'.")
            try: piq = PILIQ(None, False)
            except (FileNotFoundError, AssertionError): ...
        if piq is not None and piq.is_ready():
            if (speed := settings.get('speed', None)) is not None:
                piq.set_speed(speed)
            if (dither := settings.get('dither', None)) is not None:
                piq.set_dithering_level(dither/100.0)
            if (quality := settings.get('quality', None)) is not None:
                piq.set_quality(quality)
            cls.__piq = piq
            return True
        return False

class BuiltinQuantizer(Enum):
    LIQ = ImageQuantWrap
    HexTree = HexTreeWrap
    Qtzr = QtzrWrap
    Pillow = PillowWrap

    @classmethod
    def configure_all(cls, settings: dict[Any, Any]) -> list[bool]:
        return list(map(lambda q: q.configure(settings), cls))

    def __call__(self, image: Image.Image, colors: int, **kwargs) -> tuple[_PaletteT, _BitmapT]:
        palette, bitmap = self.value.quantize(image, colors, **kwargs)
        # Pillow can miserably fail (internal bug), so evaluate the input again with Qtzr which
        # is the only other builtin quantizer always supported (C+Python implementations)
        if self == __class__.Pillow and palette is None:
            return __class__(__class__.Qtzr)(image, colors, **kwargs)
        return palette, bitmap

    def quantize(self, image: Image.Image, colors: int, **kwargs) -> tuple[_PaletteT, _BitmapT]:
        return self(image, colors, **kwargs)

    @classmethod
    def from_name(cls, name: str) -> 'BuiltinQuantizer':
        match name.strip().lower():
            case 'hextree': return cls(cls.HexTree)
            case 'qtzr':    return cls(cls.Qtzr)
            case 'pillow' | 'pil': return cls(cls.Pillow)
            case 'pngquant' | 'libimagequant' | 'liq' | 'piliq' | 'imagequant':
                return cls(cls.LIQ)
            case _:
                return None

    @property
    def name(self) -> str:
        return self.value.name

    @classmethod
    def get_all(cls) -> list[tuple[int, 'BuiltinQuantizer', bool]]:
        # enumeration is from legacy / cli definition
        values = [3, 2, 0, 1]
        names = ['ImageQuant', 'HexTree', 'Qtzr', 'Pillow']
        enum = []
        for e, q, n in zip(values, cls, names):
            if q.value.is_ready():
                enum.append((e, n, q))
        return enum
####

class PaletteSequenceEffect:
    @staticmethod
    def solve_sequence_fast(events: list[Image.Image], colors: int, quantizer: QuantizerWrap, **kwargs):
        """
        This functions finds a solution for the provided subtitle animation.
        :param events: PIL images, stacked one after the other
        :param colors: max number of sequences usable

        :return: bitmap, sequence of palette update to obtain the said input animation.
        """

        if 1 == len(events):
            clut, img = quantizer.quantize(events[0], colors, single_bitmap=True, **kwargs)
            return img.copy(), np.expand_dims(clut, 1).copy()

        sequences = np.zeros((len(events), *events[0].size[::-1], 4), np.uint8)
        for ke, event in enumerate(events):
            clut, img = quantizer.quantize(event, colors, single_bitmap=False, **kwargs)
            sequences[ke, :, :, :] = clut[img]
        sequences = np.moveaxis(sequences, 0, 2)

        #catalog the sequences
        seq_occ: dict[int, tuple[int, np.ndarray]] = {}
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
    def solve_and_remap(cls, events: list[Image.Image], quantizer: QuantizerWrap, colors: int = 255, first_index: int = 1, **kwargs):
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

        # bitmap is (H x W), cluts is (N_c x len(events) x 4)
        bitmap, cluts = cls.solve_sequence_fast(events, colors, quantizer, **kwargs)
        transparent_id = np.nonzero(np.all(cluts[:,:,-1] == 0, axis=1))[0]

        kwargs_diff = {'matrix': kwargs.get('bt_colorspace', 'bt709')}

        #No transparency at all in this bitmap
        if 0 == len(transparent_id):
            #All colours used incl reserved transparent index. This is incorrect, requantize with colors-1
            if np.max(bitmap) + first_index == 0xFF:
                logger.ldebug("Too many colours used, lowering count.")
                bitmap, cluts = cls.solve_sequence_fast(events, colors-1, quantizer, **kwargs)
            palettes = cls.to_ycc_palettes(cluts, **kwargs_diff)
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
            palettes = cls.to_ycc_palettes(cluts, **kwargs_diff)

        for kp, pal in enumerate(palettes):
            palettes[kp] = pal.offset(first_index)
        assert len(palettes[0]) < colors
        return bitmap, palettes
    ####
    @staticmethod
    def to_ycc_palettes(cluts, /, *, matrix: str = 'bt709') -> list[Palette]:
        """
        :param cluts: RGBA Color look-up tables of the sequence, stacked one after the other.
        :param matrix: colorspace matrix name
    
        :return: N palette objects defining palette that can be converted to PDSes.
        """
        stacked_cluts = np.swapaxes(cluts, 1, 0).astype(np.int32)
        matrix = Matrix(matrix).forward()
    
        shape = stacked_cluts.shape
        stacked_cluts = np.round(np.matmul(stacked_cluts.reshape((-1, 4)), matrix.T))
        stacked_cluts += np.asarray([[16, 128, 128, 0]])
        clip_vals = (np.array([[16, 16, 16, 0]]), np.asarray([[235, 240, 240, 255]]))
        stacked_cluts = np.clip(stacked_cluts, *clip_vals).astype(np.uint8).reshape(shape)
        #YCbCrA -> YCrCbA
        stacked_cluts = stacked_cluts[:, :, [0, 2, 1, 3]]
        
        palettes = []
        for palette_array in stacked_cluts:
            new_palette = Palette()
            for ke, entry in enumerate(palette_array):
                new_palette[ke] = PaletteEntry(*entry)
            palettes.append(new_palette)
        return palettes
####

