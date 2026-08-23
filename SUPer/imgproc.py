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

import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from SSIM_PIL import compare_ssim
from typing import TypeAlias, Any, Self

from brule import HexTree as _HexTree, QtzrUTC as _Qtzr

from .internals import _classproperty, LogFacility
from .palette import Palette, Matrix

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
        if kwargs.get('single_bitmap', False):
            colors = len(image.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
        colors = max(16, min(colors, int(np.ceil(20 + colors*235/255))))
        return image, colors

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
        if kwargs.get('single_bitmap', False):
            nc = len(image.quantize(colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).palette.colors)
        else:
            nc = colors
            kwargs |= {'__orig_dither':  cls.__piq.get_dithering_level(),
                       '__orig_quality': cls.__piq.get_quality()}
            cls.__piq.set_dithering_level(kwargs['__orig_dither']*0.9)
            cls.__piq.set_quality(max(1, int(np.ceil(kwargs['__orig_quality']*0.975))))
        return image, max(2, nc), kwargs

    @classmethod
    def _palettize(cls, image: Image.Image, colors: int, settings: dict[str, Any]) -> tuple[_PaletteT, _BitmapT, dict[str, Any]]:
        palette, bitmap = cls.__piq.quantize(image, min(colors, int(np.ceil(20+colors*235/255))))
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

#%%
class ImageSequence:
    def __init__(self, n_images: int, quantizer: QuantizerWrap, matrix: Matrix):
        self.length = n_images
        self.quantizer = quantizer
        self.matrix = matrix
        self._sequence = None
        self._idx = 0
        self._bitmap = None
        self._cluts = None
        self._pg_cluts = None

    def add_to_stack(self, img: Image.Image, colors: int) -> bool:
        if self.length > 1:
            if self._idx == 0:
                self._sequence = np.zeros((self.length, *img.size[::-1], 4), np.uint8)
            clut, img = self.quantizer(img, colors)
            self._sequence[self._idx, :, :, :] = clut[img]
        else:
            self._sequence = self.quantizer(img, colors, single_bitmap=True)
        self._idx += 1
        return self._idx == self.length

    def flatten(self, colors: int = 255) -> Self:
        """
        This functions finds a solution for the provided subtitle animation.
        :param events: PIL images, stacked one after the other
        :param colors: max number of sequences usable

        :return: bitmap, sequence of palette update to obtain the said input animation.
        """

        assert self._sequence is not None, "No image lined up in the sequence."
        assert self._idx == self.length

        if 1 == self.length:
            clut, self._bitmap = self._sequence
            self._cluts = np.expand_dims(clut, 1).copy()
            self._bitmap = self._bitmap.copy()
            return self

        self._sequence = np.moveaxis(self._sequence, 0, 2)

        #catalog the sequences
        seq_occ: dict[int, list[int, np.ndarray[tuple[int, int], np.uint8]]] = {}
        for i in range(self._sequence.shape[0]):
            for j in range(self._sequence.shape[1]):
                seq = self._sequence[i, j, :, :]
                hsh = hash(seq.tobytes())
                try:
                    seq_occ[hsh][0] += 1
                except KeyError:
                    seq_occ[hsh] = [1, seq]

        #Sort sequences by commonness
        seq_sorted = {k: x[1] for k, x in sorted(seq_occ.items(), key=lambda item: item[1][0], reverse=True)}
        seq_ids = {k: z for z, k in enumerate(seq_sorted.keys())}

        #Fill a new array with kept sequences to perform fast norm calculations
        norm_mat = np.ndarray((colors, *self._sequence[i,j,:,:].shape[0:2]))

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

        bitmap = np.zeros(self._sequence.shape[0:2], dtype=np.uint8)
        for i in range(self._sequence.shape[0]):
            for j in range(self._sequence.shape[1]):
                seq = self._sequence[i, j, :, :]
                hsh = hash(seq.tobytes())
                if seq_ids[hsh] < colors:
                    bitmap[i, j] = seq_ids[hsh]
                else:
                    bitmap[i, j] = remap[seq_ids[hsh]]
        #save bitmap and the color sequence (copy only the N kept sequences)
        self._bitmap = bitmap
        self._cluts = np.asarray([seq for seq, _ in zip(seq_sorted.values(), range(colors))], dtype=np.uint8)
        return self

    def remap(self, first_index: int = 1) -> tuple[list[Palette], np.ndarray[tuple[int, int], np.uint8]]:
        assert self._bitmap is not None and self._cluts is not None
        if self._pg_cluts is not None:
            return self._pg_cluts, self._bitmap

        transparent_id = np.nonzero(np.all(self._cluts[:,:,-1] == 0, axis=1))[0]

        #No transparency at all in this bitmap
        if 0 == len(transparent_id):
            if np.max(self._bitmap) + first_index == 0xFF:
                #All colours used incl reserved transparent index. This is incorrect.
                # caller must be informed and shall decide what to do (try again with less colours)
                return None, None
            self._bitmap += first_index
            self._pg_cluts = self._cluts
        else:
            # Transparent ID is the last one and will be mapped to 0xFF by the first_index shift.
            if max(transparent_id) == (0xFF - first_index):
                transparent_id = 0xFF - first_index
                self._bitmap += first_index
            else:
                #Shift only IDs
                transparent_id = int(transparent_id[0])
                tsp_mask = (self._bitmap == transparent_id)
                smaller = self._bitmap < transparent_id
                larger = self._bitmap > transparent_id
                self._bitmap[smaller] += first_index
                self._bitmap[larger] += (first_index - 1)
                self._bitmap[tsp_mask] = 0xFF
            #logger.ldebug(f"Remapped fully transparent ID {transparent_id:02X} to FF.")
            self._pg_cluts = np.delete(self._cluts, [transparent_id], axis=0)
        return self._bitmap, list(map(lambda p: p.offset(first_index), Palette.from_stacked_rgba(self._pg_cluts, self.matrix)))
####

