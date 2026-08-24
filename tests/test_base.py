import pytest

from SUPer.geometry import Box
from SUPer.internals import TC, GraphicsDecoder
from SUPer.display.bdvideo import Framerate, Format
from SUPer.display.palette import Matrix
from SUPer.encoder.epochctx import PaddingEngine

import warnings
import numpy as np
import random

@pytest.mark.parametrize("container", [Box(0, 1080, 0, 1920), Box(0, 480, 0, 720), Box(0, 576, 0, 720), Box(0, 720, 0, 1280)])
def test_pad_box(container: Box):
    """
    Test that, whatever the margins or the input box are, the box is either
    untouched or padded to the exactly set margin.
    """
    for k in range(10000):
        mx, my = random.randint(8, 64), random.randint(8, 64)
        py, px = random.randrange(0, container.dy, 1), random.randrange(0, container.dx, 1)
        input_box = Box(py, random.randint(1, container.dy-py), px, random.randint(1, container.dx-px))
        box = PaddingEngine._pad_any_box(input_box, container, mx, my)
        if input_box.dx >= mx and input_box.dy >= my:
            assert input_box == box, f"{box} {input_box}"
        else:
            assert input_box.dx >= mx or box.dx == mx, f"{box} {input_box}"
            assert input_box.dy >= my or box.dy == my, f"{box} {input_box}"

def test_pad_centered_box():
    container = Box(0, 1080, 0, 1920)
    box = Box(538, 4, 959, 3)

    nbox = PaddingEngine._pad_any_box(box, container, 8, 8)

    #We expect perfect centering on Y axis, and accept off-by-one on X axis.
    assert abs(nbox.y-536) == 0 and abs(nbox.y2-544) == 0
    assert abs(nbox.x-956) <= 1 and abs(nbox.x2-964) <= 1

####

########################
############## BDVideo and utils
@pytest.mark.parametrize("fps", Framerate)
def test_tc_framegrid(fps: Framerate):
    #Known to produce correct result, yet totally different to SUPer implementation
    def _tc2pts(tc: TC) -> float:
        secs = round(tc.float - TC(tc.fractional_fps, '00:00:00:00', force_non_drop_frame=True).float, 6)
        scale_ntsc = not float(tc.framerate).is_integer()
        return max(0, (secs - (1/3)/GraphicsDecoder.FREQ)) * (1 if not scale_ntsc else 1.001)

    rtc = TC(fps, '00:00:00:00', force_non_drop_frame=True)
    max_frames = TC(fps, f"23:59:59:{int(np.floor(fps))}", force_non_drop_frame=True).frames

    while rtc.frames < max_frames:
        assert round(GraphicsDecoder.FREQ*_tc2pts(rtc)) == rtc.to_pts()
        rtc += random.randint(1, 600)

def test_get_matrix():
    mbt709 = Matrix('bt709').forward()
    mbt601 = Matrix('601').forward()
    mbt2020 = Matrix('bt.2020').forward()
    assert not np.array_equal(mbt709, mbt601) and not np.array_equal(mbt601, mbt2020)


@pytest.mark.parametrize("matrix", ['bt709', 'bt601', 'bt2020'])
def test_get_matrix_inverse(matrix: str):
    btd = Matrix(matrix).forward()
    bti = Matrix(matrix).inverse()
    assert bti is not None and not np.array_equal(btd, bti)

    assert np.all(np.abs(np.matmul(btd, bti) - np.eye(4)) < 12e-4)    

@pytest.mark.parametrize("plane_size_test",
                         [((1920, 1080), True),
                          (1080, True),
                          ("1080p", True),
                          ("576i", True),
                          ((720, 480), True),
                          (Format((1280, 720)), True),
                          ((1920, 1088), False),
                          ((640, 480), False),
                          (2160, False),])
def test_format(plane_size_test):
    plane_size, accepted = plane_size_test
    if accepted:
        fmt = Format(plane_size)
        if isinstance(plane_size, str):
            assert int(plane_size[:-1]) == fmt.height
        elif isinstance(plane_size, int):
            assert plane_size == fmt.height
        else:
            assert plane_size == fmt
    else:
        try:
            fmt = Format(plane_size)
        except ValueError:
            ...
        else:
            raise AssertionError(f"Video format {plane_size} should have been rejected.")

def test_brule_capabilities():
    from brule import Brule, LayoutEngine, HexTree

    cap = Brule.get_capabilities()
    if 'C' not in cap:
        warnings.warn(f"RLE codec is unoptimized on your machine: {cap}.")

    cap = LayoutEngine.get_capabilities()
    if 'C' not in cap:
        warnings.warn(f"The layout engine executes an unoptimized version on your machine: {cap}.")

    cap = HexTree.get_capabilities()
    if 'C' not in cap:
        warnings.warn(f"The HexTree quantizer executes an unoptimized version on your machine: {cap}.")
