import pytest

from SUPer.utils import Box, TC, BDVideo, MPEGTS_FREQ, get_matrix
from SUPer.render2 import PaddingEngine

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
        ge = PaddingEngine(input_box, container, 2)
        box = PaddingEngine._pad_any_box(input_box, container, mx, my)
        if input_box.dx >= mx and input_box.dy >= my:
            assert input_box == box, f"{box} {input_box}"
        else:
            assert input_box.dx >= mx or box.dx == mx, f"{box} {input_box}"
            assert input_box.dy >= my or box.dy == my, f"{box} {input_box}"

def test_pad_centered_box():
    container = Box(0, 1080, 0, 1920)
    box = Box(538, 4, 959, 3)

    ge = PaddingEngine(box, container, 1)
    nbox = PaddingEngine._pad_any_box(box, container, 8, 8)

    #We expect perfect centering on Y axis, and accept off-by-one on X axis.
    assert abs(nbox.y-536) == 0 and abs(nbox.y2-544) == 0
    assert abs(nbox.x-956) <= 1 and abs(nbox.x2-964) <= 1

####

@pytest.mark.parametrize("fps", BDVideo.FPS)
def test_tc_framegrid(fps: BDVideo.FPS):
    #Known to produce correct result, yet totally different to SUPer implementation
    def _tc2pts(tc: TC) -> float:
        secs = round(tc.float - TC(tc.fractional_fps, '00:00:00:00', force_non_drop_frame=True).float, 6)
        scale_ntsc = not float(tc.framerate).is_integer()
        return max(0, (secs - (1/3)/MPEGTS_FREQ)) * (1 if not scale_ntsc else 1.001)

    rtc = TC(fps, '00:00:00:00', force_non_drop_frame=True)
    max_frames = TC(fps, f"23:59:59:{int(np.floor(fps))}", force_non_drop_frame=True).frames

    while rtc.frames < max_frames:
        assert round(MPEGTS_FREQ*_tc2pts(rtc)) == round(MPEGTS_FREQ*rtc.to_pts())
        rtc += random.randint(1, 600)

def test_get_matrix():
    mbt709 = get_matrix('bt709', False)
    mbt601 = get_matrix('bt601', False)
    mbt2020 = get_matrix('bt2020', False)
    assert not np.array_equal(mbt709, mbt601) and not np.array_equal(mbt601, mbt2020)
    
    try:
        mbtgarb = get_matrix("12345", True)
    except NotImplementedError:
        ...

@pytest.mark.parametrize("matrix", ['bt709', 'bt601', 'bt2020'])
def test_get_matrix_inverse(matrix: str):
    btd = get_matrix(matrix, False)
    bti = get_matrix(matrix, True)
    assert not np.array_equal(btd, bti)

    assert np.all(np.abs(np.matmul(btd, bti) - np.eye(4)) < 12e-4)

def test_pcs_fps():
    vals = sorted([(fps, fps.to_pcsfps()) for fps in BDVideo.FPS], key=lambda x: x[1])

    pcs_fps = 0x10
    for k, fps in enumerate(sorted(BDVideo.FPS)):
        assert vals[k] == (fps, pcs_fps)
        pcs_fps += 0x10
        if pcs_fps == 0x50: # 30 fps does not exist
            pcs_fps += 0x10
    

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
