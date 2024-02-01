import pytest

from SUPer.utils import Box
from SUPer.render2 import GroupingEngine
from SUPer.filestreams import BDNXMLEvent

import numpy as np
import random

@pytest.mark.parametrize("container", [Box(0, 1080, 0, 1920), Box(0, 480, 0, 720), Box(0, 576, 0, 720), Box(0, 720, 0, 1280)])
def test_pad_box(container: Box):
    for k in range(10000):
        mx, my = random.randint(8, 64), random.randint(8, 64)
        py, px = random.randrange(0, container.dy, 1), random.randrange(0, container.dx, 1)
        input_box = Box(py, random.randint(1, container.dy-py), px, random.randint(1, container.dx-px))
        ge = GroupingEngine(input_box, container, 2)
        box = ge.pad_box(mx, my)
        if input_box.dx >= mx and input_box.dy >= my:
            assert input_box == box, f"{box} {input_box}"
        else:
            assert input_box.dx >= mx or box.dx == mx, f"{box} {input_box}"
            assert input_box.dy >= my or box.dy == my, f"{box} {input_box}"

def test_pad_wds():
    container = Box(0, 1080, 0, 1920)
    input_box = Box(10, 100, 10, 100)

    ge = GroupingEngine(input_box, container, 2)
    box = ge.pad_box()
    assert box == input_box

    gs_orig = np.zeros((1, box.dy, box.dx), np.uint8)
    gs_orig[0, 0, 0] = 1
    gs_orig[0,-1,-1] = 1
    windows = ge.find_layout(gs_orig)
    assert len(windows) == 2
    np_mask = np.ones((box.dy, box.dx), np.bool_)
    for wd in windows:
        np_mask[wd.y:wd.y2, wd.x:wd.x2] = False
        assert Box.union(wd, box).overlap_with(container) == 1.0
        assert wd.dx == 8 and wd.dy == 8
    assert not np.any(gs_orig[0, np_mask]), windows

def test_merge_wds():
    container = Box(0, 480, 0, 720)
    box = Box(470, 10, 710, 10)

    ge = GroupingEngine(box, container, 2)
    assert ge.pad_box() == box

    gs_orig = np.zeros((1, box.dy, box.dx), np.uint8)
    gs_orig[0, 0, 0] = 1
    gs_orig[0, 9, 9] = 1
    windows = ge.find_layout(gs_orig)
    assert len(windows) == 1, windows
    np_mask = np.ones((box.dy, box.dx), np.bool_)

    for wd in windows:
        np_mask[wd.y:wd.y2, wd.x:wd.x2] = False
        assert Box.union(wd, box).overlap_with(box) == 1.0
        assert wd.dx >= 8 and wd.dy >= 8
    assert not np.any(gs_orig[0, np_mask]), windows

def test_merge_tick_overhead():
    container = Box(0, 480, 0, 720)
    box = Box(464, 16, 710, 10)

    ge = GroupingEngine(box, container, 2)
    assert ge.pad_box() == box

    gs_orig = np.zeros((1, box.dy, box.dx), np.uint8)
    gs_orig[0, 0, 0] = 1
    gs_orig[0, -1, 0] = 1
    windows = ge.find_layout(gs_orig)
    assert len(windows) == 1, windows
    np_mask = np.ones((box.dy, box.dx), np.bool_)
    for wd in windows:
        assert Box.union(wd, box).overlap_with(box) == 1.0
        np_mask[wd.y:wd.y2, wd.x:wd.x2] = False
    assert not np.any(gs_orig[0, np_mask]), (windows, np.argwhere(gs_orig[0, :, :] == 1))

def test_split_tick_overhead():
    container = Box(0, 480, 0, 720)
    box = Box(464, 16, 0, 720)

    ge = GroupingEngine(box, container, 2)
    assert ge.pad_box() == box

    gs_orig = np.zeros((1, box.dy, box.dx), np.uint8)
    gs_orig[0, 0, :] = 1
    gs_orig[0, 8, 0] = 1
    windows = ge.find_layout(gs_orig)
    assert len(windows) == 2, windows
    np_mask = np.ones((box.dy, box.dx), np.bool_)
    for wd in windows:
        np_mask[wd.y:wd.y2, wd.x:wd.x2] = False
        assert Box.union(wd, box).overlap_with(box) == 1.0
        assert Box(wd.y+box.y, wd.dy, wd.x+box.x, wd.dx).overlap_with(container) == 1.0
    assert not np.any(gs_orig[0, np_mask])

def test_pad_marging_box():
    container = Box(0, 480, 0, 720)
    box = Box(464, 16, 0, 100)

    ge = GroupingEngine(box, container, 2)
    assert ge.pad_box() == box

    gs_orig = np.zeros((1, box.dy, box.dx), np.uint8)
    gs_orig[0, 8, 0] = 1
    gs_orig[0, 10:13, :] = 1
    windows = ge.find_layout(gs_orig)

    np_mask = np.ones((box.dy, box.dx), np.bool_)
    for wd in windows:
        np_mask[wd.y:wd.y2, wd.x:wd.x2] = False
        assert Box(wd.y+box.y, wd.dy, wd.x+box.x, wd.dx).overlap_with(container) == 1.0, windows
    assert not np.any(gs_orig[0, np_mask])
