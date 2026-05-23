import numpy as np
import pytest


# --- timestamp_from_frame ---

def test_timestamp_zero():
    from main import timestamp_from_frame
    assert timestamp_from_frame(0, 30.0) == "00:00:00"


def test_timestamp_one_minute():
    from main import timestamp_from_frame
    assert timestamp_from_frame(1800, 30.0) == "00:01:00"


def test_timestamp_one_hour():
    from main import timestamp_from_frame
    assert timestamp_from_frame(108000, 30.0) == "01:00:00"


def test_timestamp_mixed():
    from main import timestamp_from_frame
    # 25 fps × 3725 frames = 149 s = 00:02:29
    assert timestamp_from_frame(3725, 25.0) == "00:02:29"


# --- draw_result ---

def test_draw_result_adds_green_rectangle():
    from main import draw_result
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    result = draw_result(frame, (10, 10, 100, 150), 0.85, "red shirt", "00:01:23")
    assert result is not None
    assert result.shape == (200, 300, 3)
    assert result[:, :, 1].max() == 255


def test_draw_result_handles_box_at_frame_edge():
    from main import draw_result
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_result(frame, (0, 0, 100, 100), 0.5, "hat", "00:00:00")


def test_draw_result_label_clips_negative_y():
    from main import draw_result
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    draw_result(frame, (10, 0, 100, 80), 0.72, "blue jacket", "00:00:10")
