import io
import numpy as np
from PIL import Image


def _make_png_bytes(color_rgb, size=(10, 10)):
    arr = np.full((*size, 3), color_rgb, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_returns_numpy_array():
    from image_utils import uploaded_file_to_bgr
    result = uploaded_file_to_bgr(_make_png_bytes([128, 64, 32]))
    assert isinstance(result, np.ndarray)


def test_shape_is_height_width_3():
    from image_utils import uploaded_file_to_bgr
    result = uploaded_file_to_bgr(_make_png_bytes([0, 0, 0], size=(20, 30)))
    assert result.shape == (20, 30, 3)


def test_channel_order_is_bgr():
    from image_utils import uploaded_file_to_bgr
    # Pure red RGB=[255,0,0] becomes BGR=[0,0,255]
    result = uploaded_file_to_bgr(_make_png_bytes([255, 0, 0]))
    assert result[5, 5, 0] < 10   # blue channel near zero
    assert result[5, 5, 2] > 245  # red channel near max


def test_nonempty_output():
    from image_utils import uploaded_file_to_bgr
    result = uploaded_file_to_bgr(_make_png_bytes([0, 0, 0]))
    assert result.size > 0
