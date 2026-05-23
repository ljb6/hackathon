import cv2
import numpy as np
from PIL import Image


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    pil = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
