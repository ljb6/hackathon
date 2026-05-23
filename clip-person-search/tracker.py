import cv2
import numpy as np

IOU_THRESHOLD = 0.4
MAX_AGE = 30  # sampled frames before a track expires


def sharpness(crop_bgr):
    """Laplacian variance of the crop. Higher = sharper image."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """
    Tracks persons across frames by bounding box overlap.
    Calls embed_fn only when the new crop is sharper than the cached one.
    """

    def __init__(self, iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE):
        self.tracks = []  # [{box, embedding, sharpness, last_frame}]
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, box, crop, frame_count, embed_fn):
        """
        Returns (embedding, re_embedded: bool).
        re_embedded=True  → embed_fn was called (new person or sharper crop).
        re_embedded=False → cached embedding returned; skip is_match / candidate.
        """
        self._expire(frame_count)

        for track in self.tracks:
            if _iou(box, track["box"]) >= self.iou_threshold:
                track["box"] = box
                track["last_frame"] = frame_count
                new_sharpness = sharpness(crop)
                if new_sharpness > track["sharpness"]:
                    track["embedding"] = embed_fn(crop)
                    track["sharpness"] = new_sharpness
                    return track["embedding"], True
                return track["embedding"], False

        # New person
        crop_sharpness = sharpness(crop)
        emb = embed_fn(crop)
        self.tracks.append({
            "box": box,
            "embedding": emb,
            "sharpness": crop_sharpness,
            "last_frame": frame_count,
        })
        return emb, True

    def _expire(self, current_frame):
        self.tracks = [
            t for t in self.tracks
            if current_frame - t["last_frame"] <= self.max_age
        ]
