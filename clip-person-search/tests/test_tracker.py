import numpy as np
import torch
from tracker import sharpness, PersonTracker


def _uniform(value=128, h=64, w=64):
    """Uniform BGR crop — near-zero Laplacian variance."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _sharp():
    """Crop with a hard horizontal edge — high Laplacian variance."""
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:32, :] = 255
    return crop


def _embed_fn(crop):
    return torch.ones(1, 512)


# --- sharpness ---

def test_uniform_crop_has_low_sharpness():
    assert sharpness(_uniform()) < 1.0


def test_edge_crop_has_higher_sharpness_than_uniform():
    assert sharpness(_sharp()) > sharpness(_uniform())


# --- PersonTracker ---

def test_new_person_always_embeds():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    emb, re_embedded = tracker.update((10, 10, 50, 80), _sharp(), frame_count=0, embed_fn=embed_fn)
    assert re_embedded is True
    assert len(calls) == 1
    assert isinstance(emb, torch.Tensor)


def test_same_person_blurrier_crop_reuses_cached_embedding():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    box = (10, 10, 50, 80)
    tracker.update(box, _sharp(),   frame_count=0, embed_fn=embed_fn)  # sharp first
    _, re_embedded = tracker.update(box, _uniform(), frame_count=1, embed_fn=embed_fn)

    assert re_embedded is False
    assert len(calls) == 1  # embed_fn called only once


def test_same_person_sharper_crop_reembeds():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    box = (10, 10, 50, 80)
    tracker.update(box, _uniform(), frame_count=0, embed_fn=embed_fn)  # blurry first
    _, re_embedded = tracker.update(box, _sharp(),   frame_count=1, embed_fn=embed_fn)

    assert re_embedded is True
    assert len(calls) == 2


def test_expired_track_treated_as_new_person():
    tracker = PersonTracker(max_age=5)
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    box = (10, 10, 50, 80)
    tracker.update(box, _uniform(), frame_count=0,   embed_fn=embed_fn)
    _, re_embedded = tracker.update(box, _uniform(), frame_count=100, embed_fn=embed_fn)

    assert re_embedded is True
    assert len(calls) == 2  # treated as new person after expiry


def test_non_overlapping_boxes_are_independent_persons():
    tracker = PersonTracker()
    calls = []
    def embed_fn(crop):
        calls.append(1)
        return torch.ones(1, 512)

    tracker.update((0,   0,  50, 80), _uniform(), frame_count=0, embed_fn=embed_fn)
    tracker.update((200, 0, 250, 80), _uniform(), frame_count=0, embed_fn=embed_fn)

    assert len(calls) == 2
