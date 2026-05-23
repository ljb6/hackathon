import numpy as np
import torch
from unittest.mock import MagicMock, patch


def _frame(h=200, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_returns_empty_list_when_no_detections():
    with patch("detector.model") as mock_model:
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]
        from detector import extract_persons
        assert extract_persons(_frame()) == []


def test_returns_crop_and_box_for_valid_detection():
    frame = _frame(200, 200)
    frame[10:100, 10:80] = [0, 0, 255]
    with patch("detector.model") as mock_model:
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 10.0, 80.0, 100.0])]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        from detector import extract_persons
        persons = extract_persons(frame)
    assert len(persons) == 1
    crop, box = persons[0]
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert box == (10, 10, 80, 100)


def test_filters_zero_area_crops():
    with patch("detector.model") as mock_model:
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 50.0, 80.0, 50.0])]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        from detector import extract_persons
        assert extract_persons(_frame()) == []
