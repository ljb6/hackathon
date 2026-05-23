import numpy as np
import torch
from unittest.mock import patch


def _bgr_crop(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_embed_image_returns_tensor():
    with patch("embedder.model") as mock_model, \
         patch("embedder.preprocess") as mock_preprocess:
        mock_preprocess.return_value = torch.zeros(3, 224, 224)
        mock_model.encode_image.return_value = torch.zeros(1, 512)
        from embedder import embed_image
        result = embed_image(_bgr_crop())
    assert isinstance(result, torch.Tensor)
    mock_model.encode_image.assert_called_once()


def test_embed_image_accepts_bgr_without_error():
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:, :, 2] = 255
    with patch("embedder.model") as mock_model, \
         patch("embedder.preprocess") as mock_preprocess:
        mock_preprocess.return_value = torch.zeros(3, 224, 224)
        mock_model.encode_image.return_value = torch.zeros(1, 512)
        from embedder import embed_image
        embed_image(crop)  # must not raise


def test_embed_text_returns_tensor():
    with patch("embedder.model") as mock_model, \
         patch("embedder.clip") as mock_clip:
        mock_clip.tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)
        mock_model.encode_text.return_value = torch.zeros(1, 512)
        from embedder import embed_text
        result = embed_text("red shirt man")
    assert isinstance(result, torch.Tensor)
    mock_clip.tokenize.assert_called_once_with(["red shirt man"])
