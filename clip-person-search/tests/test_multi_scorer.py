import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock


@pytest.fixture
def scorer():
    mock_model = MagicMock()
    mock_model.logit_scale = torch.tensor(0.0)
    mock_model.logit_scale.exp = MagicMock(return_value=torch.tensor(1.0))
    mock_model.encode_text = MagicMock(return_value=torch.zeros(2, 512))

    with patch("embedder.model", mock_model), \
         patch("embedder.device", "cpu"), \
         patch("embedder.embed_text", return_value=torch.zeros(1, 512)):
        import importlib
        import multi_scorer as ms
        importlib.reload(ms)
        s = ms.MultiAttributeScorer()

    s._logit_scale = torch.tensor(10.0)
    s._phrase_embs["upper"]    = F.normalize(torch.eye(10, 512), dim=-1)
    s._phrase_embs["lower"]    = F.normalize(torch.eye(6,  512), dim=-1)
    s._phrase_embs["backpack"] = F.normalize(torch.eye(2,  512), dim=-1)
    s._phrase_embs["hat"]      = F.normalize(torch.eye(2,  512), dim=-1)
    s._baseline_emb = F.normalize(torch.zeros(1, 512), dim=-1)
    return s


def test_upper_color_selected(scorer):
    """Test 1 — correct upper color selected (red = index 2)."""
    img_emb = torch.eye(10, 512)[2].unsqueeze(0)  # row 2 → "red" slot
    total, breakdown = scorer.score(img_emb, upper_color="red")
    assert breakdown["upper_color"] > 0.9


def test_only_specified_attributes_in_breakdown(scorer):
    """Test 2 — only specified attributes appear in breakdown."""
    img_emb = torch.eye(10, 512)[2].unsqueeze(0)
    total, breakdown = scorer.score(img_emb, upper_color="red")
    assert set(breakdown.keys()) == {"upper_color"}
    assert total == breakdown["upper_color"]


def test_no_attributes_returns_zero(scorer):
    """Test 3 — no attributes returns zero total and empty breakdown."""
    img_emb = torch.eye(10, 512)[0].unsqueeze(0)
    total, breakdown = scorer.score(img_emb)
    assert total == 0.0
    assert breakdown == {}


def test_backpack_true_scores_high(scorer):
    """Test 4 — has_backpack=True scores high when image emb matches backpack slot."""
    img_emb = torch.eye(2, 512)[0].unsqueeze(0)  # row 0 → "has backpack" slot
    total, breakdown = scorer.score(img_emb, has_backpack=True)
    assert breakdown["backpack"] > 0.9


def test_backpack_false_scores_high(scorer):
    """Test 5 — has_backpack=False scores high when image emb matches no-backpack slot."""
    img_emb = torch.eye(2, 512)[1].unsqueeze(0)  # row 1 → "no backpack" slot
    total, breakdown = scorer.score(img_emb, has_backpack=False)
    assert breakdown["backpack"] > 0.9


def test_multi_attribute_score_is_mean(scorer):
    """Test 6 — multi-attribute total is the mean of individual attribute scores."""
    # Override upper and lower to 2×512 identity so black = index 0 for both
    scorer._phrase_embs["upper"] = F.normalize(torch.eye(2, 512), dim=-1)
    scorer._phrase_embs["lower"] = F.normalize(torch.eye(2, 512), dim=-1)
    img_emb = torch.eye(2, 512)[0].unsqueeze(0)  # row 0 → "black" for both
    total, breakdown = scorer.score(img_emb, upper_color="black", lower_color="black")
    assert breakdown["upper_color"] > 0.9
    assert breakdown["lower_color"] > 0.9
    expected_mean = (breakdown["upper_color"] + breakdown["lower_color"]) / 2
    assert total == pytest.approx(expected_mean, abs=0.01)


def test_unknown_color_ignored(scorer):
    """Test 7 — unknown color name yields zero total and empty breakdown."""
    img_emb = torch.eye(10, 512)[0].unsqueeze(0)
    total, breakdown = scorer.score(img_emb, upper_color="pink")
    assert total == 0.0
    assert breakdown == {}
