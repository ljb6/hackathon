import torch
from search import find_best_match, is_match

THRESHOLD = 0.5


def _candidate(emb, timestamp="00:00:01", frame_label="frame"):
    return (None, (0, 0, 50, 50), frame_label, emb, timestamp)


def test_returns_best_match_above_threshold():
    emb_query = torch.tensor([[1.0, 0.0]])
    candidates = [
        _candidate(torch.tensor([[0.0, 1.0]]), "00:00:01", "bad"),
        _candidate(torch.tensor([[1.0, 0.0]]), "00:00:05", "good"),
    ]
    result = find_best_match(emb_query, candidates, threshold=THRESHOLD)
    assert result is not None
    frame, box, score, timestamp = result
    assert frame == "good"
    assert score > 0.99
    assert timestamp == "00:00:05"


def test_returns_none_when_best_score_below_threshold():
    emb_query = torch.tensor([[1.0, 0.0]])
    candidates = [_candidate(torch.tensor([[0.0, 1.0]]))]
    assert find_best_match(emb_query, candidates, threshold=THRESHOLD) is None


def test_returns_none_for_empty_candidates():
    assert find_best_match(torch.tensor([[1.0, 0.0]]), [], threshold=THRESHOLD) is None


def test_result_carries_correct_timestamp():
    emb_query = torch.tensor([[1.0, 0.0]])
    candidates = [_candidate(torch.tensor([[1.0, 0.0]]), timestamp="00:02:47")]
    _, _, _, timestamp = find_best_match(emb_query, candidates, threshold=THRESHOLD)
    assert timestamp == "00:02:47"


def test_is_match_true_above_threshold():
    score, matched = is_match(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), threshold=THRESHOLD)
    assert matched is True and score > 0.99


def test_is_match_false_below_threshold():
    score, matched = is_match(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), threshold=THRESHOLD)
    assert matched is False and score < 0.01


def test_is_match_returns_float_score():
    score, _ = is_match(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), threshold=THRESHOLD)
    assert isinstance(score, float)


# --- relative scoring (baseline_embedding provided) ---

def test_find_best_match_relative_no_match_when_delta_below_threshold():
    # When query and baseline are identical, delta = 0 → no match
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[1.0, 0.0]])  # same direction as query
    emb_img      = torch.tensor([[1.0, 0.0]])  # matches both equally

    candidates = [_candidate(emb_img)]
    result = find_best_match(emb_query, candidates, threshold=0.01,
                             baseline_embedding=emb_baseline)
    assert result is None  # delta = 1.0 - 1.0 = 0.0 < 0.01


def test_find_best_match_relative_returns_match_when_delta_positive():
    # Query matches crop; baseline is orthogonal → delta ≈ 1.0
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[0.0, 1.0]])  # orthogonal
    emb_img      = torch.tensor([[1.0, 0.0]])  # matches query perfectly

    candidates = [_candidate(emb_img)]
    result = find_best_match(emb_query, candidates, threshold=0.01,
                             baseline_embedding=emb_baseline)
    assert result is not None
    _, _, score, _ = result
    assert score > 0.9  # delta ≈ 1.0 - 0.0


def test_is_match_relative_no_match_when_equal_scores():
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[1.0, 0.0]])
    emb_img      = torch.tensor([[1.0, 0.0]])

    score, matched = is_match(emb_query, emb_img, threshold=0.01,
                              baseline_embedding=emb_baseline)
    assert matched is False
    assert abs(score) < 0.001  # delta ≈ 0


def test_is_match_relative_match_when_query_more_similar():
    emb_query    = torch.tensor([[1.0, 0.0]])
    emb_baseline = torch.tensor([[0.0, 1.0]])
    emb_img      = torch.tensor([[1.0, 0.0]])

    score, matched = is_match(emb_query, emb_img, threshold=0.01,
                              baseline_embedding=emb_baseline)
    assert matched is True
    assert score > 0.9
