import torch.nn.functional as F

SIMILARITY_THRESHOLD = 0.20   # legacy absolute mode
RELATIVE_THRESHOLD   = 0.02   # delta mode: cosine_sim(crop, query) - cosine_sim(crop, baseline)


def find_best_match(query_embedding, candidates, threshold=RELATIVE_THRESHOLD,
                    baseline_embedding=None):
    """
    Batch mode. candidates: list of (crop, box, frame, image_embedding, timestamp).
    Returns (frame, box, score, timestamp) for the best candidate above threshold,
    or None if no candidate clears it.

    When baseline_embedding is provided, score = cosine_sim(img, query) - cosine_sim(img, baseline).
    This relative delta is near zero when CLIP cannot distinguish the query from a generic
    person, and positive when the crop genuinely matches the description.
    """
    if not candidates:
        return None

    best_score = -float("inf")
    best = None

    for crop, box, frame, img_emb, timestamp in candidates:
        specific = F.cosine_similarity(query_embedding, img_emb).item()
        if baseline_embedding is not None:
            generic = F.cosine_similarity(baseline_embedding, img_emb).item()
            score = specific - generic
        else:
            score = specific

        if score > best_score:
            best_score = score
            best = (frame, box, score, timestamp)

    return best if best_score >= threshold else None


def is_match(query_embedding, img_embedding, threshold=RELATIVE_THRESHOLD,
             baseline_embedding=None):
    """
    Live mode. Returns (score: float, matched: bool).
    When baseline_embedding is provided, score is the relative delta.
    """
    specific = F.cosine_similarity(query_embedding, img_embedding).item()
    if baseline_embedding is not None:
        generic = F.cosine_similarity(baseline_embedding, img_embedding).item()
        score = specific - generic
    else:
        score = specific
    return score, score >= threshold
