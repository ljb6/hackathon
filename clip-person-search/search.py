import torch.nn.functional as F

# CLIP text-image cosine similarity: ~0.10-0.20 for random pairs, 0.20-0.35 for matches.
SIMILARITY_THRESHOLD = 0.20


def find_best_match(query_embedding, candidates, threshold=SIMILARITY_THRESHOLD):
    """
    Batch mode. candidates: list of (crop, box, frame, image_embedding, timestamp).
    Returns (frame, box, score, timestamp) for the best candidate above threshold,
    or None if no candidate clears it.
    """
    if not candidates:
        return None

    best_score = -float("inf")
    best = None

    for crop, box, frame, img_emb, timestamp in candidates:
        score = F.cosine_similarity(query_embedding, img_emb).item()
        if score > best_score:
            best_score = score
            best = (frame, box, score, timestamp)

    return best if best_score >= threshold else None


def is_match(query_embedding, img_embedding, threshold=SIMILARITY_THRESHOLD):
    """Live mode. Returns (score: float, matched: bool)."""
    score = F.cosine_similarity(query_embedding, img_embedding).item()
    return score, score >= threshold
