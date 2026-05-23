import torch
import torch.nn.functional as F
from embedder import model, device, embed_text
import clip

UPPER_PHRASES = [
    "person with black upper clothing",
    "person with white upper clothing",
    "person with red upper clothing",
    "person with blue upper clothing",
    "person with gray upper clothing",
    "person with green upper clothing",
    "person with yellow upper clothing",
    "person with orange upper clothing",
    "person with purple upper clothing",
    "person with brown upper clothing",
]
UPPER_COLOR_INDEX = {
    "black": 0, "white": 1, "red": 2, "blue": 3, "gray": 4,
    "green": 5, "yellow": 6, "orange": 7, "purple": 8, "brown": 9,
}

LOWER_PHRASES = [
    "person with black pants or shorts",
    "person with white or light pants",
    "person with blue jeans",
    "person with gray pants or shorts",
    "person with brown pants",
    "person with green pants or shorts",
]
LOWER_COLOR_INDEX = {
    "black": 0, "white": 1, "blue": 2, "gray": 3, "brown": 4, "green": 5,
}

BACKPACK_PHRASES = [
    "a person carrying a backpack",
    "a person with no backpack",
]
HAT_PHRASES = [
    "a person wearing a cap or hat",
    "a person with no hat or cap",
]

ATTRIBUTE_THRESHOLD = 0.55


class MultiAttributeScorer:
    def __init__(self):
        self._logit_scale = model.logit_scale.exp().detach()
        self._phrase_embs = {}
        for key, phrases in [
            ("upper", UPPER_PHRASES),
            ("lower", LOWER_PHRASES),
            ("backpack", BACKPACK_PHRASES),
            ("hat", HAT_PHRASES),
        ]:
            tokens = clip.tokenize(phrases).to(device)
            with torch.no_grad():
                raw = model.encode_text(tokens)
            self._phrase_embs[key] = F.normalize(raw, dim=-1)
        self._baseline_emb = F.normalize(embed_text("a person"), dim=-1)

    def _softmax_prob(self, img_norm, phrase_embs_norm, target_idx):
        logits = self._logit_scale * (img_norm @ phrase_embs_norm.T)
        probs = logits.softmax(dim=-1)
        return probs[0, target_idx].item()

    def score(self, img_emb, upper_color="", lower_color="",
              has_backpack=None, has_hat=None, extra=""):
        img_norm = F.normalize(img_emb.float(), dim=-1)
        scores = {}

        if upper_color and upper_color in UPPER_COLOR_INDEX:
            scores["upper_color"] = self._softmax_prob(
                img_norm, self._phrase_embs["upper"], UPPER_COLOR_INDEX[upper_color])

        if lower_color and lower_color in LOWER_COLOR_INDEX:
            scores["lower_color"] = self._softmax_prob(
                img_norm, self._phrase_embs["lower"], LOWER_COLOR_INDEX[lower_color])

        if has_backpack is not None:
            scores["backpack"] = self._softmax_prob(
                img_norm, self._phrase_embs["backpack"], 0 if has_backpack else 1)

        if has_hat is not None:
            scores["hat"] = self._softmax_prob(
                img_norm, self._phrase_embs["hat"], 0 if has_hat else 1)

        if extra:
            query_norm = F.normalize(embed_text(extra).float(), dim=-1)
            delta = (img_norm @ query_norm.T).item() - (img_norm @ self._baseline_emb.T).item()
            scores["extra"] = max(0.0, min(1.0, delta * 3.33 + 0.5))

        if not scores:
            return 0.0, {}
        return sum(scores.values()) / len(scores), scores
