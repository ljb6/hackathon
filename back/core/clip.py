import json
import torch
import clip
from pathlib import Path
from PIL import Image

_PHRASES = {
    "tem_chapeu": [
        "a person wearing a cap or hat",
        "a person with no hat or cap",
    ],
    "tem_mochila": [
        "a person carrying a backpack",
        "a person with no backpack",
    ],
    "cor_superior": [
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
    ],
    "cor_inferior": [
        "person with black pants or shorts",
        "person with white or light pants",
        "person with blue jeans",
        "person with gray pants or shorts",
        "person with brown pants",
        "person with green pants or shorts",
    ],
}

_COLORS_SUPERIOR = ["preto", "branco", "vermelho", "azul", "cinza", "verde", "amarelo", "laranja", "roxo", "marrom"]
_COLORS_INFERIOR = ["preto", "branco", "azul", "cinza", "marrom", "verde"]


class ClipClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _classify(self, image_tensor: torch.Tensor, phrases: list[str]) -> int:
        with torch.no_grad():
            text_tokens = clip.tokenize(phrases).to(self.device)
            logits, _ = self.model(image_tensor, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
        return int(probs.argmax())

    def process(self, image_path: str, metadata: dict) -> dict:
        """
        metadata must contain: id, camera_id, camera, timestamp, timestamp_segundos, crop
        Returns a detection dict matching the deteccoes.json schema.
        """
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)

        return {
            "id": metadata["id"],
            "camera_id": metadata["camera_id"],
            "camera": metadata["camera"],
            "timestamp": metadata["timestamp"],
            "timestamp_segundos": metadata["timestamp_segundos"],
            "cor_superior": _COLORS_SUPERIOR[self._classify(image, _PHRASES["cor_superior"])],
            "cor_inferior": _COLORS_INFERIOR[self._classify(image, _PHRASES["cor_inferior"])],
            "tem_mochila": self._classify(image, _PHRASES["tem_mochila"]) == 0,
            "tem_chapeu": self._classify(image, _PHRASES["tem_chapeu"]) == 0,
            "crop": metadata["crop"],
        }


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"

    classifier = ClipClassifier()

    result = classifier.process(
        str(data_dir / "crops" / "example.png"),
        {
            "id": "example_0000_0",
            "camera_id": "example",
            "camera": "Câmera Exemplo",
            "timestamp": "00:00:00",
            "timestamp_segundos": 0,
            "crop": "crops/example.png",
        },
    )

    output_path = data_dir / "example.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
