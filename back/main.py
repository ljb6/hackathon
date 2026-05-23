import json
from pathlib import Path
from core.clip import ClipClassifier

data_dir = Path(__file__).parent / "data"

crops = [
    {
        "id": "example_0000_0",
        "camera_id": "example",
        "camera": "Câmera Exemplo",
        "timestamp": "00:00:00",
        "timestamp_segundos": 0,
        "crop": "crops/example.png",
    },
]

classifier = ClipClassifier()

results = [classifier.process(str(data_dir / entry["crop"]), entry) for entry in crops]

output_path = data_dir / "example.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(json.dumps(results, ensure_ascii=False, indent=2))
