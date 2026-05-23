from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def extract_persons(frame):
    """Returns list of (crop, box) for every person detected in frame."""
    results = model(frame, classes=[0], verbose=False)
    persons = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            persons.append((crop, (x1, y1, x2, y2)))
    return persons
