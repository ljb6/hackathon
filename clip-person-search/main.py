import cv2
from datetime import datetime

from query_builder import prompt_query
from detector import extract_persons
from embedder import embed_image, embed_text
from search import find_best_match, is_match
from tracker import PersonTracker

SAMPLE_EVERY = 15  # process every N frames


def timestamp_from_frame(frame_count, fps):
    total_seconds = int(frame_count / fps)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_result(frame, box, score, query, timestamp):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"MATCH {score:.2f} | {query} | {timestamp}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def run_video(source, query_emb, query):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    candidates = []
    frame_count = 0
    tracker = PersonTracker()

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % SAMPLE_EVERY != 0:
            continue

        timestamp = timestamp_from_frame(frame_count, fps)
        for crop, box in extract_persons(frame):
            img_emb, re_embedded = tracker.update(box, crop, frame_count, embed_image)
            if re_embedded:
                candidates.append((crop, box, frame.copy(), img_emb, timestamp))

    cap.release()

    if not candidates:
        print("No persons detected.")
        return

    print(f"Searching among {len(candidates)} detections...")
    result = find_best_match(query_emb, candidates)

    if result is None:
        print("No match found above similarity threshold.")
        return

    frame, box, score, timestamp = result
    output = draw_result(frame, box, score, query, timestamp)
    cv2.imshow(f"Best match — score: {score:.2f} @ {timestamp}", output)
    cv2.imwrite("result.jpg", output)
    print(f"Match! Score: {score:.2f} at {timestamp} — saved to result.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(query_emb, query):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    tracker = PersonTracker()

    print("Live feed active. Query stored. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % SAMPLE_EVERY == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for crop, box in extract_persons(frame):
                img_emb, re_embedded = tracker.update(box, crop, frame_count, embed_image)
                if re_embedded:
                    score, matched = is_match(query_emb, img_emb)
                    if matched:
                        output = draw_result(frame.copy(), box, score, query, timestamp)
                        cv2.imshow("MATCH FOUND", output)
                        filename = f"match_{timestamp.replace(':', '')}.jpg"
                        cv2.imwrite(filename, output)
                        print(f"Match! Score: {score:.2f} at {timestamp} — saved to {filename}")

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    source = input("Video file path (or press Enter for live webcam): ").strip()
    query = prompt_query()
    query_emb = embed_text(query)

    if source:
        run_video(source, query_emb, query)
    else:
        run_live(query_emb, query)


if __name__ == "__main__":
    main()
