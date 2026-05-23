import os
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
    label = f"MATCH {score:.3f} | {query} | {timestamp}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def _collect_candidates(source, query_emb, baseline_emb):
    """
    Process one video source. Returns (camera_label, result) where result is
    (frame, box, score, timestamp) or None if no match above threshold.
    camera_label is the filename stem (e.g. "passageway1-c1").
    """
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    camera_label = os.path.splitext(os.path.basename(source))[0]
    candidates = []
    frame_count = 0
    tracker = PersonTracker()

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
        return camera_label, None

    result = find_best_match(query_emb, candidates, baseline_embedding=baseline_emb)
    return camera_label, result


def run_video(source, query_emb, query, baseline_emb):
    """Single camera batch mode."""
    print(f"Processing {source}...")
    camera_label, result = _collect_candidates(source, query_emb, baseline_emb)

    if result is None:
        print(f"[{camera_label}] No match found above threshold.")
        return

    frame, box, score, timestamp = result
    output = draw_result(frame, box, score, query, timestamp)
    filename = f"result_{camera_label}.jpg"
    cv2.imwrite(filename, output)
    print(f"[{camera_label}] Match! Score: {score:.3f} at {timestamp} — saved to {filename}")
    cv2.imshow(f"{camera_label} — {score:.3f} @ {timestamp}", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_multi_video(sources, query_emb, query, baseline_emb):
    """Process multiple cameras, display all per-camera best matches at the end."""
    per_camera = []
    for source in sources:
        print(f"Processing {source}...")
        camera_label, result = _collect_candidates(source, query_emb, baseline_emb)
        per_camera.append((camera_label, result))
        status = f"Score: {result[2]:.3f} @ {result[3]}" if result else "no match"
        print(f"  [{camera_label}] {status}")

    matched = [(label, r) for label, r in per_camera if r is not None]

    if not matched:
        print("\nNo matches found in any camera above similarity threshold.")
        return

    print(f"\n{len(matched)}/{len(sources)} cameras matched:")
    for camera_label, (frame, box, score, timestamp) in matched:
        output = draw_result(frame, box, score, query, timestamp)
        filename = f"result_{camera_label}.jpg"
        cv2.imwrite(filename, output)
        print(f"  [{camera_label}] Score: {score:.3f} at {timestamp} → {filename}")
        cv2.imshow(f"{camera_label} — {score:.3f} @ {timestamp}", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(query_emb, query, baseline_emb):
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
                    score, matched = is_match(query_emb, img_emb,
                                              baseline_embedding=baseline_emb)
                    if matched:
                        output = draw_result(frame.copy(), box, score, query, timestamp)
                        cv2.imshow("MATCH FOUND", output)
                        filename = f"match_{timestamp.replace(':', '')}.jpg"
                        cv2.imwrite(filename, output)
                        print(f"Match! Score: {score:.3f} at {timestamp} — saved to {filename}")

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    sources_input = input(
        "Video file paths (comma-separated, or Enter for live webcam): "
    ).strip()
    query = prompt_query()

    print("Loading query embeddings...")
    query_emb    = embed_text(query)
    baseline_emb = embed_text("a person")

    if sources_input:
        sources = [s.strip() for s in sources_input.split(",") if s.strip()]
        if len(sources) == 1:
            run_video(sources[0], query_emb, query, baseline_emb)
        else:
            run_multi_video(sources, query_emb, query, baseline_emb)
    else:
        run_live(query_emb, query, baseline_emb)


if __name__ == "__main__":
    main()
