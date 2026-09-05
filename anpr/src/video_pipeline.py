import cv2
import os
import json

from detect_plate_yolo import load_plate_model, detect_plate_yolo
from crop_plate import crop_plate
from preprocess_plate import preprocess_plate, preprocess_plate_gentle
from run_ocr import load_ocr_reader, run_ocr_on_plate
from clean_text import clean_ocr_fragments, validate_plate_format
from build_output import build_anpr_result
from difflib import SequenceMatcher


def is_similar_plate(text1, text2, threshold=0.7):
    """Returns True if two plate strings are similar enough to be the same plate."""
    return SequenceMatcher(None, text1, text2).ratio() >= threshold


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def upscale_crop(cropped_image, min_width=300):
    """Upscale crop if too small for reliable OCR."""
    h, w = cropped_image.shape[:2]
    if w < min_width:
        scale = min_width / w
        new_w = min_width
        new_h = int(h * scale)
        return cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return cropped_image


def preprocess_for_video_ocr(cropped_image, min_width=300):
    """Video-specific preprocessing: sharpen, upscale, enhance contrast."""
    h, w = cropped_image.shape[:2]

    # Sharpen using unsharp mask
    gaussian = cv2.GaussianBlur(cropped_image, (0, 0), 3)
    sharpened = cv2.addWeighted(cropped_image, 1.5, gaussian, -0.5, 0)

    # Upscale
    upscaled = upscale_crop(sharpened, min_width=min_width)

    # CLAHE contrast enhancement
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) > 2 else upscaled
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Bilateral filter to reduce noise while keeping edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return thresh


def associate_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    """Match current detections to existing tracks by IoU."""
    matches = []
    unmatched_dets = list(range(len(detections)))

    for track_idx, track in enumerate(tracks):
        best_iou = 0
        best_det_idx = -1
        for det_idx in unmatched_dets:
            iou = compute_iou(track["bbox"], detections[det_idx]["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_det_idx = det_idx
        if best_det_idx >= 0 and best_iou >= iou_threshold:
            matches.append((track_idx, best_det_idx))
            unmatched_dets.remove(best_det_idx)

    return matches, unmatched_dets


def update_track_consensus(track, ocr_texts, preprocess_variant):
    """Accumulate OCR evidence for a track."""
    if "history" not in track:
        track["history"] = []

    for raw in ocr_texts:
        text, is_valid = clean_ocr_fragments([raw])
        track["history"].append({
            "raw_text": raw["text"],
            "cleaned_text": text,
            "valid": is_valid,
            "confidence": raw["confidence"],
            "variant": preprocess_variant
        })


def select_consensus_plate(track):
    """Select best plate from accumulated track history using multi-frame consensus."""
    history = track.get("history", [])
    if not history:
        return "UNKNOWN", False, 0.0

    from collections import defaultdict
    groups = defaultdict(list)
    for entry in history:
        text = entry["cleaned_text"]
        if text:
            groups[text].append(entry)

    if not groups:
        return "UNKNOWN", False, 0.0

    # Score each group
    best_text = "UNKNOWN"
    best_score = -1
    best_valid = False
    best_conf = 0.0

    for text, entries in groups.items():
        valid_count = sum(1 for e in entries if e["valid"])
        avg_conf = sum(e["confidence"] for e in entries) / len(entries)
        is_valid = validate_plate_format(text)

        # Score: validity is most important, then repetition, then confidence
        score = (valid_count * 1000) + (len(entries) * 100) + avg_conf

        if is_valid:
            score += 10000  # Strong bonus for valid format

        if score > best_score:
            best_score = score
            best_text = text
            best_valid = is_valid
            best_conf = avg_conf

    # If no valid result, try to build consensus from character-level agreement
    if not best_valid and len(groups) > 1:
        consensus = build_character_consensus(groups)
        if consensus:
            is_valid = validate_plate_format(consensus)
            if is_valid:
                best_text = consensus
                best_valid = True
                best_conf = best_conf  # Keep original confidence

    return best_text, best_valid, best_conf


def build_character_consensus(groups):
    """Build consensus plate by character-level voting across similar strings."""
    from collections import Counter

    # Only consider groups with similar strings (fuzzy match)
    all_texts = list(groups.keys())
    if len(all_texts) < 2:
        return None

    # Find the most common length
    length_counts = Counter(len(t) for t in all_texts)
    target_length = length_counts.most_common(1)[0][0]

    # Only consider texts close to target length
    candidates = [t for t in all_texts if abs(len(t) - target_length) <= 1]
    if len(candidates) < 2:
        return None

    # Character-level voting at each position
    consensus_chars = []
    max_len = max(len(t) for t in candidates)

    for pos in range(max_len):
        chars_at_pos = []
        for text in candidates:
            if pos < len(text):
                chars_at_pos.append(text[pos])

        if not chars_at_pos:
            break

        # Count characters
        char_counts = Counter(chars_at_pos)
        most_common_char, count = char_counts.most_common(1)[0]

        # Only include if majority agrees
        if count >= len(candidates) * 0.5:
            consensus_chars.append(most_common_char)
        else:
            break  # Stop at first disagreement

    consensus = "".join(consensus_chars)
    if len(consensus) >= 4:
        return consensus
    return None


def process_frame(frame, plate_model, ocr_reader):
    """Run the full ANPR pipeline on a single video frame (numpy array)."""
    detections = detect_plate_yolo(plate_model, frame)
    results = []

    for det in detections:
        cropped = crop_plate(frame, det["bbox"])

        variant_a = preprocess_plate(cropped)
        variant_b = preprocess_plate_gentle(cropped)

        ocr_texts_a = run_ocr_on_plate(ocr_reader, variant_a)
        ocr_texts_b = run_ocr_on_plate(ocr_reader, variant_b)

        plate_text_a, valid_a = clean_ocr_fragments(ocr_texts_a)
        plate_text_b, valid_b = clean_ocr_fragments(ocr_texts_b)

        if valid_a and not valid_b:
            plate_text, is_valid, ocr_texts = plate_text_a, valid_a, ocr_texts_a
        elif valid_b and not valid_a:
            plate_text, is_valid, ocr_texts = plate_text_b, valid_b, ocr_texts_b
        elif len(plate_text_a) >= len(plate_text_b):
            plate_text, is_valid, ocr_texts = plate_text_a, valid_a, ocr_texts_a
        else:
            plate_text, is_valid, ocr_texts = plate_text_b, valid_b, ocr_texts_b

        if not plate_text:
            continue  # nothing readable in this detection, skip it

        avg_conf = (
            sum(t["confidence"] for t in ocr_texts) / len(ocr_texts)
            if ocr_texts else det["confidence"]
        )

        result = build_anpr_result(plate_text, is_valid, avg_conf)
        results.append(result)

    return results, detections


def draw_frame_results(frame, results, detections):
    output = frame.copy()
    for det, r in zip(detections, results):
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if r["plate_valid_format"] else (0, 165, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{r['plate_number']} ({r['confidence']:.2f})"
        cv2.putText(output, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


def run_video_pipeline(video_path, plate_model, ocr_reader,
                        frame_skip=15, dedupe_window=10, save_output=True,
                        min_crop_width=300):
    """
    Reads a video file frame by frame, runs ANPR every `frame_skip` frames,
    tracks plates across frames, builds multi-frame consensus, and optionally
    saves an annotated output video + JSON results to output/.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    writer = None
    out_path = None
    if save_output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join("output", "videos", f"{base_name}_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    processed_count = 0
    tracks = []  # list of track dicts
    all_results = []
    last_detections = []
    next_track_id = 1
    recent_plates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_count += 1

        if frame_count % frame_skip == 0:
            processed_count += 1
            detections = detect_plate_yolo(plate_model, frame)

            # Associate detections to existing tracks
            matches, unmatched_det_idxs = associate_detections_to_tracks(detections, tracks)

            # Update matched tracks
            for track_idx, det_idx in matches:
                det = detections[det_idx]
                track = tracks[track_idx]
                track["bbox"] = det["bbox"]
                track["last_seen"] = processed_count
                track["missed"] = 0

                cropped = crop_plate(frame, det["bbox"])
                cropped = upscale_crop(cropped, min_width=min_crop_width)
                variant_a = preprocess_plate(cropped)
                variant_b = preprocess_plate_gentle(cropped)
                variant_c = preprocess_for_video_ocr(cropped, min_width=min_crop_width)

                ocr_texts_a = run_ocr_on_plate(ocr_reader, variant_a)
                ocr_texts_b = run_ocr_on_plate(ocr_reader, variant_b)
                ocr_texts_c = run_ocr_on_plate(ocr_reader, variant_c)

                update_track_consensus(track, ocr_texts_a, "adaptive_threshold")
                update_track_consensus(track, ocr_texts_b, "gentle_clahe")
                update_track_consensus(track, ocr_texts_c, "video_optimized")

            # Create new tracks for unmatched detections
            for det_idx in unmatched_det_idxs:
                det = detections[det_idx]
                cropped = crop_plate(frame, det["bbox"])
                cropped = upscale_crop(cropped, min_width=min_crop_width)
                variant_a = preprocess_plate(cropped)
                variant_b = preprocess_plate_gentle(cropped)
                variant_c = preprocess_for_video_ocr(cropped, min_width=min_crop_width)

                ocr_texts_a = run_ocr_on_plate(ocr_reader, variant_a)
                ocr_texts_b = run_ocr_on_plate(ocr_reader, variant_b)
                ocr_texts_c = run_ocr_on_plate(ocr_reader, variant_c)

                new_track = {
                    "track_id": next_track_id,
                    "bbox": det["bbox"],
                    "last_seen": processed_count,
                    "missed": 0,
                    "history": []
                }
                next_track_id += 1
                update_track_consensus(new_track, ocr_texts_a, "adaptive_threshold")
                update_track_consensus(new_track, ocr_texts_b, "gentle_clahe")
                update_track_consensus(new_track, ocr_texts_c, "video_optimized")
                tracks.append(new_track)

            # Remove stale tracks
            tracks = [t for t in tracks if processed_count - t["last_seen"] <= dedupe_window * 2]

            # Generate consensus results for all active tracks
            last_detections = []
            for track in tracks:
                plate_text, is_valid, avg_conf = select_consensus_plate(track)
                if plate_text == "UNKNOWN":
                    continue

                r = build_anpr_result(plate_text, is_valid, avg_conf)
                r["track_id"] = track["track_id"]
                r["bbox"] = track["bbox"]

                matched_existing = None
                for seen_plate, last_seen in recent_plates.items():
                    if is_similar_plate(plate_text, seen_plate) and processed_count - last_seen < dedupe_window:
                        matched_existing = seen_plate
                        break

                if matched_existing is None:
                    print(f"[frame {frame_count}] Track {track['track_id']}: {json.dumps(r)}")
                    recent_plates[plate_text] = processed_count
                    all_results.append(r)
                else:
                    recent_plates[matched_existing] = processed_count

                last_detections.append({
                    "bbox": track["bbox"],
                    "confidence": avg_conf,
                    "track_id": track["track_id"],
                    "result": r
                })

        if writer is not None:
            annotated = frame.copy()
            for det in last_detections:
                x1, y1, x2, y2 = det["bbox"]
                r = det["result"]
                color = (0, 255, 0) if r["plate_valid_format"] else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"T{det['track_id']} {r['plate_number']} ({r['confidence']:.2f})"
                cv2.putText(annotated, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            writer.write(annotated)

    cap.release()

    if writer is not None:
        writer.release()
        print(f"💾 Saved annotated video: {out_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    results_path = os.path.join("output", "results", f"{base_name}_video_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"💾 Saved results JSON: {results_path}")

    print(f"\n✅ Done. Processed {processed_count} of {frame_count} total frames.")
    return all_results


if __name__ == "__main__":
    print("Loading models...")
    plate_model = load_plate_model()
    ocr_reader = load_ocr_reader()
    print("✅ Models loaded\n")

    video_path = os.path.join("test_images", "test_video.mp4")
    run_video_pipeline(video_path, plate_model, ocr_reader)
