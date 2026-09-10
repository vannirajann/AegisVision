import cv2
import os

from detect_plate_yolo import load_plate_model, detect_plate_yolo
from crop_plate import crop_plate
from preprocess_plate import preprocess_plate, preprocess_plate_gentle, is_blurry
from run_ocr import load_ocr_reader, run_ocr_on_plate
from clean_text import clean_ocr_fragments, validate_plate_format
from build_output import build_anpr_result
from difflib import SequenceMatcher


def is_similar_plate(text1, text2, threshold=0.7):
    return SequenceMatcher(None, text1, text2).ratio() >= threshold


def compute_iou(box1, box2):
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
    h, w = cropped_image.shape[:2]
    if w < min_width:
        scale = min_width / w
        new_w = min_width
        new_h = int(h * scale)
        return cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return cropped_image


def preprocess_for_video_ocr(cropped_image, min_width=300):
    h, w = cropped_image.shape[:2]
    gaussian = cv2.GaussianBlur(cropped_image, (0, 0), 3)
    sharpened = cv2.addWeighted(cropped_image, 1.5, gaussian, -0.5, 0)
    upscaled = upscale_crop(sharpened, min_width=min_width)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) > 2 else upscaled
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def associate_detections_to_tracks(detections, tracks, iou_threshold=0.3):
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


def select_consensus_plate(track, min_valid_sightings=2):
    """
    STRICT MODE: requires at least `min_valid_sightings` corroborating valid
    reads before trusting a plate. No character-level guessing fallback —
    returns UNKNOWN if nothing meets the bar.
    """
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

    best_text = "UNKNOWN"
    best_score = -1
    best_valid = False
    best_conf = 0.0

    for text, entries in groups.items():
        valid_count = sum(1 for e in entries if e["valid"])
        avg_conf = sum(e["confidence"] for e in entries) / len(entries)
        is_valid = validate_plate_format(text)

        if is_valid and valid_count < min_valid_sightings:
            is_valid = False

        score = (valid_count * 1000) + (len(entries) * 100) + avg_conf
        if is_valid:
            score += 10000

        if score > best_score:
            best_score = score
            best_text = text
            best_valid = is_valid
            best_conf = avg_conf

    if not best_valid:
        return "UNKNOWN", False, best_conf

    return best_text, best_valid, best_conf


def process_frame(frame, plate_model, ocr_reader):
    detections = detect_plate_yolo(plate_model, frame)
    results = []

    for det in detections:
        cropped = crop_plate(frame, det["bbox"])
        if cropped is None:
            continue
        if is_blurry(cropped):
            continue

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
            continue

        avg_conf = (
            sum(t["confidence"] for t in ocr_texts) / len(ocr_texts)
            if ocr_texts else det["confidence"]
        )

        result = build_anpr_result(plate_text, is_valid, avg_conf)
        result["bbox"] = det["bbox"]
        results.append(result)

    return results


def draw_results(frame, results):
    output = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = (0, 255, 0) if r["plate_valid_format"] else (0, 165, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{r['plate_number']} ({r['confidence']:.2f})"
        cv2.putText(output, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


def run_webcam_pipeline(plate_model, ocr_reader, camera_index=0,
                         frame_skip=10, dedupe_window=15, min_crop_width=300,
                         save_frames=True, output_dir="output/webcam",
                         min_valid_sightings=2):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Could not open webcam (index {camera_index})")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")

    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    processed_count = 0
    tracks = []
    last_detections = []
    next_track_id = 1
    recent_plates = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            processed_count += 1
            frame_idx += 1
            detections = detect_plate_yolo(plate_model, frame)

            matches, unmatched_det_idxs = associate_detections_to_tracks(detections, tracks)

            for track_idx, det_idx in matches:
                det = detections[det_idx]
                track = tracks[track_idx]
                track["bbox"] = det["bbox"]
                track["last_seen"] = processed_count
                track["missed"] = 0

                cropped = crop_plate(frame, det["bbox"])
                if cropped is None:
                    continue
                cropped = upscale_crop(cropped, min_width=min_crop_width)
                if is_blurry(cropped):
                    continue

                variant_a = preprocess_plate(cropped)
                variant_b = preprocess_plate_gentle(cropped)
                variant_c = preprocess_for_video_ocr(cropped, min_width=min_crop_width)

                ocr_texts_a = run_ocr_on_plate(ocr_reader, variant_a)
                ocr_texts_b = run_ocr_on_plate(ocr_reader, variant_b)
                ocr_texts_c = run_ocr_on_plate(ocr_reader, variant_c)

                update_track_consensus(track, ocr_texts_a, "adaptive_threshold")
                update_track_consensus(track, ocr_texts_b, "gentle_clahe")
                update_track_consensus(track, ocr_texts_c, "video_optimized")

            for det_idx in unmatched_det_idxs:
                det = detections[det_idx]
                cropped = crop_plate(frame, det["bbox"])
                if cropped is None:
                    continue
                cropped = upscale_crop(cropped, min_width=min_crop_width)
                if is_blurry(cropped):
                    continue

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

            tracks = [t for t in tracks if processed_count - t["last_seen"] <= dedupe_window * 2]

            last_detections = []
            for track in tracks:
                plate_text, is_valid, avg_conf = select_consensus_plate(
                    track, min_valid_sightings=min_valid_sightings
                )
                status = "VALID" if is_valid else "UNKNOWN"
                display_text = plate_text if is_valid else ("READING..." if len(track.get("history", [])) > 0 else "UNKNOWN")

                if plate_text != "UNKNOWN":
                    matched_existing = None
                    for seen_plate, last_seen in recent_plates.items():
                        if is_similar_plate(plate_text, seen_plate) and processed_count - last_seen < dedupe_window:
                            matched_existing = seen_plate
                            break
                    if matched_existing is None:
                        print(f"[frame {frame_count}] Track {track['track_id']}: {plate_text} ({status}, conf={avg_conf:.2f})")
                        recent_plates[plate_text] = processed_count
                    else:
                        recent_plates[matched_existing] = processed_count

                last_detections.append({
                    "bbox": track["bbox"],
                    "confidence": avg_conf,
                    "track_id": track["track_id"],
                    "result": {
                        "plate_number": display_text,
                        "plate_valid_format": is_valid,
                        "confidence": avg_conf
                    }
                })

        annotated = frame.copy()
        for det in last_detections:
            x1, y1, x2, y2 = det["bbox"]
            r = det["result"]
            color = (0, 255, 0) if r["plate_valid_format"] else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"T{det['track_id']} {r['plate_number']} ({r['confidence']:.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if save_frames:
            try:
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, annotated)
            except Exception as e:
                print(f"⚠️  Failed to save frame: {e}")

        try:
            cv2.imshow("AegisVision - ANPR (press 'q' to quit)", annotated)
        except cv2.error:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    print(f"\n✅ Stopped. Processed {processed_count} of {frame_count} total frames.")
    if save_frames:
        print(f"💾 Saved frames to {output_dir}")
    return tracks, last_detections


if __name__ == "__main__":
    print("Loading models...")
    plate_model = load_plate_model()
    ocr_reader = load_ocr_reader()
    print("✅ Models loaded\n")

    run_webcam_pipeline(plate_model, ocr_reader)