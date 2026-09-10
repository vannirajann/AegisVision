import cv2
import os
import json

from detect_plate_yolo import load_plate_model, detect_plate_yolo
from crop_plate import crop_plate, save_crop, refine_plate_crop
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


def update_track_consensus(track, ocr_texts, preprocess_variant, ocr_log,
                            frame_count, track_id):
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
        ocr_log.append({
            "frame": frame_count,
            "track_id": track_id,
            "variant": preprocess_variant,
            "raw_text": raw["text"],
            "cleaned_text": text,
            "valid_format": is_valid,
            "confidence": round(raw["confidence"], 3)
        })


def select_consensus_plate(track, min_valid_sightings=2):
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


def save_ocr_debug_image(cropped_image, label_text, out_path):
    """Saves the crop with the OCR prediction burned onto it, for visual audit."""
    debug_img = cropped_image.copy()
    if len(debug_img.shape) == 2:  # grayscale -> convert so text is visible
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(debug_img, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(canvas, label_text[:40], (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, canvas)


def process_and_log_track_crop(frame, det_bbox, ocr_reader, min_crop_width,
                                track, ocr_log, frame_count, track_id,
                                crops_dir, debug_dir, skip_blur_check=True,
                                blur_threshold=60.0):
    """
    Crops the plate, tightens it with refine_plate_crop, and ALWAYS attempts
    OCR (blur no longer silently drops the crop by default). Every crop is
    saved with a status suffix so you can visually audit what happened.
    """
    cropped = crop_plate(frame, det_bbox)
    if cropped is None:
        return  # bbox itself was degenerate — nothing to save

    cropped = upscale_crop(cropped, min_width=min_crop_width)
    cropped = refine_plate_crop(cropped)  # tighten box around the actual plate rectangle

    blurry = is_blurry(cropped, threshold=blur_threshold)
    status = "blurry" if blurry else "ok"

    # Always save the raw crop, tagged with status, regardless of blur
    crop_filename = f"track{track_id}_frame{frame_count}_{status}.jpg"
    save_crop(cropped, crop_filename, output_dir=crops_dir)

    # Only SKIP OCR on blur if explicitly told to; default is to still try —
    # a flagged-blurry crop can still sometimes OCR correctly.
    if blurry and not skip_blur_check:
        return

    variant_a = preprocess_plate(cropped)
    variant_b = preprocess_plate_gentle(cropped)
    variant_c = preprocess_for_video_ocr(cropped, min_width=min_crop_width)

    ocr_texts_a = run_ocr_on_plate(ocr_reader, variant_a)
    ocr_texts_b = run_ocr_on_plate(ocr_reader, variant_b)
    ocr_texts_c = run_ocr_on_plate(ocr_reader, variant_c)

    update_track_consensus(track, ocr_texts_a, "adaptive_threshold", ocr_log, frame_count, track_id)
    update_track_consensus(track, ocr_texts_b, "gentle_clahe", ocr_log, frame_count, track_id)
    update_track_consensus(track, ocr_texts_c, "video_optimized", ocr_log, frame_count, track_id)

    # Save a debug image with the best raw OCR text burned onto it
    all_texts = ocr_texts_a + ocr_texts_b + ocr_texts_c
    best_text = max(all_texts, key=lambda t: t["confidence"])["text"] if all_texts else "(no text read)"
    debug_filename = f"track{track_id}_frame{frame_count}.jpg"
    save_ocr_debug_image(cropped, best_text, os.path.join(debug_dir, debug_filename))


def run_video_pipeline(video_path, plate_model, ocr_reader,
                        frame_skip=15, dedupe_window=10, save_output=True,
                        min_crop_width=300, min_valid_sightings=2,
                        output_base_dir=None,
                        conf_threshold=0.3, iou_threshold=0.4,
                        min_box_area=150, min_aspect_ratio=1.2, max_aspect_ratio=6.5,
                        skip_blur_check=True, blur_threshold=60.0):
    if output_base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(base_dir, "..", "output")

    videos_dir = os.path.join(output_base_dir, "videos")
    results_dir = os.path.join(output_base_dir, "results")
    crops_dir = os.path.join(output_base_dir, "crops")
    debug_dir = os.path.join(output_base_dir, "ocr_debug")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    writer = None
    out_path = None
    if save_output:
        os.makedirs(videos_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(videos_dir, f"{base_name}_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    processed_count = 0
    tracks = []
    all_results = []
    ocr_log = []
    last_detections = []
    next_track_id = 1
    recent_plates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            processed_count += 1
            detections = detect_plate_yolo(
                plate_model, frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                min_box_area=min_box_area,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio
            )

            matches, unmatched_det_idxs = associate_detections_to_tracks(detections, tracks)

            for track_idx, det_idx in matches:
                det = detections[det_idx]
                track = tracks[track_idx]
                track["bbox"] = det["bbox"]
                track["last_seen"] = processed_count
                track["missed"] = 0
                process_and_log_track_crop(
                    frame, det["bbox"], ocr_reader, min_crop_width,
                    track, ocr_log, frame_count, track["track_id"],
                    crops_dir, debug_dir, skip_blur_check, blur_threshold
                )

            for det_idx in unmatched_det_idxs:
                det = detections[det_idx]
                new_track = {
                    "track_id": next_track_id,
                    "bbox": det["bbox"],
                    "last_seen": processed_count,
                    "missed": 0,
                    "history": []
                }
                process_and_log_track_crop(
                    frame, det["bbox"], ocr_reader, min_crop_width,
                    new_track, ocr_log, frame_count, next_track_id,
                    crops_dir, debug_dir, skip_blur_check, blur_threshold
                )
                next_track_id += 1
                tracks.append(new_track)

            tracks = [t for t in tracks if processed_count - t["last_seen"] <= dedupe_window * 2]

            last_detections = []
            for track in tracks:
                plate_text, is_valid, avg_conf = select_consensus_plate(
                    track, min_valid_sightings=min_valid_sightings
                )

                display_text = plate_text if is_valid else "..."
                result_for_display = {
                    "plate_number": display_text,
                    "plate_valid_format": is_valid,
                    "confidence": avg_conf
                }

                if plate_text != "UNKNOWN" and is_valid:
                    matched_existing = None
                    for seen_plate, last_seen in recent_plates.items():
                        if is_similar_plate(plate_text, seen_plate) and processed_count - last_seen < dedupe_window:
                            matched_existing = seen_plate
                            break

                    if matched_existing is None:
                        r = build_anpr_result(plate_text, is_valid, avg_conf)
                        r["track_id"] = track["track_id"]
                        r["bbox"] = track["bbox"]
                        recent_plates[plate_text] = processed_count
                        all_results.append(r)
                    else:
                        recent_plates[matched_existing] = processed_count

                last_detections.append({
                    "bbox": track["bbox"],
                    "track_id": track["track_id"],
                    "result": result_for_display
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

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    results_path = os.path.join(results_dir, f"{base_name}_video_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    ocr_log_path = os.path.join(results_dir, f"{base_name}_ocr_log.json")
    with open(ocr_log_path, "w") as f:
        json.dump(ocr_log, f, indent=2)

    print(f"\n✅ Done. Processed {processed_count} of {frame_count} total frames.")
    print(f"💾 Annotated video: {out_path}")
    print(f"💾 Validated results: {results_path}")
    print(f"💾 Full OCR log ({len(ocr_log)} reads): {ocr_log_path}")
    print(f"💾 Raw crops (tagged ok/blurry): {crops_dir}")
    print(f"💾 OCR debug images (text overlaid): {debug_dir}")

    return all_results


if __name__ == "__main__":
    print("Loading models...")
    plate_model = load_plate_model()
    ocr_reader = load_ocr_reader()
    print("✅ Models loaded\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "..", "test_images", "test_video.mp4")

    run_video_pipeline(video_path, plate_model, ocr_reader)