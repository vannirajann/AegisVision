"""Extract ONLY the car detection + tight license-plate crop per tracked
vehicle, with the best OCR read, into runs/plate_crops/<clip>/.

Unlike the streaming ANPR run (which OCRs and votes continuously), this
tool rechecks the footage and keeps the single best plate crop per tracked
car, written with an index.json / index.csv. Word/logo reads (POLICE,
LUMA, ...) are rejected, and no plate text is ever invented.

Usage:
  python src/extract_plate_crops.py --input test_images/test_video2.mp4
  python src/extract_plate_crops.py --input test_images/test_video.mp4 \
      --frame-skip 6 --ocr --max-frames 0
"""
import argparse
import csv
import json
import os

import cv2

import config
from anpr_engine import ANPREngine, frame_luminance
from preprocess_plate import enhance_night_frame


def _score(plate_conf, reads, quality):
    mean_conf = sum(r["confidence"] for r in reads) / len(reads) if reads else 0.0
    n_valid = sum(1 for r in reads if r["valid"])
    best_len = max((len(r["cleaned_text"]) for r in reads if r["cleaned_text"]),
                   default=0)
    return (plate_conf * 40.0 + mean_conf * 30.0 + quality * 20.0
            + n_valid * 10.0 + best_len * 5.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="video file path")
    ap.add_argument("--frame-skip", type=int, default=12,
                    help="process every Nth frame (default 12)")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip OCR (geometry-only crops; default: run OCR and keep reads)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N source frames (0 = entire video)")
    args = ap.parse_args()
    args.ocr = not args.no_ocr

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    clip = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(root, "runs", "plate_crops", clip)
    os.makedirs(out_dir, exist_ok=True)

    engine = ANPREngine()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    ts = config.FRAME_SKIP
    config.FRAME_SKIP = args.frame_skip

    best = {}          # vehicle_id -> best observation
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        if frame_count % args.frame_skip != 0:
            continue
        if args.max_frames and frame_count > args.max_frames:
            break

        vehicles = engine.detect_vehicles(frame, frame_count)
        is_night = frame_luminance(frame) < config.NIGHT_LUMINANCE_THRESHOLD

        det_frames = [frame]
        if is_night:
            det_frames.append(enhance_night_frame(frame))
        plates = []
        for df in det_frames:
            conf = (config.NIGHT_PLATE_CONFIDENCE if df is not frame else None)
            plates += engine._detect_plates(df, conf=conf, crop_source=frame)
        plates = engine._dedupe_plates(plates)

        for p in plates:
            veh = engine._associate_plate_to_vehicle(p["bbox"], vehicles)
            if veh is None or not engine._plate_region_ok(p["bbox"], veh["bbox"]):
                continue
            crop = p.get("crop")
            if crop is None:
                crop = _native_crop(frame, p["bbox"])
            if crop is None or crop.size == 0:
                continue
            reads = []
            quality = 1.0
            if args.ocr:
                reads, _ = engine.read_plate_crop(
                    frame, p["bbox"], config.OCR_VARIANTS_NIGHT if is_night
                    else config.OCR_VARIANTS_VIDEO,
                    crop_img=crop, gate_quality=True, night=is_night)
                if reads and engine._reads_are_words(reads):
                    reads = []
                elif reads:
                    quality = sum(r.get("quality", 1.0) for r in reads) / len(reads)
            score = _score(p["confidence"], reads, quality)

            vid = veh["track_id"]
            prev = best.get(vid)
            if prev is None or score > prev["score"]:
                best[vid] = {
                    "score": score,
                    "vehicle_id": vid,
                    "vehicle_bbox": veh["bbox"],
                    "vehicle_type": veh.get("class_name", "vehicle"),
                    "plate_bbox": p["bbox"],
                    "plate_confidence": round(p["confidence"], 3),
                    "crop": crop,
                    "reads": reads,
                    "frame_number": frame_count,
                    "timestamp_seconds": round(frame_count / fps, 3),
                    "is_night": is_night,
                }

    cap.release()
    config.FRAME_SKIP = ts

    rows = []
    for vid in sorted(best):
        obs = best[vid]
        text = ""
        conf = None
        n_valid = 0
        best_valid = False
        if obs["reads"]:
            cleaned = [r["cleaned_text"] for r in obs["reads"] if r["cleaned_text"]]
            valid = [r for r in obs["reads"] if r["valid"]]
            best_r = max(valid or obs["reads"],
                         key=lambda r: (r["valid"], len(r["cleaned_text"]), r["confidence"]),
                         default=None)
            if best_r:
                text = best_r["cleaned_text"]
                conf = round(best_r["confidence"], 3)
                best_valid = bool(best_r["valid"])
                keep = bool(text) and (
                    len(text) >= 6
                    or (best_valid and any(ch.isdigit() for ch in text)
                        and sum(ch.isdigit() for ch in text) >= 3))
                if not keep:
                    text = ""
                    conf = None
            n_valid = len(valid)
        fname = f"car_{vid:02d}_f{obs['frame_number']:04d}_plate.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, obs["crop"])
        rows.append({
            "vehicle_id": vid,
            "vehicle_type": obs["vehicle_type"],
            "plate_text": text,
            "ocr_confidence": conf,
            "plate_valid_format": best_valid,
            "plate_text_confirmed": bool(text),
            "valid_reads": n_valid,
            "plate_confidence": obs["plate_confidence"],
            "frame_number": obs["frame_number"],
            "timestamp_seconds": obs["timestamp_seconds"],
            "is_night": obs["is_night"],
            "vehicle_bbox": obs["vehicle_bbox"],
            "plate_bbox": obs["plate_bbox"],
            "crop_file": fname,
            "reads": [{"raw": r["raw_text"], "clean": r["cleaned_text"],
                       "conf": r["confidence"], "valid": r["valid"],
                       "variant": r["variant"]} for r in obs["reads"]],
        })

    idx_path = os.path.join(out_dir, "index.json")
    csv_path = os.path.join(out_dir, "index.csv")
    with open(idx_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        cols = ["vehicle_id", "vehicle_type", "plate_text", "plate_valid_format",
                "plate_text_confirmed", "ocr_confidence", "valid_reads", "plate_confidence",
                "frame_number", "timestamp_seconds", "is_night",
                "vehicle_bbox", "plate_bbox", "crop_file"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✅ {len(rows)} car(s) with plates from {clip}:")
    for r in rows:
        print(f"  car_{r['vehicle_id']:02d} f{r['frame_number']:04d} "
              f"plate={r['plate_text'] or 'READING...'} "
              f"conf={r['ocr_confidence']} valid={r['plate_valid_format']}")
    print(f"💾 Crops: {out_dir}")
    print(f"💾 Index: {idx_path}")


def _native_crop(frame, bbox, pad=4):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return frame[y1:y2, x1:x2]


if __name__ == "__main__":
    main()