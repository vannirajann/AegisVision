"""
ANPREngine — unified Automatic Number Plate Recognition engine for AegisVision.

Single reusable implementation that supports:
    * single images (multiple vehicles / multiple plates)
    * video files (live playback + offline batch)
    * multiple vehicles + multiple plates per frame
    * moving vehicles tracked across frames
    * temporal (multi-frame) OCR confirmation via character-position voting

Pipeline (per the project spec):
    frame -> vehicle detection (COCO YOLO) -> tracking (SORT)
          -> license plate detection (YOLO, restricted to vehicle ROIs)
          -> plate crop -> preprocessing variants -> OCR
          -> text cleaning/validation -> temporal consensus -> result

No plate number or plate location is hardcoded anywhere.
"""
import os
import re
import csv
import time
import json
import queue
import threading
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

import cv2
import numpy as np
from ultralytics import YOLO

import config
from sort import Sort
from crop_plate import crop_plate, refine_plate_crop
from preprocess_plate import (
    preprocess_plate,
    preprocess_plate_gentle,
    is_blurry,
    enhance_night_frame,
    night_gamma,
    night_clahe_strong,
    night_contrast_norm,
    night_adaptive,
    night_otsu,
    night_denoise_upscaled,
)
from run_ocr import run_ocr_on_plate
from clean_text import clean_ocr_fragments, validate_plate_format


# ---------------------------------------------------------------------------
# Small geometry / string helpers
# ---------------------------------------------------------------------------
def frame_luminance(frame):
    """Mean gray level of an image; low values → night/low-light."""
    if frame is None or frame.size == 0:
        return 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(gray.mean())
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


def similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))


def _plate_format_ok(text):
    """Known-format check supporting UK, Indian, and European (German-style)
    plates. Kept here so existing clean_text validation stays untouched."""
    if not text:
        return False
    if validate_plate_format(text):
        return True
    return bool(EURO_PATTERN.match(text))


def _enforce_plate_layout(text):
    """Deterministic layout pass for the two common fixed layouts.

    UK 7-char  : LLDDPP00 (letters, 2 digits, letters)  e.g. GX15OGJ
    Indian 9/10: LL DD LLL DDDD (or LL D LLL DDDD)      e.g. TN01AK6321
    Maps lookalike chars at digit positions to digits and at letter
    positions to letters. Returns the corrected text unchanged otherwise."""
    t = text.upper()
    digits2 = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "D": "0", "Q": "0"}
    letters2 = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "B"}

    if len(t) == 7 and t[:2].isalpha():
        out = t[:2]
        out += digits2.get(t[2], t[2])
        out += digits2.get(t[3], t[3])
        out += "".join(letters2.get(ch, ch) if ch.isdigit() else ch for ch in t[4:])
        return out

    if len(t) >= 9 and t[:2].isalpha():
        n = len(t)
        head = t[2:]
        # find the first run of non-digits after the initial letters
        prefix_digits = 0
        for ch in head:
            if ch.isdigit():
                prefix_digits += 1
            else:
                break
        if not (1 <= prefix_digits <= 2):
            return t
        body = list(head)
        for i in range(prefix_digits):
            body[i] = digits2.get(body[i], body[i])
        # tail digits (last 3-4 chars must be digits)
        tail_start = max(prefix_digits, n - 4)
        for i in range(tail_start, n - 2):
            if body[i].isalpha():
                body[i] = digits2.get(body[i], body[i])
        return t[:2] + "".join(body)

    return t


def _only_alnum_upper(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())


# Stricter-than-anything European (German-style) plate: city letters,
# seal digit, optional letters, then digits — e.g. KEX3100, MM1234, DEL100
EURO_PATTERN = re.compile(r'^[A-Z]{1,3}\d[A-Z]{0,2}\d{1,4}$')


# ---------------------------------------------------------------------------
# Track representation helper
# ---------------------------------------------------------------------------
def _new_track(track_id, bbox, frame_count, kind="vehicle"):
    return {
        "track_id": track_id,
        "kind": kind,
        "bbox": bbox,
        "last_seen": frame_count,
        "missed": 0,
        "history": [],
        "last_plate_box": None,
        # per-track OCR pacing + best-frame retention (video)
        "last_ocr_processed": 0,
        "best": None,              # best observation {score, plate_conf, quality, processed_frame, ix}
        "best_reads": [],          # read dicts that produced the best observation
        "last_debug_save": 0,
    }


class ANPREngine:
    """Unified ANPR engine. Load models once, then call detect_image /
    process_video / play_video as needed."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.vehicle_model = YOLO(self.cfg.MODEL_VEHICLE)
        self.plate_model = YOLO(self.cfg.MODEL_PLATE)
        self.ocr_reader = self._load_ocr_reader()
        self.mot_tracker = Sort(
            max_age=max(8, int(self.cfg.FRAME_SKIP * 3)),
            min_hits=1,
            iou_threshold=0.2,
        )
        # per-run state
        self.tracks = {}
        self.recent_plates = {}
        self.ocr_log = []
        self.debug_stats = []
        self.next_fallback_id = 1
        self._seen_track_ids = set()

    # ------------------------------------------------------------------
    # OCR reader (PaddleOCR primary, EasyOCR fallback — lazy so engine
    # import stays instant; model warm up happens on first construction)
    # ------------------------------------------------------------------
    def _load_ocr_reader(self):
        from run_ocr import load_ocr_reader

        return load_ocr_reader(
            engine=self.cfg.OCR_ENGINE,
            gpu=self.cfg.OCR_GPU,
            lang=self.cfg.OCR_LANGUAGES[0],
            cpu_threads=self.cfg.OCR_CPU_THREADS,
            enable_mkldnn=self.cfg.OCR_PADDLE_MKLDNN,
        )

    # ------------------------------------------------------------------
    # Vehicle detection (+ SORT tracking, per processed frame)
    # ------------------------------------------------------------------
    def detect_vehicles(self, frame, frame_count):
        """Run COCO vehicle detection + SORT update. Returns list of
        dicts {track_id, bbox, confidence, class_id, class_name}."""
        h, w = frame.shape[:2]
        results = self.vehicle_model(
            frame, conf=self.cfg.VEHICLE_CONFIDENCE, classes=self.cfg.VEHICLE_CLASSES,
            verbose=False,
        )[0]
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls = int(box.cls[0])
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "class_id": cls,
            })

        tracked = self.mot_tracker.update(
            np.asarray([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                         d["confidence"]] for d in dets], dtype=float)
            if dets else np.empty((0, 5))
        )
        vehicles = []
        for row in tracked:
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            tid = int(row[4])
            bx1, by1, bx2, by2 = clip_box((x1, y1, x2, y2), w, h)
            if by2 - by1 < self.cfg.VEHICLE_MIN_HEIGHT:
                continue
            # best over the raw detections for this tracked box -> conf + class
            best_conf, best_cls = 0.0, None
            for d in dets:
                if compute_iou((bx1, by1, bx2, by2), d["bbox"]) > 0.5:
                    if d["confidence"] > best_conf:
                        best_conf = d["confidence"]
                        best_cls = d["class_id"]
            vehicles.append({
                "track_id": int(row[4]),
                "bbox": (bx1, by1, bx2, by2),
                "confidence": round(best_conf, 2),
                "class_id": best_cls,
                "class_name": self.cfg.COCO_CLASS_NAMES.get(best_cls, "vehicle"),
            })
        return vehicles

    # ------------------------------------------------------------------
    # License plate detection (restricted to vehicle ROI, upscaled)
    # ------------------------------------------------------------------
    def _detect_plates(self, frame, vehicle=None, conf=None, crop_source=None):
        """Detect plates on the full frame (upscaled for recall), or inside a
        vehicle ROI — upscaled — if given.
        Returns list of dicts {bbox (frame coords), confidence, crop} where
        crop is the upscaled plate patch used for high-quality OCR.
        `conf` overrides PLATE_CONFIDENCE (used for the enhanced night pass).
        `crop_source` is an optional same-size image to slice OCR crops from —
        used on night footage where plates are detected on a brightness-
        enhanced copy but OCR reads the original pixels far better."""
        h, w = frame.shape[:2]
        cs = frame if crop_source is None else crop_source
        plate_conf = conf if conf is not None else self.cfg.PLATE_CONFIDENCE
        plates = []
        if vehicle is not None:
            x1, y1, x2, y2 = vehicle["bbox"]
            pad = 12
            roi = frame[max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad)]
            if roi.size == 0:
                return plates
            roi_cs = cs[max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad)]
            scale = self.cfg.PLATE_ROI_UPSCALE
            roi_up = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
                                interpolation=cv2.INTER_CUBIC)
            roi_cs_up = cv2.resize(roi_cs,
                                   (int(roi_cs.shape[1] * scale),
                                    int(roi_cs.shape[0] * scale)),
                                   interpolation=cv2.INTER_CUBIC)
            results = self.plate_model(
                roi_up, conf=plate_conf, iou=0.4, verbose=False,
            )[0]
            ox, oy = max(0, x1 - pad), max(0, y1 - pad)
            for box in results.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                fx1, fy1 = ox + bx1 / scale, oy + by1 / scale
                fx2, fy2 = ox + bx2 / scale, oy + by2 / scale
                pb = clip_box((fx1, fy1, fx2, fy2), w, h)
                if not self._plate_shape_ok(pb):
                    continue
                pad_s = int(8 * scale)
                rh, rw = roi_up.shape[:2]
                crop = roi_cs_up[max(0, int(by1 - pad_s)):min(rh, int(by2 + pad_s)),
                                 max(0, int(bx1 - pad_s)):min(rw, int(bx2 + pad_s))]
                plates.append({"bbox": pb, "confidence": float(box.conf[0]),
                               "crop": crop})
        else:
            fw = int(w * self.cfg.PLATE_FRAME_UPSCALE)
            fh = int(h * self.cfg.PLATE_FRAME_UPSCALE)
            frame_up = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_CUBIC)
            cs_up = cv2.resize(cs, (fw, fh), interpolation=cv2.INTER_CUBIC)
            results = self.plate_model(
                frame_up, conf=plate_conf, iou=0.4, verbose=False,
            )[0]
            scale = self.cfg.PLATE_FRAME_UPSCALE
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pb = clip_box((x1 / scale, y1 / scale, x2 / scale, y2 / scale), w, h)
                if not self._plate_shape_ok(pb):
                    continue
                pad_s = int(8 * scale)
                crop = cs_up[max(0, int(y1 - pad_s)):min(fh, int(y2 + pad_s)),
                             max(0, int(x1 - pad_s)):min(fw, int(x2 + pad_s))]
                plates.append({"bbox": pb, "confidence": float(box.conf[0]),
                               "crop": crop})
        return plates

    def _dedupe_plates(self, plates, iou_thr=0.5):
        """Merge overlapping plate boxes from multiple detection passes
        (original + enhanced night frame), keeping the higher-confidence one."""
        kept = []
        for p in sorted(plates, key=lambda q: q["confidence"], reverse=True):
            if all(compute_iou(p["bbox"], k["bbox"]) < iou_thr for k in kept):
                kept.append(p)
        return kept

    def _plate_shape_ok(self, bbox):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return False
        if width * height < self.cfg.PLATE_MIN_AREA:
            return False
        ar = width / height
        return self.cfg.PLATE_MIN_ASPECT <= ar <= self.cfg.PLATE_MAX_ASPECT

    def _associate_plate_to_vehicle(self, plate_box, vehicles):
        px = (plate_box[0] + plate_box[2]) / 2
        py = (plate_box[1] + plate_box[3]) / 2
        best = None
        best_area = 0
        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best, best_area = v, area
        return best

    def _plate_region_ok(self, plate_box, vehicle_bbox):
        """Plates hang near the bottom bumper, not mid-body / top. Rejects
        stray lettering boxes (e.g. decals on a bus) detected as 'plates'."""
        pb = plate_box
        x1, y1, x2, y2 = vehicle_bbox
        h = y2 - y1
        if h <= 0:
            return True
        rel_bottom = (y2 - pb[3]) / h   # plate bottom vs vehicle bottom
        return rel_bottom <= 0.35

    def _plate_quality_stats(self, crop, night=False):
        """Sharpness / brightness / contrast / size metrics for a crop.

        Returns (stats dict, combined_quality 0..1). All individual metrics
        are normalised so 1.0 = ideal plate crop. `night` shifts the
        brightness reference point: a dark-but-readable night plate should
        not be penalised for being darker than a mid-day one."""
        if crop is None or crop.size == 0:
            return {}, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return {}, 0.0

        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = max(0.0, min(1.0, lap / max(self.cfg.BLUR_THRESHOLD, 1.0)))

        mean = float(gray.mean())
        if night:
            brightness = max(0.0, min(1.0, 1.0 - abs(mean - 70.0) / 90.0))
        else:
            brightness = 1.0 - abs(mean - 128.0) / 128.0

        std = float(gray.std())
        contrast = max(0.0, min(1.0, std / 60.0))

        plate_px = w * h
        size = float(w * h) / 6400.0           # ~64x100 plate = full score
        size = max(0.0, min(1.0, size))

        quality = (0.45 * sharpness + 0.20 * contrast +
                   0.15 * brightness + 0.20 * size)
        stats = {
            "sharpness": round(sharpness, 3),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "plate_size": plate_px,
            "plate_w": w,
            "plate_h": h,
            "night": bool(night),
            "quality": round(min(1.0, quality), 3),
        }
        return stats, min(1.0, quality)

    # ------------------------------------------------------------------
    # Plate preprocessing variants + OCR
    # ------------------------------------------------------------------
    def _preprocess_variants(self, crop):
        h, w = crop.shape[:2]
        target = self.cfg.OCR_TARGET_WIDTH
        scale = target / w if w > 0 else 1.0
        up = cv2.resize(crop, (target, max(1, int(h * scale))),
                        interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up
        variants = {}

        variants["gray_plain"] = gray
        variants["adaptive_threshold"] = preprocess_plate(up, target_width=target)
        variants["gentle_clahe"] = preprocess_plate_gentle(up, target_width=target)

        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants["sharpened_clahe"] = clahe.apply(sharpened)

        _, otsu = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["otsu_binary"] = otsu
        variants["binary_inv"] = cv2.bitwise_not(otsu)
        variants["trunc"] = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)[1]
        variants["tozero"] = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)[1]
        variants["tozero_inv"] = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)[1]

        if len(up.shape) == 3:
            variants["color_upscaled"] = up

        return variants

    def _night_preprocess_variants(self, crop):
        """Extra preprocessing variants for low-light / night-vision crops.
        Merged on top of the standard set when the crop is dark; OCR still
        sees the regular variants too because some night plates are bright."""
        target = self.cfg.OCR_TARGET_WIDTH
        return {
            "night_gamma": night_gamma(crop, target_width=target),
            "night_clahe_strong": night_clahe_strong(crop, target_width=target),
            "night_contrast_norm": night_contrast_norm(crop, target_width=target),
            "night_adaptive": night_adaptive(crop, target_width=target),
            "night_otsu": night_otsu(crop, target_width=target),
            "night_binary_inv": night_otsu(crop, invert=True, target_width=target),
            "night_denoise": night_denoise_upscaled(crop, target_width=target),
        }

    def _crop_is_night(self, crop):
        return frame_luminance(crop) < self.cfg.NIGHT_LUMINANCE_THRESHOLD

    def read_plate_crop(self, frame, plate_box, variant_names=None, crop_img=None,
                        gate_quality=False, night=None):
        """Crop a plate (from the native frame or an upscaled detection crop),
        run the preprocessing variants, OCR them all. Falls back to a second
        variant set if the primary set yields no usable read.

        `night` decides the low-light preprocessing set. When None it is
        inferred from the crop's own luminance; video frames it is passed
        explicitly (frame-level luminance), because a brightness-lifted crop
        from a night frame is still a night plate.

        With gate_quality=True (video path) tiny or severely degraded crops
        are rejected BEFORE OCR — a poor frame must not flood the temporal
        voter with garbage.

        Returns (reads, variants_used) where reads is a list of
        {text, confidence, variant, cleaned_text, valid, quality}."""
        if crop_img is not None and crop_img.size > 0:
            pw = max(1, plate_box[2] - plate_box[0])
            up_ratio = crop_img.shape[1] / pw
            use_crop_img = up_ratio >= 2.5
        else:
            use_crop_img = False

        if use_crop_img:
            crop = crop_img
        else:
            # native-frame crop with horizontal expansion so clipped edge
            # characters (e.g. the leading "AK" on AK64DMV) are not lost
            x1, y1, x2, y2 = plate_box
            w = x2 - x1
            ex = max(2, int(w * self.cfg.PLATE_CROP_EXPAND))
            pad = self.cfg.PLATE_CROP_PADDING
            crop = crop_plate(frame, (
                max(0, int(x1) - ex - pad), max(0, int(y1) - pad),
                min(frame.shape[1], int(x2) + ex + pad),
                min(frame.shape[0], int(y2) + pad),
            ), padding=0)
        if crop is None or crop.size == 0:
            return [], 0

        # Optional cosmetic tightening to the plate rectangle. Kept OFF by
        # default: it measurably degraded night-vision plate crops (the edge
        # detector re-crops onto a darker/noisier subregion and OCR then
        # reads nothing at all).
        if self.cfg.REFINE_PLATE_CROP:
            refined = refine_plate_crop(crop)
            if (refined is not None and refined.size > 0 and
                    refined.shape[1] >= self.cfg.PLATE_MIN_OCR_WIDTH):
                crop = refined

        is_night = bool(night) if night is not None else self._crop_is_night(crop)
        if night is None and self._crop_is_night(crop):
            is_night = True
        stats, quality = self._plate_quality_stats(crop, night=is_night)
        if gate_quality:
            h_c, w_c = crop.shape[:2]
            if min(h_c, w_c) < self.cfg.PLATE_MIN_PX:
                return [], 0
            floor = self.cfg.NIGHT_QUALITY_MIN if is_night else self.cfg.PLATE_QUALITY_MIN
            if quality < floor:
                return [], 0

        all_variants = self._preprocess_variants(crop)
        if is_night:
            all_variants.update(self._night_preprocess_variants(crop))
            names = list(variant_names or self.cfg.OCR_VARIANTS_NIGHT)
            fallback_names = self.cfg.OCR_VARIANTS_NIGHT_FALLBACK
        else:
            names = list(variant_names or all_variants.keys())
            fallback_names = self.cfg.OCR_VARIANTS_VIDEO_FALLBACK
        reads = []
        # early-exit: a validated, high-confidence read on the first variant
        # (e.g. color_upscaled) means the remaining variants add little; skip
        # them to cut CPU OCR time on long clips.
        for name in names:
            reads += self._ocr_variant(all_variants, name, quality)
            if any(r["valid"] and r["confidence"] >= 0.72 for r in reads):
                break

        # Even when the primary set returns *some* text (often garbage on
        # night plates), keep going until something validates — thresholded
        # and binary variants are frequently the only ones that read a dark
        # plate character-by-character. Daytime keeps the cheaper rule
        # (fallback only when nothing was read at all).
        need_more = is_night and not any(r["valid"] for r in reads)
        if (not reads or need_more) and names and fallback_names:
            for name in fallback_names:
                reads += self._ocr_variant(all_variants, name, quality)
        if reads and stats:
            for r in reads:
                r["quality_stats"] = stats
                r["night"] = is_night
        return reads, len(names)

    def _reads_are_words(self, reads):
        """True when the OCR reads of a plate crop are dominated by
        letters-only WORD fragments (e.g. "POLICE", "LUMA") instead of an
        alphanumeric number plate. Real plates always contain digits."""
        non_empty = [r for r in reads if r.get("cleaned_text")]
        if not non_empty:
            return False
        letter_only = [r for r in non_empty
                       if not any(ch.isdigit() for ch in r["cleaned_text"])]
        if not letter_only:
            return False
        # any solid multi-char read that DOES contain a digit means this is
        # most likely a plate after all (some variants may read the word as
        # e.g. "POL1CE", but a genuine plate read won't be pure letters)
        digit_reads = [r for r in non_empty
                       if any(ch.isdigit() for ch in r["cleaned_text"])][:]
        solid_digit = [r for r in digit_reads if len(r["cleaned_text"]) >= 5]
        if solid_digit:
            return False
        total_w = sum(r["confidence"] for r in non_empty) or 0.0
        word_w = sum(r["confidence"] for r in letter_only)
        return len(letter_only) >= 2 and (word_w / total_w) >= 0.6

    def _ocr_variant(self, all_variants, name, quality):
        variant_img = all_variants.get(name)
        if variant_img is None:
            return []
        out = []
        try:
            ocr_texts = run_ocr_on_plate(self.ocr_reader, variant_img)
        except Exception:
            ocr_texts = []
        for t in ocr_texts:
            raw = _only_alnum_upper(t["text"])
            if len(raw) < 4:
                continue
            cleaned, valid = clean_ocr_fragments([t])
            # keep every usable read in history — the temporal voter can
            # leverage reads that fail strict validation individually.
            out.append({
                "raw_text": t["text"],
                "text": raw,
                "cleaned_text": cleaned if cleaned else raw,
                "valid": valid,
                "confidence": round(float(t["confidence"]), 3),
                "variant": name,
                "quality": round(quality, 2),
            })
        return out

    # ------------------------------------------------------------------
    # Temporal consensus — character-position voting
    # ------------------------------------------------------------------
    def _cluster_reads(self, entries):
        used = set()
        clusters = []
        for i, e in enumerate(entries):
            if i in used:
                continue
            cl = [e]
            used.add(i)
            for j in range(i + 1, len(entries)):
                if j in used:
                    continue
                if similarity(e["cleaned_text"], entries[j]["cleaned_text"]) >= self.cfg.CONSENSUS_SIMILARITY:
                    cl.append(entries[j])
                    used.add(j)
            clusters.append(cl)
        return clusters

    def _cluster_score(self, cluster):
        n_valid = sum(1 for e in cluster if e["valid"])
        total_conf = sum(e["confidence"] for e in cluster)
        lens = [len(e["cleaned_text"]) for e in cluster]
        median_len = sorted(lens)[len(lens) // 2]
        full_len_bonus = 1500 if median_len in (7, 9, 10) else 0
        return total_conf + 300 * (len(cluster) - 1) + 10000 * n_valid + full_len_bonus

    def select_consensus_plate(self, track):
        """Vote over a track's OCR history and return a stable plate.

        Once a track's vote has qualified as confirmed (valid format + enough
        corroborating sightings), the result is FROZEN and returned verbatim
        forever after: a single SORT track is one physical vehicle, so later
        OCR drift must never re-confirm a different string for it."""
        frozen_c = track.get("confirmed_consensus")
        if frozen_c:
            return frozen_c
        entries = [
            e for e in track.get("history", [])
            if e.get("cleaned_text") and e.get("confidence", 0) >= self.cfg.OCR_MIN_CONFIDENCE
        ]
        if not entries:
            track.setdefault("confirmed_consensus", None)
            return None

        clusters = self._cluster_reads(entries)
        best = max(clusters, key=self._cluster_score)

        voters = []
        for e in best:
            weight = e["confidence"]
            if e.get("valid"):
                weight *= 3.0
            weight *= (0.5 + 0.5 * e.get("quality", 1.0))
            if e.get("is_best"):
                weight *= (1.0 + self.cfg.BEST_READ_BONUS)
            voters.append((e["cleaned_text"], weight, e["valid"], e["confidence"]))
        n_voters = len(voters)
        from statistics import median_low
        target_len = int(median_low([len(t) for t, _, _, _ in voters]))
        voters_f = [(t, c, f, s) for t, c, f, s in voters if abs(len(t) - target_len) <= 1]
        if voters_f:
            voters = voters_f

        chars = []
        max_len = max(len(t) for t, _, _, _ in voters)
        for pos in range(max_len):
            votes = defaultdict(float)
            coverage_n = 0
            for t, c, _, _ in voters:
                if pos < len(t):
                    votes[t[pos]] += c
                    coverage_n += 1
            if not votes:
                continue
            best_char = max(votes, key=votes.get)
            if coverage_n / n_voters >= self.cfg.CHAR_VOTE_COVERAGE:
                chars.append(best_char)

        voted = "".join(chars)
        voted = _enforce_plate_layout(voted)
        voted = voted.strip()
        n_valid = sum(1 for e in best if e["valid"])
        is_valid = _plate_format_ok(voted) and n_valid >= self.cfg.MIN_VALID_SIGHTINGS
        avg_conf = sum(s for _, _, _, s in voters) / len(voters) if voters else 0.0

        consensus = {
            "plate_text": voted,
            "plate_valid_format": is_valid,
            "ocr_confidence": round(avg_conf, 2),
            "sightings": len(best),
            "n_valid_reads": n_valid,
            "raw_reads": [t for t, _, _, _ in voters],
        }

        # Freeze only after the SAME validated read recurs across a few
        # qualifying frames. A one-frame vote (e.g. "DE723218" after 2
        # sightings) is exactly the unstable-digit state where the plate is
        # still evolving character by character — never lock that in.
        if consensus["plate_valid_format"] and \
                consensus["sightings"] >= self.cfg.MIN_VALID_SIGHTINGS and \
                consensus["ocr_confidence"] >= self.cfg.CONFIRM_MIN_CONF:
            last_seen = track.get("last_seen", 0)
            hits = [h for h in track.get("_confirm_hits", [])
                    if last_seen - h[1] <= self.cfg.CONFIRM_STABILITY_WINDOW]
            hits.append((consensus["plate_text"], last_seen))
            track["_confirm_hits"] = hits[-8:]
            if not track.get("confirmed_consensus") and \
                    sum(1 for t, _ in hits if t == consensus["plate_text"]) >= \
                    self.cfg.CONFIRM_STABILITY_REPEATS:
                track["confirmed_plate"] = consensus["plate_text"]
                track["confirmed_consensus"] = consensus
        return consensus

    # ------------------------------------------------------------------
    # Result/event builders
    # ------------------------------------------------------------------
    def _build_event(self, vehicle, plate, consensus, ocr_views,
                     frame_number=None, timestamp_seconds=None):
        now = datetime.now(timezone.utc).isoformat()
        valid = consensus["plate_valid_format"]
        status = "confirmed" if valid else "low_confidence"
        event = {
            "event_type": "anpr",
            "plate_text": consensus["plate_text"],
            "plate_number": consensus["plate_text"],
            "plate_valid_format": valid,
            "plate_status": status,
            "ocr_confidence": consensus["ocr_confidence"],
            "raw_ocr": consensus["raw_reads"],
            "vehicle_id": vehicle["track_id"],
            "vehicle_type": vehicle["class_name"],
            "vehicle_confidence": vehicle["confidence"],
            "vehicle_bbox": list(vehicle["bbox"]),
            "plate_bbox": list(plate["bbox"]),
            "plate_confidence": round(plate["confidence"], 2),
            "sightings": consensus["sightings"],
            "confirmation_count": consensus["n_valid_reads"],
            "timestamp": now,
        }
        if frame_number is not None:
            event["frame_number"] = int(frame_number)
        if timestamp_seconds is not None:
            event["timestamp_seconds"] = round(float(timestamp_seconds), 3)
        return event

    EVENT_CSV_COLUMNS = [
        "frame_number", "timestamp_seconds",
        "vehicle_id", "vehicle_type", "vehicle_confidence", "vehicle_bbox",
        "plate_bbox", "plate_confidence",
        "plate_text", "plate_valid_format", "plate_status",
        "ocr_confidence", "sightings", "confirmation_count",
        "best_plate_confidence", "best_frame",
        "timestamp",
    ]

    def _write_events_files(self, events, results_dir, base_name, tag="events"):
        """Save detection results as BOTH .json and .csv (one row per
        recognised plate, with its vehicle). Returns (json_path, csv_path)."""
        events_path = os.path.join(results_dir, f"{base_name}_{tag}.json")
        csv_path = os.path.join(results_dir, f"{base_name}_{tag}.csv")
        with open(events_path, "w") as f:
            json.dump(events, f, indent=2)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EVENT_CSV_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            for ev in events:
                row = dict(ev)
                for k in ("vehicle_bbox", "plate_bbox"):
                    if k in row:
                        row[k] = json.dumps(row[k])
                writer.writerow(row)
        return events_path, csv_path

    # ------------------------------------------------------------------
    # Annotation (matches reference image style: green vehicle box,
    # red plate box, plate text)
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_text_block(img, lines, anchor, color=(255, 255, 255),
                         font_scale=0.6, thickness=2, bg=(0, 0, 0)):
        """Write stacked text lines at `anchor` with a translucent black
        background so labels stay readable over bright video frames."""
        h, w = img.shape[:2]
        line_h = int(24 * font_scale) + 8
        x, y = int(anchor[0]), int(anchor[1])
        width = max((cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale, thickness)[0][0]
                     for ln in lines), default=0) + 14
        total_h = len(lines) * line_h
        y0 = max(0, y - total_h)
        y1 = min(h, y)
        x1 = min(w, x + width)
        if y0 < y1 and x < x1:
            overlay = img[y0:y1, x:x1].copy()
            cv2.rectangle(img, (x, y0), (x1, y1), bg, cv2.FILLED)
            cv2.addWeighted(overlay, 0.45, img[y0:y1, x:x1], 0.55, 0,
                            img[y0:y1, x:x1])
        cy = y - total_h + (line_h - 1)
        for i, ln in enumerate(lines):
            ty = max(cy + i * line_h + line_h - 6, line_h)
            cv2.putText(img, ln, (x + 7, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

    def annotate(self, frame, annotations, panel=None):
        """Draw ANPR overlays on a frame.

        GREEN box + ID/PLATE/CONF: plate confirmed (valid format, enough
        sightings). YELLOW box + READING...: plate detected, evidence still
        being gathered. WHITE box: tracked vehicle, no plate box yet. A plate
        string is NEVER drawn unless the temporal vote confirmed it."""
        out = frame.copy()
        for ann in annotations:
            vb = ann["vehicle_bbox"]
            vb = (int(vb[0]), int(vb[1]), int(vb[2]), int(vb[3]))
            status = ann.get("plate_status", "vehicle")
            valid = bool(ann["plate_valid_format"])
            vid = ann.get("vehicle_id", "?")
            conf = ann.get("ocr_confidence")

            if status == "confirmed":
                color = (0, 255, 0)
            elif status == "confirming":
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(out, (vb[0], vb[1]), (vb[2], vb[3]), color, 2)

            pb = ann.get("plate_bbox")
            if pb is not None and status in ("confirmed", "confirming"):
                pb = (int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3]))
                cv2.rectangle(out, pb, color, 2)

            if status == "confirmed":
                pct = f"{int(round(conf * 100))}%" if conf is not None else "n/a"
                label = ann["plate_text"] or "..."
                lines = [f"ID: {vid:02d}", f"PLATE: {label}", f"CONF: {pct}"]
            elif status == "confirming":
                lines = [f"ID: {vid:02d}", "PLATE: READING...", "CONF: --"]
            else:
                lines = [f"ID: {vid:02d}", f"VEHICLE: {ann.get('vehicle_type', '')}"]
            self._draw_text_block(out, lines, (vb[0], max(vb[1] - 4, 30)),
                                  color=color)
        if panel:
            self._add_info_panel(out, panel)
        return out

    def _add_info_panel(self, img, panel):
        h, w = img.shape[:2]
        bar_h = 30
        cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(img[0:bar_h, 0:w], 0.6, img[0:bar_h, 0:w], 0.4, 0,
                        img[0:bar_h, 0:w])
        lines = [
            ("AEGISVISION ANPR", 8),
            ("VIDEO ANPR", 215),
            (f"FPS: {panel.get('fps', '?')}", 330),
            (f"VEHICLES: {panel.get('vehicles', 0)}", 430),
            (f"PLATES: {panel.get('plates', 0)}", 560),
            (f"OCR: {panel.get('ocr', '?')}", 690),
        ]
        if panel.get('mode') == 'night':
            lines.append(("NIGHT-VISION PREPROC", 800))
        for text, x in lines:
            cv2.putText(img, text, (x, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------------------
    def detect_image(self, image_path, save_output=True):
        base_dir = self.cfg.OUTPUT_BASE
        annotated_dir = os.path.join(base_dir, "annotated")
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        h, w = frame.shape[:2]

        vehicles = self.detect_vehicles(frame, 0)
        plates = self._detect_plates(frame)

        results = []
        used_plates = set()
        for plate in plates:
            veh = self._associate_plate_to_vehicle(plate["bbox"], vehicles)
            key = tuple(plate["bbox"])
            if key in used_plates:
                continue
            used_plates.add(key)
            reads, n_vars = self.read_plate_crop(
                frame, plate["bbox"], self.cfg.OCR_VARIANTS_IMAGE,
                crop_img=plate.get("crop"))
            if reads and self._reads_are_words(reads):
                reads = []
            if not reads:
                continue
            consensus = self._aggregate_single_frame(reads)
            if veh is None:
                veh = {
                    "track_id": len(results) + 1,
                    "bbox": plate["bbox"],
                    "confidence": plate["confidence"],
                    "class_name": "vehicle",
                }
            results.append(self._build_event(veh, plate, consensus, reads))

        base = os.path.splitext(os.path.basename(image_path))[0]
        if save_output and results:
            annotations = [
                {
                    "vehicle_id": r["vehicle_id"],
                    "vehicle_bbox": r["vehicle_bbox"],
                    "plate_bbox": r["plate_bbox"],
                    "plate_text": r["plate_text"],
                    "plate_valid_format": r["plate_valid_format"],
                }
                for r in results
            ]
            annotated = self.annotate(frame, annotations)
            ann_path = os.path.join(annotated_dir, f"{base}_engine_result.jpg")
            cv2.imwrite(ann_path, annotated)
            res_path, csv_path = self._write_events_files(
                results, results_dir, base, tag="engine_results")
            print(f"💾 Saved: {ann_path}")
            print(f"💾 Saved: {res_path}")
            print(f"💾 Saved: {csv_path}")
        return results

    def _aggregate_single_frame(self, reads):
        """For single images there is no temporal history — pick the
        strongest validated/longest read among the variants."""
        valid = [r for r in reads if r["valid"]]
        pool = valid if valid else reads
        # prefer the longest strong read: 10-char Indian plates win over
        # 6-char fragments that also happen to match a loose tail pattern
        best = max(pool, key=lambda r: (r["valid"], len(r["cleaned_text"]), r["confidence"]))
        return {
            "plate_text": best["cleaned_text"],
            "plate_valid_format": best["valid"],
            "ocr_confidence": round(best["confidence"], 2),
            "sightings": len(pool),
            "n_valid_reads": len(valid),
            "raw_reads": [r["raw_text"] for r in reads],
        }

    # ------------------------------------------------------------------
    # Shared run state + event emission (used by batch _process_stream
    # and the live play_video player)
    # ------------------------------------------------------------------
    def _reset_run_state(self, debug=False):
        """Reset all per-run state so a single engine instance can be reused
        for many clips / playbacks cleanly."""
        self.tracks = {}
        self.recent_plates = {}
        self.ocr_log = []
        self.debug_stats = []
        self.next_fallback_id = 1
        self._seen_track_ids = set()
        self.mot_tracker.trackers = []
        self.mot_tracker.frame_count = 0
        self._debug_mode = bool(debug)
        self._debug_root = self.cfg.VIDEO_DEBUG_ROOT
        if self._debug_mode:
            os.makedirs(self._debug_root, exist_ok=True)

    def _consume_plate_events(self, annotations, source_idx, fps,
                              processed_count):
        """Turn the current per-track annotations into confirmed events.

        Reused verbatim by both the batch stream loop and the live player so
        the temporal-consensus / dedupe / emission rules stay identical.
        Evidence comes from each track's temporal vote (not from the on-screen
        label, which stays honest: plate text is only displayed once proven).
        """
        events = []
        for ann in annotations:
            if ann.get("plate_status") not in ("confirmed", "confirming"):
                continue
            veh = self.tracks.get(ann["vehicle_id"])
            if veh is None:
                continue
            consensus = self.select_consensus_plate(veh)
            if consensus is None:
                continue
            plate_text = consensus["plate_text"]
            is_valid = consensus["plate_valid_format"]
            frozen = veh.get("confirmed_consensus") is not None
            if not plate_text:
                continue
            if not self._is_new_plate_event(plate_text, source_idx):
                continue
            # Emit CONFIRMED only when the temporal vote has been stable
            # enough to freeze (same validated read recurred over multiple
            # qualifying frames). Anything weaker is reported honestly as a
            # low_confidence reading with no on-screen plate text.
            enough = consensus["sightings"] >= self.cfg.MIN_VALID_SIGHTINGS
            if not enough:
                continue
            confirmed = is_valid and frozen
            if not (confirmed or (
                    len(plate_text) >= self.cfg.EVENT_MIN_TEXT_LEN and
                    consensus["ocr_confidence"] >= self.cfg.EVENT_MIN_MEAN_CONFIDENCE)):
                continue
            # one report per real plate per vehicle: suppress re-reports whose
            # text is near-identical to an earlier event for the SAME track
            prev_emitted = veh.get("_emitted_plates", [])
            if prev_emitted and \
                    any(similarity(plate_text, p) >= self.cfg.EVENT_PLATE_DEDUPE_SIM
                        for p in prev_emitted):
                continue
            veh.setdefault("_emitted_plates", []).append(plate_text)
            plate_box = veh.get("last_plate_box")
            event = self._build_event(
                {"track_id": veh["track_id"], "bbox": veh["bbox"],
                 "class_name": veh.get("class_name", "vehicle"),
                 "confidence": veh.get("confidence", 0.0)},
                {"bbox": plate_box, "confidence": veh.get("last_plate_conf", 0.0)},
                consensus,
                veh.get("history", []),
                frame_number=source_idx,
                timestamp_seconds=source_idx / fps if fps else 0.0,
            )
            best = veh.get("best")
            if best is not None:
                event["best_plate_confidence"] = round(best.get("plate_conf", 0.0), 3)
                event["best_frame"] = best.get("processed_frame")
                event["best_quality"] = best.get("quality")
            if confirmed:
                event["plate_valid_format"] = True
                event["plate_status"] = "confirmed"
            else:
                event["plate_valid_format"] = False
                event["plate_status"] = "low_confidence"
            events.append(event)
            print(f"[frame {source_idx}] V{veh['track_id']} → "
                  f"{plate_text} (conf={consensus['ocr_confidence']:.2f}, "
                  f"sightings={consensus['sightings']}, "
                  f"{event['plate_status']})")
            self.recent_plates[plate_text] = source_idx
        return events

    # ------------------------------------------------------------------
    # VIDEO / CAMERA shared stream loop
    # ------------------------------------------------------------------
    def _process_stream(self, cap, source_name, save_output=True,
                        show_window=False, max_frames=0, debug=False):
        if not cap.isOpened():
            raise RuntimeError("Could not open video/camera source")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # clean per-run state (a single engine instance may process many clips)
        self._reset_run_state(debug=debug)

        videos_dir = os.path.join(self.cfg.OUTPUT_BASE, "videos")
        results_dir = os.path.join(self.cfg.OUTPUT_BASE, "results")
        crops_dir = os.path.join(self.cfg.OUTPUT_BASE, "crops")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

        writer = None
        out_path = None
        if save_output:
            base_name = os.path.splitext(os.path.basename(source_name))[0]
            out_path = os.path.join(videos_dir, f"{base_name}_engine_result.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))

        frame_count = 0
        processed_count = 0
        events = []
        t_start = time.time()
        last_annotations = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            frame_count += 1

            if frame_count % self.cfg.FRAME_SKIP == 0:
                processed_count += 1
                self._process_one_frame(frame, processed_count)
                last_annotations = self._current_annotations()
                events += self._consume_plate_events(
                    last_annotations, frame_count, fps, processed_count)

            if writer is not None:
                annotated = self.annotate(frame, last_annotations)
                writer.write(annotated)
            if show_window:
                cv2.imshow("AegisVision - ANPR (q to quit)",
                           self.annotate(frame, last_annotations))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        base_name = os.path.splitext(os.path.basename(source_name))[0]
        events_path, csv_path = self._write_events_files(
            events, results_dir, base_name, tag="engine_events")
        log_path = os.path.join(results_dir, f"{base_name}_engine_ocr_log.json")
        with open(log_path, "w") as f:
            json.dump(self.ocr_log, f, indent=2)

        elapsed = time.time() - t_start
        fps_proc = processed_count / elapsed if elapsed else 0.0
        print(f"\n✅ Done. {processed_count} processed of {frame_count} frames "
              f"(total in source: {total}) in {elapsed:.1f}s "
              f"({fps_proc:.2f} proc-frames/s, ~{elapsed / processed_count:.2f}s/frame)")
        print(f"💾 Annotated video: {out_path}")
        print(f"💾 Events: {events_path}")
        print(f"💾 CSV: {csv_path}")
        print(f"💾 OCR log ({len(self.ocr_log)} reads): {log_path}")
        print(f"💾 Confirmations: {len(events)}")

        if self.debug_stats:
            n_plate_frames = sum(1 for s in self.debug_stats if s["plates_with_boxes"])
            n_ocr_frames = sum(1 for s in self.debug_stats if s["ocr_calls"])
            total_reads = sum(s["reads"] for s in self.debug_stats)
            print(f"🔍 plates-detected frames: {n_plate_frames}/{len(self.debug_stats)}, "
                  f"ocr-call frames: {n_ocr_frames}, total OCR reads: {total_reads}")

        if os.path.isdir(crops_dir) and self.cfg.SAVE_DEBUG_CROPS:
            print(f"💾 Debug crops: {crops_dir}")
        return events

    def _process_one_frame(self, frame, processed_count):
        vehicles = self.detect_vehicles(frame, processed_count)

        # sync internal track registry with SORT tracks
        for v in vehicles:
            track = self.tracks.get(v["track_id"])
            if track is None:
                track = _new_track(v["track_id"], v["bbox"], processed_count,
                                   kind="vehicle")
                self.tracks[v["track_id"]] = track
                self._seen_track_ids.add(v["track_id"])
            track["bbox"] = v["bbox"]
            track["last_seen"] = processed_count
            track["missed"] = 0
            track["confidence"] = v["confidence"]
            track["class_name"] = v["class_name"]

        # prune stale tracks
        stale = [
            tid for tid, t in self.tracks.items()
            if processed_count - t["last_seen"] > self.cfg.DEDUPE_WINDOW * 2
        ]
        for tid in stale:
            del self.tracks[tid]

        # 1) full-frame plate detection once, then associate plates to vehicles.
        #    On low-light footage also run the plate detector on a brightness-
        #    enhanced copy (same coordinates) and merge the two detection sets
        #    so dark plates are not missed.
        is_night = frame_luminance(frame) < self.cfg.NIGHT_LUMINANCE_THRESHOLD
        detection_frames = [frame]
        if is_night:
            detection_frames.append(enhance_night_frame(frame))
        full_plates = []
        for df in detection_frames:
            conf = self.cfg.NIGHT_PLATE_CONFIDENCE if df is not frame else None
            full_plates += self._detect_plates(df, conf=conf, crop_source=frame)
        full_plates = self._dedupe_plates(full_plates)

        plate_to_veh = {}
        for plate in full_plates:
            veh = self._associate_plate_to_vehicle(plate["bbox"], vehicles)
            if veh is None:
                continue
            if not self._plate_region_ok(plate["bbox"], veh["bbox"]):
                continue
            plate_to_veh.setdefault(veh["track_id"], []).append(plate)

        # 2) ROI fallback for near/large vehicles without a full-frame plate
        roi_source = enhance_night_frame(frame) if is_night else frame
        for v in vehicles:
            tid = v["track_id"]
            if tid in plate_to_veh:
                continue
            h = v["bbox"][3] - v["bbox"][1]
            if h < self.cfg.PLATE_ROI_FALLBACK_MIN_HEIGHT:
                continue
            roi_plates = self._detect_plates(roi_source, vehicle=v, crop_source=frame)
            roi_plates = [p for p in roi_plates
                          if self._plate_region_ok(p["bbox"], v["bbox"])]
            if roi_plates:
                plate_to_veh[tid] = roi_plates

        # 3) OCR the best plate per vehicle, per processed frame
        #    (capped to bound runtime on CPU)
        variant_names = (self.cfg.OCR_VARIANTS_NIGHT if is_night
                         else self.cfg.OCR_VARIANTS_VIDEO)
        ranked = sorted(plate_to_veh.items(),
                        key=lambda kv: max(p["confidence"] for p in kv[1]),
                        reverse=True)
        ocr_calls = 0
        reads_total = 0
        for tid, plates in ranked[:self.cfg.MAX_OCR_PLATES_PER_FRAME]:
            track = self.tracks.get(tid)
            if track is None:
                continue
            plate = max(plates, key=lambda p: p["confidence"])
            pb = plate["bbox"]
            pw = pb[2] - pb[0]
            if pw < self.cfg.PLATE_MIN_OCR_WIDTH or pw > self.cfg.PLATE_MAX_OCR_WIDTH:
                continue
            ocr_calls += 1
            # pace OCR per track: skip if we OCR'd this track too recently
            if processed_count - track.get("last_ocr_processed", 0) < self.cfg.OCR_INTERVAL:
                continue
            track["last_ocr_processed"] = processed_count

            reads, n_vars = self.read_plate_crop(frame, pb, variant_names,
                                                 crop_img=plate.get("crop"),
                                                 gate_quality=True,
                                                 night=is_night)
            reads_total += len(reads)
            # Reject word/logo detections (e.g. "POLICE" lettering, "LUMA"
            # stickers) that the plate detector occasionally fires on: a real
            # number plate is alphanumeric, so crops that overwhelmingly read
            # back as letters-only words are NOT plates. They get no box, no
            # OCR history and no event — only the car's actual plate is cropped.
            if reads and self._reads_are_words(reads):
                reads = []
            if reads:
                track["last_plate_box"] = pb
                track["last_plate_conf"] = plate["confidence"]
                for r in reads:
                    track["history"].append(r)
                    self.ocr_log.append({
                        "frame": processed_count,
                        "track_id": track["track_id"],
                        "variant": r["variant"],
                        "raw_text": r["raw_text"],
                        "cleaned_text": r["cleaned_text"],
                        "valid_format": r["valid"],
                        "confidence": r["confidence"],
                        "quality": r.get("quality", 1.0),
                    })

            if reads:
                mean_q = sum(x.get("quality", 1.0) for x in reads) / len(reads)
                if self._update_best_observation(
                        track, plate["confidence"], reads, mean_q,
                        processed_count):
                    crop = plate.get("crop")
                    if crop is None:
                        crop = crop_plate(frame, pb, padding=6)
                    if crop is not None and self._debug_mode:
                        self._save_debug_observation(track, crop, reads, processed_count)

            if self.cfg.SAVE_DEBUG_CROPS:
                crop = plate.get("crop")
                if crop is None:
                    crop = crop_plate(frame, pb, padding=6)
                if crop is not None:
                    cv2.imwrite(os.path.join(self.cfg.OUTPUT_BASE, "crops",
                                             f"V{track['track_id']}_P{processed_count}.jpg"), crop)

        # keep history bounded
        for t in self.tracks.values():
            if len(t["history"]) > 60:
                t["history"] = t["history"][-60:]

        self.debug_stats.append({
            "frame": processed_count,
            "vehicles": len(vehicles),
            "plates_with_boxes": sum(len(v) for v in plate_to_veh.values()),
            "ocr_calls": ocr_calls,
            "reads": reads_total,
        })

    def _current_annotations(self):
        """Build per-track annotations for display.

        Honest evidence tiers:
          confirmed   — valid plate format AND enough corroborating sightings
                        (show the plate text)
          confirming  — plate box detected / OCR running, not yet proven
                        (show READING...)
          vehicle     — tracked vehicle with no plate box yet (no plate info)
        A plate string is NEVER shown unless the temporal vote confirmed it."""
        anns = []
        for track in self.tracks.values():
            vb = [int(v) for v in track.get("bbox", (0, 0, 0, 0))]
            pb = track.get("last_plate_box")
            consensus = None
            if (pb is not None and track.get("history")) or track.get("confirmed_consensus"):
                consensus = self.select_consensus_plate(track)
            frozen = track.get("confirmed_consensus")
            if frozen is not None and consensus is not None:
                status = "confirmed"
                text = consensus["plate_text"]
            elif pb is not None:
                status = "confirming"
                text = ""
            else:
                status = "vehicle"
                text = ""
            anns.append({
                "vehicle_id": track["track_id"],
                "vehicle_bbox": vb,
                "vehicle_type": track.get("class_name", "vehicle"),
                "plate_bbox": list(pb) if pb is not None else None,
                "plate_text": text,
                "plate_valid_format": frozen is not None,
                "plate_status": status,
                "ocr_confidence": consensus["ocr_confidence"] if consensus else None,
                "sightings": consensus["sightings"] if consensus else 0,
                "n_valid_reads": consensus["n_valid_reads"] if consensus else 0,
            })
        return anns

    def _update_best_observation(self, track, plate_conf, reads, quality,
                                 processed_count):
        """Remember the sharpest/highest-confidence plate observation for a
        track. Reads from the best frame are flagged `is_best` so the temporal
        voter weighs them extra (BEST_READ_BONUS). Returns True if the best
        observation changed (used to trigger debug saves)."""
        if not reads:
            return False
        lens = [len(r["cleaned_text"]) for r in reads if r["cleaned_text"]]
        mean_conf = sum(r["confidence"] for r in reads) / len(reads)
        n_valid = sum(1 for r in reads if r["valid"])
        best_len = max(lens) if lens else 0
        score = (plate_conf * 40.0 + mean_conf * 30.0 + quality * 20.0
                 + n_valid * 10.0 + best_len * 5.0)
        prev = track.get("best")
        if prev is not None and score <= prev["score"]:
            return False
        for r in track.get("best_reads", []):
            r["is_best"] = False
        for r in reads:
            r["is_best"] = True
        track["best_reads"] = list(reads)
        track["best"] = {
            "score": round(score, 2),
            "plate_conf": plate_conf,
            "quality": round(float(quality), 3),
            "processed_frame": processed_count,
        }
        return True

    def _save_debug_observation(self, track, crop, reads, processed_count):
        """Dump the plate crop variants + OCR reads for a vehicle into
        runs/video_anpr/vehicle_{id}/ — used to audit why OCR succeeded or
        failed per plate."""
        if not self._debug_mode:
            return
        if processed_count - track.get("last_debug_save", 0) < self.cfg.VIDEO_DEBUG_SAMPLING:
            return
        track["last_debug_save"] = processed_count
        folder = os.path.join(self._debug_root,
                              f"vehicle_{track['track_id']:02d}")
        os.makedirs(folder, exist_ok=True)

        def _write(key, img):
            cv2.imwrite(os.path.join(folder, f"f{processed_count:05d}_{key}.jpg"), img)

        _write("original", crop)
        variants = self._preprocess_variants(crop)
        write_map = {
            "gray_plain": "gray",
            "gentle_clahe": "clahe",
            "otsu_binary": "binary",
            "adaptive_threshold": "adaptive",
        }
        for key, suffix in write_map.items():
            variant = variants.get(key)
            if variant is None:
                continue
            if len(variant.shape) == 2:
                variant = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
            _write(suffix, variant)
        with open(os.path.join(folder, f"f{processed_count:05d}_ocr.json"), "w") as f:
            json.dump([{
                "raw_text": r.get("raw_text"),
                "cleaned_text": r.get("cleaned_text"),
                "valid": r.get("valid"),
                "confidence": r.get("confidence"),
                "variant": r.get("variant"),
                "quality": r.get("quality", 1.0),
            } for r in reads], f, indent=2)

    def _is_new_plate_event(self, plate_text, processed_count):
        for seen, last in self.recent_plates.items():
            if (similarity(plate_text, seen) >= 0.75 and
                    processed_count - last < self.cfg.DEDUPE_WINDOW):
                self.recent_plates[seen] = processed_count
                return False
        return True

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def process_video(self, video_path, save_output=True, show_window=False,
                      max_frames=0, debug=False):
        cap = cv2.VideoCapture(video_path)
        return self._process_stream(cap, video_path, save_output=save_output,
                                    show_window=show_window, max_frames=max_frames,
                                    debug=debug)

    # ------------------------------------------------------------------
    # LIVE PLAYER — video file plays on screen while ANPR runs
    # ------------------------------------------------------------------
    def play_video(self, video_path, save_output=False):
        """Play a VIDEO FILE live on the screen with ANPR overlays.

        The exact input video (VideoCapture of the file, never a webcam)
        plays continuously at its native FPS. A background thread runs
        YOLO + OCR on a paced subset of frames and updates the overlays
        (green/red boxes, ID/PLATE/CONF, top info panel).

        Controls:
            Q / ESC  quit
            SPACE    pause / resume the input video
            R        restart the video from the beginning
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        self._reset_run_state()

        lock = threading.Lock()
        end_event = threading.Event()
        frame_q = queue.Queue(maxsize=8)
        ctl = {"stop": False, "restart": False}
        proc_slot = {"idx": 0, "frame": None}
        shared = {"annotations": [], "vehicles": 0, "plates": 0}
        play_events = []

        def drain_q():
            while True:
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    break

        def reader():
            cap_local = cap
            idx = 0
            while not ctl["stop"]:
                if ctl["restart"]:
                    end_event.clear()
                    cap_local.release()
                    cap_local = cv2.VideoCapture(video_path)
                    if not cap_local.isOpened():
                        break
                    with lock:
                        self._reset_run_state()
                        shared["plates"] = 0
                        shared["annotations"] = []
                    idx = 0
                    ctl["restart"] = False
                    continue
                ret, frame = cap_local.read()
                if not ret or frame is None:
                    if not ctl["stop"]:
                        end_event.set()
                        time.sleep(0.05)   # wait for restart / quit
                    continue
                idx += 1
                with lock:
                    proc_slot["idx"] = idx
                    proc_slot["frame"] = frame
                try:
                    frame_q.put((idx, frame), timeout=0.25)
                except queue.Full:
                    if ctl["stop"]:
                        break
                    frame_q.put((idx, frame))   # block until display catches up

        def processor():
            processed = 0
            last_idx = 0
            delay = max(1, self.cfg.FRAME_SKIP)
            while not ctl["stop"]:
                with lock:
                    idx = proc_slot["idx"]
                    frame = proc_slot["frame"]
                if frame is None or (idx - last_idx) < delay:
                    if end_event.is_set():
                        break
                    time.sleep(0.05)
                    continue
                last_idx = idx
                processed += 1
                try:
                    self._process_one_frame(frame, processed)
                    anns = self._current_annotations()
                    new_events = self._consume_plate_events(
                        anns, idx, fps, processed)
                except Exception as exc:
                    print(f"[play] frame {idx} processing failed: {exc}")
                    anns = self._current_annotations()
                    new_events = []
                with lock:
                    shared["annotations"] = anns
                    shared["vehicles"] = len(self.tracks)
                    play_events.extend(new_events)
                    shared["plates"] = len(play_events)

        t_reader = threading.Thread(target=reader, name="anpr-reader", daemon=True)
        t_proc = threading.Thread(target=processor, name="anpr-processor", daemon=True)
        t_reader.start()
        t_proc.start()

        writer = None
        out_path = None
        if save_output:
            os.makedirs(self.cfg.VIDEO_DEBUG_ROOT, exist_ok=True)
            out_path = os.path.join(self.cfg.VIDEO_DEBUG_ROOT, "output.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))

        window = "AEGISVISION ANPR - VIDEO"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)

        disp_fps = 0.0
        win_t = time.time()
        win_n = 0
        frames_shown = 0
        completion_shown = False
        ocr_name = getattr(self.ocr_reader, "engine_name", self.cfg.OCR_ENGINE)

        def poll_key(in_pause=False):
            if in_pause:
                return cv2.waitKey(30) & 0xFF
            return cv2.waitKey(max(1, int(1000 / max(fps, 1.0)))) & 0xFF

        try:
            while not ctl["stop"]:
                try:
                    idx, frame = frame_q.get(timeout=0.2)
                    completion_shown = False
                except queue.Empty:
                    if end_event.is_set() and frame_q.empty():
                        if not completion_shown:
                            summary = np.zeros((height, width, 3), np.uint8)
                            lines_out = [
                                "AEGISVISION ANPR",
                                "",
                                "VIDEO COMPLETE",
                                f"Frames processed: {frames_shown}",
                                f"Vehicles detected: {len(self._seen_track_ids)}",
                                f"Confirmed plates: {len(play_events)}",
                                "",
                                "Press R to restart  |  Q / ESC to quit",
                            ]
                            for i, ln in enumerate(lines_out):
                                cv2.putText(summary, ln, (width // 2 - 260,
                                            120 + i * 45),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                            (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.imshow(window, summary)
                            completion_shown = True
                            completion_at = time.time()
                        if os.environ.get("AEGISVISION_AUTO_EXIT"):
                            if time.time() - completion_at > 3.0:
                                ctl["stop"] = True
                                continue
                        k = poll_key(in_pause=True)
                        if k in (27, ord('q'), ord('Q')):
                            ctl["stop"] = True
                        elif k in (ord('r'), ord('R')):
                            drain_q()
                            ctl["restart"] = True
                            completion_shown = False
                        continue
                    continue

                win_n += 1
                now = time.time()
                if now - win_t >= 1.0:
                    disp_fps = win_n / (now - win_t)
                    win_t = now
                    win_n = 0

                with lock:
                    anns = list(shared["annotations"])
                    panel = {
                        "fps": f"{disp_fps:.0f}",
                        "vehicles": shared["vehicles"],
                        "plates": shared["plates"],
                        "ocr": ocr_name,
                        "mode": ("night" if
                                 frame_luminance(frame) < self.cfg.NIGHT_LUMINANCE_THRESHOLD
                                 else "day"),
                    }
                annotated = self.annotate(frame, anns, panel=panel)
                cv2.imshow(window, annotated)
                if writer is not None:
                    writer.write(annotated)
                frames_shown += 1

                key = poll_key()
                if key in (27, ord('q'), ord('Q')):
                    ctl["stop"] = True
                    break
                elif key == ord(' '):
                    while not ctl["stop"]:
                        k = poll_key(in_pause=True)
                        if k in (27, ord('q'), ord('Q')):
                            ctl["stop"] = True
                            break
                        elif k == ord(' '):
                            break
                        elif k in (ord('r'), ord('R')):
                            drain_q()
                            ctl["restart"] = True
                            break
                elif key in (ord('r'), ord('R')):
                    drain_q()
                    ctl["restart"] = True
        finally:
            ctl["stop"] = True
            t_reader.join(timeout=2)
            t_proc.join(timeout=2)
            cap.release()
            if writer is not None:
                writer.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        print("\nVIDEO COMPLETE")
        print(f"Frames processed: {frames_shown}")
        print(f"Vehicles detected: {len(self._seen_track_ids)}")
        print(f"Confirmed plates: {len(play_events)}")
        if out_path:
            print(f"💾 Saved playback: {out_path}")

        os.makedirs(os.path.join(self.cfg.OUTPUT_BASE, "results"), exist_ok=True)
        results_dir = os.path.join(self.cfg.OUTPUT_BASE, "results")
        events_path, csv_path = self._write_events_files(
            play_events, results_dir, base_name, tag="play_events")
        print(f"💾 Events: {events_path}")
        print(f"💾 CSV: {csv_path}")
        return play_events


if __name__ == "__main__":
    print("Loading models (this happens once)...")
    engine = ANPREngine()
    print("✅ Models loaded\n")

    for img_path in config.TEST_IMAGES:
        print(f"--- {os.path.basename(img_path)} ---")
        res = engine.detect_image(img_path)
        for r in res:
            print(json.dumps(r, indent=2))
        print()

    print("--- video ---")
    engine.process_video(config.VIDEO_PATH, max_frames=120)