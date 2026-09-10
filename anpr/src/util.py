import csv
import cv2
from clean_text import clean_ocr_fragments


def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
                          'license_plate_bbox_score', 'license_number', 'license_number_score'])
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                entry = results[frame_nmr][car_id]
                if 'car' in entry and 'license_plate' in entry and 'text' in entry['license_plate']:
                    writer.writerow([
                        frame_nmr, car_id,
                        '[{} {} {} {}]'.format(*entry['car']['bbox']),
                        '[{} {} {} {}]'.format(*entry['license_plate']['bbox']),
                        entry['license_plate']['bbox_score'],
                        entry['license_plate']['text'],
                        entry['license_plate']['text_score']
                    ])


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1


def upscale_for_ocr(crop, min_width=280):
    """
    Plate crops from this video are as small as 40x13px — far too small
    for OCR. Upscale aggressively with cubic interpolation before reading.
    """
    h, w = crop.shape[:2]
    if w == 0 or h == 0:
        return crop
    if w < min_width:
        scale = min_width / w
        new_w = min_width
        new_h = max(1, int(h * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return crop


def enhance_for_ocr(crop):
    """Grayscale + contrast boost + sharpen — helps OCR on upscaled, originally-tiny crops."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharpened


def read_license_plate(license_plate_crop, ocr_reader):
    """
    Upscales + enhances the (often tiny) crop before OCR, tries strict
    validated cleaning first, falls back to best raw read above a low
    confidence floor.
    """
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    upscaled = upscale_for_ocr(license_plate_crop, min_width=280)
    enhanced = enhance_for_ocr(upscaled)

    # Try both the enhanced (grayscale/sharpened) and plain upscaled color
    # version — different plates respond better to one or the other.
    detections_enhanced = ocr_reader.readtext(
        enhanced, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    detections_color = ocr_reader.readtext(
        upscaled, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    all_detections = detections_enhanced + detections_color
    if not all_detections:
        return None, None

    ocr_texts = [{"text": text, "confidence": conf} for (_, text, conf) in all_detections]

    text, is_valid = clean_ocr_fragments(ocr_texts, min_confidence=0.25)
    if is_valid and text:
        best_conf = max((t["confidence"] for t in ocr_texts), default=0.0)
        return text, best_conf

    best = max(ocr_texts, key=lambda t: t["confidence"])
    if best["confidence"] >= 0.25 and len(best["text"]) >= 4:
        cleaned = ''.join(ch for ch in best["text"].upper() if ch.isalnum())
        return cleaned, best["confidence"]

    return None, None