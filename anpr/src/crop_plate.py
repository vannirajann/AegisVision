import cv2
import os
import numpy as np


def crop_plate(image, bbox, padding=8):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def refine_plate_crop(cropped_image, min_aspect=2.0, max_aspect=6.0):
    """
    Tightens a loose YOLO crop around the actual plate rectangle using
    classical edge/contour detection. Plates are high-contrast rectangles
    with a specific aspect ratio — this finds that rectangle inside the
    loose box and re-crops to it. Falls back to the original crop if no
    confident rectangle is found (never makes the crop worse).
    """
    if cropped_image is None or cropped_image.size == 0:
        return cropped_image

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cropped_image

    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    best_box = None
    best_area = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        area = w * h
        aspect = w / h

        # Must look like a plate: right aspect ratio, reasonable size
        # relative to the crop (not the whole crop, not tiny noise)
        if (min_aspect <= aspect <= max_aspect and
                0.15 * img_area <= area <= 0.95 * img_area):
            if area > best_area:
                best_area = area
                best_box = (x, y, x + w, y + h)

    if best_box is None:
        return cropped_image  # nothing confident found — keep original

    x1, y1, x2, y2 = best_box
    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)

    refined = cropped_image[y1:y2, x1:x2]
    if refined.size == 0:
        return cropped_image
    return refined


def save_crop(cropped_image, filename, output_dir="output_crops"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cropped_image)
    print(f"💾 Saved crop: {save_path}")
    return save_path