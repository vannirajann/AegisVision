import cv2
import os
import numpy as np


def deskew_plate(cropped_image):
    """
    Detects and corrects rotation/skew in the plate crop using the
    minimum-area bounding rectangle of the largest contour (the plate edge).
    Falls back to the original image if no reliable contour is found —
    never distort further on failure.
    """
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image

    # Edge detection to find the plate's outline
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cropped_image

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:  # too small to trust
        return cropped_image

    rect = cv2.minAreaRect(largest)
    angle = rect[-1]

    # Normalize angle to the smallest rotation needed
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1.0:  # already straight enough, skip to avoid unnecessary distortion
        return cropped_image
    if abs(angle) > 25:  # implausible skew for a real plate read — likely a bad contour, skip
        return cropped_image

    h, w = cropped_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        cropped_image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_plate(cropped_image, target_width=300):
    # 0. Deskew first, before any grayscale/resize work
    deskewed = deskew_plate(cropped_image)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    # 2. Resize - scale up so text is bigger and clearer for OCR
    h, w = gray.shape
    scale = target_width / w
    resized = cv2.resize(gray, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 3. Reduce noise while keeping edges sharp
    denoised = cv2.bilateralFilter(resized, 11, 17, 17)

    # 4. Adaptive threshold - converts to clean black/white
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return thresh


def preprocess_plate_gentle(cropped_image, target_width=300):
    """A softer preprocessing: deskew + grayscale + upscale + contrast boost, no thresholding."""
    deskewed = deskew_plate(cropped_image)

    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = target_width / w
    resized = cv2.resize(gray, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    return enhanced


def is_blurry(cropped_image, threshold=100.0):
    """
    Laplacian-variance blur check. Returns True if the crop is too blurred
    to reliably OCR — use this to skip OCR on this frame/crop entirely.
    """
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def enhance_night_frame(frame):
    """Brightness/contrast boost for a low-light frame, preserving size and
    colour (CLAHE on the LAB luminance channel + min-max stretch). Used to
    improve plate detection recall on night-vision footage."""
    if frame is None or frame.size == 0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _gray_upscaled(cropped_image, target_width=480):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
    h, w = gray.shape[:2]
    if w <= 0:
        return gray
    scale = target_width / w
    return cv2.resize(gray, (target_width, max(1, int(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def night_gamma(cropped_image, gamma=None, target_width=480):
    """Gamma correction lifts dark plates; brighten when gamma < 1."""
    gray = _gray_upscaled(cropped_image, target_width=target_width)
    g = gamma if gamma is not None else 0.6
    lut = np.array([np.power(i / 255.0, 1.0 / g) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(gray, lut)


def night_clahe_strong(cropped_image, target_width=480):
    gray = _gray_upscaled(cropped_image, target_width=target_width)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def night_contrast_norm(cropped_image, target_width=480):
    gray = _gray_upscaled(cropped_image, target_width=target_width)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def night_adaptive(cropped_image, target_width=480):
    gray = _gray_upscaled(cropped_image, target_width=target_width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5)


def night_otsu(cropped_image, invert=False, target_width=480):
    gray = _gray_upscaled(cropped_image, target_width=target_width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh) if invert else thresh


def night_denoise_upscaled(cropped_image, target_width=480):
    """Denoise the colour crop, then upscale — keeps text pixels consistent
    so OCR sees fewer flecks on grainy night plates."""
    if len(cropped_image.shape) == 3:
        den = cv2.fastNlMeansDenoisingColored(cropped_image, None, 7, 7, 7, 21)
    else:
        den = cv2.fastNlMeansDenoising(cropped_image, None, 7, 7, 21)
    h, w = den.shape[:2]
    if w <= 0:
        return den
    scale = target_width / w
    return cv2.resize(den, (target_width, max(1, int(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def save_preprocessed(image, filename, output_dir="output_preprocessed"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"💾 Saved preprocessed: {save_path}")
    return save_path