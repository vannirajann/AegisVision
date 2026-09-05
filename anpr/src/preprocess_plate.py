import cv2
import os

def preprocess_plate(cropped_image, target_width=300):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

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
    """A softer preprocessing: grayscale + upscale + contrast boost, no thresholding."""
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = target_width / w
    resized = cv2.resize(gray, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # CLAHE = adaptive contrast enhancement, gentler than hard thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    return enhanced


def save_preprocessed(image, filename, output_dir="output_preprocessed"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"💾 Saved preprocessed: {save_path}")
    return save_path