import cv2
import os
import re
import json
import pytesseract
from paddleocr import PaddleOCR


# ============================================================
# AEGISVISION - ANPR
# PHASE 6.6 - STEP 2 REVISED
# PLATE PREPROCESSING + OCR
# ============================================================

print("=" * 70)
print("AEGISVISION - ANPR")
print("PHASE 6.6 - STEP 2 REVISED")
print("PLATE PREPROCESSING + OCR")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "output/phase6_6/detected_plate.jpg"
OUTPUT_DIR = "output/phase6_6/ocr_tests_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_PATH):

    print("\nERROR: Plate crop not found:")
    print(INPUT_PATH)

    raise SystemExit


print("\nPlate crop found:")
print(INPUT_PATH)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(INPUT_PATH)

if image is None:

    print("ERROR: Could not read image.")

    raise SystemExit


height, width = image.shape[:2]

print(
    f"Original image size: {width} x {height}"
)


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    print("\nTesseract found.")

else:

    print("\nWARNING: Tesseract not found.")


# ============================================================
# PREPROCESSING
# ============================================================

print("\nCreating preprocessing variants...")


# ------------------------------------------------------------
# 1. Large upscale
# ------------------------------------------------------------

upscaled = cv2.resize(
    image,
    None,
    fx=8,
    fy=8,
    interpolation=cv2.INTER_CUBIC
)

cv2.imwrite(
    f"{OUTPUT_DIR}/01_upscaled.jpg",
    upscaled
)


# ------------------------------------------------------------
# 2. Grayscale
# ------------------------------------------------------------

gray = cv2.cvtColor(
    upscaled,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------------------------
# 3. CLAHE contrast
# ------------------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

contrast = clahe.apply(gray)


# ------------------------------------------------------------
# 4. Denoise
# ------------------------------------------------------------

denoised = cv2.GaussianBlur(
    contrast,
    (3, 3),
    0
)


# ------------------------------------------------------------
# 5. Otsu
# ------------------------------------------------------------

_, otsu = cv2.threshold(
    denoised,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ------------------------------------------------------------
# 6. Adaptive
# ------------------------------------------------------------

adaptive = cv2.adaptiveThreshold(
    denoised,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    7
)


# ------------------------------------------------------------
# 7. Sharpen
# ------------------------------------------------------------

sharpen_kernel = cv2.array = None

sharpen_kernel = (
    __import__("numpy").array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype="float32"
    )
)

sharpen = cv2.filter2D(
    contrast,
    -1,
    sharpen_kernel
)


# ------------------------------------------------------------
# Save grayscale versions
# ------------------------------------------------------------

cv2.imwrite(
    f"{OUTPUT_DIR}/02_gray.jpg",
    gray
)

cv2.imwrite(
    f"{OUTPUT_DIR}/03_contrast.jpg",
    contrast
)

cv2.imwrite(
    f"{OUTPUT_DIR}/04_denoised.jpg",
    denoised
)

cv2.imwrite(
    f"{OUTPUT_DIR}/05_otsu.jpg",
    otsu
)

cv2.imwrite(
    f"{OUTPUT_DIR}/06_adaptive.jpg",
    adaptive
)

cv2.imwrite(
    f"{OUTPUT_DIR}/07_sharpen.jpg",
    sharpen
)


# ============================================================
# PADDLEOCR INPUTS
#
# IMPORTANT:
# PaddleOCR pipeline expects 3-channel images here.
# Therefore grayscale images are converted back to BGR.
# ============================================================

paddle_variants = {

    "upscaled": upscaled,

    "contrast": cv2.cvtColor(
        contrast,
        cv2.COLOR_GRAY2BGR
    ),

    "otsu": cv2.cvtColor(
        otsu,
        cv2.COLOR_GRAY2BGR
    ),

    "adaptive": cv2.cvtColor(
        adaptive,
        cv2.COLOR_GRAY2BGR
    ),

    "sharpen": cv2.cvtColor(
        sharpen,
        cv2.COLOR_GRAY2BGR
    )
}


# ============================================================
# TESSERACT INPUTS
# ============================================================

tesseract_variants = {

    "upscaled": upscaled,

    "gray": gray,

    "contrast": contrast,

    "otsu": otsu,

    "adaptive": adaptive,

    "sharpen": sharpen
}


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# PADDLEOCR
# ============================================================

print("\n" + "-" * 70)
print("Loading PaddleOCR...")
print("-" * 70)

try:

    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    print("PaddleOCR loaded successfully.")

except Exception as e:

    print("PaddleOCR failed to load:")
    print(e)

    ocr = None


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# PADDLEOCR TEST
# ============================================================

print("\n" + "=" * 70)
print("PADDLEOCR RESULTS")
print("=" * 70)


if ocr is not None:

    for name, img in paddle_variants.items():

        print(f"\n[{name}]")

        try:

            results = ocr.predict(img)

            found = False

            for result in results:

                data = result.json

                if callable(data):

                    data = data()

                if isinstance(data, str):

                    data = json.loads(data)

                if not isinstance(data, dict):

                    continue

                rec_texts = data.get(
                    "rec_texts",
                    []
                )

                rec_scores = data.get(
                    "rec_scores",
                    []
                )

                for i, text in enumerate(rec_texts):

                    confidence = 0.0

                    if i < len(rec_scores):

                        try:

                            confidence = float(
                                rec_scores[i]
                            )

                        except Exception:

                            confidence = 0.0

                    cleaned = clean_text(text)

                    if cleaned:

                        print(
                            f"  Text       : {text}"
                        )

                        print(
                            f"  Cleaned    : {cleaned}"
                        )

                        print(
                            f"  Confidence : "
                            f"{confidence * 100:.2f}%"
                        )

                        all_results.append({

                            "engine": "PaddleOCR",

                            "variant": name,

                            "text": cleaned,

                            "confidence": confidence

                        })

                        found = True

            if not found:

                print("  No text detected.")

        except Exception as e:

            print(
                f"  ERROR: "
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# TESSERACT
# ============================================================

print("\n" + "=" * 70)
print("TESSERACT RESULTS")
print("=" * 70)


psm_modes = [6, 7, 8, 13]


for name, img in tesseract_variants.items():

    for psm in psm_modes:

        print(
            f"\n[{name} | PSM {psm}]"
        )

        try:

            config = (
                f"--oem 3 "
                f"--psm {psm} "
                f"-c "
                f"tessedit_char_whitelist="
                f"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            text = pytesseract.image_to_string(
                img,
                config=config
            )

            cleaned = clean_text(text)

            print(
                f"  Raw text : {text.strip()}"
            )

            print(
                f"  Cleaned  : {cleaned}"
            )

            if cleaned:

                all_results.append({

                    "engine": "Tesseract",

                    "variant": name,

                    "psm": psm,

                    "text": cleaned,

                    "confidence": 0.0

                })

        except Exception as e:

            print(
                f"  ERROR: "
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6.6 STEP 2 REVISED RESULT")
print("=" * 70)


if all_results:

    print(
        f"\nTotal OCR readings: "
        f"{len(all_results)}"
    )

    print("\nOCR readings:")

    for result in all_results:

        print(
            f"  {result['engine']:10} | "
            f"{result['variant']:10} | "
            f"{result['text']:15}"
        )

else:

    print("\nNO OCR TEXT FOUND.")


# ============================================================
# FIND MOST FREQUENT TEXT
# ============================================================

from collections import Counter

texts = [
    r["text"]
    for r in all_results
    if r.get("text")
]


if texts:

    counts = Counter(texts)

    print("\n" + "-" * 70)
    print("OCR VOTING")
    print("-" * 70)

    for text, count in counts.most_common():

        print(
            f"{text:20} -> {count} reading(s)"
        )

    best_text, best_count = (
        counts.most_common(1)[0]
    )

    print("\nMost common OCR result:")
    print(f"Plate : {best_text}")
    print(f"Votes : {best_count}")

else:

    best_text = ""
    best_count = 0


# ============================================================
# SAVE JSON
# ============================================================

result_path = (
    f"{OUTPUT_DIR}/ocr_test_results.json"
)

with open(
    result_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "input": INPUT_PATH,
            "original_size": [
                width,
                height
            ],
            "results": all_results,
            "most_common_text": best_text,
            "votes": best_count
        },
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nJSON:")
print(result_path)

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)