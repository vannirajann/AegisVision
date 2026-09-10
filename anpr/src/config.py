"""Central configuration for the AegisVision ANPR engine.

All thresholds, model paths and processing knobs live here so they can be
tuned in one place without touching the detection/OCR code.
"""
import os


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


ANPR_ROOT = _here("..")

# ---------------------------------------------------------------------------
# Model paths (relative to this project, resolved automatically)
# ---------------------------------------------------------------------------
MODEL_PLATE = _here("..", "models", "yolo_plate.pt")
MODEL_VEHICLE = _here("yolov8n.pt")

# ---------------------------------------------------------------------------
# COCO classes used for vehicle detection
#   2=car  3=motorcycle  5=bus  7=truck
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]

COCO_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ---------------------------------------------------------------------------
# Detection thresholds (single configuration section — STEP 14)
# ---------------------------------------------------------------------------
VEHICLE_CONFIDENCE = 0.25   # recall-first: let SORT + plate confirmation reject noise
PLATE_CONFIDENCE = 0.20
OCR_MIN_CONFIDENCE = 0.30

# Plate box geometry filters (reject boxes that cannot be a real plate)
PLATE_MIN_AREA = 150
PLATE_MIN_ASPECT = 1.2
PLATE_MAX_ASPECT = 7.5
PLATE_MIN_OCR_WIDTH = 26      # plates narrower than this are skipped for OCR
PLATE_MAX_OCR_WIDTH = 420     # sanity cap (very wide = not a real plate crop)
PLATE_CROP_PADDING = 10       # extra px around plate box when cropping natively
PLATE_CROP_EXPAND = 0.12      # extra fraction of plate width on each side (edges)
REFINE_PLATE_CROP = False     # cosmetic re-crop to the plate rectangle; OFF by
                              # default — it measurably hurt night-vision crops

# Small/tiny vehicle boxes cannot be read — skip plate OCR on them
VEHICLE_MIN_HEIGHT = 45          # px in the (possibly scaled) processing frame
VEHICLE_MAX_IOU_DISTANCE = None  # unused; kept as extension point

# ---------------------------------------------------------------------------
# Preprocessing / OCR
# ---------------------------------------------------------------------------
OCR_UPSCALE_MIN_WIDTH = 500      # aggressive cubic upscale before OCR reads
OCR_TARGET_WIDTH = 480           # preprocessing variants upscale target
OCR_LANGUAGES = ["en"]
OCR_ENGINE = "paddle"            # "paddle" (primary) or "easyocr" (fallback)
OCR_GPU = False
OCR_CPU_THREADS = 4              # paddle CPU inference threads
OCR_PADDLE_MKLDNN = False        # keep False: paddle 3.3 + oneDNN crashes on CPU
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_MIN_CONFIDENCE = 0.30

# Plate crop quality gates (video) — crops failing these are not OCR'd.
PLATE_MIN_PX = 20                # reject crops smaller than 20x20 px
PLATE_MAX_SHARPNESS_WEIGHT = 1.0
PLATE_QUALITY_MIN = 0.25         # combined quality floor before OCR is attempted (daytime)
NIGHT_QUALITY_MIN = 0.12         # relaxed floor for low-light crops (darkness != garbage)

# Low-light / night-vision dispatch
NIGHT_LUMINANCE_THRESHOLD = 74   # mean gray below this (frame OR crop) = night path
NIGHT_GAMMA = 0.6                # gamma for the night brightness-lift variants
NIGHT_PLATE_CONFIDENCE = 0.15    # plate detection conf used on the enhanced night frame

# Variant selection per mode. Images get the full ensemble; video gets a
# cheaper primary set, plus a fallback set only when the primary produces
# nothing usable. Multiple thresholding methods (binary, trunc, tozero …)
# mirror the reference image: no single preprocessing wins for all plates.
OCR_VARIANTS_IMAGE = [
    "adaptive_threshold", "gray_plain", "gentle_clahe",
    "sharpened_clahe", "color_upscaled", "otsu_binary",
    "binary_inv", "trunc", "tozero", "tozero_inv",
]
OCR_VARIANTS_VIDEO = [
    "color_upscaled", "gentle_clahe",
]
OCR_VARIANTS_VIDEO_FALLBACK = [
    "adaptive_threshold", "otsu_binary", "sharpened_clahe",
    "binary_inv", "trunc", "tozero", "tozero_inv",
]
# Night/low-light preprocessing set. Plates are often dark-on-dark, so the
# primary variants lift brightness/contrast (gamma, strong CLAHE, min-max
# stretch) and the fallback set adds thresholding + denoise. bright/color
# variants are still offered first because some night plates are near-white.
OCR_VARIANTS_NIGHT = [
    "color_upscaled", "night_gamma", "night_clahe_strong",
    "night_contrast_norm",
]
OCR_VARIANTS_NIGHT_FALLBACK = [
    "night_adaptive", "night_otsu", "night_binary_inv",
    "night_denoise", "gentle_clahe",
]

BLUR_THRESHOLD = 60.0            # Laplacian variance; below = flagged blurry
SKIP_BLUR_CHECK = False          # True: still OCR on flagged-blurry crops

# ---------------------------------------------------------------------------
# Temporal consensus (multi-frame confirmation — STEP 8)
# ---------------------------------------------------------------------------
FRAME_SKIP = 6                   # process every Nth frame (1 = every frame)
OCR_INTERVAL = 2                 # min processed-frames between OCR per track
DEDUPE_WINDOW = 30               # processed-frame window to suppress repeat events
MIN_VALID_SIGHTINGS = 2          # corroborating valid reads before accepting a plate
CONFIRM_STABILITY_WINDOW = 6     # processed-frame window for repeated confirm votes
CONFIRM_STABILITY_REPEATS = 2    # same validated vote must recur this many times before freezing
CONFIRM_MIN_CONF = 0.55          # mean OCR confidence floor to freeze/confirm a plate
EVENT_PLATE_DEDUPE_SIM = 0.7     # same-track events with similarity >= this are one plate
CONSENSUS_SIMILARITY = 0.7       # min similarity to cluster reads (SequenceMatcher)
CHAR_VOTE_COVERAGE = 0.6         # min fraction of reads agreeing on a char position
BEST_READ_BONUS = 0.10           # extra consensus weight for reads on the best frame

# ---------------------------------------------------------------------------
# Processing / IO
# ---------------------------------------------------------------------------
VIDEO_WORK_SCALE = 1.0           # downscale processing scale factor (1.0 = native)
PLATE_FRAME_UPSCALE = 1.5        # upscale full frame before plate detection (recall)
PLATE_ROI_UPSCALE = 3.0          # upscale vehicle ROI before plate detection (video)
PLATE_ROI_FALLBACK_MIN_HEIGHT = 60   # only run ROI detection on near/large vehicles
ROI_PLATE_OCR = True             # run OCR on the best plate per vehicle per frame
MAX_OCR_PLATES_PER_FRAME = 8     # cap on OCR calls per processed frame (runtime)
# event emission: a non-validated vote must be this long & confident to emit
EVENT_MIN_TEXT_LEN = 5
EVENT_MIN_MEAN_CONFIDENCE = 0.60
SAVE_VIDEO = True
SAVE_DEBUG_CROPS = False
VIDEO_DEBUG_MODE = False              # dump per-vehicle crops + ocr.json
VIDEO_DEBUG_ROOT = os.path.join(ANPR_ROOT, "runs", "video_anpr")
VIDEO_DEBUG_SAMPLING = 4              # processed-frame step for debug dump
OUTPUT_BASE = os.path.join(ANPR_ROOT, "output")

VIDEO_PATH = os.path.join(ANPR_ROOT, "test_images", "test_video.mp4")
TEST_IMAGES = [
    os.path.join(ANPR_ROOT, "test_images", "car.jpeg"),
    os.path.join(ANPR_ROOT, "test_images", "car2.jpeg"),
    os.path.join(ANPR_ROOT, "test_images", "car3.jpeg"),
]