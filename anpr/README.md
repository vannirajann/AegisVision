ANPR Module — AegisVision

Automatic Number Plate Recognition module for the AegisVision surveillance platform.

Owner: Person 3 Folder: anpr/ Status: Working prototype (image-based). Video/live camera support planned next.

What this module does

Given an image containing a vehicle, this module:

Detects the license plate region using a YOLOv8 model.
Crops the plate out of the full image.
Preprocesses the crop (grayscale, resize, denoise, threshold) to improve OCR accuracy.
Runs OCR (EasyOCR) to extract the plate text.
Cleans the OCR output and validates it against a typical Indian plate format.
Returns a structured JSON result for downstream consumption.

Pipeline:

Image → Plate Detection (YOLO) → Crop → Preprocess → OCR → Clean/Validate → JSON Output
Folder structure
anpr/
├── requirements.txt              # Python dependencies
├── ground_truth.json             # Expected plate numbers for benchmark
├── benchmark.py                  # Accuracy measurement script
├── tests/
│   ├── __init__.py
│   ├── test_clean_text.py
│   ├── test_preprocess.py
│   ├── test_build_output.py
│   ├── test_crop_plate.py
│   └── test_pipeline_integration.py
├── src/
│   ├── __init__.py
│   ├── load_image.py              # Load/display an image (dev/test utility)
│   ├── detect_plate_yolo.py       # YOLO-based plate detection
│   ├── detect_plate.py            # Haar cascade plate detection (legacy)
│   ├── crop_plate.py              # Crop plate region from full image
│   ├── preprocess_plate.py        # Grayscale/resize/denoise/threshold before OCR
│   ├── run_ocr.py                 # EasyOCR wrapper
│   ├── clean_text.py              # Clean + validate OCR text against plate format
│   ├── build_output.py            # Build final structured JSON output
│   ├── compare_ocr_methods.py     # Compare preprocessing variants
│   ├── anpr_pipeline.py           # Full pipeline: image path in → JSON out
│   ├── video_pipeline.py          # Video file pipeline with deduplication
│   └── webcam_pipeline.py         # Live webcam pipeline with deduplication
├── models/
│   ├── yolo_plate.pt              # YOLOv8 license plate detector weights
│   └── haarcascade_russian_plate_number.xml
├── test_images/                   # Sample test images (car.jpg, bike.jpg, bus.jpg)
├── output_crops/                  # Saved cropped plate images (for inspection)
└── output/                        # Annotated images, results, and videos
Setup
bash
cd anpr
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

requirements.txt includes: opencv-contrib-python, ultralytics, huggingface_hub, easyocr, pytest.

Running tests
bash
pytest tests/ -v

Running the benchmark
Before running the benchmark, edit ground_truth.json and add the expected plate numbers
for each test image and video:

json
{
  "car.jpg": ["TN01AK6321"],
  "bike.jpg": [],
  "bus.jpg": [],
  "test_video.mp4": []
}

Then run:
bash
python benchmark.py                  # all modes
python benchmark.py --mode image     # image only
python benchmark.py --mode video     # video only

The benchmark reports exact-match rate, fuzzy-match rate, and valid-format rate.

Running the pipeline

Run the full pipeline against the sample test images:

bash
python src/anpr_pipeline.py

To use it in code:

python
from anpr_pipeline import run_anpr_pipeline
from detect_plate_yolo import load_plate_model
from run_ocr import load_ocr_reader

plate_model = load_plate_model()
ocr_reader = load_ocr_reader()

results = run_anpr_pipeline("path/to/image.jpg", plate_model, ocr_reader)
# results is a list of dicts — one per plate detected in the image

Load plate_model and ocr_reader once at startup, not per image/frame — model loading is slow, inference is fast.

Output format

Each detected plate produces one JSON object:

json
{
  "event_type": "anpr",
  "plate_number": "TN01AK6321",
  "plate_valid_format": true,
  "confidence": 0.89,
  "timestamp": "2026-09-05T11:07:31.527117+00:00"
}
Field	Description
event_type	Always "anpr" — used by the backend to route the event.
plate_number	Best-guess cleaned plate text (uppercase, alphanumeric only).
plate_valid_format	true if the text matches a standard Indian plate pattern, false if not. Downstream consumers should treat false results as lower-trust / needing review.
confidence	Average OCR confidence (0–1) across the fragments used.
timestamp	ISO 8601 UTC timestamp of when the result was generated.

This matches the common event format used across the project and is intended to be sent to Person 5's backend (POST /events).

Known limitations (current prototype)
OCR accuracy is ~70–90% depending on image quality, plate condition, and angle. Clear, well-lit, frontal plates (e.g. the car test image) read reliably (~89% confidence, correct text). Small, stylized, or bilingual plates (e.g. Tamil + English plates) are harder and may produce partial or incorrect reads.
The pipeline runs two preprocessing variants (preprocess_plate — adaptive threshold, and preprocess_plate_gentle — CLAHE contrast enhancement) per plate and automatically keeps whichever produces a valid-format or more coherent result. This measurably improved results over a single fixed preprocessing method, without needing per-image manual tuning.
plate_valid_format: false does not always mean detection failed — it can also mean the plate uses a non-standard format not covered by the current regex. This is a deliberate safety behavior: it's better to flag an uncertain read than confidently report a wrong plate number.
Currently image-only. Video frame support is the next planned step (see Development Order below).
Single-plate-per-crop assumption; if a YOLO model detects multiple plates in one image, each is processed independently and returned as a separate JSON object.
Next steps
Extend anpr_pipeline.py to accept a video frame (numpy array) directly, not just an image file path — needed for live camera integration.
Add a lightweight temporal filter for video (avoid re-reporting the same plate every frame).
Tune/replace the validation regex to support more plate formats as more real-world samples are tested.
Hand off run_anpr_pipeline() output directly to Person 5's POST /events endpoint.