"""OCR reader for the AegisVision ANPR engine.

PaddleOCR is the PRIMARY engine (better robustness on blurred / dark /
low-resolution CCTV plates). EasyOCR is used as an automatic fallback when
PaddleOCR is unavailable or fails to initialise.

Both engines expose the same interface:

    reader.readtext(image) -> [{"text": str, "confidence": float}]

so callers (anpr_engine and the legacy pipelines) never need to know which
engine is in use.
"""
import os

import cv2
import numpy as np

_MIN_CONFIDENCE = 0.30


class _PaddleOCRReader:
    """Thin adapter around PaddleOCR 3.x (paddlex dict-based results)."""

    def __init__(self, lang="en", gpu=False, cpu_threads=4,
                 enable_mkldnn=False, min_confidence=_MIN_CONFIDENCE):
        from paddleocr import PaddleOCR

        # enable_mkldnn defaults to False: paddle 3.3.x + oneDNN raises
        # "ConvertPirAttribute2RuntimeAttribute not supported" on CPU.
        self._reader = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=enable_mkldnn,
            cpu_threads=cpu_threads,
            text_rec_score_thresh=min_confidence,
        )
        self.min_confidence = min_confidence
        self.engine_name = "paddle"

    def readtext(self, image):
        texts = []
        try:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                if image.ndim == 2:
                    # PaddleOCR (paddlex) crashes on 2D gray arrays
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            results = list(self._reader.predict(image))
        except Exception as exc:  # never let a bad frame kill the pipeline
            print(f"[paddleocr] predict failed ({exc}); treating crop as empty")
            return texts
        if not results:
            return texts
        rec_texts = results[0].get("rec_texts") or []
        rec_scores = results[0].get("rec_scores") or []
        for text, score in zip(rec_texts, rec_scores):
            if score is not None and float(score) >= self.min_confidence:
                texts.append({"text": str(text).strip(), "confidence": float(score)})
        return texts


class _EasyOCRReader:
    """Thin adapter around EasyOCR (kept as a CPU-friendly fallback)."""

    def __init__(self, lang="en", gpu=False):
        import easyocr

        self._reader = easyocr.Reader([lang], gpu=gpu)
        self.gpu = gpu
        self.engine_name = "easyocr"

    def readtext(self, image):
        if not isinstance(image, np.ndarray):
            return []
        results = self._reader.readtext(
            image,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        return [{"text": str(text).strip(), "confidence": float(conf)}
                for (_, text, conf) in results]


def load_paddle_reader(lang="en", gpu=False, cpu_threads=4, enable_mkldnn=False):
    return _PaddleOCRReader(lang=lang, gpu=gpu, cpu_threads=cpu_threads,
                            enable_mkldnn=enable_mkldnn)


def load_easyocr_reader(lang="en", gpu=False):
    return _EasyOCRReader(lang=lang, gpu=gpu)


def load_ocr_reader(engine="paddle", gpu=False, lang="en",
                    cpu_threads=4, enable_mkldnn=False):
    """Load the requested OCR reader.

    engine: "paddle" (default) or "easyocr". If "paddle" fails to
    initialise, silently falls back to EasyOCR so existing callers keep
    working on machines without PaddlePaddle.
    """
    if engine == "easyocr":
        return load_easyocr_reader(lang=lang, gpu=gpu)

    if engine == "paddle":
        try:
            return load_paddle_reader(lang=lang, gpu=gpu,
                                      cpu_threads=cpu_threads,
                                      enable_mkldnn=enable_mkldnn)
        except Exception as exc:
            print(f"[ocr] PaddleOCR init failed ({exc}); falling back to EasyOCR")
            return load_easyocr_reader(lang=lang, gpu=gpu)

    raise ValueError(f"Unknown OCR engine: {engine!r}")


def run_ocr_on_plate(reader, cropped_image, min_confidence=_MIN_CONFIDENCE):
    """Unified OCR entry point.

    Accepts any object with a ``readtext(image)`` method (a PaddleOCR or
    EasyOCR wrapper) and returns [{text, confidence}]. Returns [] on any
    failure so no bad crop can crash the video pipeline.
    """
    if cropped_image is None or getattr(cropped_image, "size", 0) == 0:
        return []
    try:
        if isinstance(cropped_image, np.ndarray) and cropped_image.dtype != np.uint8:
            cropped_image = np.clip(cropped_image, 0, 255).astype(np.uint8)
        results = reader.readtext(cropped_image)
    except Exception as exc:
        print(f"[ocr] OCR call failed ({exc}); skipping crop")
        return []
    out = []
    for r in results:
        text = str(r.get("text", "")).strip()
        conf = float(r.get("confidence", 0.0) or 0.0)
        if text and conf >= min_confidence:
            out.append({"text": text, "confidence": conf})
    return out


if __name__ == "__main__":
    reader = load_ocr_reader()
    print("✅ OCR reader ready")

    crop_files = ["car_plate0.jpg", "bike_plate0.jpg", "bus_plate0.jpg"]

    for filename in crop_files:
        path = os.path.join("output_crops", filename)
        image = cv2.imread(path)

        if image is None:
            print(f"❌ Could not load {filename}")
            continue

        print(f"--- {filename} ---")
        texts = run_ocr_on_plate(reader, image)

        if not texts:
            print("  ⚠️ No text detected")
        for t in texts:
            print(f"  📝 '{t['text']}'  (confidence: {t['confidence']:.2f})")