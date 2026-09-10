"""CLI entry point for the AegisVision ANPREngine.

Usage:
    python src/run_anpr.py image           --input test_images/car.jpeg
    python src/run_anpr.py video           --source test_images/test_video.mp4
    python src/run_anpr.py video --batch   --source test_images/test_video.mp4

The video mode opens the exact VIDEO FILE and plays it live on screen while
ANPR runs (boxes + plate text drawn over the playing video). No webcam is
used anywhere. `--batch` runs the non-interactive offline pipeline instead
(saved annotated video + events JSON).

Controls while playing:  Q / ESC = quit,  SPACE = pause/resume,
R = restart the video from the beginning.
"""
import argparse
import json
import os
import sys
import site

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def _prefer_gui_cv2():
    """Make `import cv2` resolve to a GUI-capable OpenCV without touching
    any installed package. Several machines have both opencv-python (GUI)
    and opencv-python-headless installed; the headless one usually shadows
    the GUI build and `cv2.imshow/namedWindow` then can't open a window.
    This fronts the site-packages directory that contains the non-headless
    opencv distribution so the video player window works."""
    paths = [p for p in (site.getsitepackages() + [site.getusersitepackages()])
             if p and os.path.isdir(p)]
    for p in paths:
        try:
            names = set(os.listdir(p))
        except OSError:
            continue
        if "cv2" not in names:
            continue
        gui = any(n.startswith("opencv_python-") and "headless" not in n
                  for n in names)
        headless = any(n.startswith("opencv_python_headless-") for n in names)
        if gui and not headless and p in sys.path:
            sys.path.remove(p)
            sys.path.insert(0, p)
            return True
    return False


_prefer_gui_cv2()

import config
from anpr_engine import ANPREngine


def main():
    parser = argparse.ArgumentParser(description="AegisVision ANPR engine")
    parser.add_argument("mode", choices=["image", "video"],
                        help="processing mode")
    parser.add_argument("--input", help="image path (image mode)")
    parser.add_argument("--source", help="video file path (video mode)")
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP,
                        help="process every Nth frame (1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames of the source (0 = all)")
    parser.add_argument("--no-save", action="store_true",
                        help="disable saving annotated outputs")
    parser.add_argument("--batch", action="store_true",
                        help="video mode: run offline (no live window)")
    parser.add_argument("--save", action="store_true",
                        help="video mode: also save the playback to "
                             "runs/video_anpr/output.mp4")

    args = parser.parse_args()

    config.FRAME_SKIP = max(1, args.frame_skip)

    print("Loading models (this happens once)...")
    engine = ANPREngine()
    print("✅ Models loaded\n")

    if args.mode == "image":
        path = args.input or config.TEST_IMAGES[0]
        if not os.path.exists(path):
            print(f"❌ Image not found: {path}")
            return 1
        results = engine.detect_image(path, save_output=not args.no_save)
        print(f"\nFound {len(results)} plate result(s):")
        for r in results:
            print(json.dumps(r, indent=2))
        return 0

    # video mode — the source must be a video FILE (never a webcam)
    path = args.source or config.VIDEO_PATH
    if not os.path.exists(path):
        print(f"❌ Video file not found: {path}")
        return 1
    if not os.path.isfile(path):
        print(f"❌ Not a file: {path}")
        return 1

    import cv2
    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        print(f"❌ Could not open video file (is it a valid video?): {path}")
        return 1
    probe.release()

    if args.batch:
        engine.process_video(path, save_output=not args.no_save,
                             max_frames=args.max_frames)
    else:
        engine.play_video(path, save_output=args.save and not args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())