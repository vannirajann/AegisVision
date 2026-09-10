"""VIDEO-only CLI for the AegisVision ANPREngine.

Usage:
    python src/run_video_anpr.py --input test_images/test_video.mp4
    python src/run_video_anpr.py --input clip.mp4 --debug --max-frames 300

Focuses on the offline video path (SORT tracking + temporal consensus).
Images are handled by run_anpr.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from anpr_engine import ANPREngine


def main():
    parser = argparse.ArgumentParser(description="AegisVision video ANPR")
    parser.add_argument("--input", help="video path (default: test video)")
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP,
                        help="process every Nth frame (1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N source frames (0 = entire video)")
    parser.add_argument("--ocr-interval", type=int, default=config.OCR_INTERVAL,
                        help="min processed-frames between OCR per track")
    parser.add_argument("--no-save", action="store_true",
                        help="disable writing annotated video/results")
    parser.add_argument("--show", action="store_true",
                        help="show live annotated window")
    parser.add_argument("--debug", action="store_true",
                        help=f"dump per-vehicle plate crops into "
                             f"{config.VIDEO_DEBUG_ROOT}")
    args = parser.parse_args()

    path = args.input or config.VIDEO_PATH
    if not os.path.exists(path):
        print(f"❌ Video not found: {path}")
        return 1

    config.FRAME_SKIP = max(1, args.frame_skip)
    config.OCR_INTERVAL = max(0, args.ocr_interval)
    if args.debug:
        config.VIDEO_DEBUG_MODE = True

    print("Loading models (this happens once)...")
    engine = ANPREngine()
    print("✅ Models loaded\n")

    events = engine.process_video(path, save_output=not args.no_save,
                                  show_window=args.show,
                                  max_frames=args.max_frames,
                                  debug=args.debug)
    print(f"\nConfirmed plate events: {len(events)}")
    for e in events:
        print(f"  V{e['vehicle_id']:>2} | {e['plate_text']:<12} "
              f"conf={e['ocr_confidence']:.2f} "
              f"@frame {e.get('frame_number', '?')} "
              f"({e.get('timestamp_seconds', 0):.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())