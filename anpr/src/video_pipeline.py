import cv2
import os
import json
import time

from detect_plate_yolo import (
    load_plate_model,
    detect_plate_yolo,
    draw_plate_boxes
)

from run_ocr import (
    load_ocr_reader,
    run_ocr_on_plate
)

from build_output import build_anpr_result

from backend_sender import send_anpr_event


def run_video_pipeline(
    video_path,
    plate_model,
    ocr_reader,
    frame_skip=15,
    dedupe_window=10,
    save_output=True,
    min_crop_width=50
):
    """
    AegisVision ANPR Video Pipeline

    Video
        ↓
    YOLO Plate Detection
        ↓
    OCR
        ↓
    ANPR Event
        ↓
    FastAPI Backend
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    print(f"✅ Video opened: {video_path}")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    # --------------------------------------------------
    # OUTPUT FOLDERS
    # --------------------------------------------------

    os.makedirs(
        "output/videos",
        exist_ok=True
    )

    os.makedirs(
        "output/results",
        exist_ok=True
    )

    # --------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------

    writer = None
    out_path = None

    if save_output:

        base_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        out_path = os.path.join(
            "output",
            "videos",
            f"{base_name}_result.mp4"
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            out_path,
            fourcc,
            fps,
            (width, height)
        )

    # --------------------------------------------------
    # RESULT STORAGE
    # --------------------------------------------------

    all_results = []

    recent_tracks = {}

    frame_index = 0
    processed_frames = 0

    # --------------------------------------------------
    # VIDEO PROCESSING LOOP
    # --------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # Process every 15th frame
        if frame_index % frame_skip != 0:

            if writer is not None:
                writer.write(frame)

            continue

        processed_frames += 1

        # --------------------------------------------------
        # YOLO PLATE DETECTION
        # --------------------------------------------------

        detections = detect_plate_yolo(
            plate_model,
            frame,
            conf_threshold=0.25
        )

        annotated_frame = draw_plate_boxes(
            frame,
            detections
        )

        # --------------------------------------------------
        # PROCESS EACH DETECTED PLATE
        # --------------------------------------------------

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            detection_confidence = det["confidence"]

            # --------------------------------------------------
            # CROP PLATE
            # --------------------------------------------------

            plate_crop = frame[
                max(0, y1):min(
                    frame.shape[0],
                    y2
                ),
                max(0, x1):min(
                    frame.shape[1],
                    x2
                )
            ]

            if plate_crop.size == 0:
                continue

            crop_width = plate_crop.shape[1]

            if crop_width < min_crop_width:
                continue

            # --------------------------------------------------
            # OCR
            # --------------------------------------------------

            ocr_result = run_ocr_on_plate(
                ocr_reader,
                plate_crop
            )

            plate_text = ""
            ocr_confidence = 0.0

            # --------------------------------------------------
            # OCR RESULT: LIST
            # --------------------------------------------------

            if isinstance(
                ocr_result,
                list
            ):

                best_text = ""
                best_confidence = 0.0

                for item in ocr_result:

                    if isinstance(
                        item,
                        dict
                    ):

                        text = str(
                            item.get(
                                "text",
                                ""
                            )
                        ).strip()

                        try:
                            confidence = float(
                                item.get(
                                    "confidence",
                                    0.0
                                )
                            )

                        except Exception:
                            confidence = 0.0

                        if (
                            text
                            and confidence > best_confidence
                        ):

                            best_text = text
                            best_confidence = confidence

                plate_text = best_text
                ocr_confidence = best_confidence

            # --------------------------------------------------
            # OCR RESULT: TUPLE
            # --------------------------------------------------

            elif isinstance(
                ocr_result,
                tuple
            ):

                if len(ocr_result) >= 1:

                    plate_text = str(
                        ocr_result[0]
                    ).strip()

                if len(ocr_result) >= 2:

                    try:
                        ocr_confidence = float(
                            ocr_result[1]
                        )

                    except Exception:
                        ocr_confidence = 0.0

            # --------------------------------------------------
            # OCR RESULT: DICTIONARY
            # --------------------------------------------------

            elif isinstance(
                ocr_result,
                dict
            ):

                plate_text = str(
                    ocr_result.get(
                        "plate_number",
                        ocr_result.get(
                            "text",
                            ""
                        )
                    )
                ).strip()

                try:

                    ocr_confidence = float(
                        ocr_result.get(
                            "confidence",
                            0.0
                        )
                    )

                except Exception:
                    ocr_confidence = 0.0

            # --------------------------------------------------
            # OCR RESULT: TEXT
            # --------------------------------------------------

            else:

                plate_text = str(
                    ocr_result
                ).strip()

            if not plate_text:
                continue

            # --------------------------------------------------
            # CLEAN OCR TEXT
            # --------------------------------------------------

            plate_text = (
                plate_text
                .replace(
                    " ",
                    ""
                )
                .replace(
                    "\n",
                    ""
                )
                .upper()
            )

            # --------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------

            avg_confidence = max(
                detection_confidence,
                ocr_confidence
            )

            # --------------------------------------------------
            # PLATE VALIDATION
            # --------------------------------------------------

            is_valid = (
                len(plate_text) >= 6
                and len(plate_text) <= 12
            )

            # --------------------------------------------------
            # BUILD ANPR EVENT
            # --------------------------------------------------

            event = build_anpr_result(
                plate_text,
                is_valid,
                avg_confidence
            )

            # --------------------------------------------------
            # ADD EXTRA INFORMATION
            # --------------------------------------------------

            event["timestamp"] = (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
            )

            event["track_id"] = 1

            event["bbox"] = [
                x1,
                y1,
                x2,
                y2
            ]

            # --------------------------------------------------
            # PRINT EVENT
            # --------------------------------------------------

            print(
                f"[frame {frame_index}] "
                f"Track {event['track_id']}: "
                f"{json.dumps(event)}"
            )

            # --------------------------------------------------
            # SAVE LOCAL RESULT
            # --------------------------------------------------

            all_results.append(event)

            # --------------------------------------------------
            # SEND EVENT TO BACKEND
            # --------------------------------------------------

            plate_key = plate_text.upper()

            current_time = time.time()

            last_sent = recent_tracks.get(
                plate_key,
                0
            )

            if (
                current_time - last_sent
                >= dedupe_window
            ):

                success = send_anpr_event(
                    event
                )

                if success:

                    print(
                        "ANPR EVENT SENT SUCCESSFULLY"
                    )

                    print(
                        f"✅ ANPR backend event sent: "
                        f"{plate_text}"
                    )

                recent_tracks[
                    plate_key
                ] = current_time

            # --------------------------------------------------
            # DRAW OCR RESULT ON VIDEO
            # --------------------------------------------------

            label = (
                f"{plate_text} "
                f"({avg_confidence:.2f})"
            )

            cv2.putText(
                annotated_frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # --------------------------------------------------
        # SAVE ANNOTATED FRAME
        # --------------------------------------------------

        if writer is not None:

            writer.write(
                annotated_frame
            )

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    cap.release()

    if writer is not None:

        writer.release()

        print(
            f"💾 Saved annotated video: "
            f"{out_path}"
        )

    # --------------------------------------------------
    # SAVE JSON RESULTS
    # --------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    results_path = os.path.join(
        "output",
        "results",
        f"{base_name}_video_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"💾 Saved results JSON: "
        f"{results_path}"
    )

    print()

    print(
        f"✅ Done. Processed "
        f"{processed_frames} of "
        f"{total_frames} total frames."
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("Loading models...")

    plate_model = load_plate_model()

    ocr_reader = load_ocr_reader()

    print("✅ Models loaded")
    print()

    # Actual AegisVision video
    video_path = (
        r"D:\AegisVision\detection\videos\test.mp4"
    )

    run_video_pipeline(
        video_path,
        plate_model,
        ocr_reader
    )