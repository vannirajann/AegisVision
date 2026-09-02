import os
import cv2
import json
import traceback
from collections import defaultdict, Counter

from anpr_engine import process_image


# ============================================================
# AEGISVISION - ANPR
# PHASE 5.4 - VIDEO / FRAME PROCESSING
# MULTI-FRAME OCR VOTING
# ============================================================

INPUT_VIDEO = os.path.join(
    "input",
    "test_video.mp4"
)

OUTPUT_DIR = os.path.join(
    "output",
    "phase5_4"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "anpr_result.mp4"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "phase5_4_report.json"
)

# Process one frame every N frames
FRAME_SKIP = 5

# Number of previous OCR observations used for voting
VOTING_HISTORY = 15

# Minimum number of characters required
MIN_PLATE_LENGTH = 5


# ============================================================
# TRACK STORAGE
# ============================================================

plate_tracks = {}

next_track_id = 1


# ============================================================
# CLEAN PLATE TEXT
# ============================================================

def normalize_plate(text):
    """
    Normalize OCR plate text.
    """

    if not text:
        return ""

    return "".join(
        character
        for character in str(text).upper()
        if character.isalnum()
    )


# ============================================================
# PLATE SIMILARITY
# ============================================================

def plate_similarity(text1, text2):
    """
    Calculate simple character similarity between
    two OCR results.
    """

    if not text1 or not text2:
        return 0.0

    length = min(
        len(text1),
        len(text2)
    )

    if length == 0:
        return 0.0

    matches = 0

    for index in range(length):

        if text1[index] == text2[index]:
            matches += 1

    return matches / max(
        len(text1),
        len(text2)
    )


# ============================================================
# FIND EXISTING TRACK
# ============================================================

def find_track(plate_text):
    """
    Try to associate OCR text with an existing plate track.
    """

    global next_track_id

    plate_text = normalize_plate(
        plate_text
    )

    if not plate_text:
        return None

    best_track = None
    best_similarity = 0.0

    for track_id, track in plate_tracks.items():

        history = track.get(
            "history",
            []
        )

        if not history:
            continue

        previous_text = history[-1].get(
            "text",
            ""
        )

        similarity = plate_similarity(
            plate_text,
            previous_text
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_track = track_id

    # --------------------------------------------------------
    # Existing track
    # --------------------------------------------------------

    if (
        best_track is not None
        and best_similarity >= 0.55
    ):

        return best_track

    # --------------------------------------------------------
    # Create new track
    # --------------------------------------------------------

    track_id = next_track_id

    next_track_id += 1

    plate_tracks[track_id] = {
        "history": [],
        "frames": [],
        "best_text": "",
        "best_confidence": 0.0,
        "status": "UNKNOWN"
    }

    print(
        f"    New plate track: {track_id}"
    )

    return track_id


# ============================================================
# ADD OCR OBSERVATION
# ============================================================

def add_observation(
    track_id,
    frame_number,
    text,
    confidence,
    status
):
    """
    Add OCR observation to a plate track.
    """

    if track_id not in plate_tracks:
        return

    text = normalize_plate(
        text
    )

    if not text:
        return

    track = plate_tracks[
        track_id
    ]

    observation = {
        "frame": frame_number,
        "text": text,
        "confidence": float(
            confidence
        ),
        "status": status
    }

    track["history"].append(
        observation
    )

    track["frames"].append(
        frame_number
    )

    # --------------------------------------------------------
    # Keep recent observations only
    # --------------------------------------------------------

    if len(track["history"]) > VOTING_HISTORY:

        track["history"] = (
            track["history"]
            [-VOTING_HISTORY:]
        )

        track["frames"] = (
            track["frames"]
            [-VOTING_HISTORY:]
        )


# ============================================================
# MULTI-FRAME OCR VOTING
# ============================================================

def get_voted_plate(track_id):
    """
    Select the most reliable OCR plate using
    multi-frame voting.

    The score considers:
        - frequency
        - OCR confidence
    """

    if track_id not in plate_tracks:
        return {
            "plate_number": "",
            "confidence": 0.0,
            "votes": 0,
            "status": "FAILED"
        }

    history = plate_tracks[
        track_id
    ].get(
        "history",
        []
    )

    if not history:

        return {
            "plate_number": "",
            "confidence": 0.0,
            "votes": 0,
            "status": "FAILED"
        }

    # --------------------------------------------------------
    # Count OCR results
    # --------------------------------------------------------

    votes = Counter()

    confidence_sum = defaultdict(
        float
    )

    confidence_count = defaultdict(
        int
    )

    for observation in history:

        text = observation[
            "text"
        ]

        confidence = observation[
            "confidence"
        ]

        if not text:
            continue

        votes[text] += 1

        confidence_sum[text] += (
            confidence
        )

        confidence_count[text] += 1

    if not votes:

        return {
            "plate_number": "",
            "confidence": 0.0,
            "votes": 0,
            "status": "FAILED"
        }

    # --------------------------------------------------------
    # Calculate voting score
    # --------------------------------------------------------

    candidates = []

    for text, vote_count in votes.items():

        average_confidence = (
            confidence_sum[text]
            /
            confidence_count[text]
        )

        # Frequency has more importance than
        # one isolated high-confidence result.

        score = (
            vote_count * 100
            + average_confidence
        )

        candidates.append(
            (
                score,
                text,
                vote_count,
                average_confidence
            )
        )

    candidates.sort(
        reverse=True
    )

    (
        _,
        best_text,
        best_votes,
        best_confidence
    ) = candidates[0]

    # --------------------------------------------------------
    # Determine status
    # --------------------------------------------------------

    if best_votes >= 3:

        status = "VOTED"

    elif best_confidence >= 60:

        status = "HIGH_CONFIDENCE"

    else:

        status = "REVIEW"

    return {
        "plate_number": best_text,
        "confidence": round(
            best_confidence,
            2
        ),
        "votes": best_votes,
        "status": status
    }


# ============================================================
# DRAW ANPR RESULTS
# ============================================================

def draw_results(
    frame,
    result,
    frame_number
):
    """
    Draw ANPR detection results.
    """

    for plate in result.get(
        "plates",
        []
    ):

        bbox = plate.get(
            "bounding_box"
        )

        if not bbox:
            continue

        x1 = int(
            bbox.get(
                "x1",
                0
            )
        )

        y1 = int(
            bbox.get(
                "y1",
                0
            )
        )

        x2 = int(
            bbox.get(
                "x2",
                0
            )
        )

        y2 = int(
            bbox.get(
                "y2",
                0
            )
        )

        plate_number = plate.get(
            "plate_number",
            ""
        )

        if not plate_number:

            plate_number = (
                plate.get(
                    "corrected_ocr_text",
                    "UNKNOWN"
                )
            )

        status = plate.get(
            "status",
            "UNKNOWN"
        )

        confidence = plate.get(
            "confidence",
            {}
        ).get(
            "overall",
            0
        )

        # ----------------------------------------------------
        # Display color
        # ----------------------------------------------------

        if status == "VALID":

            color = (
                0,
                255,
                0
            )

        elif status == "REVIEW":

            color = (
                0,
                255,
                255
            )

        else:

            color = (
                0,
                0,
                255
            )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{plate_number} | "
            f"{status} | "
            f"{confidence:.1f}%"
        )

        text_y = max(
            y1 - 10,
            30
        )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )[0]

        cv2.rectangle(
            frame,
            (
                x1,
                text_y
                - text_size[1]
                - 10
            ),
            (
                x1
                + text_size[0]
                + 10,
                text_y + 5
            ),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (
                x1 + 5,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                0,
                0
            ),
            2,
            cv2.LINE_AA
        )

    return frame


# ============================================================
# DRAW SYSTEM INFORMATION
# ============================================================

def draw_system_info(
    frame,
    frame_number,
    processed_frames
):
    """
    Draw ANPR system information.
    """

    cv2.putText(
        frame,
        "AEGISVISION - ANPR",
        (
            30,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (
            30,
            80
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Processed: {processed_frames}",
        (
            30,
            110
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "AEGISVISION - ANPR"
    )

    print(
        "PHASE 5.4 - VIDEO / FRAME PROCESSING"
    )

    print(
        "MULTI-FRAME OCR VOTING"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_VIDEO
    ):

        print(
            "\nERROR: Video not found:"
        )

        print(
            os.path.abspath(
                INPUT_VIDEO
            )
        )

        return

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "\nTesting video:"
    )

    print(
        os.path.abspath(
            INPUT_VIDEO
        )
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )

    if not cap.isOpened():

        print(
            "\nERROR: Could not open video."
        )

        return

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        fps = 25.0

    duration = (
        total_frames / fps
    )

    print(
        "\nVIDEO INFORMATION"
    )

    print("-" * 60)

    print(
        f"Video: "
        f"{os.path.basename(INPUT_VIDEO)}"
    )

    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    print(
        f"FPS: "
        f"{fps:.2f}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )

    print(
        f"Duration: "
        f"{duration:.2f} seconds"
    )

    print(
        f"Frame skip: "
        f"{FRAME_SKIP}"
    )

    print(
        f"Voting history: "
        f"{VOTING_HISTORY}"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        print(
            "\nERROR: Could not create output video."
        )

        cap.release()

        return

    # --------------------------------------------------------
    # Processing variables
    # --------------------------------------------------------

    frame_number = 0

    processed_frames = 0

    total_detections = 0

    valid_plates = set()

    frame_results = []

    last_result = {
        "plates": []
    }

    detection_display_frames = 0

    # Reset tracks
    plate_tracks.clear()

    global next_track_id

    next_track_id = 1

    # --------------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------------

    try:

        print(
            "\nPROCESSING VIDEO"
        )

        print("-" * 60)

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            frame_number += 1

            # ------------------------------------------------
            # Process selected frames
            # ------------------------------------------------

            if (
                frame_number
                % FRAME_SKIP
                == 0
            ):

                processed_frames += 1

                print(
                    f"\nFrame "
                    f"{frame_number} | "
                    f"Processed "
                    f"{processed_frames}"
                )

                try:

                    # ========================================
                    # IMPORTANT
                    # ========================================
                    #
                    # process_image() expects:
                    #
                    # OpenCV BGR image
                    #
                    # Therefore pass FRAME directly.
                    #
                    # ========================================

                    result = process_image(
                        frame
                    )

                    detected_count = result.get(
                        "plate_count",
                        0
                    )

                    total_detections += (
                        detected_count
                    )

                    print(
                        f"Detected plates: "
                        f"{detected_count}"
                    )

                    # ------------------------------------------------
                    # Process detected plates
                    # ------------------------------------------------

                    for plate in result.get(
                        "plates",
                        []
                    ):

                        number = normalize_plate(
                            plate.get(
                                "plate_number",
                                ""
                            )
                        )

                        status = plate.get(
                            "status",
                            "UNKNOWN"
                        )

                        confidence = plate.get(
                            "confidence",
                            {}
                        ).get(
                            "overall",
                            0
                        )

                        print(
                            f"  Plate "
                            f"{plate.get('plate_id')}"
                        )

                        print(
                            f"    Number: "
                            f"{number}"
                        )

                        print(
                            f"    Status: "
                            f"{status}"
                        )

                        print(
                            f"    Overall confidence: "
                            f"{confidence}"
                        )

                        # ------------------------------------------------
                        # Ignore extremely short OCR
                        # ------------------------------------------------

                        if len(number) < MIN_PLATE_LENGTH:

                            print(
                                "    OCR ignored: "
                                "too short"
                            )

                            continue

                        # ------------------------------------------------
                        # Find/create track
                        # ------------------------------------------------

                        track_id = find_track(
                            number
                        )

                        if track_id is None:

                            continue

                        add_observation(
                            track_id,
                            frame_number,
                            number,
                            confidence,
                            status
                        )

                        print(
                            f"    Track: "
                            f"{track_id}"
                        )

                        # ------------------------------------------------
                        # Voting result
                        # ------------------------------------------------

                        voted = get_voted_plate(
                            track_id
                        )

                        voted_number = voted.get(
                            "plate_number",
                            ""
                        )

                        voted_confidence = voted.get(
                            "confidence",
                            0
                        )

                        votes = voted.get(
                            "votes",
                            0
                        )

                        print(
                            f"    Multi-frame result: "
                            f"{voted_number}"
                        )

                        print(
                            f"    Votes: "
                            f"{votes}"
                        )

                        print(
                            f"    Voting confidence: "
                            f"{voted_confidence}"
                        )

                        # ------------------------------------------------
                        # Store voting result
                        # ------------------------------------------------

                        plate["track_id"] = (
                            track_id
                        )

                        plate[
                            "multi_frame_result"
                        ] = {

                            "plate_number":
                                voted_number,

                            "confidence":
                                voted_confidence,

                            "votes":
                                votes,

                            "status":
                                voted.get(
                                    "status",
                                    "REVIEW"
                                )
                        }

                        # ------------------------------------------------
                        # Valid plate
                        # ------------------------------------------------

                        if (
                            status == "VALID"
                            and number
                        ):

                            valid_plates.add(
                                number
                            )

                        if (
                            votes >= 3
                            and voted_number
                        ):

                            valid_plates.add(
                                voted_number
                            )

                    # ------------------------------------------------
                    # Save result
                    # ------------------------------------------------

                    last_result = result

                    detection_display_frames = (
                        FRAME_SKIP
                    )

                    frame_results.append(
                        {
                            "frame":
                                frame_number,

                            "result":
                                result
                        }
                    )

                except Exception as frame_error:

                    print(
                        f"WARNING: Frame "
                        f"{frame_number} "
                        f"processing failed."
                    )

                    print(
                        f"Reason: "
                        f"{frame_error}"
                    )

                    traceback.print_exc()

            # ------------------------------------------------
            # Draw results
            # ------------------------------------------------

            if (
                detection_display_frames
                > 0
            ):

                frame = draw_results(
                    frame,
                    last_result,
                    frame_number
                )

                detection_display_frames -= 1

            # ------------------------------------------------
            # System information
            # ------------------------------------------------

            frame = draw_system_info(
                frame,
                frame_number,
                processed_frames
            )

            # ------------------------------------------------
            # Write frame
            # ------------------------------------------------

            writer.write(
                frame
            )

        # ====================================================
        # CLEANUP
        # ====================================================

        cap.release()

        writer.release()

        # ====================================================
        # FINAL TRACK RESULTS
        # ====================================================

        track_results = []

        for track_id, track in plate_tracks.items():

            voted = get_voted_plate(
                track_id
            )

            track_results.append(
                {
                    "track_id":
                        track_id,

                    "plate_number":
                        voted.get(
                            "plate_number",
                            ""
                        ),

                    "confidence":
                        voted.get(
                            "confidence",
                            0
                        ),

                    "votes":
                        voted.get(
                            "votes",
                            0
                        ),

                    "status":
                        voted.get(
                            "status",
                            "FAILED"
                        ),

                    "frames":
                        track.get(
                            "frames",
                            []
                        ),

                    "observations":
                        track.get(
                            "history",
                            []
                        )
                }
            )

        # ====================================================
        # JSON REPORT
        # ====================================================

        report = {

            "project":
                "AegisVision",

            "module":
                "ANPR",

            "phase":
                "5.4",

            "phase_name":
                "Video / Frame Processing",

            "feature":
                "Multi-Frame OCR Voting",

            "input_video":
                os.path.basename(
                    INPUT_VIDEO
                ),

            "video_information": {

                "width":
                    width,

                "height":
                    height,

                "fps":
                    round(
                        fps,
                        2
                    ),

                "total_frames":
                    total_frames,

                "duration_seconds":
                    round(
                        duration,
                        2
                    )
            },

            "processing": {

                "frame_skip":
                    FRAME_SKIP,

                "processed_frames":
                    processed_frames,

                "voting_history":
                    VOTING_HISTORY
            },

            "statistics": {

                "total_detection_events":
                    total_detections,

                "unique_valid_plates":
                    len(
                        valid_plates
                    ),

                "plate_tracks":
                    len(
                        plate_tracks
                    )
            },

            "valid_plates":
                sorted(
                    list(
                        valid_plates
                    )
                ),

            "track_results":
                track_results,

            "frame_results":
                frame_results,

            "output_video":
                os.path.abspath(
                    OUTPUT_VIDEO
                )
        }

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        with open(
            OUTPUT_JSON,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print("\n")

        print("=" * 60)

        print(
            "PHASE 5.4 VIDEO SUMMARY"
        )

        print("=" * 60)

        print(
            f"Total frames       : "
            f"{total_frames}"
        )

        print(
            f"Processed frames   : "
            f"{processed_frames}"
        )

        print(
            f"Detection events   : "
            f"{total_detections}"
        )

        print(
            f"Plate tracks       : "
            f"{len(plate_tracks)}"
        )

        print(
            f"Unique valid plates: "
            f"{len(valid_plates)}"
        )

        # ----------------------------------------------------
        # Track summary
        # ----------------------------------------------------

        print(
            "\nMULTI-FRAME RESULTS"
        )

        print("-" * 60)

        for track in track_results:

            print(
                f"Track "
                f"{track['track_id']}: "
                f"{track['plate_number']} "
                f"| Votes: "
                f"{track['votes']} "
                f"| Confidence: "
                f"{track['confidence']}"
            )

        # ----------------------------------------------------
        # Valid plates
        # ----------------------------------------------------

        if valid_plates:

            print(
                "\nVALID PLATES"
            )

            for plate in sorted(
                valid_plates
            ):

                print(
                    f"  {plate}"
                )

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        print(
            "\nOutput video:"
        )

        print(
            os.path.abspath(
                OUTPUT_VIDEO
            )
        )

        print(
            "\nJSON report:"
        )

        print(
            os.path.abspath(
                OUTPUT_JSON
            )
        )

        print("\n")

        print("=" * 60)

        print(
            "PHASE 5.4 COMPLETED SUCCESSFULLY"
        )

        print(
            "Video ANPR processing is ready."
        )

        print("=" * 60)

    except Exception as error:

        cap.release()

        writer.release()

        print("\n")

        print("=" * 60)

        print(
            "PHASE 5.4 FAILED"
        )

        print("=" * 60)

        print(
            f"Error: {error}"
        )

        traceback.print_exc()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()