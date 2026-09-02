import os
import json
import csv
import re
from collections import defaultdict, Counter
from difflib import SequenceMatcher


# ============================================================
# AEGISVISION - ANPR
# PHASE 5.5 - RESULT STABILIZATION
# ============================================================
#
# Purpose:
#   Stabilize noisy OCR results from Phase 5.4.
#
# Input:
#   output/phase5_4/phase5_4_report.json
#
# Output:
#   output/phase5_5/phase5_5_report.json
#   output/phase5_5/final_plates.csv
#
# This phase DOES NOT modify anpr_video.py.
# ============================================================


INPUT_REPORT = os.path.join(
    "output",
    "phase5_4",
    "phase5_4_report.json"
)


OUTPUT_DIR = os.path.join(
    "output",
    "phase5_5"
)


OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "phase5_5_report.json"
)


OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "final_plates.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 12

# Similarity threshold used to group OCR readings.
SIMILARITY_THRESHOLD = 0.70

# Minimum confidence required for a strong result.
HIGH_CONFIDENCE = 60.0

# Confidence below this is considered weak.
LOW_CONFIDENCE = 20.0


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_plate(text):
    """
    Clean OCR text.

    Example:

        "ap-os jed" -> "APOSJED"
        " MWSIYSU " -> "MWSIYSU"
        "mwsiy-su" -> "MWSIYSU"
    """

    if text is None:
        return ""

    text = str(text).upper()

    # Keep only letters and numbers.
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# OCR VALIDITY CHECK
# ============================================================

def is_candidate(text):
    """
    Check whether OCR text is long enough
    to be considered as a possible number plate.
    """

    if not text:
        return False

    if len(text) < MIN_PLATE_LENGTH:
        return False

    if len(text) > MAX_PLATE_LENGTH:
        return False

    # Must contain only alphanumeric characters.
    if not re.fullmatch(
        r"[A-Z0-9]+",
        text
    ):
        return False

    # Reject strings containing only letters.
    # Most vehicle plates contain at least one number.
    if text.isalpha():
        return False

    return True


# ============================================================
# SIMILARITY
# ============================================================

def similarity(text1, text2):
    """
    Calculate similarity between two OCR readings.
    """

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# ============================================================
# GROUP SIMILAR OCR READINGS
# ============================================================

def group_similar_readings(readings):
    """
    Group similar OCR strings.

    Example:

        MWSIYSU
        MWSIVSU
        MWSWSU

    may be grouped together because they are
    similar OCR readings of the same physical plate.
    """

    groups = []

    for reading in readings:

        placed = False

        for group in groups:

            # Compare against the strongest/current
            # representative of the group.
            representative = group["representative"]

            score = similarity(
                reading["text"],
                representative
            )

            if score >= SIMILARITY_THRESHOLD:

                group["items"].append(
                    reading
                )

                placed = True

                # Update representative if the new
                # reading has higher confidence.
                if (
                    reading["confidence"]
                    >
                    group["representative_confidence"]
                ):

                    group["representative"] = (
                        reading["text"]
                    )

                    group["representative_confidence"] = (
                        reading["confidence"]
                    )

                break

        if not placed:

            groups.append(
                {
                    "representative":
                        reading["text"],

                    "representative_confidence":
                        reading["confidence"],

                    "items":
                        [reading]
                }
            )

    return groups


# ============================================================
# CALCULATE GROUP SCORE
# ============================================================

def calculate_group_score(group):
    """
    Calculate score using:

        frequency
        +
        average confidence
        +
        maximum confidence
    """

    items = group["items"]

    if not items:
        return 0.0

    confidences = [
        item["confidence"]
        for item in items
    ]

    count = len(items)

    average_confidence = (
        sum(confidences)
        /
        len(confidences)
    )

    maximum_confidence = max(
        confidences
    )

    # Weighted score.
    #
    # Repeated observations are important,
    # but confidence is also important.
    #
    score = (
        count * 20.0
        +
        average_confidence * 0.5
        +
        maximum_confidence * 0.3
    )

    return round(
        score,
        2
    )


# ============================================================
# DETERMINE FINAL STATUS
# ============================================================

def determine_status(
    readings,
    final_confidence,
    valid_seen,
    review_seen
):
    """
    Determine final plate status.
    """

    # If Phase 5.4 saw a VALID result for this
    # track, preserve that information.
    if valid_seen:

        return "VALID"

    # Strong repeated result.
    if (
        len(readings) >= 3
        and final_confidence >= HIGH_CONFIDENCE
    ):

        return "VALID"

    # Reasonable result that needs review.
    if (
        final_confidence >= 35.0
        or review_seen
    ):

        return "REVIEW"

    return "INVALID"


# ============================================================
# EXTRACT READINGS FROM PHASE 5.4 REPORT
# ============================================================

def extract_track_readings(report):
    """
    Extract all usable OCR readings from
    frame_results.

    Returns:

        {
            track_id: [
                {
                    frame,
                    text,
                    confidence,
                    status
                }
            ]
        }
    """

    tracks = defaultdict(list)

    frame_results = report.get(
        "frame_results",
        []
    )

    for frame_entry in frame_results:

        frame_number = frame_entry.get(
            "frame",
            0
        )

        result = frame_entry.get(
            "result",
            {}
        )

        plates = result.get(
            "plates",
            []
        )

        for plate in plates:

            track_id = plate.get(
                "track_id",
                plate.get(
                    "track",
                    None
                )
            )

            # Skip detections without a track.
            if track_id is None:
                continue

            raw_text = plate.get(
                "plate_number",
                plate.get(
                    "corrected_ocr_text",
                    ""
                )
            )

            text = normalize_plate(
                raw_text
            )

            confidence_data = plate.get(
                "confidence",
                {}
            )

            if isinstance(
                confidence_data,
                dict
            ):

                confidence = confidence_data.get(
                    "overall",
                    0
                )

            else:

                confidence = confidence_data

            try:

                confidence = float(
                    confidence
                )

            except (
                ValueError,
                TypeError
            ):

                confidence = 0.0

            status = plate.get(
                "status",
                "UNKNOWN"
            )

            # Ignore extremely short OCR.
            if not is_candidate(text):

                continue

            tracks[
                str(track_id)
            ].append(
                {
                    "frame":
                        frame_number,

                    "text":
                        text,

                    "confidence":
                        confidence,

                    "status":
                        status
                }
            )

    return tracks


# ============================================================
# STABILIZE ONE TRACK
# ============================================================

def stabilize_track(
    track_id,
    readings
):
    """
    Stabilize OCR results for one plate track.
    """

    if not readings:

        return None

    # Group similar OCR readings.
    groups = group_similar_readings(
        readings
    )

    # Calculate score for each group.
    for group in groups:

        group["score"] = (
            calculate_group_score(
                group
            )
        )

    # Sort strongest group first.
    groups.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best_group = groups[0]

    best_items = best_group["items"]

    # --------------------------------------------------------
    # Choose final OCR text
    # --------------------------------------------------------

    # Count occurrences.
    text_counter = Counter(
        item["text"]
        for item in best_items
    )

    # Find most frequently observed text.
    most_common = (
        text_counter
        .most_common()
    )

    # Highest frequency.
    highest_count = (
        most_common[0][1]
    )

    frequency_candidates = [
        text
        for text, count
        in most_common
        if count == highest_count
    ]

    # If multiple texts have same frequency,
    # choose the one with highest confidence.
    final_text = max(
        frequency_candidates,
        key=lambda text: max(
            item["confidence"]
            for item in best_items
            if item["text"] == text
        )
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    final_text_items = [
        item
        for item in best_items
        if item["text"] == final_text
    ]

    confidences = [
        item["confidence"]
        for item in final_text_items
    ]

    average_confidence = (
        sum(confidences)
        /
        len(confidences)
    )

    maximum_confidence = max(
        confidences
    )

    # Combine average and maximum.
    final_confidence = (
        average_confidence * 0.6
        +
        maximum_confidence * 0.4
    )

    final_confidence = round(
        final_confidence,
        2
    )

    # --------------------------------------------------------
    # Status information
    # --------------------------------------------------------

    valid_seen = any(
        item["status"] == "VALID"
        for item in readings
    )

    review_seen = any(
        item["status"] == "REVIEW"
        for item in readings
    )

    final_status = determine_status(
        readings,
        final_confidence,
        valid_seen,
        review_seen
    )

    # --------------------------------------------------------
    # Unique OCR readings
    # --------------------------------------------------------

    unique_readings = sorted(
        set(
            item["text"]
            for item in readings
        )
    )

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------

    frames = sorted(
        set(
            item["frame"]
            for item in readings
        )
    )

    return {

        "track_id":
            str(track_id),

        "final_plate":
            final_text,

        "status":
            final_status,

        "final_confidence":
            final_confidence,

        "observations":
            len(readings),

        "votes":
            highest_count,

        "unique_ocr_readings":
            unique_readings,

        "frames":
            frames,

        "average_confidence":
            round(
                average_confidence,
                2
            ),

        "maximum_confidence":
            round(
                maximum_confidence,
                2
            ),

        "group_size":
            len(best_items),

        "group_score":
            best_group["score"]
    }


# ============================================================
# MERGE DUPLICATE FINAL PLATES
# ============================================================

def merge_duplicate_final_plates(results):
    """
    Merge tracks that eventually produce the same
    final plate number.

    Example:

        Track 2 -> MWSIYSU
        Track 5 -> MWSIYSU

    becomes one final plate.
    """

    merged = {}

    for result in results:

        plate = result[
            "final_plate"
        ]

        if plate not in merged:

            merged[plate] = {
                "final_plate":
                    plate,

                "status":
                    result["status"],

                "confidence":
                    result["final_confidence"],

                "total_observations":
                    result["observations"],

                "votes":
                    result["votes"],

                "tracks":
                    [
                        result["track_id"]
                    ],

                "frames":
                    list(
                        result["frames"]
                    ),

                "ocr_readings":
                    list(
                        result[
                            "unique_ocr_readings"
                        ]
                    )
            }

        else:

            existing = merged[
                plate
            ]

            existing[
                "total_observations"
            ] += result[
                "observations"
            ]

            existing[
                "votes"
            ] += result[
                "votes"
            ]

            existing[
                "tracks"
            ].append(
                result[
                    "track_id"
                ]
            )

            existing[
                "frames"
            ].extend(
                result[
                    "frames"
                ]
            )

            existing[
                "ocr_readings"
            ].extend(
                result[
                    "unique_ocr_readings"
                ]
            )

            # Keep strongest confidence.
            if (
                result["final_confidence"]
                >
                existing["confidence"]
            ):

                existing[
                    "confidence"
                ] = result[
                    "final_confidence"
                ]

            # VALID has priority.
            if result["status"] == "VALID":

                existing[
                    "status"
                ] = "VALID"

            elif (
                result["status"] == "REVIEW"
                and
                existing["status"] != "VALID"
            ):

                existing[
                    "status"
                ] = "REVIEW"

    # Clean duplicates in arrays.
    final_results = []

    for plate, result in merged.items():

        result["tracks"] = sorted(
            set(
                result["tracks"]
            ),
            key=lambda x: int(x)
        )

        result["frames"] = sorted(
            set(
                result["frames"]
            )
        )

        result["ocr_readings"] = sorted(
            set(
                result["ocr_readings"]
            )
        )

        result["confidence"] = round(
            result["confidence"],
            2
        )

        final_results.append(
            result
        )

    # Highest confidence first.
    final_results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return final_results


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):
    """
    Save final plate results to CSV.
    """

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Plate",
                "Status",
                "Confidence",
                "Observations",
                "Votes",
                "Tracks",
                "Frames",
                "OCR Readings"
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result[
                        "final_plate"
                    ],

                    result[
                        "status"
                    ],

                    result[
                        "confidence"
                    ],

                    result[
                        "total_observations"
                    ],

                    result[
                        "votes"
                    ],

                    ",".join(
                        result[
                            "tracks"
                        ]
                    ),

                    ",".join(
                        map(
                            str,
                            result[
                                "frames"
                            ]
                        )
                    ),

                    ",".join(
                        result[
                            "ocr_readings"
                        ]
                    )
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "AEGISVISION - ANPR"
    )

    print(
        "PHASE 5.5 - RESULT STABILIZATION"
    )

    print(
        "OCR CLEANING + TRACK AGGREGATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_REPORT
    ):

        print(
            "\nERROR: Phase 5.4 report not found."
        )

        print(
            os.path.abspath(
                INPUT_REPORT
            )
        )

        print(
            "\nRun Phase 5.4 first:"
        )

        print(
            "python anpr_video.py"
        )

        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "\nInput report:"
    )

    print(
        os.path.abspath(
            INPUT_REPORT
        )
    )

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with open(
            INPUT_REPORT,
            "r",
            encoding="utf-8"
        ) as file:

            report = json.load(
                file
            )

    except Exception as error:

        print(
            "\nERROR: Could not read Phase 5.4 report."
        )

        print(
            f"Reason: {error}"
        )

        return

    # --------------------------------------------------------
    # Extract readings
    # --------------------------------------------------------

    print(
        "\nExtracting OCR readings..."
    )

    tracks = extract_track_readings(
        report
    )

    print(
        f"Plate tracks found: "
        f"{len(tracks)}"
    )

    # --------------------------------------------------------
    # Stabilize each track
    # --------------------------------------------------------

    stabilized_tracks = []

    for track_id in sorted(
        tracks.keys(),
        key=lambda x: int(x)
    ):

        readings = tracks[
            track_id
        ]

        result = stabilize_track(
            track_id,
            readings
        )

        if result:

            stabilized_tracks.append(
                result
            )

            print(
                f"\nTrack {track_id}"
            )

            print(
                f"  Final plate : "
                f"{result['final_plate']}"
            )

            print(
                f"  Status      : "
                f"{result['status']}"
            )

            print(
                f"  Confidence  : "
                f"{result['final_confidence']:.2f}%"
            )

            print(
                f"  Observations: "
                f"{result['observations']}"
            )

            print(
                f"  Votes       : "
                f"{result['votes']}"
            )

            print(
                f"  OCR readings: "
                f"{', '.join(result['unique_ocr_readings'])}"
            )

    # --------------------------------------------------------
    # Merge duplicate final plates
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "MERGING DUPLICATE FINAL PLATES"
    )

    print(
        "-" * 60
    )

    final_plates = merge_duplicate_final_plates(
        stabilized_tracks
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_plates = [
        result
        for result in final_plates
        if result["status"] == "VALID"
    ]

    review_plates = [
        result
        for result in final_plates
        if result["status"] == "REVIEW"
    ]

    invalid_plates = [
        result
        for result in final_plates
        if result["status"] == "INVALID"
    ]

    # --------------------------------------------------------
    # Create final report
    # --------------------------------------------------------

    phase54_info = report.get(
        "video_information",
        {}
    )

    final_report = {

        "project":
            "AegisVision",

        "module":
            "ANPR",

        "phase":
            "5.5",

        "phase_name":
            "Result Stabilization",

        "input_phase":
            "5.4",

        "input_report":
            os.path.abspath(
                INPUT_REPORT
            ),

        "video_information":
            phase54_info,

        "processing": {

            "minimum_plate_length":
                MIN_PLATE_LENGTH,

            "maximum_plate_length":
                MAX_PLATE_LENGTH,

            "similarity_threshold":
                SIMILARITY_THRESHOLD,

            "high_confidence_threshold":
                HIGH_CONFIDENCE,

            "low_confidence_threshold":
                LOW_CONFIDENCE
        },

        "statistics": {

            "input_tracks":
                len(tracks),

            "stabilized_tracks":
                len(
                    stabilized_tracks
                ),

            "final_unique_plates":
                len(
                    final_plates
                ),

            "valid_plates":
                len(
                    valid_plates
                ),

            "review_plates":
                len(
                    review_plates
                ),

            "invalid_plates":
                len(
                    invalid_plates
                )
        },

        "final_plates":
            final_plates,

        "track_results":
            stabilized_tracks,

        "output_files": {

            "json":
                os.path.abspath(
                    OUTPUT_JSON
                ),

            "csv":
                os.path.abspath(
                    OUTPUT_CSV
                )
        }
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    try:

        with open(
            OUTPUT_JSON,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                final_report,
                file,
                indent=4
            )

    except Exception as error:

        print(
            "\nERROR: Could not save JSON report."
        )

        print(
            f"Reason: {error}"
        )

        return

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    try:

        save_csv(
            final_plates
        )

    except Exception as error:

        print(
            "\nWARNING: Could not save CSV."
        )

        print(
            f"Reason: {error}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
    )

    print("=" * 60)

    print(
        "PHASE 5.5 FINAL SUMMARY"
    )

    print("=" * 60)

    print(
        f"Input tracks       : "
        f"{len(tracks)}"
    )

    print(
        f"Stabilized tracks  : "
        f"{len(stabilized_tracks)}"
    )

    print(
        f"Unique final plates: "
        f"{len(final_plates)}"
    )

    print(
        f"Valid plates       : "
        f"{len(valid_plates)}"
    )

    print(
        f"Review plates      : "
        f"{len(review_plates)}"
    )

    print(
        f"Invalid plates     : "
        f"{len(invalid_plates)}"
    )

    # --------------------------------------------------------
    # Final plates
    # --------------------------------------------------------

    if final_plates:

        print(
            "\nFINAL STABILIZED PLATES"
        )

        print(
            "-" * 60
        )

        for index, result in enumerate(
            final_plates,
            start=1
        ):

            print(
                f"{index}. "
                f"{result['final_plate']} | "
                f"{result['status']} | "
                f"{result['confidence']:.2f}% | "
                f"Observations: "
                f"{result['total_observations']}"
            )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print(
        "\nJSON report:"
    )

    print(
        os.path.abspath(
            OUTPUT_JSON
        )
    )

    print(
        "\nCSV report:"
    )

    print(
        os.path.abspath(
            OUTPUT_CSV
        )
    )

    print(
        "\n"
    )

    print("=" * 60)

    print(
        "PHASE 5.5 COMPLETED SUCCESSFULLY"
    )

    print(
        "ANPR results have been stabilized."
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()