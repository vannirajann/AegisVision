import os
import json
import csv
import traceback


# ============================================================
# AEGISVISION - ANPR
# PHASE 5.6 - FINAL ANPR DECISION & EXPORT
# ============================================================

INPUT_REPORT = os.path.join(
    "output",
    "phase5_5",
    "phase5_5_report.json"
)

OUTPUT_DIR = os.path.join(
    "output",
    "phase5_6"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "phase5_6_report.json"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "final_anpr_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Confidence thresholds
VALID_CONFIDENCE = 50.0
REVIEW_CONFIDENCE = 30.0

# Minimum number of observations for extra reliability
MULTI_FRAME_BONUS = 2.0

# Maximum reliability score
MAX_RELIABILITY = 100.0


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_plate_text(text):
    """
    Clean OCR plate text.
    """

    if text is None:
        return ""

    text = str(text).upper().strip()

    # Remove spaces and common separators
    text = (
        text
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )

    return text


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(value, default=0.0):

    try:
        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):

    try:
        return int(value)

    except Exception:
        return default


# ============================================================
# CALCULATE RELIABILITY
# ============================================================

def calculate_reliability(
    confidence,
    observations,
    votes,
    status
):
    """
    Calculate final reliability.

    The original Phase 5.5 confidence is preserved.
    Multi-frame observations provide a small bonus.
    Voting consistency also contributes slightly.
    """

    confidence = safe_float(
        confidence
    )

    observations = safe_int(
        observations
    )

    votes = safe_int(
        votes
    )

    reliability = confidence

    # --------------------------------------------------------
    # Multi-frame observation bonus
    # --------------------------------------------------------

    if observations > 1:

        reliability += min(
            observations - 1,
            5
        ) * MULTI_FRAME_BONUS

    # --------------------------------------------------------
    # Voting bonus
    # --------------------------------------------------------

    if votes > 1:

        reliability += min(
            votes - 1,
            3
        )

    # --------------------------------------------------------
    # Valid result should retain its confidence
    # but reliability cannot exceed 100
    # --------------------------------------------------------

    reliability = min(
        reliability,
        MAX_RELIABILITY
    )

    return round(
        reliability,
        2
    )


# ============================================================
# FINAL STATUS
# ============================================================

def determine_status(
    original_status,
    confidence,
    observations,
    votes
):
    """
    Decide final ANPR status.

    IMPORTANT:
    If Phase 5.5 already marked a result VALID,
    preserve VALID unless confidence is completely unusable.

    This prevents accidental VALID -> REVIEW changes.
    """

    original_status = str(
        original_status or ""
    ).upper()

    confidence = safe_float(
        confidence
    )

    observations = safe_int(
        observations
    )

    votes = safe_int(
        votes
    )

    # --------------------------------------------------------
    # Preserve VALID from Phase 5.5
    # --------------------------------------------------------

    if original_status == "VALID":

        return "VALID"

    # --------------------------------------------------------
    # Strong confidence can become VALID
    # --------------------------------------------------------

    if (
        confidence >= VALID_CONFIDENCE
        and (
            observations >= 2
            or votes >= 2
        )
    ):

        return "VALID"

    # --------------------------------------------------------
    # Review range
    # --------------------------------------------------------

    if confidence >= REVIEW_CONFIDENCE:

        return "REVIEW"

    # --------------------------------------------------------
    # Low confidence
    # --------------------------------------------------------

    return "INVALID"


# ============================================================
# EXTRACT FINAL PLATES
# ============================================================

def extract_final_plates(data):

    """
    Extract stabilized results from Phase 5.5.

    Supports the structure produced by the current
    anpr_result_stabilizer.py.
    """

    # --------------------------------------------------------
    # Preferred structure
    # --------------------------------------------------------

    if isinstance(
        data.get("stabilized_tracks"),
        list
    ):

        return data[
            "stabilized_tracks"
        ]

    # --------------------------------------------------------
    # Alternative structure
    # --------------------------------------------------------

    if isinstance(
        data.get("final_plates"),
        list
    ):

        return data[
            "final_plates"
        ]

    # --------------------------------------------------------
    # Another possible structure
    # --------------------------------------------------------

    if isinstance(
        data.get("tracks"),
        list
    ):

        return data[
            "tracks"
        ]

    return []


# ============================================================
# NORMALIZE TRACK
# ============================================================

def normalize_track(track, fallback_id):

    """
    Convert Phase 5.5 track information into a
    consistent structure.
    """

    track_id = (
        track.get("track_id")
        if track.get("track_id") is not None
        else track.get("id")
    )

    if track_id is None:

        track_id = fallback_id

    # --------------------------------------------------------
    # Plate number
    # --------------------------------------------------------

    plate = (
        track.get("final_plate")
        or track.get("plate_number")
        or track.get("plate")
        or track.get("text")
        or ""
    )

    plate = clean_plate_text(
        plate
    )

    # --------------------------------------------------------
    # Original plate
    # --------------------------------------------------------

    original_plate = (
        track.get("original_plate")
        or track.get("plate_number")
        or plate
    )

    original_plate = clean_plate_text(
        original_plate
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = (
        track.get("status")
        or track.get("final_status")
        or "REVIEW"
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = track.get(
        "confidence"
    )

    if isinstance(
        confidence,
        dict
    ):

        confidence = confidence.get(
            "overall",
            0
        )

    if confidence is None:

        confidence = track.get(
            "voting_confidence",
            track.get(
                "final_confidence",
                0
            )
        )

    confidence = safe_float(
        confidence
    )

    # --------------------------------------------------------
    # Observations
    # --------------------------------------------------------

    observations = track.get(
        "observations"
    )

    if observations is None:

        observations = track.get(
            "observation_count",
            0
        )

    # If observations is actually a list
    if isinstance(
        observations,
        list
    ):

        observations = len(
            observations
        )

    observations = safe_int(
        observations
    )

    # --------------------------------------------------------
    # Votes
    # --------------------------------------------------------

    votes = track.get(
        "votes",
        track.get(
            "vote_count",
            0
        )
    )

    votes = safe_int(
        votes
    )

    # --------------------------------------------------------
    # OCR readings
    # --------------------------------------------------------

    ocr_readings = track.get(
        "ocr_readings",
        []
    )

    if not isinstance(
        ocr_readings,
        list
    ):

        ocr_readings = []

    # --------------------------------------------------------
    # Calculate reliability
    # --------------------------------------------------------

    reliability = calculate_reliability(
        confidence,
        observations,
        votes,
        status
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    final_status = determine_status(
        status,
        confidence,
        observations,
        votes
    )

    return {

        "track_id":
            track_id,

        "original_plate":
            original_plate,

        "final_plate":
            plate,

        "status":
            final_status,

        "original_status":
            str(status).upper(),

        "confidence":
            round(
                confidence,
                2
            ),

        "reliability":
            reliability,

        "observations":
            observations,

        "votes":
            votes,

        "ocr_readings":
            ocr_readings
    }


# ============================================================
# MERGE DUPLICATE PLATES
# ============================================================

def merge_duplicate_plates(results):

    """
    Merge identical final plate numbers.

    The strongest result is retained.
    """

    merged = {}

    for result in results:

        plate = result.get(
            "final_plate",
            ""
        )

        if not plate:
            continue

        if plate not in merged:

            merged[
                plate
            ] = result

            continue

        existing = merged[
            plate
        ]

        # ----------------------------------------------------
        # Combine observations
        # ----------------------------------------------------

        existing[
            "observations"
        ] += result.get(
            "observations",
            0
        )

        # ----------------------------------------------------
        # Combine votes
        # ----------------------------------------------------

        existing[
            "votes"
        ] += result.get(
            "votes",
            0
        )

        # ----------------------------------------------------
        # Keep strongest confidence
        # ----------------------------------------------------

        if result.get(
            "confidence",
            0
        ) > existing.get(
            "confidence",
            0
        ):

            existing[
                "confidence"
            ] = result[
                "confidence"
            ]

            existing[
                "track_id"
            ] = result[
                "track_id"
            ]

        # ----------------------------------------------------
        # VALID takes priority
        # ----------------------------------------------------

        if result.get(
            "status"
        ) == "VALID":

            existing[
                "status"
            ] = "VALID"

        # ----------------------------------------------------
        # Recalculate reliability
        # ----------------------------------------------------

        existing[
            "reliability"
        ] = calculate_reliability(
            existing[
                "confidence"
            ],
            existing[
                "observations"
            ],
            existing[
                "votes"
            ],
            existing[
                "status"
            ]
        )

    return list(
        merged.values()
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    results,
    input_data
):

    valid_count = sum(
        1
        for r in results
        if r["status"] == "VALID"
    )

    review_count = sum(
        1
        for r in results
        if r["status"] == "REVIEW"
    )

    invalid_count = sum(
        1
        for r in results
        if r["status"] == "INVALID"
    )

    report = {

        "project":
            "AegisVision",

        "module":
            "ANPR",

        "phase":
            "5.6",

        "phase_name":
            "Final ANPR Decision & Export",

        "input_report":
            os.path.abspath(
                INPUT_REPORT
            ),

        "input_tracks":
            len(results),

        "final_results":
            len(results),

        "unique_final_plates":
            len(results),

        "statistics": {

            "valid_plates":
                valid_count,

            "review_plates":
                review_count,

            "invalid_plates":
                invalid_count
        },

        "final_plates":
            results
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Rank",
                "Plate Number",
                "Status",
                "Confidence (%)",
                "Reliability Score (%)",
                "Observations",
                "Votes",
                "Track ID"
            ]
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            writer.writerow(
                [
                    rank,

                    result.get(
                        "final_plate",
                        ""
                    ),

                    result.get(
                        "status",
                        "UNKNOWN"
                    ),

                    result.get(
                        "confidence",
                        0
                    ),

                    result.get(
                        "reliability",
                        0
                    ),

                    result.get(
                        "observations",
                        0
                    ),

                    result.get(
                        "votes",
                        0
                    ),

                    result.get(
                        "track_id",
                        0
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
        "PHASE 5.6 - FINAL ANPR DECISION & EXPORT"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_REPORT
    ):

        print(
            "\nERROR: Phase 5.5 report not found:"
        )

        print(
            os.path.abspath(
                INPUT_REPORT
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
        ) as f:

            data = json.load(
                f
            )

    except Exception as error:

        print(
            "\nERROR: Could not read Phase 5.5 report."
        )

        print(
            error
        )

        return

    # --------------------------------------------------------
    # Extract tracks
    # --------------------------------------------------------

    print(
        "\nExtracting stabilized tracks..."
    )

    tracks = extract_final_plates(
        data
    )

    if not tracks:

        print(
            "\nWARNING: No stabilized tracks found."
        )

        print(
            "Checking final plates..."
        )

        tracks = data.get(
            "final_plates",
            []
        )

    if not tracks:

        print(
            "\nERROR: No Phase 5.5 results found."
        )

        print(
            "Please check phase5_5_report.json"
        )

        return

    print(
        f"Input tracks: {len(tracks)}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    results = []

    print(
        "\nPROCESSING FINAL RESULTS"
    )

    print(
        "-" * 60
    )

    for index, track in enumerate(
        tracks,
        start=1
    ):

        try:

            result = normalize_track(
                track,
                index
            )

            print(
                f"\nTrack {result['track_id']}"
            )

            print(
                f"  Original plate : "
                f"{result['original_plate']}"
            )

            print(
                f"  Final plate    : "
                f"{result['final_plate']}"
            )

            print(
                f"  Status         : "
                f"{result['status']}"
            )

            print(
                f"  Confidence     : "
                f"{result['confidence']:.2f}%"
            )

            print(
                f"  Reliability    : "
                f"{result['reliability']:.2f}%"
            )

            print(
                f"  Observations   : "
                f"{result['observations']}"
            )

            print(
                f"  Votes          : "
                f"{result['votes']}"
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"\nWARNING: Could not process track {index}"
            )

            print(
                f"Reason: {error}"
            )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print(
        "\n\nMERGING DUPLICATE FINAL PLATES"
    )

    print(
        "-" * 60
    )

    before_merge = len(
        results
    )

    results = merge_duplicate_plates(
        results
    )

    after_merge = len(
        results
    )

    print(
        f"Before merging: {before_merge}"
    )

    print(
        f"After merging : {after_merge}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda x: (
            x.get(
                "status"
            ) != "VALID",

            -x.get(
                "reliability",
                0
            )
        )
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    save_json(
        results,
        data
    )

    save_csv(
        results
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid = [
        r for r in results
        if r["status"] == "VALID"
    ]

    review = [
        r for r in results
        if r["status"] == "REVIEW"
    ]

    invalid = [
        r for r in results
        if r["status"] == "INVALID"
    ]

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        "PHASE 5.6 FINAL SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Input tracks        : "
        f"{len(tracks)}"
    )

    print(
        f"Final results       : "
        f"{len(results)}"
    )

    print(
        f"Unique final plates : "
        f"{len(results)}"
    )

    print(
        f"Valid plates        : "
        f"{len(valid)}"
    )

    print(
        f"Review plates       : "
        f"{len(review)}"
    )

    print(
        f"Invalid plates      : "
        f"{len(invalid)}"
    )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print(
        "\nFINAL ANPR PLATES"
    )

    print(
        "-" * 60
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['final_plate']} | "
            f"{result['status']} | "
            f"Confidence: "
            f"{result['confidence']:.2f}% | "
            f"Reliability: "
            f"{result['reliability']:.2f}% | "
            f"Observations: "
            f"{result['observations']} | "
            f"Votes: "
            f"{result['votes']} | "
            f"Track: "
            f"{result['track_id']}"
        )

    # --------------------------------------------------------
    # Valid plates
    # --------------------------------------------------------

    if valid:

        print(
            "\nVALID PLATES"
        )

        print(
            "-" * 60
        )

        for result in valid:

            print(
                f"  {result['final_plate']}"
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

    print(
        "=" * 60
    )

    print(
        "PHASE 5.6 COMPLETED SUCCESSFULLY"
    )

    print(
        "Final ANPR decision and export completed."
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            "\n"
        )

        print(
            "=" * 60
        )

        print(
            "PHASE 5.6 FAILED"
        )

        print(
            "=" * 60
        )

        print(
            f"Error: {error}"
        )

        traceback.print_exc()