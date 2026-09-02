import json
import csv
import time
from pathlib import Path
from statistics import mean


# ============================================================
# AEGISVISION - ANPR
# PHASE 5.8 - ANPR EVALUATION & PERFORMANCE REPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_REPORT = (
    PROJECT_ROOT
    / "output"
    / "phase5_7"
    / "phase5_7_report.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "phase5_8"
)

OUTPUT_JSON = OUTPUT_DIR / "phase5_8_report.json"
OUTPUT_CSV = OUTPUT_DIR / "evaluation_results.csv"


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_list(data, key):
    value = data.get(key, [])
    return value if isinstance(value, list) else []


# ------------------------------------------------------------
# Load Phase 5.7 report
# ------------------------------------------------------------

def load_report():

    if not INPUT_REPORT.exists():
        print("ERROR: Phase 5.7 report not found.")
        print(f"Expected:")
        print(INPUT_REPORT)
        return None

    try:
        with open(INPUT_REPORT, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        print("ERROR: Invalid JSON report.")
        print(error)
        return None

    except Exception as error:
        print("ERROR while reading report:")
        print(error)
        return None


# ------------------------------------------------------------
# Extract video information
# ------------------------------------------------------------

def extract_video_information(report):

    video = report.get("video_information", {})

    return {
        "width": safe_int(video.get("width")),
        "height": safe_int(video.get("height")),
        "fps": safe_float(video.get("fps")),
        "total_frames": safe_int(video.get("total_frames")),
        "duration_seconds": safe_float(
            video.get("duration_seconds")
        ),
    }


# ------------------------------------------------------------
# Extract processing information
# ------------------------------------------------------------

def extract_processing_information(report):

    processing = report.get("processing", {})

    return {
        "minimum_plate_length": safe_int(
            processing.get("minimum_plate_length")
        ),
        "maximum_plate_length": safe_int(
            processing.get("maximum_plate_length")
        ),
        "similarity_threshold": safe_float(
            processing.get("similarity_threshold")
        ),
        "high_confidence_threshold": safe_float(
            processing.get("high_confidence_threshold")
        ),
        "low_confidence_threshold": safe_float(
            processing.get("low_confidence_threshold")
        ),
    }


# ------------------------------------------------------------
# Evaluate plates
# ------------------------------------------------------------

def evaluate_plates(report):

    plates = get_list(report, "final_plates")

    evaluations = []

    for index, plate in enumerate(plates, start=1):

        final_plate = str(
            plate.get("improved_plate")
            or plate.get("final_plate")
            or ""
        )

        status = str(
            plate.get("status", "UNKNOWN")
        ).upper()

        accuracy = safe_float(
            plate.get("accuracy_score")
            or plate.get("accuracy")
            or plate.get("confidence")
        )

        consistency = safe_float(
            plate.get("consistency")
            or plate.get("consistency_score")
        )

        observations = safe_int(
            plate.get("total_observations")
            or plate.get("observations")
        )

        votes = safe_int(
            plate.get("votes")
        )

        tracks = plate.get("tracks", [])

        if not isinstance(tracks, list):
            tracks = []

        track_id = tracks[0] if tracks else ""

        format_score = safe_float(
            plate.get("format_score")
        )

        evaluations.append({
            "rank": index,
            "plate_number": final_plate,
            "status": status,
            "accuracy_score": round(accuracy, 2),
            "consistency": round(consistency, 2),
            "format_score": round(format_score, 2),
            "observations": observations,
            "votes": votes,
            "track_id": track_id,
        })

    return evaluations


# ------------------------------------------------------------
# Calculate evaluation metrics
# ------------------------------------------------------------

def calculate_metrics(
    report,
    plates,
    processing_time
):

    video = extract_video_information(report)

    total_frames = video["total_frames"]
    duration = video["duration_seconds"]

    # --------------------------------------------------------
    # Status counts
    # --------------------------------------------------------

    valid_count = sum(
        1 for plate in plates
        if plate["status"] == "VALID"
    )

    review_count = sum(
        1 for plate in plates
        if plate["status"] == "REVIEW"
    )

    invalid_count = sum(
        1 for plate in plates
        if plate["status"] == "INVALID"
    )

    unique_plates = len(plates)

    # --------------------------------------------------------
    # Score calculations
    # --------------------------------------------------------

    accuracy_scores = [
        plate["accuracy_score"]
        for plate in plates
        if plate["accuracy_score"] > 0
    ]

    consistency_scores = [
        plate["consistency"]
        for plate in plates
        if plate["consistency"] > 0
    ]

    format_scores = [
        plate["format_score"]
        for plate in plates
        if plate["format_score"] > 0
    ]

    average_accuracy = (
        mean(accuracy_scores)
        if accuracy_scores
        else 0.0
    )

    average_consistency = (
        mean(consistency_scores)
        if consistency_scores
        else 0.0
    )

    average_format = (
        mean(format_scores)
        if format_scores
        else 0.0
    )

    # --------------------------------------------------------
    # Best plate
    # --------------------------------------------------------

    best_plate = None

    if plates:
        best_plate = max(
            plates,
            key=lambda x: x["accuracy_score"]
        )

    # --------------------------------------------------------
    # Multi-observation plates
    # --------------------------------------------------------

    multi_observation_count = sum(
        1 for plate in plates
        if plate["observations"] > 1
    )

    total_observations = sum(
        plate["observations"]
        for plate in plates
    )

    total_votes = sum(
        plate["votes"]
        for plate in plates
    )

    # --------------------------------------------------------
    # Processing performance
    # --------------------------------------------------------

    processing_fps = 0.0

    if processing_time > 0 and total_frames > 0:
        processing_fps = total_frames / processing_time

    frame_coverage = 0.0

    # Phase 5.7 does not necessarily store processed frame count.
    # Therefore this metric is only calculated if available.

    statistics = report.get("statistics", {})

    processed_frames = safe_int(
        statistics.get("processed_frames")
    )

    if processed_frames > 0 and total_frames > 0:
        frame_coverage = (
            processed_frames / total_frames
        ) * 100

    # --------------------------------------------------------
    # Overall evaluation score
    # --------------------------------------------------------

    overall_score = (
        average_accuracy * 0.50
        + average_consistency * 0.30
        + average_format * 0.20
    )

    # --------------------------------------------------------
    # Performance grade
    # --------------------------------------------------------

    if overall_score >= 90:
        grade = "A+"
    elif overall_score >= 80:
        grade = "A"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    elif overall_score >= 50:
        grade = "D"
    else:
        grade = "E"

    return {
        "total_frames": total_frames,
        "video_duration_seconds": duration,
        "unique_plates": unique_plates,
        "valid_plates": valid_count,
        "review_plates": review_count,
        "invalid_plates": invalid_count,
        "total_observations": total_observations,
        "total_votes": total_votes,
        "multi_observation_plates": multi_observation_count,
        "average_accuracy_score": round(
            average_accuracy, 2
        ),
        "average_consistency": round(
            average_consistency, 2
        ),
        "average_format_score": round(
            average_format, 2
        ),
        "overall_evaluation_score": round(
            overall_score, 2
        ),
        "performance_grade": grade,
        "processing_time_seconds": round(
            processing_time, 3
        ),
        "processing_fps": round(
            processing_fps, 2
        ),
        "frame_coverage_percent": round(
            frame_coverage, 2
        ),
        "best_plate": (
            best_plate["plate_number"]
            if best_plate
            else None
        ),
        "best_plate_accuracy": (
            best_plate["accuracy_score"]
            if best_plate
            else 0.0
        ),
    }


# ------------------------------------------------------------
# Export CSV
# ------------------------------------------------------------

def export_csv(plates):

    fields = [
        "rank",
        "plate_number",
        "status",
        "accuracy_score",
        "consistency",
        "format_score",
        "observations",
        "votes",
        "track_id",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for plate in plates:
            writer.writerow(plate)


# ------------------------------------------------------------
# Export JSON
# ------------------------------------------------------------

def export_json(
    report,
    video_info,
    processing_info,
    metrics,
    plates
):

    output = {
        "project": "AegisVision",
        "module": "ANPR",
        "phase": "5.8",
        "phase_name": "ANPR Evaluation & Performance Report",
        "input_phase": "5.7",
        "input_report": str(INPUT_REPORT),

        "video_information": video_info,

        "processing": processing_info,

        "evaluation": metrics,

        "final_plates": plates,

        "output_files": {
            "json": str(OUTPUT_JSON),
            "csv": str(OUTPUT_CSV),
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


# ------------------------------------------------------------
# Display report
# ------------------------------------------------------------

def display_report(
    video_info,
    metrics,
    plates
):

    print()
    print("=" * 60)
    print("PHASE 5.8 EVALUATION SUMMARY")
    print("=" * 60)

    print()
    print("VIDEO")
    print("-" * 60)

    print(
        f"Resolution       : "
        f"{video_info['width']} x {video_info['height']}"
    )

    print(
        f"FPS              : "
        f"{video_info['fps']:.2f}"
    )

    print(
        f"Total frames     : "
        f"{video_info['total_frames']}"
    )

    print(
        f"Duration         : "
        f"{video_info['duration_seconds']:.2f} seconds"
    )

    print()
    print("ANPR PERFORMANCE")
    print("-" * 60)

    print(
        f"Unique plates    : "
        f"{metrics['unique_plates']}"
    )

    print(
        f"Valid plates     : "
        f"{metrics['valid_plates']}"
    )

    print(
        f"Review plates    : "
        f"{metrics['review_plates']}"
    )

    print(
        f"Invalid plates   : "
        f"{metrics['invalid_plates']}"
    )

    print(
        f"Observations     : "
        f"{metrics['total_observations']}"
    )

    print(
        f"Votes            : "
        f"{metrics['total_votes']}"
    )

    print(
        f"Multi-observed   : "
        f"{metrics['multi_observation_plates']}"
    )

    print()
    print("QUALITY METRICS")
    print("-" * 60)

    print(
        f"Average accuracy : "
        f"{metrics['average_accuracy_score']:.2f}%"
    )

    print(
        f"Average consistency : "
        f"{metrics['average_consistency']:.2f}%"
    )

    print(
        f"Average format   : "
        f"{metrics['average_format_score']:.2f}%"
    )

    print(
        f"Overall score    : "
        f"{metrics['overall_evaluation_score']:.2f}%"
    )

    print(
        f"Performance grade: "
        f"{metrics['performance_grade']}"
    )

    print()
    print("PROCESSING PERFORMANCE")
    print("-" * 60)

    print(
        f"Processing time  : "
        f"{metrics['processing_time_seconds']:.3f} seconds"
    )

    print(
        f"Processing FPS   : "
        f"{metrics['processing_fps']:.2f}"
    )

    if metrics["frame_coverage_percent"] > 0:
        print(
            f"Frame coverage   : "
            f"{metrics['frame_coverage_percent']:.2f}%"
        )
    else:
        print(
            "Frame coverage   : Not available"
        )

    print()
    print("BEST RESULT")
    print("-" * 60)

    if metrics["best_plate"]:
        print(
            f"Best plate       : "
            f"{metrics['best_plate']}"
        )

        print(
            f"Best accuracy    : "
            f"{metrics['best_plate_accuracy']:.2f}%"
        )
    else:
        print("No plate detected.")

    print()
    print("FINAL PLATES")
    print("-" * 60)

    for index, plate in enumerate(plates, start=1):

        print(
            f"{index}. "
            f"{plate['plate_number']} | "
            f"{plate['status']} | "
            f"Accuracy: {plate['accuracy_score']:.2f}% | "
            f"Consistency: {plate['consistency']:.2f}% | "
            f"Observations: {plate['observations']}"
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("AEGISVISION - ANPR")
    print("PHASE 5.8 - ANPR EVALUATION & PERFORMANCE REPORT")
    print("=" * 60)

    print()
    print("Input report:")
    print(INPUT_REPORT)

    print()
    print("Loading Phase 5.7 results...")

    report = load_report()

    if report is None:
        return

    print("Phase 5.7 report loaded successfully.")

    start_time = time.perf_counter()

    video_info = extract_video_information(report)

    processing_info = extract_processing_information(report)

    print()
    print("Extracting final ANPR results...")

    plates = evaluate_plates(report)

    print(
        f"Plates found: {len(plates)}"
    )

    # Evaluation calculation
    processing_time = (
        time.perf_counter() - start_time
    )

    metrics = calculate_metrics(
        report,
        plates,
        processing_time
    )

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Export files
    export_csv(plates)

    export_json(
        report,
        video_info,
        processing_info,
        metrics,
        plates
    )

    # Display
    display_report(
        video_info,
        metrics,
        plates
    )

    print()
    print("=" * 60)
    print("PHASE 5.8 COMPLETED SUCCESSFULLY")
    print("ANPR evaluation and performance report generated.")
    print("=" * 60)

    print()
    print("JSON report:")
    print(OUTPUT_JSON)

    print()
    print("CSV report:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()