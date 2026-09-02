import os
import json
import csv
import math
from datetime import datetime

import matplotlib.pyplot as plt


# ============================================================
# AEGISVISION - ANPR
# PHASE 5.9 - FINAL VISUALIZATION & EVIDENCE GENERATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "output", "phase5_8")
INPUT_REPORT = os.path.join(INPUT_DIR, "phase5_8_report.json")

PHASE57_DIR = os.path.join(BASE_DIR, "output", "phase5_7")
PHASE57_REPORT = os.path.join(PHASE57_DIR, "phase5_7_report.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "phase5_9")

FINAL_JSON = os.path.join(
    OUTPUT_DIR,
    "phase5_9_final_report.json"
)

FINAL_CSV = os.path.join(
    OUTPUT_DIR,
    "final_anpr_results.csv"
)

DASHBOARD_IMAGE = os.path.join(
    OUTPUT_DIR,
    "anpr_dashboard.png"
)

ACCURACY_CHART = os.path.join(
    OUTPUT_DIR,
    "accuracy_chart.png"
)

STATUS_CHART = os.path.join(
    OUTPUT_DIR,
    "status_distribution.png"
)

HTML_REPORT = os.path.join(
    OUTPUT_DIR,
    "anpr_final_report.html"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

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


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input report not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_status(status):
    if not status:
        return "REVIEW"

    status = str(status).upper().strip()

    if status in ["VALID", "REVIEW", "INVALID"]:
        return status

    return "REVIEW"


# ============================================================
# EXTRACT RESULTS
# ============================================================

def extract_results(report):
    """
    Extract final ANPR results from Phase 5.8.

    Phase 5.8 may store results under different keys depending
    on the previous implementation, so several possibilities
    are supported.
    """

    possible_keys = [
        "final_results",
        "final_plates",
        "results",
        "plates",
        "evaluation_results"
    ]

    results = []

    for key in possible_keys:
        value = report.get(key)

        if isinstance(value, list) and value:
            results = value
            break

    # Some versions may store results inside another section.
    if not results:
        for section_name in [
            "evaluation",
            "evaluation_results",
            "performance",
            "data"
        ]:
            section = report.get(section_name)

            if isinstance(section, dict):
                for key in possible_keys:
                    value = section.get(key)

                    if isinstance(value, list) and value:
                        results = value
                        break

            if results:
                break

    return results


def get_plate(item):
    return (
        item.get("final_plate")
        or item.get("improved_plate")
        or item.get("plate")
        or item.get("plate_number")
        or item.get("text")
        or "UNKNOWN"
    )


def get_status(item):
    return normalize_status(
        item.get("status")
    )


def get_accuracy(item):
    return safe_float(
        item.get("accuracy",
        item.get("accuracy_score",
        item.get("final_confidence",
        item.get("confidence", 0))))
    )


def get_consistency(item):
    return safe_float(
        item.get(
            "consistency",
            item.get("consistency_score", 0)
        )
    )


def get_format_score(item):
    return safe_float(
        item.get(
            "format_score",
            item.get("format", 0)
        )
    )


def get_confidence(item):
    return safe_float(
        item.get(
            "confidence",
            item.get("final_confidence", 0)
        )
    )


def get_observations(item):
    return safe_int(
        item.get(
            "observations",
            item.get("total_observations", 0)
        )
    )


def get_votes(item):
    return safe_int(
        item.get("votes", 0)
    )


def get_track(item):
    track = item.get("track_id")

    if track is None:
        tracks = item.get("tracks", [])

        if isinstance(tracks, list) and tracks:
            track = tracks[0]

    if track is None:
        return ""

    return str(track)


# ============================================================
# BUILD NORMALIZED RESULTS
# ============================================================

def build_normalized_results(raw_results):

    normalized = []

    for item in raw_results:

        if not isinstance(item, dict):
            continue

        result = {
            "plate_number": str(
                get_plate(item)
            ).strip().upper(),

            "status": get_status(item),

            "accuracy": round(
                get_accuracy(item), 2
            ),

            "consistency": round(
                get_consistency(item), 2
            ),

            "format_score": round(
                get_format_score(item), 2
            ),

            "confidence": round(
                get_confidence(item), 2
            ),

            "observations": get_observations(item),

            "votes": get_votes(item),

            "track_id": get_track(item)
        }

        normalized.append(result)

    return normalized


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(results):

    total = len(results)

    valid = sum(
        1 for r in results
        if r["status"] == "VALID"
    )

    review = sum(
        1 for r in results
        if r["status"] == "REVIEW"
    )

    invalid = sum(
        1 for r in results
        if r["status"] == "INVALID"
    )

    observations = sum(
        r["observations"] for r in results
    )

    votes = sum(
        r["votes"] for r in results
    )

    multi_observed = sum(
        1 for r in results
        if r["observations"] > 1
    )

    if total > 0:

        avg_accuracy = sum(
            r["accuracy"] for r in results
        ) / total

        avg_consistency = sum(
            r["consistency"] for r in results
        ) / total

        avg_format = sum(
            r["format_score"] for r in results
        ) / total

        overall_score = (
            avg_accuracy * 0.50
            + avg_consistency * 0.30
            + avg_format * 0.20
        )

    else:
        avg_accuracy = 0
        avg_consistency = 0
        avg_format = 0
        overall_score = 0

    if overall_score >= 80:
        grade = "A"

    elif overall_score >= 70:
        grade = "B"

    elif overall_score >= 60:
        grade = "C"

    elif overall_score >= 50:
        grade = "D"

    else:
        grade = "F"

    best_result = None

    if results:
        best_result = max(
            results,
            key=lambda x: x["accuracy"]
        )

    return {
        "unique_plates": total,
        "valid_plates": valid,
        "review_plates": review,
        "invalid_plates": invalid,
        "observations": observations,
        "votes": votes,
        "multi_observed": multi_observed,
        "average_accuracy": round(
            avg_accuracy, 2
        ),
        "average_consistency": round(
            avg_consistency, 2
        ),
        "average_format": round(
            avg_format, 2
        ),
        "overall_score": round(
            overall_score, 2
        ),
        "performance_grade": grade,
        "best_result": best_result
    }


# ============================================================
# VIDEO INFORMATION
# ============================================================

def extract_video_information(report):

    video = report.get(
        "video_information",
        report.get("video", {})
    )

    if not isinstance(video, dict):
        video = {}

    return {
        "width": safe_int(
            video.get("width", 0)
        ),

        "height": safe_int(
            video.get("height", 0)
        ),

        "fps": safe_float(
            video.get("fps", 0)
        ),

        "total_frames": safe_int(
            video.get("total_frames", 0)
        ),

        "duration_seconds": safe_float(
            video.get("duration_seconds", 0)
        )
    }


# ============================================================
# CSV EXPORT
# ============================================================

def create_csv(results):

    fieldnames = [
        "Rank",
        "Plate Number",
        "Status",
        "Accuracy (%)",
        "Consistency (%)",
        "Format Score (%)",
        "Confidence (%)",
        "Observations",
        "Votes",
        "Track ID"
    ]

    sorted_results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    with open(
        FINAL_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(fieldnames)

        for rank, result in enumerate(
            sorted_results,
            start=1
        ):

            writer.writerow([
                rank,
                result["plate_number"],
                result["status"],
                result["accuracy"],
                result["consistency"],
                result["format_score"],
                result["confidence"],
                result["observations"],
                result["votes"],
                result["track_id"]
            ])


# ============================================================
# ACCURACY CHART
# ============================================================

def create_accuracy_chart(results):

    if not results:
        return

    sorted_results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    plates = [
        r["plate_number"]
        for r in sorted_results
    ]

    accuracies = [
        r["accuracy"]
        for r in sorted_results
    ]

    plt.figure(figsize=(14, 7))

    bars = plt.bar(
        range(len(plates)),
        accuracies
    )

    plt.xticks(
        range(len(plates)),
        plates,
        rotation=60,
        ha="right"
    )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Plate Number")
    plt.title(
        "AegisVision ANPR - OCR Accuracy"
    )

    plt.ylim(0, 100)

    for bar, value in zip(
        bars,
        accuracies
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(
        ACCURACY_CHART,
        dpi=180
    )

    plt.close()


# ============================================================
# STATUS CHART
# ============================================================

def create_status_chart(stats):

    labels = [
        "VALID",
        "REVIEW",
        "INVALID"
    ]

    values = [
        stats["valid_plates"],
        stats["review_plates"],
        stats["invalid_plates"]
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Number of Plates")
    plt.xlabel("ANPR Status")
    plt.title(
        "AegisVision ANPR - Result Status"
    )

    for index, value in enumerate(values):

        plt.text(
            index,
            value + 0.1,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        STATUS_CHART,
        dpi=180
    )

    plt.close()


# ============================================================
# DASHBOARD IMAGE
# ============================================================

def create_dashboard(
    stats,
    video_info,
    results
):

    fig = plt.figure(
        figsize=(16, 10)
    )

    fig.suptitle(
        "AEGISVISION - ANPR FINAL DASHBOARD",
        fontsize=22,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_text = (
        f"Unique Plates : {stats['unique_plates']}\n"
        f"Valid Plates  : {stats['valid_plates']}\n"
        f"Review Plates : {stats['review_plates']}\n"
        f"Invalid Plates: {stats['invalid_plates']}\n\n"
        f"Average Accuracy    : "
        f"{stats['average_accuracy']:.2f}%\n"
        f"Average Consistency : "
        f"{stats['average_consistency']:.2f}%\n"
        f"Average Format      : "
        f"{stats['average_format']:.2f}%\n"
        f"Overall Score       : "
        f"{stats['overall_score']:.2f}%\n"
        f"Performance Grade   : "
        f"{stats['performance_grade']}"
    )

    fig.text(
        0.05,
        0.78,
        summary_text,
        fontsize=13,
        va="top"
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    video_text = (
        "VIDEO INFORMATION\n\n"
        f"Resolution : "
        f"{video_info['width']} x "
        f"{video_info['height']}\n"
        f"FPS        : "
        f"{video_info['fps']:.2f}\n"
        f"Frames     : "
        f"{video_info['total_frames']}\n"
        f"Duration   : "
        f"{video_info['duration_seconds']:.2f} sec"
    )

    fig.text(
        0.52,
        0.78,
        video_text,
        fontsize=13,
        va="top"
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    best = stats["best_result"]

    if best:

        best_text = (
            "BEST OCR RESULT\n\n"
            f"Plate       : "
            f"{best['plate_number']}\n"
            f"Accuracy    : "
            f"{best['accuracy']:.2f}%\n"
            f"Consistency : "
            f"{best['consistency']:.2f}%\n"
            f"Status      : "
            f"{best['status']}"
        )

    else:

        best_text = (
            "BEST OCR RESULT\n\n"
            "No result available."
        )

    fig.text(
        0.05,
        0.48,
        best_text,
        fontsize=13,
        va="top"
    )

    # --------------------------------------------------------
    # Top results
    # --------------------------------------------------------

    top_results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True
    )[:8]

    lines = [
        "TOP ANPR RESULTS",
        ""
    ]

    for index, result in enumerate(
        top_results,
        start=1
    ):

        lines.append(
            f"{index}. "
            f"{result['plate_number']} | "
            f"{result['status']} | "
            f"{result['accuracy']:.2f}%"
        )

    fig.text(
        0.52,
        0.48,
        "\n".join(lines),
        fontsize=12,
        va="top"
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.04,
        "Phase 5.9 - Final Visualization & Evidence Generation",
        ha="center",
        fontsize=11
    )

    plt.axis("off")

    plt.savefig(
        DASHBOARD_IMAGE,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# HTML REPORT
# ============================================================

def create_html_report(
    stats,
    video_info,
    results
):

    sorted_results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    rows = ""

    for rank, result in enumerate(
        sorted_results,
        start=1
    ):

        rows += f"""
        <tr>
            <td>{rank}</td>
            <td><strong>{result['plate_number']}</strong></td>
            <td>{result['status']}</td>
            <td>{result['accuracy']:.2f}%</td>
            <td>{result['consistency']:.2f}%</td>
            <td>{result['format_score']:.2f}%</td>
            <td>{result['confidence']:.2f}%</td>
            <td>{result['observations']}</td>
            <td>{result['votes']}</td>
            <td>{result['track_id']}</td>
        </tr>
        """

    best = stats["best_result"]

    if best:
        best_plate = best["plate_number"]
        best_accuracy = f"{best['accuracy']:.2f}%"
    else:
        best_plate = "N/A"
        best_accuracy = "N/A"

    html = f"""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<title>AegisVision ANPR Final Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f4f4f4;
}}

.container {{
    max-width: 1200px;
    margin: auto;
    background: white;
    padding: 30px;
}}

h1 {{
    text-align: center;
}}

h2 {{
    margin-top: 35px;
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 15px;
}}

.card {{
    padding: 20px;
    border: 1px solid #ddd;
    text-align: center;
}}

.card h3 {{
    margin: 5px;
}}

.card p {{
    font-size: 24px;
    font-weight: bold;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 10px;
    text-align: center;
}}

th {{
    background: #eee;
}}

img {{
    max-width: 100%;
    margin-top: 20px;
}}

.footer {{
    margin-top: 40px;
    text-align: center;
    color: #666;
}}

</style>

</head>

<body>

<div class="container">

<h1>
AEGISVISION - ANPR FINAL REPORT
</h1>

<p style="text-align:center;">
Phase 5.9 - Final Visualization & Evidence Generation
</p>

<h2>Performance Summary</h2>

<div class="cards">

<div class="card">
<h3>Unique Plates</h3>
<p>{stats['unique_plates']}</p>
</div>

<div class="card">
<h3>Valid</h3>
<p>{stats['valid_plates']}</p>
</div>

<div class="card">
<h3>Review</h3>
<p>{stats['review_plates']}</p>
</div>

<div class="card">
<h3>Invalid</h3>
<p>{stats['invalid_plates']}</p>
</div>

</div>

<h2>Quality Metrics</h2>

<table>

<tr>
<th>Metric</th>
<th>Value</th>
</tr>

<tr>
<td>Average Accuracy</td>
<td>{stats['average_accuracy']:.2f}%</td>
</tr>

<tr>
<td>Average Consistency</td>
<td>{stats['average_consistency']:.2f}%</td>
</tr>

<tr>
<td>Average Format Score</td>
<td>{stats['average_format']:.2f}%</td>
</tr>

<tr>
<td>Overall Score</td>
<td>{stats['overall_score']:.2f}%</td>
</tr>

<tr>
<td>Performance Grade</td>
<td><strong>{stats['performance_grade']}</strong></td>
</tr>

</table>

<h2>Video Information</h2>

<table>

<tr>
<td>Resolution</td>
<td>
{video_info['width']} x
{video_info['height']}
</td>
</tr>

<tr>
<td>FPS</td>
<td>{video_info['fps']:.2f}</td>
</tr>

<tr>
<td>Total Frames</td>
<td>{video_info['total_frames']}</td>
</tr>

<tr>
<td>Duration</td>
<td>
{video_info['duration_seconds']:.2f} seconds
</td>
</tr>

</table>

<h2>Best OCR Result</h2>

<table>

<tr>
<th>Plate</th>
<th>Accuracy</th>
<th>Status</th>
</tr>

<tr>
<td><strong>{best_plate}</strong></td>
<td>{best_accuracy}</td>
<td>
{best['status'] if best else 'N/A'}
</td>
</tr>

</table>

<h2>Accuracy Chart</h2>

<img src="accuracy_chart.png">

<h2>Status Distribution</h2>

<img src="status_distribution.png">

<h2>ANPR Results</h2>

<table>

<tr>
<th>Rank</th>
<th>Plate Number</th>
<th>Status</th>
<th>Accuracy</th>
<th>Consistency</th>
<th>Format</th>
<th>Confidence</th>
<th>Observations</th>
<th>Votes</th>
<th>Track</th>
</tr>

{rows}

</table>

<div class="footer">

Generated by AegisVision ANPR Phase 5.9<br>

Generated at:
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

</div>

</div>

</body>
</html>
"""

    with open(
        HTML_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(html)


# ============================================================
# FINAL JSON REPORT
# ============================================================

def create_final_json(
    stats,
    video_info,
    results
):

    report = {
        "project": "AegisVision",
        "module": "ANPR",

        "phase": "5.9",

        "phase_name":
            "Final Visualization & Evidence Generation",

        "input_phase": "5.8",

        "input_report": INPUT_REPORT,

        "video_information": video_info,

        "processing_performance": {
            "processing_time_seconds":
                None,

            "processing_fps":
                None,

            "frame_coverage":
                "Not available",

            "note":
                "Phase 5.9 evaluates previously "
                "generated ANPR results and does "
                "not reprocess the source video."
        },

        "statistics": stats,

        "final_results": results,

        "output_files": {
            "json": FINAL_JSON,
            "csv": FINAL_CSV,
            "dashboard": DASHBOARD_IMAGE,
            "accuracy_chart": ACCURACY_CHART,
            "status_chart": STATUS_CHART,
            "html_report": HTML_REPORT
        },

        "generated_at":
            datetime.now().isoformat()
    }

    with open(
        FINAL_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AEGISVISION - ANPR")
    print("PHASE 5.9 - FINAL VISUALIZATION")
    print("& EVIDENCE GENERATION")
    print("=" * 60)

    print()

    print("Input report:")
    print(INPUT_REPORT)

    print()

    ensure_output_directory()

    # --------------------------------------------------------
    # Load Phase 5.8
    # --------------------------------------------------------

    print("Loading Phase 5.8 evaluation report...")

    try:
        report = load_json(
            INPUT_REPORT
        )

    except Exception as error:

        print()
        print("ERROR:")
        print(error)
        return

    print("Phase 5.8 report loaded successfully.")

    print()

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    print("Extracting final ANPR results...")

    raw_results = extract_results(
        report
    )

    # If Phase 5.8 does not contain a usable result list,
    # try Phase 5.7 directly.

    if not raw_results:

        print(
            "Phase 5.8 result list not found."
        )

        if os.path.exists(
            PHASE57_REPORT
        ):

            print(
                "Checking Phase 5.7 report..."
            )

            try:

                phase57 = load_json(
                    PHASE57_REPORT
                )

                raw_results = extract_results(
                    phase57
                )

            except Exception as error:

                print(
                    "Could not load Phase 5.7:"
                )

                print(error)

    if not raw_results:

        print()
        print(
            "ERROR: No ANPR results found."
        )

        print(
            "Expected results in Phase 5.8 "
            "or Phase 5.7 report."
        )

        return

    print(
        f"Results found: {len(raw_results)}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    results = build_normalized_results(
        raw_results
    )

    if not results:

        print()
        print(
            "ERROR: Results could not be normalized."
        )

        return

    print(
        f"Normalized results: {len(results)}"
    )

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    video_info = extract_video_information(
        report
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_statistics(
        results
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 5.9 VISUALIZATION SUMMARY")
    print("=" * 60)

    print()

    print("ANPR RESULTS")
    print("-" * 60)

    print(
        f"Unique plates : "
        f"{stats['unique_plates']}"
    )

    print(
        f"Valid plates  : "
        f"{stats['valid_plates']}"
    )

    print(
        f"Review plates : "
        f"{stats['review_plates']}"
    )

    print(
        f"Invalid plates: "
        f"{stats['invalid_plates']}"
    )

    print()

    print("QUALITY METRICS")
    print("-" * 60)

    print(
        f"Average accuracy    : "
        f"{stats['average_accuracy']:.2f}%"
    )

    print(
        f"Average consistency : "
        f"{stats['average_consistency']:.2f}%"
    )

    print(
        f"Average format      : "
        f"{stats['average_format']:.2f}%"
    )

    print(
        f"Overall score       : "
        f"{stats['overall_score']:.2f}%"
    )

    print(
        f"Performance grade   : "
        f"{stats['performance_grade']}"
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    print()
    print("BEST RESULT")
    print("-" * 60)

    if stats["best_result"]:

        best = stats["best_result"]

        print(
            f"Plate       : "
            f"{best['plate_number']}"
        )

        print(
            f"Accuracy    : "
            f"{best['accuracy']:.2f}%"
        )

        print(
            f"Consistency : "
            f"{best['consistency']:.2f}%"
        )

        print(
            f"Status      : "
            f"{best['status']}"
        )

    # --------------------------------------------------------
    # Generate files
    # --------------------------------------------------------

    print()
    print(
        "Generating final evidence..."
    )

    print()

    print(
        "[1/6] Creating CSV..."
    )

    create_csv(
        results
    )

    print(
        "[2/6] Creating accuracy chart..."
    )

    create_accuracy_chart(
        results
    )

    print(
        "[3/6] Creating status chart..."
    )

    create_status_chart(
        stats
    )

    print(
        "[4/6] Creating dashboard..."
    )

    create_dashboard(
        stats,
        video_info,
        results
    )

    print(
        "[5/6] Creating HTML report..."
    )

    create_html_report(
        stats,
        video_info,
        results
    )

    print(
        "[6/6] Creating JSON report..."
    )

    create_final_json(
        stats,
        video_info,
        results
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 5.9 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()

    print("Generated files:")
    print()

    print(
        f"JSON report:"
    )

    print(
        FINAL_JSON
    )

    print()

    print(
        f"CSV report:"
    )

    print(
        FINAL_CSV
    )

    print()

    print(
        f"Dashboard:"
    )

    print(
        DASHBOARD_IMAGE
    )

    print()

    print(
        f"Accuracy chart:"
    )

    print(
        ACCURACY_CHART
    )

    print()

    print(
        f"Status chart:"
    )

    print(
        STATUS_CHART
    )

    print()

    print(
        f"HTML report:"
    )

    print(
        HTML_REPORT
    )

    print()

    print("=" * 60)
    print(
        "FINAL ANPR VISUALIZATION AND "
        "EVIDENCE GENERATION COMPLETED."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()