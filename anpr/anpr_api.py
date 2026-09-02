# ============================================================
# AEGISVISION - ANPR
# PHASE 5.3 - ANPR API / INTEGRATION WRAPPER
#
# Purpose:
#   Provide a simple interface for the AegisVision backend
#   to call the existing ANPR engine.
#
# Existing engine functions:
#   process_image(image)
#   process_image_file(image_path)
#
# ============================================================

import json
from pathlib import Path

from anpr_engine import process_image_file


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"

TEST_IMAGE = INPUT_DIR / "car.jpg"


# ============================================================
# ANPR API FUNCTION
# ============================================================

def detect_number_plate(image_path):
    """
    Process an image using the existing ANPR engine.

    Parameters:
        image_path: Path to input image.

    Returns:
        Structured ANPR result dictionary.
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        return {
            "success": False,
            "message": "Input image not found",
            "image_path": str(image_path)
        }

    # --------------------------------------------------------
    # Process image using ANPR engine
    # --------------------------------------------------------

    try:

        result = process_image_file(
            str(image_path)
        )

        return result

    except Exception as error:

        return {
            "success": False,
            "message": "ANPR processing failed",
            "error": str(error)
        }


# ============================================================
# GET VALID PLATES
# ============================================================

def get_valid_plates(result):
    """
    Extract only VALID number plates from
    the ANPR result.
    """

    valid_plates = []

    if not result:
        return valid_plates

    plates = result.get(
        "plates",
        []
    )

    for plate in plates:

        if plate.get("status") == "VALID":

            valid_plates.append(
                plate.get(
                    "plate_number",
                    ""
                )
            )

    return valid_plates


# ============================================================
# API RESPONSE
# ============================================================

def create_api_response(result):
    """
    Create a simplified response for
    integration with the main AegisVision system.
    """

    if not result.get("success", False):

        return {
            "success": False,
            "message": result.get(
                "message",
                "ANPR processing failed"
            )
        }

    valid_plates = get_valid_plates(
        result
    )

    return {
        "success": True,
        "message": "ANPR processing completed",

        "plate_count": result.get(
            "plate_count",
            0
        ),

        "valid_plate_count": len(
            valid_plates
        ),

        "valid_plates": valid_plates,

        "plates": result.get(
            "plates",
            []
        )
    }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print(
        "AEGISVISION - ANPR API"
    )
    print(
        "PHASE 5.3 - INTEGRATION WRAPPER TEST"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Check test image
    # --------------------------------------------------------

    print("\nTesting image:")
    print(TEST_IMAGE)

    if not TEST_IMAGE.exists():

        print(
            "\nERROR: Test image not found."
        )

        print(
            f"Expected:\n{TEST_IMAGE}"
        )

        return

    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    print(
        "\nProcessing image through ANPR API..."
    )

    result = detect_number_plate(
        TEST_IMAGE
    )

    # --------------------------------------------------------
    # Print complete engine result
    # --------------------------------------------------------

    print("\nANPR ENGINE RESULT")
    print("-" * 60)

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Create simplified API response
    # --------------------------------------------------------

    api_response = create_api_response(
        result
    )

    print("\nANPR API RESPONSE")
    print("-" * 60)

    print(
        json.dumps(
            api_response,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if api_response.get("success"):

        print(
            "ANPR API TEST PASSED"
        )

        print(
            "Phase 5.3 integration wrapper "
            "is working successfully."
        )

    else:

        print(
            "ANPR API TEST FAILED"
        )

        print(
            api_response.get(
                "message",
                "Unknown error"
            )
        )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()
