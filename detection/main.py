import cv2

from multi_camera import (
    multi_camera_manager
)


# ==========================================
# START FOUR-CAMERA SYSTEM
# ==========================================

print(
    "🚀 Starting AegisVision "
    "Four-Camera Detection..."
)


multi_camera_manager.start()


# ==========================================
# DISPLAY
# ==========================================

while True:

    frame = (
        multi_camera_manager.get_grid()
    )


    if frame is None:

        continue


    cv2.imshow(

        "AegisVision - "
        "4 Camera Border Surveillance",

        frame

    )


    # ======================================
    # QUIT
    # ======================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        print(
            "ℹ️ Detection stopped."
        )

        break


# ==========================================
# CLEANUP
# ==========================================

multi_camera_manager.stop()

cv2.destroyAllWindows()

print(
    "✅ AegisVision Detection Module Closed."
)