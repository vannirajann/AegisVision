import cv2
from ultralytics import YOLO
import time
import os


# ==========================================
# AEGISVISION MULTI-CAMERA BORDER SURVEILLANCE
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Model loaded successfully.")


# ==========================================
# CAMERA CONFIGURATION
# ==========================================

VIDEO_PATH = "videos/test.mp4"

CAMERAS = [
    {
        "name": "CCTV-01",
        "source": VIDEO_PATH,
        "border_y": 250,
        "alert": False,
        "last_alert": 0
    },
    {
        "name": "CCTV-02",
        "source": VIDEO_PATH,
        "border_y": 250,
        "alert": False,
        "last_alert": 0
    },
    {
        "name": "CCTV-03",
        "source": VIDEO_PATH,
        "border_y": 250,
        "alert": False,
        "last_alert": 0
    },
    {
        "name": "CCTV-04",
        "source": VIDEO_PATH,
        "border_y": 250,
        "alert": False,
        "last_alert": 0
    }
]


# ==========================================
# CHECK VIDEO FILE
# ==========================================

if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Video not found: {VIDEO_PATH}")
    print("Make sure test.mp4 is inside the videos folder.")
    exit()


# ==========================================
# OPEN ALL CAMERAS
# ==========================================

captures = []

for camera in CAMERAS:

    cap = cv2.VideoCapture(camera["source"])

    if not cap.isOpened():
        print(
            f"[ERROR] Cannot open "
            f"{camera['name']} -> {camera['source']}"
        )
    else:
        captures.append(cap)
        print(
            f"[OK] {camera['name']} connected"
        )


if len(captures) == 0:
    print("No camera sources available.")
    exit()


print("\nAegisVision Multi-Camera Monitoring Started")
print("Press Q to quit.\n")


# ==========================================
# SETTINGS
# ==========================================

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

ALERT_COOLDOWN = 3

# YOLO COCO class ID for person
PERSON_CLASS = 0


# ==========================================
# CREATE CAMERA FRAME
# ==========================================

def process_camera(camera, cap):

    ret, frame = cap.read()

    # --------------------------------------
    # LOOP VIDEO WHEN IT ENDS
    # --------------------------------------

    if not ret:

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = cap.read()

        if not ret:
            return None

    # --------------------------------------
    # RESIZE
    # --------------------------------------

    frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    camera_name = camera["name"]

    border_y = camera["border_y"]

    intrusion_detected = False

    # --------------------------------------
    # YOLO DETECTION
    # --------------------------------------

    results = model(
        frame,
        classes=[PERSON_CLASS],
        conf=0.45,
        verbose=False
    )

    # --------------------------------------
    # DRAW VIRTUAL BORDER
    # --------------------------------------

    cv2.line(
        frame,
        (0, border_y),
        (DISPLAY_WIDTH, border_y),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "VIRTUAL BORDER",
        (10, border_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # --------------------------------------
    # DETECT PEOPLE
    # --------------------------------------

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            confidence = float(box.conf[0])

            # Center point of detected person

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # ----------------------------------
            # INTRUSION CHECK
            #
            # Person crosses below border line
            # ----------------------------------

            if center_y > border_y:

                intrusion_detected = True

                box_color = (0, 0, 255)

                label = (
                    f"INTRUDER "
                    f"{confidence:.2f}"
                )

            else:

                box_color = (0, 255, 0)

                label = (
                    f"PERSON "
                    f"{confidence:.2f}"
                )

            # Bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # Label background

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 30)),
                (x2, y1),
                box_color,
                -1
            )

            # Label

            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # Center point

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                box_color,
                -1
            )

    # --------------------------------------
    # CAMERA STATUS
    # --------------------------------------

    if intrusion_detected:

        camera["alert"] = True

        current_time = time.time()

        # Print alert with cooldown

        if (
            current_time - camera["last_alert"]
            > ALERT_COOLDOWN
        ):

            print(
                f"\n!!! ALERT !!! "
                f"{camera_name}: "
                f"BORDER BREACH DETECTED!"
            )

            camera["last_alert"] = current_time

        # Red border around CCTV

        cv2.rectangle(
            frame,
            (0, 0),
            (
                DISPLAY_WIDTH - 1,
                DISPLAY_HEIGHT - 1
            ),
            (0, 0, 255),
            8
        )

        # Alert banner

        cv2.rectangle(
            frame,
            (0, 0),
            (DISPLAY_WIDTH, 55),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"{camera_name} - INTRUDER ALERT!",
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        camera["alert"] = False

        # Normal camera border

        cv2.rectangle(
            frame,
            (0, 0),
            (
                DISPLAY_WIDTH - 1,
                DISPLAY_HEIGHT - 1
            ),
            (0, 255, 0),
            3
        )

        # Camera name

        cv2.rectangle(
            frame,
            (0, 0),
            (DISPLAY_WIDTH, 45),
            (0, 80, 0),
            -1
        )

        cv2.putText(
            frame,
            f"{camera_name} - MONITORING",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return frame


# ==========================================
# MAIN MONITORING LOOP
# ==========================================

while True:

    camera_frames = []

    for i, cap in enumerate(captures):

        camera = CAMERAS[i]

        processed_frame = process_camera(
            camera,
            cap
        )

        if processed_frame is not None:
            camera_frames.append(
                processed_frame
            )

    # --------------------------------------
    # MAKE 2 x 2 CCTV GRID
    # --------------------------------------

    if len(camera_frames) >= 4:

        top_row = cv2.hconcat([
            camera_frames[0],
            camera_frames[1]
        ])

        bottom_row = cv2.hconcat([
            camera_frames[2],
            camera_frames[3]
        ])

        dashboard = cv2.vconcat([
            top_row,
            bottom_row
        ])

    else:

        dashboard = camera_frames[0]

    # --------------------------------------
    # ADD MAIN TITLE
    # --------------------------------------

    title_bar = dashboard.copy()

    cv2.rectangle(
        title_bar,
        (0, 0),
        (title_bar.shape[1], 50),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        title_bar,
        "AEGISVISION - MULTI CAMERA BORDER SURVEILLANCE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------
    # COUNT ACTIVE ALERTS
    # --------------------------------------

    active_alerts = sum(
        1
        for camera in CAMERAS
        if camera["alert"]
    )

    # --------------------------------------
    # GLOBAL ALERT
    # --------------------------------------

    if active_alerts > 0:

        cv2.rectangle(
            title_bar,
            (
                title_bar.shape[1] - 300,
                0
            ),
            (
                title_bar.shape[1],
                50
            ),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            title_bar,
            f"ALERTS: {active_alerts}",
            (
                title_bar.shape[1] - 270,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            title_bar,
            "SYSTEM SECURE",
            (
                title_bar.shape[1] - 250,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # --------------------------------------
    # SHOW DASHBOARD
    # --------------------------------------

    cv2.imshow(
        "AegisVision Command Center",
        title_bar
    )

    # --------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

for cap in captures:
    cap.release()

cv2.destroyAllWindows()

print("\nAegisVision Multi-Camera Monitoring Stopped.")