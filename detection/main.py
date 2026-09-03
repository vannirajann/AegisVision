import cv2
import os
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# VIDEO INPUT CONFIGURATION
# ==========================================

# CURRENT SOURCE: VIDEO FILE
SOURCE = "videos/test.mp4"

# ------------------------------------------
# OTHER OPTIONS
# ------------------------------------------

# Webcam:
# SOURCE = 0

# Future RTSP / IP CCTV:
# SOURCE = "rtsp://username:password@192.168.1.100:554/stream"


# ==========================================
# OPEN VIDEO SOURCE
# ==========================================

video = cv2.VideoCapture(SOURCE)

if not video.isOpened():
    print(f"❌ Cannot open video source: {SOURCE}")
    exit()

print(f"✅ Video source connected: {SOURCE}")


# ==========================================
# DETECT THESE CLASSES
#
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
# ==========================================

allowed_classes = [0, 1, 2, 3, 5, 7]


# ==========================================
# STORE OBJECTS THAT ALREADY CREATED ALERTS
# ==========================================

alerted_objects = set()


# ==========================================
# CREATE ALERTS FOLDER
# ==========================================

os.makedirs("alerts", exist_ok=True)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # ==========================================
    # READ VIDEO FRAME
    # ==========================================

    success, frame = video.read()

    # Stop when video ends
    if not success:
        print("ℹ️ Video stream ended or connection lost.")
        break


    # ==========================================
    # FLIP VIDEO LEFT-RIGHT
    # ==========================================

    frame = cv2.flip(frame, 1)


    # ==========================================
    # GET FRAME DIMENSIONS
    # ==========================================

    height, width = frame.shape[:2]


    # ==========================================
    # CENTER RESTRICTED LINE
    # ==========================================

    ZONE_Y = height // 2


    # ==========================================
    # YOLO DETECTION + TRACKING
    # ==========================================

    results = model.track(
        frame,
        persist=True,
        classes=allowed_classes,
        tracker="bytetrack.yaml",
        verbose=False
    )


    # ==========================================
    # COPY FRAME FOR DRAWING
    # ==========================================

    output_frame = frame.copy()


    # ==========================================
    # DRAW TOP RESTRICTED AREA
    # ==========================================

    cv2.rectangle(
        output_frame,
        (0, 0),
        (width, ZONE_Y),
        (0, 0, 255),
        2
    )

    # Draw center line
    cv2.line(
        output_frame,
        (0, ZONE_Y),
        (width, ZONE_Y),
        (0, 0, 255),
        5
    )

    # Restricted zone text
    cv2.putText(
        output_frame,
        "RESTRICTED ZONE",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )


    # ==========================================
    # CHECK EVERY DETECTED OBJECT
    # ==========================================

    for box in results[0].boxes:

        # ==========================================
        # GET OBJECT CLASS
        # ==========================================

        class_id = int(box.cls[0])
        class_name = model.names[class_id]


        # ==========================================
        # GET CONFIDENCE
        # ==========================================

        confidence = float(box.conf[0])


        # ==========================================
        # GET BOUNDING BOX COORDINATES
        # ==========================================

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # ==========================================
        # GET TRACKING ID
        # ==========================================

        if box.id is not None:
            track_id = int(box.id[0])
        else:
            track_id = -1


        # ==========================================
        # UNIQUE OBJECT KEY
        # ==========================================

        if track_id != -1:
            object_key = f"{class_name}_{track_id}"
        else:
            object_key = f"{class_name}_{x1}_{y1}"


        # ==========================================
        # DEFAULT = SAFE
        # ==========================================

        color = (0, 255, 0)

        if track_id != -1:
            label = (
                f"{class_name.upper()} "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )
        else:
            label = (
                f"{class_name.upper()} "
                f"{confidence:.2f}"
            )


        # ==========================================
        # INTRUSION DETECTION
        #
        # TOP OF OBJECT ENTERS TOP ZONE
        # ==========================================

        is_intruder = False

        if y1 < ZONE_Y:
            is_intruder = True


        # ==========================================
        # INTRUDER ACTION
        # ==========================================

        if is_intruder:

            # Change box to RED
            color = (0, 0, 255)

            # Change label
            label = (
                f"INTRUDER: "
                f"{class_name.upper()}"
            )

            # Add ID if available
            if track_id != -1:
                label += f" ID:{track_id}"

            # Add confidence
            label += f" {confidence:.2f}"


            # ==========================================
            # SAVE ALERT ONLY ONCE
            # ==========================================

            if object_key not in alerted_objects:

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                filename = (
                    f"alerts/intruder_"
                    f"{class_name}_"
                    f"ID_{track_id}_"
                    f"{timestamp}.jpg"
                )

                # Save evidence image
                cv2.imwrite(
                    filename,
                    output_frame
                )

                print("\n🚨 INTRUSION DETECTED!")

                print(
                    f"Object: "
                    f"{class_name.upper()}"
                )

                print(
                    f"Confidence: "
                    f"{confidence:.2f}"
                )

                print(
                    f"ID: "
                    f"{track_id}"
                )

                print(
                    f"Evidence: "
                    f"{filename}\n"
                )

                # Remember alerted object
                alerted_objects.add(object_key)


        # ==========================================
        # DRAW OBJECT BOX
        # ==========================================

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )


        # ==========================================
        # DRAW LABEL
        # ==========================================

        cv2.putText(
            output_frame,
            label,
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # ==========================================
    # SHOW VIDEO
    # ==========================================

    cv2.imshow(
        "AegisVision - Border Surveillance",
        output_frame
    )


    # ==========================================
    # PRESS Q TO STOP
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("ℹ️ Detection stopped by user.")
        break


# ==========================================
# CLEANUP
# ==========================================

video.release()
cv2.destroyAllWindows()

print("✅ AegisVision Detection Module Closed.")

