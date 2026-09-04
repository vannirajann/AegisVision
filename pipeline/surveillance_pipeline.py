import sys
import os
import time

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================
# AegisVision Analytics Modules
# ==============================

from analytics.video.video_source import VideoSource
from analytics.face.face_detector import FaceDetector
from analytics.movement.movement_detector import MovementDetector
from analytics.night.night_detector import NightDetector
from analytics.tracking.person_tracker import PersonTracker

from analytics.events.event_generator import EventGenerator
from analytics.alerts.alert_manager import AlertManager
from analytics.evidence.evidence_manager import EvidenceManager

from analytics.summary.analytics_summary import AnalyticsSummary

# Loitering and suspicious activity
from analytics.loitering.loitering_detector import LoiteringDetector
from analytics.suspicious.suspicious_activity_detector import (
    SuspiciousActivityDetector
)


class SurveillancePipeline:

    def __init__(self):

        print("Initializing AegisVision pipeline...")

        # ==============================
        # Video Input
        # ==============================

        self.video_source = VideoSource(0)

        # ==============================
        # Detection Modules
        # ==============================

        self.face_detector = FaceDetector()
        self.movement_detector = MovementDetector()
        self.night_detector = NightDetector()

        # ==============================
        # Tracking
        # ==============================

        self.person_tracker = PersonTracker()

        # ==============================
        # Security Modules
        # ==============================

        self.event_generator = EventGenerator()
        self.alert_manager = AlertManager()
        self.evidence_manager = EvidenceManager()

        # ==============================
        # Analytics
        # ==============================

        self.analytics_summary = AnalyticsSummary()

        # ==============================
        # Loitering Detection
        # ==============================

        self.loitering_detector = LoiteringDetector(
            threshold_seconds=10
        )

        # ==============================
        # Suspicious Activity Detection
        # ==============================

        self.suspicious_detector = SuspiciousActivityDetector()

        # ==============================
        # Event Cooldown
        # ==============================

        self.last_event_time = 0
        self.event_cooldown = 2.0

        print("All AegisVision modules loaded successfully")


    def initialize(self):

        self.video_source.open()

        print("Video source initialized successfully")
        print("Surveillance pipeline ready")


    def process_frame(self, frame):

        # ==========================================
        # 1. Face / Person Detection
        # ==========================================

        faces = self.face_detector.detect_faces(frame)

        # ==========================================
        # 2. Movement Detection
        # ==========================================

        movement_detected = (
            self.movement_detector.detect_movement(frame)
        )

        # ==========================================
        # 3. Low-Light / Night Detection
        # ==========================================

        night_result = self.night_detector.detect(frame)

        low_light = night_result["low_light"]
        brightness = night_result["brightness"]
        light_status = night_result["status"]

        # ==========================================
        # 4. Person Tracking
        # ==========================================

        people = self.person_tracker.update(faces)

        # ==========================================
        # 5. Event Generation
        # ==========================================

        event = None

        current_time = time.time()

        if movement_detected:

            if (
                current_time - self.last_event_time
                >= self.event_cooldown
            ):

                person_id = None

                if len(people) > 0:
                    person_id = people[0].get("id")

                # ----------------------------------
                # Night Movement
                # ----------------------------------

                if low_light:

                    event_type = "night_movement"
                    message = "Night-time movement detected"

                # ----------------------------------
                # Normal Movement
                # ----------------------------------

                else:

                    event_type = "movement"
                    message = "Movement detected"

                # ----------------------------------
                # Create Event
                # ----------------------------------

                event = self.event_generator.create_event(
                    event_type=event_type,
                    person_id=person_id,
                    details={
                        "message": message,
                        "people_count": len(people),
                        "brightness": brightness,
                        "light_status": light_status
                    }
                )

                self.last_event_time = current_time

                print("EVENT GENERATED:", event)

                # ----------------------------------
                # Generate Alert
                # ----------------------------------

                alert = self.alert_manager.create_alert(
                    event_type=event_type,
                    person_id=person_id
                )

                print("ALERT GENERATED:", alert)

                # ----------------------------------
                # Save Evidence
                # ----------------------------------

                evidence_path = (
                    self.evidence_manager.save_evidence(
                        event_type=event_type,
                        person_id=person_id,
                        details={
                            "message": message,
                            "people_count": len(people),
                            "brightness": brightness,
                            "light_status": light_status,
                            "alert_level": alert["level"]
                        }
                    )
                )

                print("EVIDENCE SAVED:", evidence_path)

        # ==========================================
        # 6. Analytics Summary
        # ==========================================

        self.analytics_summary.update_people(
            len(people)
        )

        if movement_detected:

            self.analytics_summary.add_movement()

        if event is not None:

            self.analytics_summary.add_event()

        # ==========================================
        # 7. Return Results
        # ==========================================

        return {

            "faces": faces,

            "people": people,

            "movement": movement_detected,

            "low_light": low_light,

            "brightness": brightness,

            "light_status": light_status,

            "event": event,

            "summary":
                self.analytics_summary.get_summary()
        }


    def release(self):

        self.video_source.release()

        print("Surveillance pipeline stopped")


# ==================================================
# Main Live Surveillance
# ==================================================

def main():

    import cv2

    print(
        "Starting AegisVision surveillance pipeline..."
    )

    pipeline = SurveillancePipeline()

    try:

        # ==========================================
        # Initialize Pipeline
        # ==========================================

        pipeline.initialize()

        print("Live surveillance started")
        print("Press Q to quit")

        # ==========================================
        # Main Camera Loop
        # ==========================================

        while True:

            ret, frame = (
                pipeline.video_source.read()
            )

            if not ret or frame is None:

                print(
                    "Could not read camera frame"
                )

                break

            # ======================================
            # Process Frame
            # ======================================

            results = pipeline.process_frame(
                frame
            )

            faces = results["faces"]
            people = results["people"]
            movement = results["movement"]
            event = results["event"]

            # ======================================
            # Draw Face / Person Boxes
            # ======================================

            for face in faces:

                x = face["x"]
                y = face["y"]
                w = face["width"]
                h = face["height"]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                if "id" in face:

                    cv2.putText(
                        frame,
                        f"ID:{face['id']}",
                        (
                            x,
                            max(y - 10, 20)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # ======================================
            # Face Count
            # ======================================

            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ======================================
            # Person Count
            # ======================================

            cv2.putText(
                frame,
                f"People: {len(people)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ======================================
            # Movement Status
            # ======================================

            movement_text = (
                "MOVEMENT DETECTED"
                if movement
                else "NO MOVEMENT"
            )

            cv2.putText(
                frame,
                movement_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ======================================
            # Light Status
            # ======================================

            cv2.putText(
                frame,
                results["light_status"],
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ======================================
            # Event Notification
            # ======================================

            if event is not None:

                cv2.putText(
                    frame,
                    "EVENT GENERATED",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # ======================================
            # Show Live Camera
            # ======================================

            try:

                cv2.imshow(
                    "AegisVision - Surveillance Pipeline",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            except cv2.error:

                print(
                    "[INFO] OpenCV GUI backend unavailable. "
                    "Running in headless terminal mode."
                )

                print(
                    "Summary:",
                    results["summary"]
                )

                max_headless_frames = 5

                if (
                    getattr(
                        main,
                        "headless_frame_count",
                        0
                    )
                    >= max_headless_frames
                ):

                    print(
                        "Headless mode limit reached. "
                        "Stopping gracefully."
                    )

                    break

                main.headless_frame_count = (
                    getattr(
                        main,
                        "headless_frame_count",
                        0
                    ) + 1
                )

                continue

    finally:

        pipeline.release()

        try:

            cv2.destroyAllWindows()

        except cv2.error:

            pass

        print(
            "AegisVision surveillance stopped"
        )


# ==================================================
# Program Entry Point
# ==================================================

if __name__ == "__main__":

    main()