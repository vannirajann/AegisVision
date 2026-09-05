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

        # Prevent repeated suspicious alerts
        # for the same person after loitering starts
        self.loitering_alerted_ids = set()

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
        # 5. Loitering Detection
        # ==========================================

        loitering_people = []

        current_track_ids = set()

        for person in people:

            person_id = person.get("id")

            if person_id is None:
                continue

            current_track_ids.add(person_id)

            # Start timer for newly detected person
            self.loitering_detector.person_entered(person_id)

            # Get current loitering status
            loitering_status = (
                self.loitering_detector.get_status(person_id)
            )

            if loitering_status["loitering"]:

                loitering_people.append(
                    loitering_status
                )

        # Remove timers for people who are no longer tracked
        tracked_ids = set(
            self.person_tracker.tracked_people.keys()
        )

        for track_id in list(
            self.loitering_detector.person_start_times.keys()
        ):

            if track_id not in tracked_ids:

                self.loitering_detector.person_left(
                    track_id
                )

                self.loitering_alerted_ids.discard(
                    track_id
                )

        # ==========================================
        # 6. Suspicious Activity Detection
        # ==========================================

        suspicious_events = []

        # Check each currently tracked person
        for person in people:

            person_id = person.get("id")

            if person_id is None:
                continue

            loitering_status = (
                self.loitering_detector.get_status(
                    person_id
                )
            )

            is_loitering = loitering_status["loitering"]

            # Night movement is considered suspicious
            night_movement = (
                movement_detected and low_light
            )

            suspicious_event = (
                self.suspicious_detector.analyze(
                    track_id=person_id,
                    loitering=is_loitering,
                    repeated_entry=False,
                    night_movement=night_movement
                )
            )

            if suspicious_event is not None:

                suspicious_events.append(
                    suspicious_event
                )

        # ==========================================
        # 7. Event Generation
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

                print(
                    "EVIDENCE SAVED:",
                    evidence_path
                )

        # ==========================================
        # 8. Loitering Events / Alerts
        # ==========================================

        for loitering_status in loitering_people:

            person_id = loitering_status["track_id"]

            # Generate only once for each loitering episode
            if person_id in self.loitering_alerted_ids:
                continue

            self.loitering_alerted_ids.add(person_id)

            print(
                "LOITERING DETECTED:",
                loitering_status
            )

            loitering_event = (
                self.event_generator.create_event(
                    event_type="loitering",
                    person_id=person_id,
                    details={
                        "message": (
                            "Person remained in "
                            "camera view for too long"
                        ),
                        "duration_seconds": (
                            loitering_status[
                                "duration_seconds"
                            ]
                        ),
                        "threshold_seconds": (
                            self.loitering_detector
                            .threshold_seconds
                        )
                    }
                )
            )

            print(
                "LOITERING EVENT GENERATED:",
                loitering_event
            )

            loitering_alert = (
                self.alert_manager.create_alert(
                    event_type="loitering",
                    person_id=person_id
                )
            )

            print(
                "LOITERING ALERT GENERATED:",
                loitering_alert
            )

            evidence_path = (
                self.evidence_manager.save_evidence(
                    event_type="loitering",
                    person_id=person_id,
                    details={
                        "duration_seconds": (
                            loitering_status[
                                "duration_seconds"
                            ]
                        ),
                        "threshold_seconds": (
                            self.loitering_detector
                            .threshold_seconds
                        ),
                        "alert_level": (
                            loitering_alert["level"]
                        )
                    }
                )
            )

            print(
                "LOITERING EVIDENCE SAVED:",
                evidence_path
            )

        # ==========================================
        # 9. Suspicious Activity Events
        # ==========================================

        for suspicious_event in suspicious_events:

            person_id = suspicious_event.get(
                "track_id"
            )

            reasons = suspicious_event.get(
                "reasons",
                []
            )

            # Avoid generating repeated suspicious
            # events continuously for loitering.
            if (
                "Person stayed too long in "
                "restricted zone" in reasons
                and person_id in self.loitering_alerted_ids
            ):
                continue

            print(
                "SUSPICIOUS ACTIVITY DETECTED:",
                suspicious_event
            )

            suspicious_alert = (
                self.alert_manager.create_alert(
                    event_type="suspicious_activity",
                    person_id=person_id
                )
            )

            print(
                "SUSPICIOUS ALERT GENERATED:",
                suspicious_alert
            )

            evidence_path = (
                self.evidence_manager.save_evidence(
                    event_type="suspicious_activity",
                    person_id=person_id,
                    details={
                        "severity": (
                            suspicious_event[
                                "severity"
                            ]
                        ),
                        "reasons": reasons,
                        "alert_level": (
                            suspicious_alert["level"]
                        )
                    }
                )
            )

            print(
                "SUSPICIOUS EVIDENCE SAVED:",
                evidence_path
            )

        # ==========================================
        # 10. Analytics Summary
        # ==========================================

        self.analytics_summary.update_people(
            len(people)
        )

        if movement_detected:

            self.analytics_summary.add_movement()

        if event is not None:

            self.analytics_summary.add_event()

        # ==========================================
        # 11. Return Results
        # ==========================================

        return {

            "faces": faces,

            "people": people,

            "movement": movement_detected,

            "low_light": low_light,

            "brightness": brightness,

            "light_status": light_status,

            "event": event,

            "loitering": loitering_people,

            "suspicious_activity": suspicious_events,

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
            # Loitering Status
            # ======================================

            loitering_count = len(
                results["loitering"]
            )

            cv2.putText(
                frame,
                f"Loitering: {loitering_count}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ======================================
            # Suspicious Activity Status
            # ======================================

            suspicious_count = len(
                results["suspicious_activity"]
            )

            cv2.putText(
                frame,
                f"Suspicious: {suspicious_count}",
                (20, 210),
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
                    (20, 245),
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