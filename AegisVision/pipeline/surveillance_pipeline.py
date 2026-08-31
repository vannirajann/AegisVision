import sys
import os
import time

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from analytics.video.video_source import VideoSource
from analytics.face.face_detector import FaceDetector
from analytics.movement.movement_detector import MovementDetector
from analytics.night.night_detector import NightDetector
from analytics.tracking.person_tracker import PersonTracker
from analytics.events.event_generator import EventGenerator
from analytics.alerts.alert_manager import AlertManager
from analytics.evidence.evidence_manager import EvidenceManager
from analytics.summary.analytics_summary import AnalyticsSummary


class SurveillancePipeline:

    def __init__(self):

        print("Initializing AegisVision pipeline...")

        # Video input
        self.video_source = VideoSource(0)

        # Detection modules
        self.face_detector = FaceDetector()
        self.movement_detector = MovementDetector()
        self.night_detector = NightDetector()

        # Tracking
        self.person_tracker = PersonTracker()

        # Security modules
        self.event_generator = EventGenerator()
        self.alert_manager = AlertManager()
        self.evidence_manager = EvidenceManager()

        # Analytics
        self.analytics_summary = AnalyticsSummary()

        # Event cooldown
        self.last_event_time = 0
        self.event_cooldown = 2.0

        print("All AegisVision modules loaded successfully")

    def initialize(self):

        self.video_source.open()

        print("Video source initialized successfully")
        print("Surveillance pipeline ready")

    def process_frame(self, frame):

        # 1. Face detection
        faces = self.face_detector.detect_faces(frame)

        # 2. Movement detection
        movement_detected = self.movement_detector.detect_movement(frame)

        # 3. Low-light / night detection
        night_result = self.night_detector.detect(frame)

        low_light = night_result["low_light"]
        brightness = night_result["brightness"]
        light_status = night_result["status"]

        # 4. Person tracking
        people = self.person_tracker.update(faces)

        # 5. Event generation
        event = None

        current_time = time.time()

        if movement_detected:

            if current_time - self.last_event_time >= self.event_cooldown:

                person_id = None

                if len(people) > 0:
                    person_id = people[0].get("id")

                # Night movement
                if low_light:

                    event_type = "night_movement"
                    message = "Night-time movement detected"

                # Normal movement
                else:

                    event_type = "movement"
                    message = "Movement detected"

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

                # Generate alert
                alert = self.alert_manager.create_alert(
                    event_type=event_type,
                    person_id=person_id
                )

                print("ALERT GENERATED:", alert)

                # Save evidence
                evidence_path = self.evidence_manager.save_evidence(
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

                print("EVIDENCE SAVED:", evidence_path)

        return {
            "faces": faces,
            "people": people,
            "movement": movement_detected,
            "low_light": low_light,
            "brightness": brightness,
            "light_status": light_status,
            "event": event
        }

    def release(self):

        self.video_source.release()

        print("Surveillance pipeline stopped")