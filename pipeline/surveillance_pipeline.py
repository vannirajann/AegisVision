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

        # Detection and tracking
        self.face_detector = FaceDetector()
        self.movement_detector = MovementDetector()
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

        # 3. Person tracking
        people = self.person_tracker.update(faces)

        event = None
        alert = None
        evidence = None

        current_time = time.time()

        # 4. Generate event
        if movement_detected:

            if current_time - self.last_event_time >= self.event_cooldown:

                person_id = None

                if len(people) > 0:
                    person_id = people[0].get("id")

                event = self.event_generator.create_event(
                    event_type="movement",
                    person_id=person_id,
                    details={
                        "message": "Movement detected",
                        "people_count": len(people)
                    }
                )

                print("EVENT GENERATED:", event)

                # 5. Generate alert
                alert = self.alert_manager.create_alert(
                    event_type="movement",
                    person_id=person_id
                )

                print("ALERT GENERATED:", alert)

                # 6. Save evidence
                evidence = self.evidence_manager.save_evidence(
                    event_type="movement",
                    person_id=person_id,
                    details={
                        "message": "Movement detected",
                        "people_count": len(people),
                        "alert_level": alert["level"]
                    }
                )

                print("EVIDENCE SAVED:", evidence)

                self.last_event_time = current_time

        return {
            "faces": faces,
            "people": people,
            "movement": movement_detected,
            "event": event,
            "alert": alert,
            "evidence": evidence
        }

    def release(self):

        self.video_source.release()

        print("Surveillance pipeline stopped")