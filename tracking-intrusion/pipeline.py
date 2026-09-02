from geometry import get_center_point
from tracker import CentroidTracker
from fence import VirtualLine, LineCrossingDetector, RestrictedZone, ZoneEntryDetector
from events import create_intrusion_event


class IntrusionPipeline:
    """
    The main entry point for the tracking-intrusion module.
    Wires together tracking, fence/zone checks, and event generation.
    Person 1 -> feeds detections in here.
    Person 5 -> receives the events this produces.
    """

    def __init__(self, line_points=None, zone_points=None):
        self.tracker = CentroidTracker()

        self.crossing_detector = None
        if line_points:
            line = VirtualLine(line_points[0], line_points[1])
            self.crossing_detector = LineCrossingDetector(line)

        self.zone_detector = None
        if zone_points:
            zone = RestrictedZone(zone_points)
            self.zone_detector = ZoneEntryDetector(zone)

    def process_frame(self, detections):
        """
        detections: list of Person 1's detection dicts, e.g.
        [{"class": "person", "confidence": 0.95, "bbox": {...}}, ...]

        Returns: list of intrusion event dicts (empty list if nothing happened this frame)
        """
        # Step A: turn each detection's bbox into a center point
        points = []
        detection_info = []  # keep class + confidence lined up with each point
        for det in detections:
            point = get_center_point(det["bbox"])
            points.append(point)
            detection_info.append({
                "object_type": det["class"],
                "confidence": det["confidence"]
            })

        # Step B: run tracking -> get track_id for each point
        tracked_objects = self.tracker.update(points)
        # tracked_objects = {track_id: point}

        events = []

        # Step C: for each currently tracked object, check crossing/entry
        # We match tracked points back to their detection info by point value.
        for track_id, point in tracked_objects.items():
            # Find the detection info matching this point (same frame, same position)
            info = None
            for i, p in enumerate(points):
                if p == point:
                    info = detection_info[i]
                    break
            if info is None:
                continue  # object wasn't seen this frame (e.g. briefly missing)

            if self.crossing_detector:
                if self.crossing_detector.check_crossing(track_id, point):
                    events.append(create_intrusion_event(
                        object_type=info["object_type"],
                        track_id=track_id,
                        confidence=info["confidence"]
                    ))

            if self.zone_detector:
                if self.zone_detector.check_entry(track_id, point):
                    events.append(create_intrusion_event(
                        object_type=info["object_type"],
                        track_id=track_id,
                        confidence=info["confidence"]
                    ))

        return events