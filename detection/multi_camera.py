import cv2
import os
import time
import threading
import numpy as np
from datetime import datetime

from ultralytics import YOLO

from config import (
    MODEL_PATH,
    ALLOWED_CLASSES,
    CAMERA_CONFIGS,
    TILE_WIDTH,
    TILE_HEIGHT,
    ALERTS_DIR,
    DETECTION_CONFIDENCE,
    DETECTION_IMAGE_SIZE
)


# ============================================================
# AEGISVISION - 4 CAMERA BORDER SURVEILLANCE
# ============================================================

model = YOLO(MODEL_PATH)

# Dog = 16
YOLO_CLASSES = sorted(
    set(ALLOWED_CLASSES + [16])
)

model_lock = threading.Lock()


# ============================================================
# CLASSES
# ============================================================

PERSON_CLASS = "person"

VEHICLE_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}

DOG_CLASS = "dog"


# ============================================================
# INTERNAL TRACK SETTINGS
# ============================================================

# Maximum distance an object can move between frames
# and still be considered the same object when YOLO
# does not provide a track ID.
MAX_TRACK_DISTANCE = 140

# A very large movement is useful for detecting a jump.
JUMP_DISTANCE = 25


# ============================================================
# CAMERA PROCESSOR
# ============================================================

class CameraProcessor:

    def __init__(
        self,
        camera_id,
        camera_config
    ):

        self.camera_id = camera_id
        self.name = camera_config["name"]
        self.source = camera_config["source"]

        # ====================================================
        # IMPORTANT
        #
        # FENCE IS TAKEN DIRECTLY FROM CONFIG.PY.
        # NO FENCE COORDINATES ARE CHANGED.
        # ====================================================

        self.fence_normalized = camera_config["fence"]

        self.cap = None
        self.running = False
        self.thread = None

        self.lock = threading.Lock()
        self.latest_frame = None

        self.frame_id = 0

        # ====================================================
        # YOLO TRACKS
        # ====================================================

        self.seen_tracks = set()
        self.seen_person_tracks = set()
        self.seen_vehicle_tracks = set()
        self.seen_dog_tracks = set()

        # ====================================================
        # INTERNAL TRACKER
        #
        # This is the important fix.
        #
        # Even if YOLO says:
        #
        # Track ID: None
        #
        # we create our own ID.
        # ====================================================

        self.next_internal_id = 1

        self.internal_tracks = {}

        # Format:
        #
        # internal_id: {
        #     "class": "person",
        #     "center": (x,y),
        #     "bottom": (x,y),
        #     "side": -1/1/0,
        #     "last_frame": frame_number
        # }

        # ====================================================
        # INTRUDERS
        # ====================================================

        self.intruder_tracks = set()

        self.total_intruders = 0

        # ====================================================
        # POSITION HISTORY
        # ====================================================

        self.previous_points = {}
        self.previous_sides = {}
        self.previous_centers = {}

        # ====================================================
        # CURRENT COUNTS
        # ====================================================

        self.current_detections = []

        self.current_objects = 0
        self.current_persons = 0
        self.current_vehicles = 0
        self.current_dogs = 0

        # ====================================================
        # TOTAL COUNTS
        # ====================================================

        self.total_objects = 0
        self.total_persons = 0
        self.total_vehicles = 0
        self.total_dogs = 0


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    def open_video(self):

        if self.cap is not None:

            try:
                self.cap.release()
            except:
                pass

            self.cap = None

        self.cap = cv2.VideoCapture(
            self.source
        )

        if not self.cap.isOpened():

            print(
                f"❌ Camera {self.camera_id}: "
                f"Cannot open:"
            )

            print(self.source)

            return False

        try:

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        except:
            pass

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = self.cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        print(
            f"✅ Camera {self.camera_id}: Connected"
        )

        print(
            f"   Source: {self.source}"
        )

        print(
            f"   FPS: {fps:.2f}"
        )

        print(
            f"   Frames: {int(frame_count)}"
        )

        return True


    # ========================================================
    # GET FENCE POINTS
    # ========================================================

    def get_fence_points(
        self,
        width,
        height
    ):

        # ====================================================
        # EXACT VALUES FROM CONFIG.PY
        # ====================================================

        (x1, y1), (x2, y2) = (
            self.fence_normalized
        )

        p1 = (
            int(x1 * width),
            int(y1 * height)
        )

        p2 = (
            int(x2 * width),
            int(y2 * height)
        )

        return p1, p2


    # ========================================================
    # SIDE OF FENCE
    # ========================================================

    def get_side(
        self,
        point,
        fence_p1,
        fence_p2
    ):

        px, py = point

        x1, y1 = fence_p1
        x2, y2 = fence_p2

        value = (
            (x2 - x1) * (py - y1)
            -
            (y2 - y1) * (px - x1)
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0


    # ========================================================
    # LINE INTERSECTION
    # ========================================================

    def line_segments_intersect(
        self,
        p1,
        p2,
        q1,
        q2
    ):

        def orientation(a, b, c):

            value = (
                (b[1] - a[1]) *
                (c[0] - b[0])
                -
                (b[0] - a[0]) *
                (c[1] - b[1])
            )

            if abs(value) < 0.00001:
                return 0

            if value > 0:
                return 1

            return 2

        def on_segment(a, b, c):

            return (
                min(a[0], c[0])
                <= b[0]
                <= max(a[0], c[0])
                and
                min(a[1], c[1])
                <= b[1]
                <= max(a[1], c[1])
            )

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True

        if (
            o1 == 0
            and
            on_segment(p1, q1, p2)
        ):
            return True

        if (
            o2 == 0
            and
            on_segment(p1, q2, p2)
        ):
            return True

        if (
            o3 == 0
            and
            on_segment(q1, p1, q2)
        ):
            return True

        if (
            o4 == 0
            and
            on_segment(q1, p2, q2)
        ):
            return True

        return False


    # ========================================================
    # FIND INTERNAL TRACK
    #
    # THIS FIXES Track ID: None
    # ========================================================

    def find_internal_track(
        self,
        class_name,
        center,
        bottom,
        current_side
    ):

        best_id = None
        best_distance = float("inf")

        # ----------------------------------------------------
        # Search existing tracks
        # ----------------------------------------------------

        for internal_id, data in self.internal_tracks.items():

            if data["class"] != class_name:
                continue

            # Ignore very old tracks
            if (
                self.frame_id
                -
                data["last_frame"]
                > 10
            ):
                continue

            old_center = data["center"]

            distance = np.hypot(
                center[0] - old_center[0],
                center[1] - old_center[1]
            )

            if (
                distance
                <
                MAX_TRACK_DISTANCE
                and
                distance
                <
                best_distance
            ):

                best_distance = distance
                best_id = internal_id

        # ----------------------------------------------------
        # Existing track found
        # ----------------------------------------------------

        if best_id is not None:

            self.internal_tracks[
                best_id
            ] = {

                "class":
                    class_name,

                "center":
                    center,

                "bottom":
                    bottom,

                "side":
                    current_side,

                "last_frame":
                    self.frame_id
            }

            return best_id

        # ----------------------------------------------------
        # New internal track
        # ----------------------------------------------------

        internal_id = self.next_internal_id

        self.next_internal_id += 1

        self.internal_tracks[
            internal_id
        ] = {

            "class":
                class_name,

            "center":
                center,

            "bottom":
                bottom,

            "side":
                current_side,

            "last_frame":
                self.frame_id
        }

        return internal_id


    # ========================================================
    # CLEAN OLD INTERNAL TRACKS
    # ========================================================

    def clean_old_tracks(self):

        remove_ids = []

        for internal_id, data in (
            self.internal_tracks.items()
        ):

            if (
                self.frame_id
                -
                data["last_frame"]
                >
                30
            ):

                remove_ids.append(
                    internal_id
                )

        for internal_id in remove_ids:

            self.internal_tracks.pop(
                internal_id,
                None
            )


    # ========================================================
    # SAVE ALERT SCREENSHOT
    # ========================================================

    def save_evidence(
        self,
        frame,
        class_name,
        internal_id,
        yolo_track_id
    ):

        camera_folder = os.path.join(
            ALERTS_DIR,
            "multi_camera",
            f"camera_{self.camera_id}"
        )

        os.makedirs(
            camera_folder,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        if yolo_track_id is not None:

            track_text = (
                f"yolo_{yolo_track_id}_"
                f"internal_{internal_id}"
            )

        else:

            track_text = (
                f"internal_{internal_id}"
            )

        filename = (
            f"camera_{self.camera_id}_"
            f"INTRUDER_{class_name}_"
            f"{track_text}_"
            f"{timestamp}.jpg"
        )

        path = os.path.join(
            camera_folder,
            filename
        )

        # IMPORTANT:
        # frame already contains:
        # RED BOX
        # INTRUDER LABEL
        # RED ALERT BANNER
        # VIRTUAL FENCE

        success = cv2.imwrite(
            path,
            frame
        )

        if success:

            print(
                f"📸 ALERT SCREENSHOT SAVED:"
            )

            print(
                f"   {path}"
            )

        else:

            print(
                "❌ Failed to save alert screenshot"
            )

        return path


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        self.frame_id += 1

        height, width = (
            frame.shape[:2]
        )

        # ====================================================
        # FENCE
        # ====================================================

        fence_p1, fence_p2 = (
            self.get_fence_points(
                width,
                height
            )
        )

        # ====================================================
        # YOLO
        # ====================================================

        try:

            with model_lock:

                results = model.track(
                    frame,
                    persist=True,
                    classes=YOLO_CLASSES,
                    tracker="bytetrack.yaml",
                    conf=DETECTION_CONFIDENCE,
                    imgsz=DETECTION_IMAGE_SIZE,
                    verbose=False
                )

        except Exception as error:

            print(
                f"❌ Camera {self.camera_id} "
                f"YOLO error: {error}"
            )

            return frame.copy()

        output_frame = frame.copy()

        # ====================================================
        # DRAW FENCE
        # ====================================================

        cv2.line(
            output_frame,
            fence_p1,
            fence_p2,
            (0, 0, 255),
            5
        )

        cv2.circle(
            output_frame,
            fence_p1,
            7,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            output_frame,
            fence_p2,
            7,
            (0, 0, 255),
            -1
        )

        # ====================================================
        # FENCE LABEL
        # ====================================================

        label_x = (
            fence_p1[0]
            +
            fence_p2[0]
        ) // 2

        label_y = (
            fence_p1[1]
            +
            fence_p2[1]
        ) // 2

        cv2.putText(
            output_frame,
            "VIRTUAL FENCE",
            (
                max(label_x - 75, 10),
                max(label_y - 12, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        # ====================================================
        # COUNTERS
        # ====================================================

        current_objects = 0
        current_persons = 0
        current_vehicles = 0
        current_dogs = 0

        frame_detections = []

        intrusion_now = False

        # ====================================================
        # DETECTION LOOP
        # ====================================================

        if (
            results
            and
            len(results) > 0
            and
            results[0].boxes is not None
        ):

            for box in results[0].boxes:

                # =================================================
                # CLASS
                # =================================================

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                confidence = float(
                    box.conf[0]
                )

                if confidence < DETECTION_CONFIDENCE:
                    continue

                # =================================================
                # BOUNDING BOX
                # =================================================

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                )

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # =================================================
                # YOLO TRACK ID
                # =================================================

                yolo_track_id = None

                if box.id is not None:

                    yolo_track_id = int(
                        box.id[0]
                    )

                # =================================================
                # COUNTS
                # =================================================

                current_objects += 1

                if class_name == PERSON_CLASS:

                    current_persons += 1

                elif class_name in VEHICLE_CLASSES:

                    current_vehicles += 1

                elif class_name == DOG_CLASS:

                    current_dogs += 1

                # =================================================
                # POSITION
                # =================================================

                center = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                bottom_center = (
                    (x1 + x2) // 2,
                    y2
                )

                # =================================================
                # SIDE
                # =================================================

                current_side = self.get_side(
                    bottom_center,
                    fence_p1,
                    fence_p2
                )

                # =================================================
                # INTERNAL TRACK
                #
                # THIS WORKS EVEN IF YOLO TRACK ID = NONE.
                # =================================================

                internal_id = (
                    self.find_internal_track(
                        class_name,
                        center,
                        bottom_center,
                        current_side
                    )
                )

                track_key = (
                    f"{self.camera_id}_"
                    f"{class_name}_"
                    f"internal_{internal_id}"
                )

                # =================================================
                # TOTAL TRACKS
                # =================================================

                self.seen_tracks.add(
                    track_key
                )

                if class_name == PERSON_CLASS:

                    self.seen_person_tracks.add(
                        track_key
                    )

                elif class_name in VEHICLE_CLASSES:

                    self.seen_vehicle_tracks.add(
                        track_key
                    )

                elif class_name == DOG_CLASS:

                    self.seen_dog_tracks.add(
                        track_key
                    )

                # =================================================
                # PREVIOUS POSITION
                # =================================================

                previous_point = (
                    self.previous_points.get(
                        track_key
                    )
                )

                previous_side = (
                    self.previous_sides.get(
                        track_key
                    )
                )

                previous_center = (
                    self.previous_centers.get(
                        track_key
                    )
                )

                crossed_fence = False

                # =================================================
                # METHOD 1
                #
                # TRAJECTORY CROSSES FENCE
                #
                # MOST IMPORTANT FOR JUMPING.
                # =================================================

                if previous_point is not None:

                    if self.line_segments_intersect(
                        previous_point,
                        bottom_center,
                        fence_p1,
                        fence_p2
                    ):

                        crossed_fence = True

                # =================================================
                # METHOD 2
                #
                # SIDE CHANGED
                # =================================================

                if not crossed_fence:

                    if (
                        previous_side is not None
                        and
                        previous_side != 0
                        and
                        current_side != 0
                        and
                        previous_side != current_side
                    ):

                        crossed_fence = True

                # =================================================
                # METHOD 3
                #
                # CENTER CROSSED FENCE
                # =================================================

                if (
                    not crossed_fence
                    and
                    previous_center is not None
                ):

                    old_center_side = (
                        self.get_side(
                            previous_center,
                            fence_p1,
                            fence_p2
                        )
                    )

                    new_center_side = (
                        self.get_side(
                            center,
                            fence_p1,
                            fence_p2
                        )
                    )

                    if (
                        old_center_side != 0
                        and
                        new_center_side != 0
                        and
                        old_center_side
                        !=
                        new_center_side
                    ):

                        crossed_fence = True

                # =================================================
                # METHOD 4
                #
                # LARGE JUMP + FENCE INTERSECTION
                # =================================================

                if (
                    not crossed_fence
                    and
                    previous_point is not None
                ):

                    movement = np.hypot(
                        bottom_center[0]
                        -
                        previous_point[0],

                        bottom_center[1]
                        -
                        previous_point[1]
                    )

                    if movement >= JUMP_DISTANCE:

                        if self.line_segments_intersect(
                            previous_point,
                            bottom_center,
                            fence_p1,
                            fence_p2
                        ):

                            crossed_fence = True

                # =================================================
                # UPDATE HISTORY
                # =================================================

                self.previous_points[
                    track_key
                ] = bottom_center

                self.previous_sides[
                    track_key
                ] = current_side

                self.previous_centers[
                    track_key
                ] = center

                # =================================================
                # CHECK EXISTING INTRUDER
                # =================================================

                is_intruder = (
                    track_key
                    in
                    self.intruder_tracks
                )

                # =================================================
                # NEW CROSSING
                # =================================================

                if (
                    crossed_fence
                    and
                    track_key
                    not in
                    self.intruder_tracks
                ):

                    # =================================================
                    # ANY REAL CROSSING = INTRUSION
                    #
                    # No restricted-side filter.
                    # =================================================

                    is_intruder = True

                    self.intruder_tracks.add(
                        track_key
                    )

                    self.total_intruders += 1

                    intrusion_now = True

                    # =================================================
                    # RED BOX FIRST
                    # =================================================

                    cv2.rectangle(
                        output_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        7
                    )

                    # =================================================
                    # RED INTRUDER LABEL
                    # =================================================

                    alert_label = (
                        "INTRUDER: "
                        f"{class_name.upper()}"
                    )

                    alert_label += (
                        f" INT-ID:{internal_id}"
                    )

                    cv2.putText(
                        output_frame,
                        alert_label,
                        (
                            x1,
                            max(y1 - 15, 30)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        3
                    )

                    # =================================================
                    # RED ALERT BANNER
                    # =================================================

                    cv2.rectangle(
                        output_frame,
                        (0, 0),
                        (width, 55),
                        (0, 0, 180),
                        -1
                    )

                    cv2.putText(
                        output_frame,
                        "!!! INTRUSION ALERT !!!",
                        (15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.78,
                        (255, 255, 255),
                        2
                    )

                    # =================================================
                    # ALERT INFORMATION
                    # =================================================

                    cv2.putText(
                        output_frame,
                        "FENCE CROSSED",
                        (
                            x1,
                            min(
                                y2 + 30,
                                height - 20
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 0, 255),
                        3
                    )

                    # =================================================
                    # SAVE THE ACTUAL ALERT IMAGE
                    #
                    # It contains:
                    # - Person
                    # - RED BOX
                    # - INTRUDER LABEL
                    # - RED ALERT
                    # - Fence
                    # =================================================

                    evidence = (
                        self.save_evidence(
                            output_frame,
                            class_name,
                            internal_id,
                            yolo_track_id
                        )
                    )

                    # =================================================
                    # TERMINAL ALERT
                    # =================================================

                    print()
                    print(
                        "🚨🚨🚨 INTRUSION DETECTED 🚨🚨🚨"
                    )

                    print(
                        f"Camera: {self.name}"
                    )

                    print(
                        f"Object: "
                        f"{class_name.upper()}"
                    )

                    print(
                        f"Confidence: "
                        f"{confidence:.2f}"
                    )

                    print(
                        f"YOLO Track ID: "
                        f"{yolo_track_id}"
                    )

                    print(
                        f"Internal Track ID: "
                        f"{internal_id}"
                    )

                    print(
                        "Action: "
                        "Object crossed virtual fence"
                    )

                    print(
                        f"TOTAL INTRUDERS: "
                        f"{self.total_intruders}"
                    )

                    print(
                        f"Evidence: "
                        f"{evidence}"
                    )

                    print()

                # =================================================
                # PERSISTENT INTRUDER
                #
                # Once crossed:
                # ALWAYS RED while tracked.
                # =================================================

                if is_intruder:

                    box_color = (
                        0,
                        0,
                        255
                    )

                    label = (
                        "INTRUDER: "
                        f"{class_name.upper()}"
                    )

                    label += (
                        f" INT-ID:{internal_id}"
                    )

                    intrusion_now = True

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    label = (
                        class_name.upper()
                    )

                    label += (
                        f" INT-ID:{internal_id}"
                    )

                # =================================================
                # CONFIDENCE
                # =================================================

                label += (
                    f" {confidence:.2f}"
                )

                # =================================================
                # YOLO ID IF AVAILABLE
                # =================================================

                if yolo_track_id is not None:

                    label += (
                        f" ID:{yolo_track_id}"
                    )

                # =================================================
                # DRAW BOX
                # =================================================

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    5 if is_intruder else 3
                )

                # =================================================
                # DRAW LABEL
                # =================================================

                cv2.putText(
                    output_frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                # =================================================
                # BOTTOM CENTER
                # =================================================

                cv2.circle(
                    output_frame,
                    bottom_center,
                    5,
                    box_color,
                    -1
                )

                # =================================================
                # STRUCTURED OUTPUT
                # =================================================

                frame_detections.append({

                    "camera_id":
                        self.camera_id,

                    "class":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            2
                        ),

                    "track_id":
                        yolo_track_id,

                    "internal_track_id":
                        internal_id,

                    "bbox": {

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },

                    "crossed_fence":
                        crossed_fence,

                    "intruder":
                        is_intruder
                })

        # ========================================================
        # CLEAN OLD INTERNAL TRACKS
        # ========================================================

        self.clean_old_tracks()

        # ========================================================
        # CURRENT COUNTS
        # ========================================================

        self.current_objects = (
            current_objects
        )

        self.current_persons = (
            current_persons
        )

        self.current_vehicles = (
            current_vehicles
        )

        self.current_dogs = (
            current_dogs
        )

        self.current_detections = (
            frame_detections
        )

        # ========================================================
        # TOTAL COUNTS
        # ========================================================

        self.total_objects = len(
            self.seen_tracks
        )

        self.total_persons = len(
            self.seen_person_tracks
        )

        self.total_vehicles = len(
            self.seen_vehicle_tracks
        )

        self.total_dogs = len(
            self.seen_dog_tracks
        )

        # ========================================================
        # INTRUSION STATUS
        # ========================================================

        if intrusion_now:

            status_text = (
                "STATUS: INTRUSION"
            )

            status_color = (
                0,
                0,
                255
            )

            # ====================================================
            # RED SCREEN BORDER
            # ====================================================

            cv2.rectangle(
                output_frame,
                (3, 3),
                (
                    width - 3,
                    height - 3
                ),
                (0, 0, 255),
                8
            )

            # ====================================================
            # RED ALERT BANNER
            # ====================================================

            cv2.rectangle(
                output_frame,
                (0, 0),
                (width, 55),
                (0, 0, 180),
                -1
            )

            cv2.putText(
                output_frame,
                "!!! INTRUSION ALERT !!!",
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2
            )

        else:

            status_text = (
                "STATUS: SECURE"
            )

            status_color = (
                0,
                255,
                0
            )

        # ========================================================
        # CAMERA NAME
        # ========================================================

        cv2.rectangle(
            output_frame,
            (0, 55),
            (250, 92),
            (15, 15, 15),
            -1
        )

        cv2.putText(
            output_frame,
            self.name,
            (12, 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        # ========================================================
        # STATUS
        # ========================================================

        cv2.putText(
            output_frame,
            status_text,
            (15, height - 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2
        )

        # ========================================================
        # LIVE OBJECTS
        # ========================================================

        cv2.putText(
            output_frame,
            f"Live Objects: {current_objects}",
            (15, height - 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # ========================================================
        # PERSONS
        # ========================================================

        cv2.putText(
            output_frame,
            f"Persons: {current_persons}",
            (15, height - 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # ========================================================
        # VEHICLES
        # ========================================================

        cv2.putText(
            output_frame,
            f"Vehicles: {current_vehicles}",
            (160, height - 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # ========================================================
        # DOGS
        # ========================================================

        cv2.putText(
            output_frame,
            f"Dogs: {current_dogs}",
            (160, height - 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # ========================================================
        # TOTAL OBJECTS
        # ========================================================

        cv2.putText(
            output_frame,
            f"TOTAL: {self.total_objects}",
            (width - 150, height - 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ========================================================
        # TOTAL INTRUDERS
        # ========================================================

        cv2.putText(
            output_frame,
            f"INTRUDERS: {self.total_intruders}",
            (width - 205, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        return output_frame


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    def run(self):

        if not self.open_video():
            return

        self.running = True

        while self.running:

            success, frame = (
                self.cap.read()
            )

            if not success:

                print(
                    f"🔄 Camera {self.camera_id}: "
                    f"Video ended. Reopening..."
                )

                # Reset only movement history.
                #
                # Intruder history is NOT reset.

                self.previous_points.clear()
                self.previous_sides.clear()
                self.previous_centers.clear()

                self.internal_tracks.clear()

                if self.cap is not None:

                    self.cap.release()
                    self.cap = None

                time.sleep(0.3)

                if not self.open_video():

                    time.sleep(1)
                    continue

                continue

            # =================================================
            # PROCESS
            # =================================================

            output_frame = (
                self.process_frame(
                    frame
                )
            )

            with self.lock:

                self.latest_frame = (
                    output_frame.copy()
                )

        if self.cap is not None:

            self.cap.release()
            self.cap = None


    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.thread is not None:
            return

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )

        self.thread.start()


    # ========================================================
    # GET FRAME
    # ========================================================

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False


# ============================================================
# MULTI CAMERA MANAGER
# ============================================================

class MultiCameraManager:

    def __init__(self):

        self.cameras = {}

        for camera_id, config in (
            CAMERA_CONFIGS.items()
        ):

            self.cameras[
                camera_id
            ] = CameraProcessor(
                camera_id,
                config
            )


    # ========================================================
    # START
    # ========================================================

    def start(self):

        print()
        print(
            "=========================================="
        )

        print(
            " AEGISVISION 4-CAMERA SURVEILLANCE"
        )

        print(
            "=========================================="
        )

        print(
            f"Detection confidence: "
            f"{DETECTION_CONFIDENCE}"
        )

        print(
            f"Detection image size: "
            f"{DETECTION_IMAGE_SIZE}"
        )

        print(
            f"YOLO classes: "
            f"{YOLO_CLASSES}"
        )

        print()

        for camera in (
            self.cameras.values()
        ):

            camera.start()


    # ========================================================
    # 2x2 GRID
    # ========================================================

    def get_grid(self):

        frames = []

        for camera_id in sorted(
            self.cameras.keys()
        ):

            camera = self.cameras[
                camera_id
            ]

            frame = camera.get_frame()

            if frame is None:

                frame = (
                    self.create_waiting_frame(
                        camera.name
                    )
                )

            try:

                frame = cv2.resize(
                    frame,
                    (
                        TILE_WIDTH,
                        TILE_HEIGHT
                    )
                )

            except Exception:

                frame = (
                    self.create_waiting_frame(
                        camera.name
                    )
                )

            frames.append(frame)

        while len(frames) < 4:

            frames.append(
                self.create_waiting_frame(
                    f"Camera {len(frames) + 1}"
                )
            )

        # Make sure exactly four frames exist.

        frames = frames[:4]

        top = cv2.hconcat([
            frames[0],
            frames[1]
        ])

        bottom = cv2.hconcat([
            frames[2],
            frames[3]
        ])

        grid = cv2.vconcat([
            top,
            bottom
        ])

        # ====================================================
        # GLOBAL TOTALS
        # ====================================================

        total_intruders = sum(
            camera.total_intruders
            for camera
            in self.cameras.values()
        )

        total_objects = sum(
            camera.total_objects
            for camera
            in self.cameras.values()
        )

        total_persons = sum(
            camera.total_persons
            for camera
            in self.cameras.values()
        )

        total_dogs = sum(
            camera.total_dogs
            for camera
            in self.cameras.values()
        )

        # ====================================================
        # HEADER
        # ====================================================

        cv2.rectangle(
            grid,
            (0, 0),
            (1280, 42),
            (15, 15, 15),
            -1
        )

        cv2.putText(
            grid,
            "AEGISVISION - 4 CAMERA BORDER SURVEILLANCE",
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            grid,
            f"OBJECTS:{total_objects}",
            (700, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            grid,
            f"INTRUDERS:{total_intruders}",
            (850, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        cv2.putText(
            grid,
            f"PERSON:{total_persons}",
            (1010, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.putText(
            grid,
            f"DOG:{total_dogs}",
            (1140, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        return grid


    # ========================================================
    # WAITING SCREEN
    # ========================================================

    def create_waiting_frame(
        self,
        name
    ):

        frame = np.zeros(
            (
                TILE_HEIGHT,
                TILE_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        cv2.putText(
            frame,
            name,
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "WAITING FOR VIDEO...",
            (25, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2
        )

        return frame


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        for camera in (
            self.cameras.values()
        ):

            camera.stop()


# ============================================================
# GLOBAL MANAGER
# ============================================================

multi_camera_manager = (
    MultiCameraManager()
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    multi_camera_manager.start()

    print()
    print(
        "✅ Four cameras started."
    )

    print(
        "✅ Existing fence positions preserved."
    )

    print(
        "✅ Person detection enabled."
    )

    print(
        "✅ Vehicle detection enabled."
    )

    print(
        "✅ Dog detection enabled."
    )

    print(
        "✅ Internal tracking enabled."
    )

    print(
        "✅ Jump/crossing detection enabled."
    )

    print(
        "✅ Persistent red intruder detection enabled."
    )

    print(
        "✅ Alert screenshot capture enabled."
    )

    print()
    print(
        "Press Q to stop."
    )
    print()

    try:

        while True:

            grid = (
                multi_camera_manager.get_grid()
            )

            cv2.imshow(
                "AegisVision - 4 Camera Border Surveillance",
                grid
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                break

    except KeyboardInterrupt:

        print()
        print(
            "ℹ️ Stopping AegisVision..."
        )

    finally:

        multi_camera_manager.stop()

        cv2.destroyAllWindows()

        print(
            "✅ AegisVision stopped."
        )