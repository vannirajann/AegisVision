import os


# ============================================================
# AEGISVISION CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
STATIC_DIR = os.path.join(BASE_DIR, "static")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

os.makedirs(ALERTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


# ============================================================
# YOLO
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)


# ============================================================
# YOLO CLASSES
#
# 0  person
# 1  bicycle
# 2  car
# 3  motorcycle
# 5  bus
# 7  truck
# 16 dog
# ============================================================

ALLOWED_CLASSES = [
    0,
    1,
    2,
    3,
    5,
    7,
    16
]


# ============================================================
# DETECTION SETTINGS
# ============================================================

# Lower value helps detect objects in dark CCTV footage.
DETECTION_CONFIDENCE = 0.15

# Larger image improves detection of smaller objects such as dogs.
DETECTION_IMAGE_SIZE = 960


# ============================================================
# SINGLE CAMERA
# ============================================================

INPUT_SOURCE_TYPE = "video"

VIDEO_PATH = os.path.join(
    VIDEOS_DIR,
    "test.mp4"
)

RTSP_URL = ""


# ============================================================
# 4 CAMERA CONFIGURATION
#
# Coordinates are normalized:
#
# x = 0.0 left
# x = 1.0 right
#
# y = 0.0 top
# y = 1.0 bottom
#
# These coordinates are specifically adjusted according
# to the physical fences visible in your four videos.
# ============================================================

CAMERA_CONFIGS = {

    # ========================================================
    # CAMERA 1
    # ========================================================
    #
    # Physical fence:
    # upper-left -> lower-middle
    #
    # The old line was going in the opposite direction.
    # ========================================================

    1: {
        "name": "Camera 1",

        "source": os.path.join(
            VIDEOS_DIR,
            "camera1.mp4"
        ),

        "fence": (
            (0.20, 0.22),
            (0.66, 0.98)
        )
    },


    # ========================================================
    # CAMERA 2
    # ========================================================
    #
    # Physical fence:
    # lower-left -> upper-right
    # ========================================================

    2: {
        "name": "Camera 2",

        "source": os.path.join(
            VIDEOS_DIR,
            "camera2.mp4"
        ),

        "fence": (
            (0.06, 0.78),
            (0.68, 0.14)
        )
    },


    # ========================================================
    # CAMERA 3
    # ========================================================
    #
    # Physical fence is almost horizontal.
    #
    # IMPORTANT:
    # The old red line was too LOW.
    # The actual fence is around the upper-middle part.
    # ========================================================

    3: {
        "name": "Camera 3",

        "source": os.path.join(
            VIDEOS_DIR,
            "camera3.mp4"
        ),

        "fence": (
            (0.08, 0.34),
            (0.94, 0.34)
        )
    },


    # ========================================================
    # CAMERA 4
    # ========================================================
    #
    # Physical fence:
    # lower-left -> upper-right
    # ========================================================

    4: {
        "name": "Camera 4",

        "source": os.path.join(
            VIDEOS_DIR,
            "camera4.mp4"
        ),

        "fence": (
            (0.04, 0.68),
            (0.72, 0.17)
        )
    }
}


# ============================================================
# DISPLAY
# ============================================================

TILE_WIDTH = 640
TILE_HEIGHT = 360

GRID_WIDTH = 1280
GRID_HEIGHT = 720