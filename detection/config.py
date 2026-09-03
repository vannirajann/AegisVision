import os


# ==========================================
# BASE DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================
# PROJECT FOLDERS
# ==========================================

ALERTS_DIR = os.path.join(
    BASE_DIR,
    "alerts"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


# ==========================================
# YOLO MODEL
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)


# ==========================================
# ALLOWED DETECTION CLASSES
# ==========================================

ALLOWED_CLASSES = [
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7   # truck
]


# ==========================================
# INPUT SOURCE
#
# Choose:
# "video"
# "rtsp"
# ==========================================

INPUT_SOURCE_TYPE = "video"


# ==========================================
# VIDEO FILE
# ==========================================

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)


# ==========================================
# RTSP / IP CCTV
# ==========================================

RTSP_URL = ""


# ==========================================
# CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs(
    ALERTS_DIR,
    exist_ok=True
)

os.makedirs(
    STATIC_DIR,
    exist_ok=True
)