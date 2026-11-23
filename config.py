"""Centralized configuration and constants for 'Esquiva la Silueta'."""

from pathlib import Path

# Virtual 16:9 HD arena & display settings
VIRTUAL_W = 1280
VIRTUAL_H = 720
WINDOW_WIDTH = VIRTUAL_W
WINDOW_HEIGHT = VIRTUAL_H
WINDOW_TITLE = "Esquiva la Silueta"
TARGET_FPS = 60
DEFAULT_FULLSCREEN = True

# Webcam and segmentation settings (16:9 ratio matching 1280x720)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
SEG_WIDTH = 480
SEG_HEIGHT = 270
MASK_THRESHOLD = 0.55
MASK_TEMPORAL_ALPHA = 0.65
SILHOUETTE_SCALE = 1.0

# Vital Core, Pose limbs, and Hands gesture settings
CORE_RADIUS = 22
CORE_ORB_RADIUS = 38
CORE_LERP_ALPHA = 0.35
POSE_MIN_VISIBILITY = 0.50
POSE_MIN_TRACKING_CONF = 0.45
LIMB_MIN_VISIBILITY = 0.45
LIMB_LERP_ALPHA = 0.55
CHEST_DROP_RATIO = 0.15
MASK_CENTROID_MIN_PIXELS = 40

HAND_MIN_DETECTION_CONF = 0.50
HAND_MIN_TRACKING_CONF = 0.45
HAND_LERP_ALPHA = 0.45
HAND_LOST_GRACE_FRAMES = 3
HAND_MAGNET_RADIUS = 130.0
ORB_ABSORB_RADIUS = 46.0
CORE_GRAB_RADIUS = 85.0
CORE_MAX_TORSO_RADIUS = 240.0

# Legacy pixel-mask collision thresholds (retained for compatibility)
COLLISION_PIXEL_THRESHOLD = 180
ORB_COLLISION_PIXEL_THRESHOLD = 110

# Core gameplay rules (5-minute arcade run)
GAME_DURATION_SECONDS = 300.0
MAX_LIVES = 5
INVULNERABILITY_DURATION = 1.5
LASER_WARNING_DURATION = 1.5

# Scoring
SURVIVAL_POINTS_PER_SECOND = 15.0
WALL_DODGE_BONUS = 300
ORB_GOLD_POINTS = 250
ORB_HEAL_FULL_BONUS = 400

# Model asset location and download source
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "selfie_segmenter_landscape.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Futuristic neon palette (RGB)
COLOR_BG = (7, 11, 22)
COLOR_GRID = (17, 29, 56)
COLOR_PANEL_BG = (10, 16, 32)
COLOR_PANEL_BORDER = (36, 92, 140)
COLOR_SILHOUETTE_CYAN = (0, 240, 255)
COLOR_EDGE_BRIGHT = (145, 255, 255)
COLOR_DANGER = (255, 42, 95)
COLOR_LASER = (255, 59, 48)
COLOR_LASER_WARN = (255, 175, 64)
COLOR_SAFE = (0, 255, 136)
COLOR_ORB_GOLD = (255, 215, 0)
COLOR_ORB_HEAL = (43, 255, 136)
COLOR_TEXT = (234, 246, 255)
COLOR_MUTED = (148, 172, 198)
COLOR_PHASE_BANNER = (0, 212, 255)
COLOR_HEART = (255, 58, 122)
COLOR_HEART_EMPTY = (42, 54, 78)
COLOR_HAND_OPEN = (0, 235, 255)
COLOR_HAND_FIST = (255, 195, 45)
COLOR_HAND_FIST_CORE = (255, 55, 195)
COLOR_LIMB_GLOW = (50, 235, 255)

# Phase definitions covering the 5-minute (300s) progression curve.
# Fractions scale automatically if --duration is overridden for quick testing.
PHASES = [
    {
        "number": 1,
        "start_frac": 0.00,
        "end_frac": 0.20,  # 0:00 - 1:00 of 5:00
        "name": "FASE 1 · CALENTAMIENTO",
        "subtitle": "Esquiva los meteoritos que caen desde arriba",
        "color": (0, 230, 255),
    },
    {
        "number": 2,
        "start_frac": 0.20,
        "end_frac": 0.50,  # 1:00 - 2:30 of 5:00
        "name": "FASE 2 · LÁSERES",
        "subtitle": "Atento a las líneas de advertencia (1.5s)",
        "color": (255, 175, 55),
    },
    {
        "number": 3,
        "start_frac": 0.50,
        "end_frac": 0.80,  # 2:30 - 4:00 of 5:00
        "name": "FASE 3 · HOLE IN THE WALL",
        "subtitle": "Muévete hacia la zona segura verde antes del cierre",
        "color": (0, 255, 136),
    },
    {
        "number": 4,
        "start_frac": 0.80,
        "end_frac": 1.01,  # 4:00 - 5:00 of 5:00
        "name": "FASE 4 · CLÍMAX FINAL",
        "subtitle": "¡Sobrevive a la tormenta final hasta 05:00!",
        "color": (255, 65, 130),
    },
]
