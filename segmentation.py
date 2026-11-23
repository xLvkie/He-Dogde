"""Webcam capture, MediaPipe Segmentation + Pose + Hands tracking, and neon rendering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHEST_DROP_RATIO,
    COLOR_BG,
    CORE_LERP_ALPHA,
    CORE_RADIUS,
    HAND_LERP_ALPHA,
    HAND_LOST_GRACE_FRAMES,
    HAND_MIN_DETECTION_CONF,
    HAND_MIN_TRACKING_CONF,
    HAND_MODEL_PATH,
    HAND_MODEL_URL,
    LIMB_LERP_ALPHA,
    LIMB_MIN_VISIBILITY,
    MASK_CENTROID_MIN_PIXELS,
    MASK_TEMPORAL_ALPHA,
    MASK_THRESHOLD,
    MODEL_PATH,
    MODEL_URL,
    POSE_MIN_TRACKING_CONF,
    POSE_MIN_VISIBILITY,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
    SEG_HEIGHT,
    SEG_WIDTH,
    SILHOUETTE_SCALE,
    VIRTUAL_H,
    VIRTUAL_W,
)

# Avoid noisy stderr warnings when ~/.config is read-only (e.g., restricted environments)
if "MPLCONFIGDIR" not in os.environ:
    _cfg_dir = Path.home() / ".config"
    if not os.access(_cfg_dir, os.W_OK):
        os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import mediapipe as mp  # noqa: E402


_TINT_PALETTES = {
    "normal": {
        "fill": np.array([0, 178, 232], dtype=np.uint8),
        "glow": np.array([0.0, 0.55, 0.88], dtype=np.float32),
        "edge": np.array([150, 255, 255], dtype=np.uint8),
    },
    "hit_flash": {
        "fill": np.array([255, 45, 95], dtype=np.uint8),
        "glow": np.array([1.0, 0.16, 0.34], dtype=np.float32),
        "edge": np.array([255, 200, 215], dtype=np.uint8),
    },
    "invuln_dim": {
        "fill": np.array([0, 75, 108], dtype=np.uint8),
        "glow": np.array([0.0, 0.22, 0.38], dtype=np.float32),
        "edge": np.array([85, 175, 200], dtype=np.uint8),
    },
    "heal_glow": {
        "fill": np.array([38, 225, 138], dtype=np.uint8),
        "glow": np.array([0.12, 0.95, 0.52], dtype=np.float32),
        "edge": np.array([195, 255, 225], dtype=np.uint8),
    },
    "gold_glow": {
        "fill": np.array([245, 198, 42], dtype=np.uint8),
        "glow": np.array([0.98, 0.78, 0.15], dtype=np.float32),
        "edge": np.array([255, 245, 180], dtype=np.uint8),
    },
}

# Pose limb indices & connections (shoulders, elbows, wrists, hips, knees, ankles)
_LIMB_JOINT_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
_LIMB_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)

# Hand landmark indices for palm center and fist detection
_PALM_INDICES = (0, 5, 9, 13, 17)
_FINGER_TRIPLETS = (
    (5, 6, 8),    # Index
    (9, 10, 12),  # Middle
    (13, 14, 16), # Ring
    (17, 18, 20), # Pinky
)


@dataclass
class HandState:
    """Tracked hand cursor and fist gesture state in virtual canvas coordinates."""

    hand_id: str
    pos: tuple[float, float]
    is_fist: bool
    confidence: float = 1.0


@dataclass
class TrackingFrame:
    """Snapshot of stylized silhouette, binary mask, chest core, limbs, and hands."""

    binary_mask: np.ndarray
    stylized_rgb: np.ndarray
    coverage: float
    camera_ok: bool
    error_msg: str
    core_pos: tuple[float, float]
    tracking_source: str  # "pose", "mask", or "memory"
    core_trail: list[tuple[float, float]]
    hands: list[HandState] = field(default_factory=list)
    limb_segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )
    limb_joints: list[tuple[float, float]] = field(default_factory=list)


def _landmark_xy(lm: Any) -> tuple[float, float]:
    """Extract (x, y) from either a MediaPipe landmark object or a sequence."""
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return float(lm.x), float(lm.y)
    return float(lm[0]), float(lm[1])


def detect_fist(landmarks: Sequence[Any]) -> bool:
    """Return True if at least 3 of 4 non-thumb fingers are curled toward the wrist."""
    if landmarks is None or len(landmarks) < 21:
        return False

    wx, wy = _landmark_xy(landmarks[0])
    mx, my = _landmark_xy(landmarks[9])
    palm_size = max(math.hypot(mx - wx, my - wy), 1e-5)

    curled = 0
    for mcp_idx, pip_idx, tip_idx in _FINGER_TRIPLETS:
        mcx, mcy = _landmark_xy(landmarks[mcp_idx])
        pix, piy = _landmark_xy(landmarks[pip_idx])
        tix, tiy = _landmark_xy(landmarks[tip_idx])

        d_tip = math.hypot(tix - wx, tiy - wy)
        d_mcp = math.hypot(mcx - wx, mcy - wy)
        d_pip = math.hypot(pix - wx, piy - wy)

        if (d_tip < d_mcp) or (d_tip < d_mcp + 0.12 * palm_size and d_tip < d_pip):
            curled += 1

    return curled >= 3


def ensure_model_downloaded(
    model_path: Path = MODEL_PATH,
    url: str = MODEL_URL,
) -> Path:
    """Download a MediaPipe model asset if missing or empty."""
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded model file is empty: {model_path}")
        os.replace(tmp_path, model_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return model_path


def _aspect_fill_resize(
    frame_bgr: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Center-crop to match target aspect ratio (e.g. 4:3 -> 16:9) without stretching."""
    fh, fw = frame_bgr.shape[:2]
    if fw == target_w and fh == target_h:
        return frame_bgr

    target_aspect = target_w / max(1, target_h)
    src_aspect = fw / max(1, fh)
    if abs(src_aspect - target_aspect) > 0.02:
        if src_aspect < target_aspect:
            new_h = max(1, int(round(fw / target_aspect)))
            y0 = max(0, (fh - new_h) // 2)
            frame_bgr = frame_bgr[y0 : y0 + new_h, :]
        else:
            new_w = max(1, int(round(fh * target_aspect)))
            x0 = max(0, (fw - new_w) // 2)
            frame_bgr = frame_bgr[:, x0 : x0 + new_w]

    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


class SilhouetteSegmenter:
    """Real-time webcam/demo segmenter + Pose limbs + Hands fist gesture tracker."""

    def __init__(
        self,
        camera_index: int = 0,
        display_size: tuple[int, int] = (VIRTUAL_W, VIRTUAL_H),
        seg_size: tuple[int, int] = (SEG_WIDTH, SEG_HEIGHT),
        silhouette_scale: float = SILHOUETTE_SCALE,
        demo_mode: bool = False,
        auto_start: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self._disp_w, self._disp_h = display_size
        self._seg_w, self._seg_h = seg_size
        self._scale = float(np.clip(silhouette_scale, 0.45, 1.0))
        self.demo_mode = demo_mode

        # Scaled silhouette placement rectangle at internal seg_size (bottom-center anchor)
        self._sw = max(1, int(round(self._seg_w * self._scale)))
        self._sh = max(1, int(round(self._seg_h * self._scale)))
        self._sx = (self._seg_w - self._sw) // 2
        self._sy = self._seg_h - self._sh

        self._lock = threading.Lock()
        self._seg_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # MediaPipe segmentation, pose & hands backends
        self._backend: Optional[str] = None
        self._task_segmenter = None
        self._solution_segmenter = None
        self._pose_backend: Optional[str] = None
        self._pose_landmarker = None
        self._solution_pose = None
        self._hand_backend: Optional[str] = None
        self._hand_landmarker = None
        self._solution_hands = None
        self._last_ts_ms = 0

        # Smoothing and morphological kernels at seg_size
        self._alpha = MASK_TEMPORAL_ALPHA
        self._threshold = MASK_THRESHOLD
        self._prev_conf: Optional[np.ndarray] = None
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Shared small-resolution buffers (seg_h, seg_w)
        self._frame_rgb_small = np.zeros((self._seg_h, self._seg_w, 3), dtype=np.uint8)
        self._mask_small = np.zeros((self._seg_h, self._seg_w), dtype=np.uint8)
        self._edge_small = np.zeros((self._seg_h, self._seg_w), dtype=np.uint8)
        self._glow_small = np.zeros((self._seg_h, self._seg_w), dtype=np.uint8)
        self._raw_coverage = 0.0

        # Vital-core tracking state in logical virtual display coords
        self._core_x = self._disp_w * 0.5
        self._core_y = self._disp_h * 0.55
        self._core_locked = False
        self._tracking_source = "memory"
        self._core_trail: deque[tuple[float, float]] = deque(maxlen=10)

        # Limb joint smoothing state (joint_idx -> (vx, vy), miss_count)
        self._limb_joints_smooth: dict[int, tuple[float, float]] = {}
        self._limb_miss_counts: dict[int, int] = {}
        self._limb_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._limb_joints: list[tuple[float, float]] = []

        # Hand smoothing & grace-period tracking state
        self._hand_smooth: dict[str, HandState] = {}
        self._hand_miss_counts: dict[str, int] = {}
        self._hands: list[HandState] = []

        # Reusable full-resolution dark background template
        self._dark_bg_full = np.full(
            (self._disp_h, self._disp_w, 3), COLOR_BG, dtype=np.uint8
        )

        self._has_frame = False
        self._camera_ok = True if demo_mode else False
        self._error_msg = ""

        # Demo silhouette and interactive hand state
        self._demo_x = 0.5
        self._demo_y = 0.64
        self._demo_crouch = False
        self._demo_signal_lost = False
        self._demo_target_x: Optional[float] = None
        self._demo_target_y: Optional[float] = None
        self._demo_hand_x = 0.65
        self._demo_hand_y = 0.52
        self._demo_fist = False
        self._demo_hands_override: Optional[list[HandState]] = None
        self._demo_bg_small = self._build_demo_background()

        if not self.demo_mode:
            self._init_mediapipe()
        else:
            self._update_demo_frame()

        if auto_start:
            self.start()

    @property
    def silhouette_scale(self) -> float:
        """Configured scale factor for the segmented player silhouette."""
        return self._scale

    def get_silhouette_bounds(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) of the silhouette zone in logical display coords."""
        if self._scale >= 0.999:
            return (0, 0, self._disp_w, self._disp_h)
        dx = int(round(self._sx * self._disp_w / self._seg_w))
        dy = int(round(self._sy * self._disp_h / self._seg_h))
        dw = int(round(self._sw * self._disp_w / self._seg_w))
        dh = self._disp_h - dy
        return (dx, dy, dw, dh)

    def _place_scaled_silhouette(
        self, raw_mask_small: np.ndarray, raw_rgb_small: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Scale mask and RGB frame to `silhouette_scale` and anchor bottom-center."""
        raw_coverage = float(np.count_nonzero(raw_mask_small)) / float(raw_mask_small.size)
        if self._scale >= 0.999:
            return raw_mask_small, raw_rgb_small, raw_coverage

        scaled_mask = cv2.resize(
            raw_mask_small, (self._sw, self._sh), interpolation=cv2.INTER_AREA
        )
        scaled_mask = np.where(scaled_mask > 127, np.uint8(255), np.uint8(0))
        placed_mask = np.zeros((self._seg_h, self._seg_w), dtype=np.uint8)
        placed_mask[self._sy : self._sy + self._sh, self._sx : self._sx + self._sw] = scaled_mask

        scaled_rgb = cv2.resize(
            raw_rgb_small, (self._sw, self._sh), interpolation=cv2.INTER_AREA
        )
        placed_rgb = np.full((self._seg_h, self._seg_w, 3), COLOR_BG, dtype=np.uint8)
        placed_rgb[self._sy : self._sy + self._sh, self._sx : self._sx + self._sw] = scaled_rgb
        return placed_mask, placed_rgb, raw_coverage

    def _build_demo_background(self) -> np.ndarray:
        """Create a subtle futuristic studio gradient for --demo mode."""
        bg = np.zeros((self._seg_h, self._seg_w, 3), dtype=np.uint8)
        y_grad = np.linspace(14, 42, self._seg_h, dtype=np.uint8)[:, None]
        bg[:, :, 0] = 10
        bg[:, :, 1] = y_grad // 2
        bg[:, :, 2] = y_grad + 20
        return bg

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe ImageSegmenter, PoseLandmarker, and HandLandmarker."""
        tasks_error: Optional[Exception] = None
        try:
            model_path = ensure_model_downloaded(MODEL_PATH, MODEL_URL)
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self._task_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
                options
            )
            self._backend = "tasks"
        except Exception as exc:
            tasks_error = exc

        if self._backend is None:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
                self._solution_segmenter = (
                    mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                )
                self._backend = "solutions"
            else:
                raise RuntimeError(
                    f"No se pudo inicializar MediaPipe Selfie Segmentation: {tasks_error}"
                )

        # Lightweight Pose tracker (for chest core & limb skeletal reinforcement)
        try:
            pose_path = ensure_model_downloaded(POSE_MODEL_PATH, POSE_MODEL_URL)
            pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=POSE_MIN_TRACKING_CONF,
                output_segmentation_masks=False,
            )
            self._pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                pose_opts
            )
            self._pose_backend = "tasks"
        except Exception:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                try:
                    self._solution_pose = mp.solutions.pose.Pose(
                        static_image_mode=False,
                        model_complexity=0,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=POSE_MIN_TRACKING_CONF,
                    )
                    self._pose_backend = "solutions"
                except Exception:
                    self._pose_backend = None

        # Lightweight Hands tracker (2 hands + fist gesture)
        try:
            hand_path = ensure_model_downloaded(HAND_MODEL_PATH, HAND_MODEL_URL)
            hand_opts = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=HAND_MIN_DETECTION_CONF,
                min_hand_presence_confidence=HAND_MIN_DETECTION_CONF,
                min_tracking_confidence=HAND_MIN_TRACKING_CONF,
            )
            self._hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                hand_opts
            )
            self._hand_backend = "tasks"
        except Exception:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                try:
                    self._solution_hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=0,
                        min_detection_confidence=HAND_MIN_DETECTION_CONF,
                        min_tracking_confidence=HAND_MIN_TRACKING_CONF,
                    )
                    self._hand_backend = "solutions"
                except Exception:
                    self._hand_backend = None

    def start(self) -> None:
        """Start the background capture/segmentation thread (if not already running)."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        if self.demo_mode:
            self._camera_ok = True
            self._error_msg = ""
            self._update_demo_frame()
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            self._camera_ok = False
            self._error_msg = (
                f"No se pudo abrir la cámara web (índice {self.camera_index}). "
                "Conecta una cámara o ejecuta con --demo."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._camera_ok = True
        self._error_msg = ""
        self._thread = threading.Thread(
            target=self._capture_loop, name="SegmentationCapture", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        """Continuously grab webcam frames, mirror them, segment, and track pose/hands."""
        consecutive_failures = 0
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                with self._lock:
                    self._camera_ok = False
                    self._error_msg = "La cámara web se desconectó."
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    with self._lock:
                        self._camera_ok = False
                        self._error_msg = "Error leyendo frames de la cámara web."
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            try:
                self.process_frame(frame_bgr, flip=True)
            except Exception as exc:
                with self._lock:
                    self._camera_ok = False
                    self._error_msg = f"Error de segmentación: {exc}"
                break

    def _run_mediapipe_frame(
        self, small_rgb: np.ndarray
    ) -> tuple[
        np.ndarray,
        Optional[tuple[float, float]],
        dict[int, tuple[float, float]],
        list[HandState],
    ]:
        """Run Selfie Segmentation + Pose + Hands on small_rgb."""
        shoulders_chest: Optional[tuple[float, float]] = None
        raw_joints: dict[int, tuple[float, float]] = {}
        raw_hands: list[HandState] = []

        with self._seg_lock:
            ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms

            mp_image: Optional[mp.Image] = None
            if (
                self._backend == "tasks"
                or self._pose_backend == "tasks"
                or self._hand_backend == "tasks"
            ):
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)

            # 1. Selfie Segmentation
            if self._backend == "tasks" and self._task_segmenter is not None:
                assert mp_image is not None
                result = self._task_segmenter.segment_for_video(mp_image, ts_ms)
                masks = result.confidence_masks
                if not masks:
                    conf = np.zeros((self._seg_h, self._seg_w), dtype=np.float32)
                else:
                    person_idx = 1 if len(masks) >= 2 else 0
                    conf = (
                        masks[person_idx]
                        .numpy_view()
                        .squeeze()
                        .astype(np.float32, copy=True)
                    )
            elif self._backend == "solutions" and self._solution_segmenter is not None:
                result = self._solution_segmenter.process(small_rgb)
                if result.segmentation_mask is None:
                    conf = np.zeros((self._seg_h, self._seg_w), dtype=np.float32)
                else:
                    conf = result.segmentation_mask.astype(np.float32, copy=True)
            else:
                raise RuntimeError("Backend de MediaPipe no inicializado.")

            # 2. Lightweight Pose Landmarker (chest + limbs)
            pose_lms: Optional[Sequence[Any]] = None
            if self._pose_backend == "tasks" and self._pose_landmarker is not None:
                assert mp_image is not None
                pose_res = self._pose_landmarker.detect_for_video(mp_image, ts_ms)
                if pose_res.pose_landmarks:
                    pose_lms = pose_res.pose_landmarks[0]
            elif self._pose_backend == "solutions" and self._solution_pose is not None:
                pose_res = self._solution_pose.process(small_rgb)
                if pose_res.pose_landmarks is not None:
                    pose_lms = pose_res.pose_landmarks.landmark

            if pose_lms is not None and len(pose_lms) > 28:
                shoulders_chest = self._chest_from_landmarks(pose_lms[11], pose_lms[12])
                raw_joints = self._extract_limb_joints(pose_lms)

            # 3. Lightweight Hands Landmarker (2 hands + fist gesture)
            if self._hand_backend == "tasks" and self._hand_landmarker is not None:
                assert mp_image is not None
                hand_res = self._hand_landmarker.detect_for_video(mp_image, ts_ms)
                raw_hands = self._extract_hands_from_result(
                    hand_res.hand_landmarks, hand_res.handedness
                )
            elif self._hand_backend == "solutions" and self._solution_hands is not None:
                hand_res = self._solution_hands.process(small_rgb)
                if hand_res.multi_hand_landmarks:
                    raw_hands = self._extract_hands_from_solutions(
                        hand_res.multi_hand_landmarks, hand_res.multi_handedness
                    )

        return conf, shoulders_chest, raw_joints, raw_hands

    def _to_virtual_xy(self, nx: float, ny: float) -> tuple[float, float]:
        """Map normalized [0..1] camera coords into virtual 1280x720 bounds."""
        bx, by, bw, bh = self.get_silhouette_bounds()
        return (bx + nx * bw, by + ny * bh)

    def _extract_limb_joints(
        self, pose_lms: Sequence[Any]
    ) -> dict[int, tuple[float, float]]:
        """Extract visible limb joints mapped into virtual canvas coordinates."""
        joints: dict[int, tuple[float, float]] = {}
        for idx in _LIMB_JOINT_INDICES:
            if idx >= len(pose_lms):
                continue
            lm = pose_lms[idx]
            vis = getattr(lm, "visibility", 1.0)
            vis_val = float(1.0 if vis is None else vis)
            if vis_val < LIMB_MIN_VISIBILITY:
                continue
            nx, ny = _landmark_xy(lm)
            if -0.10 <= nx <= 1.10 and -0.10 <= ny <= 1.10:
                vx, vy = self._to_virtual_xy(nx, ny)
                joints[idx] = (
                    float(np.clip(vx, 0.0, self._disp_w)),
                    float(np.clip(vy, 0.0, self._disp_h)),
                )
        return joints

    def _extract_hands_from_result(
        self,
        hand_landmarks_list: Optional[Sequence[Sequence[Any]]],
        handedness_list: Optional[Sequence[Sequence[Any]]],
    ) -> list[HandState]:
        """Parse MediaPipe Tasks HandLandmarkerResult into smoothed HandState inputs."""
        if not hand_landmarks_list:
            return []

        hands: list[HandState] = []
        used_ids: set[str] = set()
        for idx, lms in enumerate(hand_landmarks_list):
            if not lms or len(lms) < 21:
                continue
            label = f"Hand{idx}"
            conf = 1.0
            if handedness_list and idx < len(handedness_list) and handedness_list[idx]:
                cat = handedness_list[idx][0]
                label = str(getattr(cat, "category_name", label) or label)
                conf = float(getattr(cat, "score", 1.0))
            if label in used_ids:
                label = f"{label}_{idx}"
            used_ids.add(label)

            px_sum = 0.0
            py_sum = 0.0
            for p_idx in _PALM_INDICES:
                nx, ny = _landmark_xy(lms[p_idx])
                px_sum += nx
                py_sum += ny
            vx, vy = self._to_virtual_xy(
                px_sum / len(_PALM_INDICES), py_sum / len(_PALM_INDICES)
            )
            vx = float(np.clip(vx, 8.0, self._disp_w - 8.0))
            vy = float(np.clip(vy, 8.0, self._disp_h - 8.0))

            hands.append(
                HandState(
                    hand_id=label,
                    pos=(vx, vy),
                    is_fist=detect_fist(lms),
                    confidence=conf,
                )
            )
        return hands

    def _extract_hands_from_solutions(
        self,
        multi_landmarks: Sequence[Any],
        multi_handedness: Optional[Sequence[Any]],
    ) -> list[HandState]:
        """Fallback parser for legacy mp.solutions.hands."""
        lms_lists = [list(hl.landmark) for hl in multi_landmarks]
        hand_cats: list[list[Any]] = []
        if multi_handedness:
            for mh in multi_handedness:
                if mh.classification:
                    c = mh.classification[0]
                    # Legacy solutions classification uses .label instead of .category_name
                    hand_cats.append(
                        [
                            type(
                                "_Cat",
                                (),
                                {
                                    "category_name": getattr(c, "label", "Hand"),
                                    "score": getattr(c, "score", 1.0),
                                },
                            )()
                        ]
                    )
                else:
                    hand_cats.append([])
        return self._extract_hands_from_result(lms_lists, hand_cats)

    def _update_limb_smoothing(
        self, raw_joints: dict[int, tuple[float, float]]
    ) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[float, float]]]:
        """Smooth visible limb joints and build glowing skeletal segments."""
        for idx in _LIMB_JOINT_INDICES:
            if idx in raw_joints:
                tx, ty = raw_joints[idx]
                if idx in self._limb_joints_smooth:
                    px, py = self._limb_joints_smooth[idx]
                    self._limb_joints_smooth[idx] = (
                        px + LIMB_LERP_ALPHA * (tx - px),
                        py + LIMB_LERP_ALPHA * (ty - py),
                    )
                else:
                    self._limb_joints_smooth[idx] = (tx, ty)
                self._limb_miss_counts[idx] = 0
            elif idx in self._limb_joints_smooth:
                miss = self._limb_miss_counts.get(idx, 0) + 1
                self._limb_miss_counts[idx] = miss
                if miss >= 3:
                    self._limb_joints_smooth.pop(idx, None)
                    self._limb_miss_counts.pop(idx, None)

        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for a, b in _LIMB_CONNECTIONS:
            if a in self._limb_joints_smooth and b in self._limb_joints_smooth:
                segments.append(
                    (self._limb_joints_smooth[a], self._limb_joints_smooth[b])
                )
        joints = list(self._limb_joints_smooth.values())
        self._limb_segments = segments
        self._limb_joints = joints
        return segments, joints

    def _update_hand_smoothing(self, raw_hands: Sequence[HandState]) -> list[HandState]:
        """Smooth hand positions and preserve tracking across brief detection dropouts."""
        seen_ids: set[str] = set()
        for h in raw_hands:
            seen_ids.add(h.hand_id)
            if h.hand_id in self._hand_smooth:
                prev = self._hand_smooth[h.hand_id]
                sx = prev.pos[0] + HAND_LERP_ALPHA * (h.pos[0] - prev.pos[0])
                sy = prev.pos[1] + HAND_LERP_ALPHA * (h.pos[1] - prev.pos[1])
                self._hand_smooth[h.hand_id] = HandState(
                    hand_id=h.hand_id,
                    pos=(sx, sy),
                    is_fist=h.is_fist,
                    confidence=h.confidence,
                )
            else:
                self._hand_smooth[h.hand_id] = HandState(
                    hand_id=h.hand_id,
                    pos=h.pos,
                    is_fist=h.is_fist,
                    confidence=h.confidence,
                )
            self._hand_miss_counts[h.hand_id] = 0

        stale: list[str] = []
        for hid in list(self._hand_smooth.keys()):
            if hid not in seen_ids:
                miss = self._hand_miss_counts.get(hid, 0) + 1
                self._hand_miss_counts[hid] = miss
                if miss >= HAND_LOST_GRACE_FRAMES:
                    stale.append(hid)
        for hid in stale:
            self._hand_smooth.pop(hid, None)
            self._hand_miss_counts.pop(hid, None)

        self._hands = [
            HandState(
                hand_id=hs.hand_id,
                pos=hs.pos,
                is_fist=hs.is_fist,
                confidence=hs.confidence,
            )
            for hs in self._hand_smooth.values()
        ]
        return self._hands

    def _chest_from_landmarks(
        self, left_shoulder: object, right_shoulder: object
    ) -> Optional[tuple[float, float]]:
        """Compute virtual-canvas chest center from left/right shoulder landmarks."""
        vis_l = getattr(left_shoulder, "visibility", 1.0)
        vis_r = getattr(right_shoulder, "visibility", 1.0)
        vis_l_val = float(1.0 if vis_l is None else vis_l)
        vis_r_val = float(1.0 if vis_r is None else vis_r)
        if vis_l_val < POSE_MIN_VISIBILITY or vis_r_val < POSE_MIN_VISIBILITY:
            return None

        lx_n, ly_n = _landmark_xy(left_shoulder)
        rx_n, ry_n = _landmark_xy(right_shoulder)
        if not (-0.08 <= lx_n <= 1.08 and -0.08 <= rx_n <= 1.08):
            return None
        if not (-0.08 <= ly_n <= 1.08 and -0.08 <= ry_n <= 1.08):
            return None

        lx, ly = self._to_virtual_xy(lx_n, ly_n)
        rx, ry = self._to_virtual_xy(rx_n, ry_n)
        return self._chest_from_virtual_shoulders((lx, ly), (rx, ry))

    @staticmethod
    def _chest_from_virtual_shoulders(
        left_xy: tuple[float, float], right_xy: tuple[float, float]
    ) -> tuple[float, float]:
        """Midpoint of shoulders shifted downward by 15% of inter-shoulder distance."""
        mid_x = 0.5 * (left_xy[0] + right_xy[0])
        mid_y = 0.5 * (left_xy[1] + right_xy[1])
        dist = math.hypot(right_xy[0] - left_xy[0], right_xy[1] - left_xy[1])
        return (mid_x, mid_y + CHEST_DROP_RATIO * dist)

    def _centroid_from_mask(
        self, placed_mask: np.ndarray
    ) -> Optional[tuple[float, float]]:
        """Tier-2 fallback: compute upper/central torso centroid from silhouette mask."""
        ys, xs = np.nonzero(placed_mask)
        if len(xs) < MASK_CENTROID_MIN_PIXELS:
            return None

        y_min = int(ys.min())
        y_max = int(ys.max())
        h_span = max(1, y_max - y_min)
        band = (ys >= y_min + 0.18 * h_span) & (ys <= y_min + 0.58 * h_span)
        if int(np.count_nonzero(band)) >= 15:
            cx_s = float(xs[band].mean())
            cy_s = float(ys[band].mean())
        else:
            cx_s = float(xs.mean())
            cy_s = float(ys.mean())

        return (
            cx_s * self._disp_w / max(1, self._seg_w),
            cy_s * self._disp_h / max(1, self._seg_h),
        )

    def _apply_core_target(
        self,
        pose_target: Optional[tuple[float, float]],
        placed_mask: np.ndarray,
    ) -> tuple[tuple[float, float], str]:
        """3-tier tracking (Pose -> Mask Centroid -> Memory) with Lerp smoothing."""
        target: Optional[tuple[float, float]] = None
        source = "memory"

        if pose_target is not None:
            target = pose_target
            source = "pose"
        else:
            centroid = self._centroid_from_mask(placed_mask)
            if centroid is not None:
                target = centroid
                source = "mask"

        if target is not None:
            margin = float(CORE_RADIUS + 8)
            tx = float(np.clip(target[0], margin, self._disp_w - margin))
            ty = float(np.clip(target[1], margin, self._disp_h - margin))
            if not self._core_locked:
                self._core_x = tx
                self._core_y = ty
                self._core_locked = True
            else:
                self._core_x += CORE_LERP_ALPHA * (tx - self._core_x)
                self._core_y += CORE_LERP_ALPHA * (ty - self._core_y)

        self._tracking_source = source
        self._core_trail.append((self._core_x, self._core_y))
        return (self._core_x, self._core_y), source

    def _compute_edges_and_glow(self, mask_small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute clean neon edge and soft outer halo at internal resolution."""
        dilated = cv2.dilate(mask_small, self._edge_kernel, iterations=2)
        edge_small = cv2.morphologyEx(mask_small, cv2.MORPH_GRADIENT, self._edge_kernel)
        glow_ring = cv2.subtract(dilated, mask_small)
        glow_small = cv2.GaussianBlur(glow_ring, (11, 11), 0)
        return edge_small, glow_small

    def process_frame(
        self, frame_bgr: np.ndarray, flip: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronously process a BGR frame and update silhouette, limbs, hands, and core."""
        if flip:
            frame_bgr = cv2.flip(frame_bgr, 1)

        small_bgr = _aspect_fill_resize(frame_bgr, self._seg_w, self._seg_h)
        small_rgb = np.ascontiguousarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))

        if self._backend is None:
            self._init_mediapipe()

        conf, pose_target, raw_joints, raw_hands = self._run_mediapipe_frame(small_rgb)
        if conf.shape != (self._seg_h, self._seg_w):
            conf = cv2.resize(
                conf, (self._seg_w, self._seg_h), interpolation=cv2.INTER_LINEAR
            )

        if self._prev_conf is None:
            smooth = conf
        else:
            smooth = self._alpha * conf + (1.0 - self._alpha) * self._prev_conf
        self._prev_conf = smooth

        raw_mask = (smooth >= self._threshold).astype(np.uint8) * 255
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self._morph_kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        # Light dilation + Gaussian smoothing preserves fast-moving thin arms and hands
        raw_mask = cv2.dilate(raw_mask, self._morph_kernel, iterations=1)
        raw_mask = cv2.GaussianBlur(raw_mask, (5, 5), 0)
        raw_mask = np.where(raw_mask > 110, np.uint8(255), np.uint8(0))

        mask_small, placed_rgb, raw_coverage = self._place_scaled_silhouette(
            raw_mask, small_rgb
        )
        edge_small, glow_small = self._compute_edges_and_glow(mask_small)

        with self._lock:
            self._apply_core_target(pose_target, mask_small)
            self._update_limb_smoothing(raw_joints)
            self._update_hand_smoothing(raw_hands)
            self._frame_rgb_small = placed_rgb
            self._mask_small = mask_small
            self._edge_small = edge_small
            self._glow_small = glow_small
            self._raw_coverage = raw_coverage
            self._has_frame = True

        return mask_small, placed_rgb

    # ------------------------------------------------------------------
    # Demo mode synthetic silhouette & hand controls (for --demo without webcam)
    # ------------------------------------------------------------------
    def set_demo_signal_lost(self, lost: bool) -> None:
        """Simulate camera obstruction in --demo mode to test Tier-3 memory fallback."""
        if not self.demo_mode:
            return
        with self._lock:
            self._demo_signal_lost = bool(lost)
        self._update_demo_frame()

    def set_demo_fist(self, closed: bool) -> None:
        """Open or close the synthetic demo hand fist."""
        if not self.demo_mode:
            return
        with self._lock:
            self._demo_fist = bool(closed)
        self._update_demo_frame()

    def set_demo_hand_pos(self, norm_x: float, norm_y: float) -> None:
        """Move the synthetic demo hand cursor to normalized virtual coordinates."""
        if not self.demo_mode:
            return
        with self._lock:
            self._demo_hand_x = float(np.clip(norm_x, 0.02, 0.98))
            self._demo_hand_y = float(np.clip(norm_y, 0.05, 0.95))
        self._update_demo_frame()

    def set_demo_hands(self, hands: Optional[list[HandState]]) -> None:
        """Override demo hands explicitly (used for deterministic tests) or pass None to clear."""
        if not self.demo_mode:
            return
        with self._lock:
            self._demo_hands_override = hands
        self._update_demo_frame()

    def set_demo_target(self, norm_x: float, norm_y: float) -> None:
        """Move synthetic demo silhouette directly toward normalized screen coords."""
        if not self.demo_mode:
            return
        with self._lock:
            self._demo_target_x = float(np.clip(norm_x, 0.08, 0.92))
            self._demo_target_y = float(np.clip(norm_y, 0.22, 0.90))

    def step_demo(
        self, dx: float, dy: float, crouch: bool, dt: float
    ) -> None:
        """Update synthetic silhouette position from keyboard/mouse input."""
        if not self.demo_mode:
            return
        with self._lock:
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                self._demo_target_x = None
                self._demo_target_y = None
                speed = 0.72
                self._demo_x = float(np.clip(self._demo_x + dx * speed * dt, 0.08, 0.92))
                self._demo_y = float(np.clip(self._demo_y + dy * speed * dt, 0.22, 0.90))
            elif self._demo_target_x is not None and self._demo_target_y is not None:
                blend = min(1.0, dt * 14.0)
                self._demo_x += (self._demo_target_x - self._demo_x) * blend
                self._demo_y += (self._demo_target_y - self._demo_y) * blend
            self._demo_crouch = crouch
        self._update_demo_frame()

    def _update_demo_frame(self) -> None:
        """Rasterize synthetic silhouette, limbs, chest core, and demo hand."""
        with self._lock:
            nx = self._demo_x
            ny = self._demo_y
            crouch = self._demo_crouch
            signal_lost = self._demo_signal_lost
            hand_nx = self._demo_hand_x
            hand_ny = self._demo_hand_y
            demo_fist = self._demo_fist
            hands_override = self._demo_hands_override

        mask = np.zeros((self._seg_h, self._seg_w), dtype=np.uint8)
        pose_target: Optional[tuple[float, float]] = None
        raw_joints: dict[int, tuple[float, float]] = {}
        raw_hands: list[HandState] = []

        if not signal_lost:
            cx = int(nx * self._seg_w)
            feet_y = int(ny * self._seg_h) + int(self._seg_h * 0.28)
            feet_y = min(self._seg_h - 2, max(int(self._seg_h * 0.42), feet_y))

            scale = 0.64 if crouch else 1.0
            body_h = int(self._seg_h * 0.56 * scale)
            top_y = max(4, feet_y - body_h)
            torso_w = max(14, int(self._seg_w * (0.068 if crouch else 0.058)))

            head_r = max(10, int(self._seg_h * 0.085))
            head_y = top_y + head_r
            cv2.circle(mask, (cx, head_y), head_r, 255, -1)

            neck_top = head_y + head_r - 3
            shoulder_y = neck_top + max(6, int(body_h * 0.08))
            cv2.rectangle(
                mask,
                (cx - head_r // 2, neck_top),
                (cx + head_r // 2, shoulder_y + 4),
                255,
                -1,
            )

            hip_y = top_y + int(body_h * 0.56)
            torso_pts = np.array(
                [
                    [cx - torso_w, shoulder_y],
                    [cx + torso_w, shoulder_y],
                    [cx + int(torso_w * 0.85), hip_y],
                    [cx - int(torso_w * 0.85), hip_y],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [torso_pts], 255)
            cv2.ellipse(
                mask,
                (cx, shoulder_y + 4),
                (torso_w, max(6, int(head_r * 0.55))),
                0,
                0,
                360,
                255,
                -1,
            )

            arm_thick = max(8, int(torso_w * 0.44))
            arm_span_x = int(torso_w * (1.75 if not crouch else 1.55))
            arm_end_y = hip_y + (0 if crouch else int(body_h * 0.08))
            l_sh = (cx - torso_w + 3, shoulder_y + 6)
            r_sh = (cx + torso_w - 3, shoulder_y + 6)
            l_el = ((l_sh[0] + cx - arm_span_x) // 2, (l_sh[1] + arm_end_y) // 2 + 4)
            r_el = ((r_sh[0] + cx + arm_span_x) // 2, (r_sh[1] + arm_end_y) // 2 + 4)
            l_wr = (cx - arm_span_x, arm_end_y)
            r_wr = (cx + arm_span_x, arm_end_y)
            cv2.line(mask, l_sh, l_wr, 255, arm_thick)
            cv2.line(mask, r_sh, r_wr, 255, arm_thick)

            leg_thick = max(10, int(torso_w * 0.50))
            leg_dx = int(torso_w * 0.48)
            l_hip = (cx - leg_dx, hip_y - 4)
            r_hip = (cx + leg_dx, hip_y - 4)
            l_ank = (cx - leg_dx - 5, feet_y)
            r_ank = (cx + leg_dx + 5, feet_y)
            l_kne = ((l_hip[0] + l_ank[0]) // 2, (l_hip[1] + l_ank[1]) // 2)
            r_kne = ((r_hip[0] + r_ank[0]) // 2, (r_hip[1] + r_ank[1]) // 2)
            cv2.line(mask, l_hip, l_ank, 255, leg_thick)
            cv2.line(mask, r_hip, r_ank, 255, leg_thick)

            bx, by, bw, bh = self.get_silhouette_bounds()

            def to_v(pt: tuple[int, int]) -> tuple[float, float]:
                return (
                    bx + (pt[0] / self._seg_w) * bw,
                    by + (pt[1] / self._seg_h) * bh,
                )

            pose_target = self._chest_from_virtual_shoulders(
                to_v((cx - torso_w, shoulder_y)), to_v((cx + torso_w, shoulder_y))
            )
            raw_joints = {
                11: to_v((cx - torso_w, shoulder_y)),
                12: to_v((cx + torso_w, shoulder_y)),
                13: to_v(l_el),
                14: to_v(r_el),
                15: to_v(l_wr),
                16: to_v(r_wr),
                23: to_v(l_hip),
                24: to_v(r_hip),
                25: to_v(l_kne),
                26: to_v(r_kne),
                27: to_v(l_ank),
                28: to_v(r_ank),
            }

            if hands_override is not None:
                raw_hands = list(hands_override)
            else:
                raw_hands = [
                    HandState(
                        hand_id="Right",
                        pos=(hand_nx * self._disp_w, hand_ny * self._disp_h),
                        is_fist=demo_fist,
                        confidence=1.0,
                    )
                ]

        mask_small, placed_rgb, raw_coverage = self._place_scaled_silhouette(
            mask, self._demo_bg_small
        )
        edge_small, glow_small = self._compute_edges_and_glow(mask_small)

        with self._lock:
            self._apply_core_target(pose_target, mask_small)
            if signal_lost:
                self._limb_joints_smooth.clear()
                self._limb_miss_counts.clear()
                self._limb_segments = []
                self._limb_joints = []
                self._hand_smooth.clear()
                self._hand_miss_counts.clear()
                self._hands = []
            else:
                self._update_limb_smoothing(raw_joints)
                if hands_override is not None:
                    self._hand_smooth.clear()
                    self._hand_miss_counts.clear()
                    self._hands = list(hands_override)
                else:
                    self._update_hand_smoothing(raw_hands)
            self._frame_rgb_small = placed_rgb
            self._mask_small = mask_small
            self._edge_small = edge_small
            self._glow_small = glow_small
            self._raw_coverage = raw_coverage
            self._has_frame = True
            self._camera_ok = True

    def get_tracking_frame(
        self,
        show_camera_bg: bool = True,
        tint_mode: str = "normal",
    ) -> TrackingFrame:
        """Return latest stylized silhouette, binary mask, core, limbs, and hands."""
        with self._lock:
            frame_small = self._frame_rgb_small.copy()
            mask_small = self._mask_small.copy()
            edge_small = self._edge_small.copy()
            glow_small = self._glow_small.copy()
            raw_coverage = self._raw_coverage
            has_frame = self._has_frame
            camera_ok = self._camera_ok
            error_msg = self._error_msg
            core_pos = (self._core_x, self._core_y)
            tracking_source = self._tracking_source
            core_trail = list(self._core_trail)
            hands = [
                HandState(h.hand_id, h.pos, h.is_fist, h.confidence)
                for h in self._hands
            ]
            limb_segments = list(self._limb_segments)
            limb_joints = list(self._limb_joints)

        if not has_frame:
            empty_mask = np.zeros((self._disp_h, self._disp_w), dtype=np.uint8)
            return TrackingFrame(
                binary_mask=empty_mask,
                stylized_rgb=self._dark_bg_full.copy(),
                coverage=0.0,
                camera_ok=camera_ok,
                error_msg=error_msg,
                core_pos=core_pos,
                tracking_source=tracking_source,
                core_trail=core_trail,
                hands=hands,
                limb_segments=limb_segments,
                limb_joints=limb_joints,
            )

        palette = _TINT_PALETTES.get(tint_mode, _TINT_PALETTES["normal"])

        mask_up = cv2.resize(
            mask_small, (self._disp_w, self._disp_h), interpolation=cv2.INTER_LINEAR
        )
        fg = mask_up > 127
        binary_mask_full = fg.astype(np.uint8) * 255

        if show_camera_bg:
            if self._scale < 0.999:
                bg_small = np.full((self._seg_h, self._seg_w, 3), COLOR_BG, dtype=np.uint8)
                zone = (
                    frame_small[
                        self._sy : self._sy + self._sh, self._sx : self._sx + self._sw
                    ]
                    >> 2
                ).astype(np.uint8)
                zone[:, :, 2] = np.minimum(
                    255, zone[:, :, 2].astype(np.uint16) + 18
                ).astype(np.uint8)
                bg_small[
                    self._sy : self._sy + self._sh, self._sx : self._sx + self._sw
                ] = zone
            else:
                bg_small = (frame_small >> 2).astype(np.uint8)
                bg_small[:, :, 2] = np.minimum(
                    255, bg_small[:, :, 2].astype(np.uint16) + 18
                ).astype(np.uint8)
            canvas = cv2.resize(
                bg_small, (self._disp_w, self._disp_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            canvas = self._dark_bg_full.copy()

        if np.any(fg):
            fill_rgb = palette["fill"]
            if show_camera_bg:
                canvas[fg] = (canvas[fg] >> 1) + (fill_rgb >> 1)
            else:
                canvas[fg] = fill_rgb

        glow_rgb_small = np.clip(
            glow_small[:, :, None].astype(np.float32) * palette["glow"], 0, 255
        ).astype(np.uint8)
        glow_rgb_full = cv2.resize(
            glow_rgb_small, (self._disp_w, self._disp_h), interpolation=cv2.INTER_LINEAR
        )
        cv2.add(canvas, glow_rgb_full, dst=canvas)

        edge_up = cv2.resize(
            edge_small, (self._disp_w, self._disp_h), interpolation=cv2.INTER_LINEAR
        )
        edge_fg = edge_up > 85
        if np.any(edge_fg):
            canvas[edge_fg] = palette["edge"]

        return TrackingFrame(
            binary_mask=np.ascontiguousarray(binary_mask_full),
            stylized_rgb=np.ascontiguousarray(canvas),
            coverage=raw_coverage,
            camera_ok=camera_ok,
            error_msg=error_msg,
            core_pos=core_pos,
            tracking_source=tracking_source,
            core_trail=core_trail,
            hands=hands,
            limb_segments=limb_segments,
            limb_joints=limb_joints,
        )

    def get_latest(
        self,
        show_camera_bg: bool = True,
        tint_mode: str = "normal",
    ) -> tuple[np.ndarray, np.ndarray, float, bool, str]:
        """Legacy tuple accessor wrapping get_tracking_frame()."""
        tf = self.get_tracking_frame(
            show_camera_bg=show_camera_bg, tint_mode=tint_mode
        )
        return (
            tf.binary_mask,
            tf.stylized_rgb,
            tf.coverage,
            tf.camera_ok,
            tf.error_msg,
        )

    def close(self) -> None:
        """Stop capture thread and release webcam and MediaPipe resources."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        with self._seg_lock:
            for attr in (
                "_task_segmenter",
                "_solution_segmenter",
                "_pose_landmarker",
                "_solution_pose",
                "_hand_landmarker",
                "_solution_hands",
            ):
                obj = getattr(self, attr, None)
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
                    setattr(self, attr, None)
