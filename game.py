"""Core game loop, Virtual Canvas fullscreen scaling, Vital Core, and Hand Gestures."""

from __future__ import annotations

from collections import deque
import math
import random
from typing import Optional

import cv2
import numpy as np
import pygame

from audio import SoundFX
from config import (
    COLOR_BG,
    COLOR_DANGER,
    COLOR_EDGE_BRIGHT,
    COLOR_GRID,
    COLOR_HAND_FIST,
    COLOR_HAND_FIST_CORE,
    COLOR_HAND_OPEN,
    COLOR_HEART,
    COLOR_HEART_EMPTY,
    COLOR_LIMB_GLOW,
    COLOR_MUTED,
    COLOR_ORB_GOLD,
    COLOR_ORB_HEAL,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_SAFE,
    COLOR_SILHOUETTE_CYAN,
    COLOR_TEXT,
    CORE_GRAB_RADIUS,
    CORE_MAX_TORSO_RADIUS,
    CORE_RADIUS,
    DEFAULT_FULLSCREEN,
    GAME_DURATION_SECONDS,
    HAND_MAGNET_RADIUS,
    INVULNERABILITY_DURATION,
    LASER_WARNING_DURATION,
    MAX_LIVES,
    ORB_ABSORB_RADIUS,
    ORB_GOLD_POINTS,
    ORB_HEAL_FULL_BONUS,
    PHASES,
    SURVIVAL_POINTS_PER_SECOND,
    TARGET_FPS,
    VIRTUAL_H,
    VIRTUAL_W,
    WALL_DODGE_BONUS,
    WINDOW_TITLE,
)
from obstacles import FloatingText, Laser, Meteor, Obstacle, Orb, Particle, SafeZoneWall
from segmentation import HandState, SilhouetteSegmenter, TrackingFrame


def format_mm_ss(seconds: float) -> str:
    """Format non-negative seconds as MM:SS."""
    total = max(0, int(math.ceil(seconds)))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


class EsquivaLaSiluetaGame:
    """Main arcade coordinator with 1280x720 virtual canvas, limbs, and hand gestures."""

    STATE_MENU = "MENU"
    STATE_PLAYING = "PLAYING"
    STATE_PAUSED = "PAUSED"
    STATE_GAME_OVER = "GAME_OVER"
    STATE_VICTORY = "VICTORY"

    def __init__(
        self,
        segmenter: SilhouetteSegmenter,
        duration_seconds: float = GAME_DURATION_SECONDS,
        width: int = VIRTUAL_W,
        height: int = VIRTUAL_H,
        fullscreen: bool = DEFAULT_FULLSCREEN,
        seed: Optional[int] = None,
    ) -> None:
        if not pygame.get_init():
            pygame.init()

        self.width = width
        self.height = height
        self.duration = max(10.0, float(duration_seconds))
        self.segmenter = segmenter
        self.rng = random.Random(seed)

        # Detect monitor size BEFORE creating display surface
        self._monitor_size = self._detect_monitor_size()
        self._windowed_size = self._fit_windowed_size(self._monitor_size)
        self.is_fullscreen = bool(fullscreen)

        # Virtual 16:9 HD render target + shake buffer + reusable alpha FX overlay
        self._virtual_canvas = pygame.Surface((self.width, self.height))
        self._shake_surface = pygame.Surface((self.width, self.height))
        self._fx_overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._viewport_rect = pygame.Rect(0, 0, self.width, self.height)
        self._scaled_surface: Optional[pygame.Surface] = None
        self._last_screen_size: tuple[int, int] = (0, 0)
        self.screen: pygame.Surface
        self._apply_display_mode(self.is_fullscreen)

        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.sfx = SoundFX()

        # Pre-cached fonts (for 1280x720 virtual canvas)
        self.font_title = pygame.font.SysFont("dejavusans,liberation sans,arial", 48, bold=True)
        self.font_h1 = pygame.font.SysFont("dejavusans,liberation sans,arial", 38, bold=True)
        self.font_h2 = pygame.font.SysFont("dejavusans,liberation sans,arial", 22, bold=True)
        self.font_body = pygame.font.SysFont("dejavusans,liberation sans,arial", 18, bold=False)
        self.font_bold = pygame.font.SysFont("dejavusans,liberation sans,arial", 18, bold=True)
        self.font_small = pygame.font.SysFont("dejavusans,liberation sans,arial", 14, bold=False)
        self.font_tiny = pygame.font.SysFont("dejavusans,liberation sans,arial", 12, bold=True)
        self.font_timer = pygame.font.SysFont("dejavusans,liberation sans,arial", 40, bold=True)
        self.font_score = pygame.font.SysFont("dejavusans,liberation sans,arial", 30, bold=True)

        # Pre-rendered subtle neon grid overlay (computed once)
        self._grid_surf = self._build_grid_surface()

        # Visual and debug toggles
        self.show_camera_bg = True
        self.debug_mask = False
        self.running = True

        # Session tracking
        self.state = self.STATE_MENU
        self.best_score = 0
        self.current_fps = float(TARGET_FPS)

        # Run state
        self.time_remaining = self.duration
        self.elapsed = 0.0
        self.lives = MAX_LIVES
        self.score = 0.0
        self.current_phase_idx = 0
        self.invuln_timer = 0.0
        self.hit_flash_timer = 0.0
        self.orb_flash_timer = 0.0
        self.orb_flash_kind = "gold"
        self.screen_shake_timer = 0.0
        self.shake_offset: tuple[int, int] = (0, 0)
        self.phase_banner_timer = 0.0
        self.phase_banner_phase = PHASES[0]

        self.obstacles: list[Obstacle] = []
        self.particles: list[Particle] = []
        self.floating_texts: list[FloatingText] = []

        # Dynamic Vital Core & hand gesture interaction state
        self.active_core_pos: tuple[float, float] = (self.width * 0.5, self.height * 0.55)
        self.dragging_core = False
        self.core_drag_hand_id: Optional[str] = None
        self._core_returning = False
        self._active_core_trail: deque[tuple[float, float]] = deque(maxlen=10)
        self.last_hands: list[HandState] = []

        # Spawner cooldown timers
        self._meteor_timer = 0.8
        self._laser_timer = 1.6
        self._wall_timer = 2.0
        self._orb_timer = 7.5
        self._last_wall_pattern: Optional[str] = None
        self._last_laser_orient = "horizontal"

        # Metrics & debug inspection
        self.orbs_collected = 0
        self.hits_taken = 0
        self.last_player_px = 0
        self.last_coverage = 0.0
        self.last_core_pos: tuple[float, float] = self.active_core_pos
        self.last_tracking_source = "memory"

        # Clickable UI buttons updated on draw (in virtual coordinates)
        self._start_btn_rect: Optional[pygame.Rect] = None
        self._restart_btn_rect: Optional[pygame.Rect] = None
        self._menu_btn_rect: Optional[pygame.Rect] = None

    def _detect_monitor_size(self) -> tuple[int, int]:
        """Detect native desktop monitor resolution before creating a display."""
        try:
            sizes = pygame.display.get_desktop_sizes()
            if sizes and sizes[0][0] > 0 and sizes[0][1] > 0:
                return int(sizes[0][0]), int(sizes[0][1])
        except Exception:
            pass
        try:
            info = pygame.display.Info()
            if info.current_w > 0 and info.current_h > 0:
                return int(info.current_w), int(info.current_h)
        except Exception:
            pass
        return (self.width, self.height)

    def _fit_windowed_size(self, monitor_size: tuple[int, int]) -> tuple[int, int]:
        """Choose an initial windowed size that fits comfortably inside the monitor."""
        mw, mh = monitor_size
        if mw >= self.width and mh >= self.height:
            return (self.width, self.height)
        scale = min((mw * 0.90) / max(1, self.width), (mh * 0.90) / max(1, self.height))
        return (
            max(640, int(round(self.width * scale))),
            max(360, int(round(self.height * scale))),
        )

    def _apply_display_mode(self, fullscreen: bool) -> None:
        """Switch between native fullscreen and resizable windowed mode."""
        self.is_fullscreen = bool(fullscreen)
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            sw, sh = self.screen.get_size()
            if sw <= 0 or sh <= 0:
                self.screen = pygame.display.set_mode(
                    self._monitor_size, pygame.FULLSCREEN
                )
        else:
            self.screen = pygame.display.set_mode(
                self._windowed_size, pygame.RESIZABLE
            )
        self._recompute_viewport(*self.screen.get_size())

    def _recompute_viewport(self, sw: int, sh: int) -> None:
        """Compute proportional centered 16:9 viewport inside physical screen."""
        sw = max(1, int(sw))
        sh = max(1, int(sh))
        self._last_screen_size = (sw, sh)
        scale = min(sw / max(1, self.width), sh / max(1, self.height))
        vw = max(1, int(round(self.width * scale)))
        vh = max(1, int(round(self.height * scale)))
        vx = (sw - vw) // 2
        vy = (sh - vh) // 2
        self._viewport_rect = pygame.Rect(vx, vy, vw, vh)
        if (vw, vh) != (self.width, self.height):
            self._scaled_surface = pygame.Surface((vw, vh))
        else:
            self._scaled_surface = None

    def toggle_fullscreen(self) -> bool:
        """Toggle fullscreen/windowed mode and return the new fullscreen state."""
        self._apply_display_mode(not self.is_fullscreen)
        return self.is_fullscreen

    def _screen_to_logical(self, pos: tuple[int, int]) -> tuple[float, float]:
        """Map physical screen mouse coordinates into the virtual 1280x720 arena."""
        vr = self._viewport_rect
        lx = (pos[0] - vr.x) * (self.width / max(1, vr.w))
        ly = (pos[1] - vr.y) * (self.height / max(1, vr.h))
        return (lx, ly)

    def _reset_grabs(self) -> None:
        """Release core dragging and any grabbed orbs safely."""
        self.dragging_core = False
        self.core_drag_hand_id = None
        self._core_returning = False
        for obs in self.obstacles:
            if isinstance(obs, Orb):
                obs.grabbed_by = None

    def _clear_transient_effects(self, clear_invuln: bool = True) -> None:
        """Immediately halt screen shake, flashes, grabs, and optionally invulnerability."""
        self.screen_shake_timer = 0.0
        self.shake_offset = (0, 0)
        self.hit_flash_timer = 0.0
        self.orb_flash_timer = 0.0
        self._reset_grabs()
        if clear_invuln:
            self.invuln_timer = 0.0

    def _build_grid_surface(self) -> pygame.Surface:
        """Create a pre-rendered futuristic coordinate grid with subtle alpha."""
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        step = 64
        for x in range(0, self.width, step):
            pygame.draw.line(surf, (*COLOR_GRID, 70), (x, 0), (x, self.height), 1)
        for y in range(0, self.height, step):
            pygame.draw.line(surf, (*COLOR_GRID, 70), (0, y), (self.width, y), 1)
        return surf

    def reset_run(self, start_immediately: bool = True) -> None:
        """Reset all gameplay timers, entities, lives, and score for a new run."""
        self.time_remaining = self.duration
        self.elapsed = 0.0
        self.lives = MAX_LIVES
        self.score = 0.0
        self.current_phase_idx = 0
        self._clear_transient_effects(clear_invuln=True)
        self.phase_banner_timer = 2.6
        self.phase_banner_phase = PHASES[0]

        self.obstacles.clear()
        self.particles.clear()
        self.floating_texts.clear()

        self._meteor_timer = 0.9
        self._laser_timer = 1.4
        self._wall_timer = 1.8
        self._orb_timer = 8.0
        self._last_wall_pattern = None
        self._last_laser_orient = "horizontal"

        self.orbs_collected = 0
        self.hits_taken = 0

        if start_immediately:
            self.state = self.STATE_PLAYING
            self.sfx.play("phase_up")

    def _get_phase_for_elapsed(self, elapsed: float) -> dict:
        frac = min(0.9999, max(0.0, elapsed / max(0.01, self.duration)))
        for phase in PHASES:
            if phase["start_frac"] <= frac < phase["end_frac"]:
                return phase
        return PHASES[-1]

    def _active_wall(self) -> Optional[SafeZoneWall]:
        for obs in self.obstacles:
            if isinstance(obs, SafeZoneWall) and obs.alive:
                return obs
        return None

    def _update_demo_controls(self, dt: float) -> None:
        """Forward keyboard arrows/WASD/C/F & mouse fist to demo mode."""
        if not self.segmenter.demo_mode:
            return
        keys = pygame.key.get_pressed()
        dx = 0.0
        dy = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1.0
        crouch = bool(keys[pygame.K_c] or keys[pygame.K_LCTRL] or keys[pygame.K_DOWN])
        mouse_pressed = bool(pygame.mouse.get_pressed(3)[0])
        fist_closed = bool(keys[pygame.K_f] or mouse_pressed)
        self.segmenter.set_demo_fist(fist_closed)
        self.segmenter.step_demo(dx, dy, crouch, dt)

    def handle_events(self) -> None:
        """Process keyboard, window resize, and mouse input events."""
        resize_events = {pygame.VIDEORESIZE}
        for attr in ("WINDOWRESIZED", "WINDOWSIZECHANGED"):
            ev_type = getattr(pygame, attr, None)
            if ev_type is not None:
                resize_events.add(ev_type)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type in resize_events:
                disp = pygame.display.get_surface()
                if disp is not None:
                    self.screen = disp
                    sw, sh = disp.get_size()
                else:
                    sw = int(getattr(event, "w", getattr(event, "x", self.width)))
                    sh = int(getattr(event, "h", getattr(event, "y", self.height)))
                if not self.is_fullscreen and sw > 0 and sh > 0:
                    self._windowed_size = (max(320, sw), max(180, sh))
                self._recompute_viewport(sw, sh)
                continue

            if event.type == pygame.MOUSEMOTION and self.segmenter.demo_mode:
                if pygame.mouse.get_focused():
                    lx, ly = self._screen_to_logical(event.pos)
                    self.segmenter.set_demo_hand_pos(lx / self.width, ly / self.height)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                lpos = self._screen_to_logical(event.pos)
                if (
                    self.state == self.STATE_MENU
                    and self._start_btn_rect
                    and self._start_btn_rect.collidepoint(lpos)
                ):
                    self.reset_run(start_immediately=True)
                elif self.state in (self.STATE_GAME_OVER, self.STATE_VICTORY):
                    if self._restart_btn_rect and self._restart_btn_rect.collidepoint(lpos):
                        self.reset_run(start_immediately=True)
                    elif self._menu_btn_rect and self._menu_btn_rect.collidepoint(lpos):
                        self._clear_transient_effects(clear_invuln=True)
                        self.state = self.STATE_MENU

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_F11:
                    self.toggle_fullscreen()
                    continue

                if key == pygame.K_ESCAPE:
                    if self.state == self.STATE_PLAYING:
                        self._clear_transient_effects(clear_invuln=False)
                        self.state = self.STATE_PAUSED
                    else:
                        self.running = False
                        return
                    continue

                if key == pygame.K_b:
                    self.show_camera_bg = not self.show_camera_bg
                elif key == pygame.K_m:
                    self.sfx.toggle_mute()
                elif key == pygame.K_d and not self.segmenter.demo_mode:
                    self.debug_mask = not self.debug_mask
                elif key == pygame.K_F3:
                    self.debug_mask = not self.debug_mask

                if self.state == self.STATE_MENU:
                    if key in (pygame.K_SPACE, pygame.K_RETURN):
                        self.reset_run(start_immediately=True)

                elif self.state == self.STATE_PLAYING:
                    if key == pygame.K_p:
                        self._clear_transient_effects(clear_invuln=False)
                        self.state = self.STATE_PAUSED

                elif self.state == self.STATE_PAUSED:
                    if key in (pygame.K_p, pygame.K_SPACE):
                        self.state = self.STATE_PLAYING
                    elif key == pygame.K_r:
                        self.reset_run(start_immediately=True)
                    elif key == pygame.K_v:
                        self._clear_transient_effects(clear_invuln=True)
                        self.state = self.STATE_MENU

                elif self.state in (self.STATE_GAME_OVER, self.STATE_VICTORY):
                    if key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                        self.reset_run(start_immediately=True)
                    elif key in (pygame.K_v, pygame.K_BACKSPACE):
                        self._clear_transient_effects(clear_invuln=True)
                        self.state = self.STATE_MENU

    def _clamp_to_silhouette(
        self,
        target_pos: tuple[float, float],
        chest_anchor: tuple[float, float],
        binary_mask: np.ndarray,
    ) -> tuple[float, float]:
        """Keep a dragged Vital Core strictly inside valid silhouette pixels (with torso fallback)."""
        margin = float(CORE_RADIUS + 8)
        tx = float(np.clip(target_pos[0], margin, self.width - margin))
        ty = float(np.clip(target_pos[1], margin, self.height - margin))
        ax = float(np.clip(chest_anchor[0], margin, self.width - margin))
        ay = float(np.clip(chest_anchor[1], margin, self.height - margin))

        if binary_mask.size > 0:
            mh, mw = binary_mask.shape
            ix = int(np.clip(round(tx * mw / max(1, self.width)), 0, mw - 1))
            iy = int(np.clip(round(ty * mh / max(1, self.height)), 0, mh - 1))
            if binary_mask[iy, ix] > 0:
                return (tx, ty)

            # Ray-march from target toward chest anchor to find exact silhouette boundary
            steps = 24
            for i in range(1, steps + 1):
                t = i / steps
                px = tx + t * (ax - tx)
                py = ty + t * (ay - ty)
                mx = int(np.clip(round(px * mw / max(1, self.width)), 0, mw - 1))
                my = int(np.clip(round(py * mh / max(1, self.height)), 0, mh - 1))
                if binary_mask[my, mx] > 0:
                    return (
                        float(np.clip(px, margin, self.width - margin)),
                        float(np.clip(py, margin, self.height - margin)),
                    )

        # Fallback if silhouette mask flickered empty: cap within torso safety radius
        dx = tx - ax
        dy = ty - ay
        dist = math.hypot(dx, dy)
        if dist > CORE_MAX_TORSO_RADIUS and dist > 1e-6:
            scale = CORE_MAX_TORSO_RADIUS / dist
            tx = ax + dx * scale
            ty = ay + dy * scale
        return (
            float(np.clip(tx, margin, self.width - margin)),
            float(np.clip(ty, margin, self.height - margin)),
        )

    def _update_hand_interactions(self, dt: float, tf: TrackingFrame) -> None:
        """Process fist orb attraction (Mech A) & elastic Vital Core dragging (Mech B)."""
        hand_map: dict[str, HandState] = {h.hand_id: h for h in tf.hands}
        busy_hands: set[str] = set()

        # 1. Maintain or release already-grabbed orbs
        for obs in self.obstacles:
            if not obs.alive or not isinstance(obs, Orb) or obs.grabbed_by is None:
                continue
            hand = hand_map.get(obs.grabbed_by)
            if hand is not None and hand.is_fist:
                obs.pull_toward(hand.pos[0], hand.pos[1], dt)
                busy_hands.add(hand.hand_id)
            else:
                obs.release()

        # 2. Maintain or release dragged Vital Core
        if self.dragging_core:
            drag_hand = (
                hand_map.get(self.core_drag_hand_id)
                if self.core_drag_hand_id is not None
                else None
            )
            if drag_hand is not None and drag_hand.is_fist:
                busy_hands.add(drag_hand.hand_id)
            else:
                self.dragging_core = False
                self.core_drag_hand_id = None
                self._core_returning = True
                self.sfx.play("release")

        # 3. Start new fist interactions for free closed-fist hands
        for hand in tf.hands:
            if not hand.is_fist or hand.hand_id in busy_hands:
                continue

            hx, hy = hand.pos
            best_orb: Optional[Orb] = None
            best_orb_dist = HAND_MAGNET_RADIUS + 1.0
            for obs in self.obstacles:
                if not obs.alive or not isinstance(obs, Orb) or obs.grabbed_by is not None:
                    continue
                d = math.hypot(hx - obs.x, hy - obs.y)
                if d <= HAND_MAGNET_RADIUS and d < best_orb_dist:
                    best_orb = obs
                    best_orb_dist = d

            dist_core = math.hypot(
                hx - self.active_core_pos[0], hy - self.active_core_pos[1]
            )

            # Prefer core grab when fist closes right on the chest core (unless an orb is closer)
            if (
                not self.dragging_core
                and dist_core <= CORE_GRAB_RADIUS
                and (best_orb is None or dist_core <= best_orb_dist)
            ):
                self.dragging_core = True
                self.core_drag_hand_id = hand.hand_id
                self._core_returning = False
                busy_hands.add(hand.hand_id)
                self.sfx.play("grab")
            elif best_orb is not None:
                best_orb.grabbed_by = hand.hand_id
                busy_hands.add(hand.hand_id)
                self.sfx.play("grab")

        # 4. Update effective active_core_pos (dragged, elastic return, or anchored)
        ax, ay = tf.core_pos
        if (
            self.dragging_core
            and self.core_drag_hand_id is not None
            and self.core_drag_hand_id in hand_map
        ):
            clamped = self._clamp_to_silhouette(
                hand_map[self.core_drag_hand_id].pos, tf.core_pos, tf.binary_mask
            )
            blend = 1.0 if dt <= 0.0 else min(1.0, 1.0 - math.exp(-22.0 * dt))
            self.active_core_pos = (
                self.active_core_pos[0] + (clamped[0] - self.active_core_pos[0]) * blend,
                self.active_core_pos[1] + (clamped[1] - self.active_core_pos[1]) * blend,
            )
        elif self._core_returning:
            blend = 1.0 if dt <= 0.0 else min(1.0, 1.0 - math.exp(-10.0 * dt))
            nx = self.active_core_pos[0] + (ax - self.active_core_pos[0]) * blend
            ny = self.active_core_pos[1] + (ay - self.active_core_pos[1]) * blend
            if math.hypot(ax - nx, ay - ny) < 3.0:
                self.active_core_pos = (ax, ay)
                self._core_returning = False
            else:
                self.active_core_pos = (nx, ny)
        else:
            self.active_core_pos = (ax, ay)

        self._active_core_trail.append(self.active_core_pos)

    def _spawn_meteor(self, speed_min: float, speed_max: float) -> None:
        radius = self.rng.randint(26, 40)
        x = self.rng.uniform(radius + 24, self.width - radius - 24)
        vx = self.rng.uniform(-65.0, 65.0)
        vy = self.rng.uniform(speed_min, speed_max)
        self.obstacles.append(
            Meteor(
                x=x,
                y=-radius - 10,
                radius=radius,
                vx=vx,
                vy=vy,
                rng=self.rng,
                screen_size=(self.width, self.height),
            )
        )

    def _spawn_laser(
        self,
        avoid_rect: Optional[pygame.Rect] = None,
        allow_double: bool = False,
    ) -> None:
        orient = "vertical" if self._last_laser_orient == "horizontal" else "horizontal"
        self._last_laser_orient = orient
        bx, by, bw, _bh = self.segmenter.get_silhouette_bounds()

        def pick_pos(orientation: str) -> Optional[int]:
            if orientation == "horizontal":
                lo = max(130, by + 20)
                hi = max(lo + 10, self.height - 62)
            else:
                lo = max(95, bx + 40)
                hi = max(lo + 10, min(self.width - 95, bx + bw - 40))

            for _ in range(12):
                p = self.rng.randint(lo, hi)
                if avoid_rect is None:
                    return p
                if orientation == "horizontal" and not (
                    avoid_rect.top - 28 <= p <= avoid_rect.bottom + 28
                ):
                    return p
                if orientation == "vertical" and not (
                    avoid_rect.left - 28 <= p <= avoid_rect.right + 28
                ):
                    return p
            return None

        pos = pick_pos(orient)
        if pos is None:
            return
        thickness = self.rng.randint(52, 66)
        self.obstacles.append(
            Laser(
                orientation=orient,
                pos=pos,
                thickness=thickness,
                warning_duration=LASER_WARNING_DURATION,
                screen_size=(self.width, self.height),
            )
        )
        self.sfx.play("laser_warn")

        if allow_double and avoid_rect is None:
            pos2 = pick_pos(orient)
            if pos2 is not None and abs(pos2 - pos) > 210:
                self.obstacles.append(
                    Laser(
                        orientation=orient,
                        pos=pos2,
                        thickness=54,
                        warning_duration=LASER_WARNING_DURATION,
                        screen_size=(self.width, self.height),
                    )
                )

    def _spawn_wall(self, warning_duration: float) -> None:
        patterns = list(SafeZoneWall.PATTERN_SPECS.keys())
        choices = [p for p in patterns if p != self._last_wall_pattern]
        pattern = self.rng.choice(choices)
        self._last_wall_pattern = pattern
        self.obstacles.append(
            SafeZoneWall(
                pattern=pattern,
                warning_duration=warning_duration,
                screen_size=(self.width, self.height),
            )
        )
        self.sfx.play("wall_warn")

    def _spawn_orb(self) -> None:
        if self.lives < MAX_LIVES and self.rng.random() < 0.42:
            kind = "heal"
        else:
            kind = "gold"
        bx, by, bw, bh = self.segmenter.get_silhouette_bounds()
        x_lo = min(self.width - 120, max(120, bx + 90))
        x_hi = max(x_lo + 10, min(self.width - 120, bx + bw - 90))
        y_lo = min(self.height - 110, max(145, by + 60))
        y_hi = max(y_lo + 10, min(self.height - 85, by + bh - 80))
        x = self.rng.uniform(x_lo, x_hi)
        y = self.rng.uniform(y_lo, y_hi)
        self.obstacles.append(Orb(x=x, y=y, kind=kind))

    def _update_spawner(self, dt: float) -> None:
        """Advance difficulty director according to the 4 phases across 5 minutes."""
        frac = min(1.0, max(0.0, self.elapsed / max(0.01, self.duration)))
        phase = self._get_phase_for_elapsed(self.elapsed)
        phase_idx = phase["number"] - 1

        if phase_idx != self.current_phase_idx:
            self.current_phase_idx = phase_idx
            self.phase_banner_phase = phase
            self.phase_banner_timer = 2.6
            self.sfx.play("phase_up")
            self._laser_timer = min(self._laser_timer, 1.1)
            self._wall_timer = min(self._wall_timer, 1.4)

        active_wall = self._active_wall()
        wall_closing_soon = (
            active_wall is not None
            and (
                active_wall.state == "ACTIVE"
                or (active_wall.state == "WARNING" and active_wall.timer < 1.25)
            )
        )

        self._meteor_timer -= dt
        self._laser_timer -= dt
        self._wall_timer -= dt
        self._orb_timer -= dt

        if phase["number"] == 1:
            local_t = frac / 0.20
            interval = 1.25 - 0.42 * local_t
            if self._meteor_timer <= 0.0:
                self._spawn_meteor(210.0 + 45.0 * local_t, 270.0 + 60.0 * local_t)
                self._meteor_timer = interval * self.rng.uniform(0.82, 1.18)

        elif phase["number"] == 2:
            local_t = (frac - 0.20) / 0.30
            if self._meteor_timer <= 0.0:
                self._spawn_meteor(230.0 + 45.0 * local_t, 305.0 + 55.0 * local_t)
                self._meteor_timer = (1.35 - 0.30 * local_t) * self.rng.uniform(0.85, 1.15)
            if self._laser_timer <= 0.0:
                self._spawn_laser(
                    allow_double=(local_t > 0.65 and self.rng.random() < 0.35)
                )
                self._laser_timer = (4.1 - 1.10 * local_t) * self.rng.uniform(0.88, 1.12)

        elif phase["number"] == 3:
            local_t = (frac - 0.50) / 0.30
            if active_wall is None and self._wall_timer <= 0.0:
                self._spawn_wall(warning_duration=2.6 - 0.30 * local_t)
                self._wall_timer = 6.2 - 0.8 * local_t
            if self._meteor_timer <= 0.0:
                if not wall_closing_soon:
                    self._spawn_meteor(245.0 + 40.0 * local_t, 325.0 + 40.0 * local_t)
                self._meteor_timer = 1.55 * self.rng.uniform(0.85, 1.15)
            if self._laser_timer <= 0.0:
                if active_wall is None:
                    self._spawn_laser(allow_double=False)
                    self._laser_timer = 4.3
                else:
                    self._laser_timer = 1.6

        else:
            local_t = (frac - 0.80) / 0.20
            if active_wall is None and self._wall_timer <= 0.0:
                self._spawn_wall(warning_duration=2.15 - 0.15 * local_t)
                self._wall_timer = 5.6
            if self._meteor_timer <= 0.0:
                if not (active_wall is not None and active_wall.state == "ACTIVE"):
                    self._spawn_meteor(300.0 + 50.0 * local_t, 380.0 + 55.0 * local_t)
                self._meteor_timer = (0.95 - 0.18 * local_t) * self.rng.uniform(0.82, 1.12)
            if self._laser_timer <= 0.0:
                avoid = active_wall.safe_rect if active_wall is not None else None
                self._spawn_laser(
                    avoid_rect=avoid,
                    allow_double=(avoid is None and self.rng.random() < 0.45),
                )
                self._laser_timer = (3.0 - 0.45 * local_t) * self.rng.uniform(0.88, 1.12)

        if self._orb_timer <= 0.0:
            active_orbs = sum(1 for o in self.obstacles if isinstance(o, Orb) and o.alive)
            if active_orbs < 2:
                self._spawn_orb()
            self._orb_timer = self.rng.uniform(10.0, 14.0)

    def _emit_burst(
        self, x: float, y: float, color: tuple[int, int, int], count: int = 18
    ) -> None:
        for _ in range(count):
            ang = self.rng.uniform(0.0, math.tau)
            spd = self.rng.uniform(80.0, 290.0)
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    color=color,
                    life=self.rng.uniform(0.28, 0.55),
                    radius=self.rng.randint(3, 6),
                )
            )

    def _handle_collisions(self, core_pos: tuple[float, float]) -> None:
        """Bullet-hell precision collision against the active Vital Core."""
        cx, cy = core_pos

        # 1. Orb absorption at Vital Core (direct touch or dragged via fist tractor)
        for obs in self.obstacles:
            if not obs.alive or obs.is_hazard:
                continue
            if obs.collides_with_circle(cx, cy, ORB_ABSORB_RADIUS) and isinstance(obs, Orb):
                obs.grabbed_by = None
                obs.alive = False
                self.orbs_collected += 1
                if obs.kind == "heal":
                    if self.lives < MAX_LIVES:
                        self.lives += 1
                        self.floating_texts.append(
                            FloatingText("+1 VIDA", obs.x, obs.y, COLOR_ORB_HEAL, life=1.1, size=26)
                        )
                    else:
                        self.score += ORB_HEAL_FULL_BONUS
                        self.floating_texts.append(
                            FloatingText(
                                f"+{ORB_HEAL_FULL_BONUS}", obs.x, obs.y, COLOR_ORB_HEAL, life=1.1, size=26
                            )
                        )
                    self.orb_flash_kind = "heal"
                    self.orb_flash_timer = 0.28
                    self._emit_burst(obs.x, obs.y, COLOR_ORB_HEAL, count=20)
                    self.sfx.play("orb_heal")
                else:
                    self.score += ORB_GOLD_POINTS
                    self.orb_flash_kind = "gold"
                    self.orb_flash_timer = 0.25
                    self.floating_texts.append(
                        FloatingText(
                            f"+{ORB_GOLD_POINTS}", obs.x, obs.y, COLOR_ORB_GOLD, life=1.0, size=26
                        )
                    )
                    self._emit_burst(obs.x, obs.y, COLOR_ORB_GOLD, count=18)
                    self.sfx.play("orb_gold")

        # 2. Active hazards ONLY damage if they touch the active Vital Core
        if self.invuln_timer > 0.0:
            return

        for obs in self.obstacles:
            if not obs.is_hazard or not obs.is_dangerous_now():
                continue
            if obs.collides_with_circle(cx, cy, CORE_RADIUS):
                self.lives = max(0, self.lives - 1)
                self.hits_taken += 1
                self.sfx.play("hit")

                if isinstance(obs, Meteor):
                    obs.alive = False
                    fx_x, fx_y = obs.x, obs.y
                elif isinstance(obs, Laser):
                    fx_x = cx if obs.orientation == "horizontal" else float(obs.pos)
                    fx_y = float(obs.pos) if obs.orientation == "horizontal" else cy
                elif isinstance(obs, SafeZoneWall):
                    obs.hit_player = True
                    fx_x, fx_y = cx, cy
                else:
                    fx_x, fx_y = cx, cy

                self._emit_burst(fx_x, fx_y, COLOR_DANGER, count=24)

                if self.lives <= 0:
                    self._clear_transient_effects(clear_invuln=True)
                    self.floating_texts.clear()
                    self.best_score = max(self.best_score, int(self.score))
                    self.state = self.STATE_GAME_OVER
                    self.sfx.play("game_over")
                else:
                    self.invuln_timer = INVULNERABILITY_DURATION
                    self.hit_flash_timer = 0.30
                    self.screen_shake_timer = 0.26
                    self.floating_texts.append(
                        FloatingText("-1 VIDA", fx_x, max(115.0, fx_y - 26.0), COLOR_DANGER, life=1.1, size=28)
                    )
                break

    def _update_playing(self, dt: float, core_pos: tuple[float, float]) -> None:
        """Advance match timers, obstacles, particles, and core collisions."""
        self.time_remaining = max(0.0, self.time_remaining - dt)
        self.elapsed = self.duration - self.time_remaining
        self.score += SURVIVAL_POINTS_PER_SECOND * dt

        if self.invuln_timer > 0.0:
            self.invuln_timer = max(0.0, self.invuln_timer - dt)
        if self.hit_flash_timer > 0.0:
            self.hit_flash_timer = max(0.0, self.hit_flash_timer - dt)
        if self.orb_flash_timer > 0.0:
            self.orb_flash_timer = max(0.0, self.orb_flash_timer - dt)
        if self.screen_shake_timer > 0.0:
            self.screen_shake_timer = max(0.0, self.screen_shake_timer - dt)
            if self.screen_shake_timer <= 0.0:
                self.shake_offset = (0, 0)
        if self.phase_banner_timer > 0.0:
            self.phase_banner_timer = max(0.0, self.phase_banner_timer - dt)

        self._update_spawner(dt)

        for obs in self.obstacles:
            obs.update(dt)
            if obs.just_activated:
                if isinstance(obs, Laser):
                    self.sfx.play("laser_fire")
                elif isinstance(obs, SafeZoneWall):
                    self.sfx.play("wall_impact")
            if isinstance(obs, SafeZoneWall) and obs.dodged:
                obs.dodged = False
                self.score += WALL_DODGE_BONUS
                self.floating_texts.append(
                    FloatingText(
                        f"¡ESQUIVADO! +{WALL_DODGE_BONUS}",
                        obs.safe_rect.centerx,
                        max(130.0, float(obs.safe_rect.centery)),
                        COLOR_SAFE,
                        life=1.2,
                        size=26,
                    )
                )

        self._handle_collisions(core_pos)

        self.obstacles = [o for o in self.obstacles if o.alive]

        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

        for ft in self.floating_texts:
            ft.update(dt)
        self.floating_texts = [ft for ft in self.floating_texts if ft.alive]

        if self.state == self.STATE_PLAYING and self.time_remaining <= 0.0:
            self._clear_transient_effects(clear_invuln=True)
            self.floating_texts.clear()
            self.best_score = max(self.best_score, int(self.score))
            self.state = self.STATE_VICTORY
            self.sfx.play("victory")

    def _current_tint_mode(self) -> str:
        if self.state == self.STATE_PLAYING:
            if self.hit_flash_timer > 0.0:
                return "hit_flash"
            if self.invuln_timer > 0.0 and int(self.invuln_timer * 14.0) % 2 == 0:
                return "invuln_dim"
            if self.orb_flash_timer > 0.0:
                return "heal_glow" if self.orb_flash_kind == "heal" else "gold_glow"
        return "normal"

    def _draw_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        border_color: tuple[int, int, int] = COLOR_PANEL_BORDER,
        alpha: int = 215,
        radius: int = 12,
    ) -> None:
        panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (*COLOR_PANEL_BG, alpha), (0, 0, rect.w, rect.h), border_radius=radius)
        pygame.draw.rect(panel, (*border_color, 235), (0, 0, rect.w, rect.h), width=2, border_radius=radius)
        surface.blit(panel, rect.topleft)

    def _draw_heart(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        size: int,
        filled: bool,
        fill_override: Optional[tuple[int, int, int]] = None,
        edge_override: Optional[tuple[int, int, int]] = None,
    ) -> None:
        """Draw a crisp neon heart icon centered at (cx, cy)."""
        r = max(4, int(size * 0.28))
        fill_col = fill_override if fill_override is not None else (COLOR_HEART if filled else COLOR_HEART_EMPTY)
        edge_col = edge_override if edge_override is not None else ((255, 175, 205) if filled else (72, 92, 124))

        left_c = (cx - r + 1, cy - int(r * 0.45))
        right_c = (cx + r - 1, cy - int(r * 0.45))
        pts = [
            (cx - 2 * r + 1, cy - int(r * 0.1)),
            (cx, cy + int(size * 0.46)),
            (cx + 2 * r - 1, cy - int(r * 0.1)),
        ]
        pygame.draw.circle(surface, fill_col, left_c, r)
        pygame.draw.circle(surface, fill_col, right_c, r)
        pygame.draw.polygon(surface, fill_col, pts)

        outline = [
            (cx, cy - int(r * 0.45)),
            (cx - r, cy - int(r * 1.15)),
            (cx - 2 * r, cy - int(r * 0.25)),
            (cx, cy + int(size * 0.48)),
            (cx + 2 * r, cy - int(r * 0.25)),
            (cx + r, cy - int(r * 1.15)),
        ]
        pygame.draw.polygon(surface, edge_col, outline, width=2)
        if filled:
            pygame.draw.circle(surface, (255, 225, 238), (cx - r, cy - int(r * 0.6)), max(2, r // 3))

    def _draw_limb_glow(
        self,
        surface: pygame.Surface,
        limb_segments: list[tuple[tuple[float, float], tuple[float, float]]],
        limb_joints: list[tuple[float, float]],
    ) -> None:
        """Render glowing skeletal energy lines over arms/legs to reinforce fast motion."""
        if not limb_segments and not limb_joints:
            return
        overlay = self._fx_overlay
        overlay.fill((0, 0, 0, 0))
        for p1, p2 in limb_segments:
            a = (int(round(p1[0])), int(round(p1[1])))
            b = (int(round(p2[0])), int(round(p2[1])))
            pygame.draw.line(overlay, (*COLOR_LIMB_GLOW, 75), a, b, 10)
            pygame.draw.line(overlay, (*COLOR_EDGE_BRIGHT, 165), a, b, 3)
        for pt in limb_joints:
            c = (int(round(pt[0])), int(round(pt[1])))
            pygame.draw.circle(overlay, (*COLOR_LIMB_GLOW, 110), c, 7)
            pygame.draw.circle(overlay, (220, 255, 255, 200), c, 3)
        surface.blit(overlay, (0, 0))

    def _draw_tractor_beams(
        self, surface: pygame.Surface, hands: list[HandState]
    ) -> None:
        """Draw luminous energy beams connecting closed fists to magnetized orbs."""
        hand_map = {h.hand_id: h for h in hands}
        now = pygame.time.get_ticks() / 1000.0
        for obs in self.obstacles:
            if not obs.alive or not isinstance(obs, Orb) or obs.grabbed_by is None:
                continue
            hand = hand_map.get(obs.grabbed_by)
            if hand is None:
                continue
            col = COLOR_ORB_GOLD if obs.kind == "gold" else COLOR_ORB_HEAL
            hx, hy = int(round(hand.pos[0])), int(round(hand.pos[1]))
            ox, oy = int(round(obs.x)), int(round(obs.y))
            pygame.draw.line(surface, col, (hx, hy), (ox, oy), 5)
            pygame.draw.line(surface, (255, 255, 255), (hx, hy), (ox, oy), 2)
            pulse_t = (now * 2.5) % 1.0
            px = int(hx + (ox - hx) * pulse_t)
            py = int(hy + (oy - hy) * pulse_t)
            pygame.draw.circle(surface, (255, 255, 255), (px, py), 5)

    def _draw_core_tether(
        self,
        surface: pygame.Surface,
        anchor_pos: tuple[float, float],
        active_pos: tuple[float, float],
    ) -> None:
        """Render elastic tether and chest ghost marker when Vital Core is displaced."""
        dist = math.hypot(active_pos[0] - anchor_pos[0], active_pos[1] - anchor_pos[1])
        if not (self.dragging_core or self._core_returning) or dist < 6.0:
            return

        ax, ay = int(round(anchor_pos[0])), int(round(anchor_pos[1]))
        cx, cy = int(round(active_pos[0])), int(round(active_pos[1]))

        # Ghost ring at natural chest position
        pygame.draw.circle(surface, (80, 175, 220), (ax, ay), CORE_RADIUS - 4, width=1)
        pygame.draw.circle(surface, COLOR_SILHOUETTE_CYAN, (ax, ay), 4)

        # Elastic dashed line from chest anchor to displaced core
        steps = max(4, int(dist // 16))
        for i in range(0, steps, 2):
            t1 = i / steps
            t2 = min(1.0, (i + 1) / steps)
            p1 = (int(ax + (cx - ax) * t1), int(ay + (cy - ay) * t1))
            p2 = (int(ax + (cx - ax) * t2), int(ay + (cy - ay) * t2))
            pygame.draw.line(surface, COLOR_HAND_FIST_CORE, p1, p2, 3)

    def _draw_hands(
        self, surface: pygame.Surface, hands: list[HandState]
    ) -> None:
        """Render interactive open/fist hand cursors and magnetic reach cues."""
        if not hands:
            return
        now = pygame.time.get_ticks() / 1000.0
        has_nearby_orb = any(
            isinstance(o, Orb) and o.alive for o in self.obstacles
        )
        for hand in hands:
            hx, hy = int(round(hand.pos[0])), int(round(hand.pos[1]))
            if hand.is_fist:
                is_core_hand = self.dragging_core and hand.hand_id == self.core_drag_hand_id
                outer_col = COLOR_HAND_FIST_CORE if is_core_hand else COLOR_HAND_FIST
                r = 16 + int(2 * math.sin(now * 12.0))
                pygame.draw.circle(surface, outer_col, (hx, hy), r, width=3)
                pygame.draw.circle(surface, (255, 245, 200), (hx, hy), 7)
                tag_txt = "ESCUDO" if is_core_hand else "PUÑO"
                tag = self.font_tiny.render(tag_txt, True, outer_col)
                surface.blit(tag, tag.get_rect(center=(hx, max(18, hy - 26))))
            else:
                if has_nearby_orb:
                    # Subtle magnetic reach indicator when orbs are on screen
                    aura = self._fx_overlay
                    aura.fill((0, 0, 0, 0))
                    pygame.draw.circle(
                        aura,
                        (*COLOR_HAND_OPEN, 30),
                        (hx, hy),
                        int(HAND_MAGNET_RADIUS),
                        width=2,
                    )
                    surface.blit(aura, (0, 0))
                pygame.draw.circle(surface, COLOR_HAND_OPEN, (hx, hy), 21, width=2)
                pygame.draw.circle(surface, (215, 255, 255), (hx, hy), 4)

    def _draw_vital_core(
        self,
        surface: pygame.Surface,
        core_pos: tuple[float, float],
        tracking_source: str,
        trail: list[tuple[float, float]],
    ) -> None:
        """Render the glowing bullet-hell Vital Core, ring, trail, and fallback badge."""
        cx, cy = int(round(core_pos[0])), int(round(core_pos[1]))
        now = pygame.time.get_ticks() / 1000.0

        invuln_blink = self.state == self.STATE_PLAYING and self.invuln_timer > 0.0
        blink_white = invuln_blink and (int(self.invuln_timer * 16.0) % 2 == 0)

        if invuln_blink:
            ring_col = (255, 255, 255) if blink_white else COLOR_DANGER
            heart_fill = (255, 255, 255) if blink_white else COLOR_DANGER
            heart_edge = COLOR_DANGER if blink_white else (255, 230, 235)
        elif self.dragging_core:
            ring_col = COLOR_HAND_FIST
            heart_fill = COLOR_HAND_FIST_CORE
            heart_edge = (255, 240, 190)
        elif tracking_source == "memory":
            ring_col = (255, 195, 65)
            heart_fill = (255, 145, 55)
            heart_edge = (255, 240, 175)
        elif tracking_source == "mask":
            ring_col = (70, 240, 215)
            heart_fill = COLOR_HEART
            heart_edge = (255, 215, 230)
        else:
            ring_col = COLOR_SILHOUETTE_CYAN
            heart_fill = COLOR_HEART
            heart_edge = (255, 220, 235)

        if len(trail) > 1:
            n_t = len(trail)
            for idx, (tx, ty) in enumerate(trail[:-1]):
                frac = (idx + 1) / n_t
                tr = max(3, int(CORE_RADIUS * 0.45 * frac))
                t_surf = pygame.Surface((tr * 2 + 2, tr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(t_surf, (*ring_col, int(90 * frac)), (tr + 1, tr + 1), tr)
                surface.blit(t_surf, (int(tx - tr - 1), int(ty - tr - 1)))

        pulse = 1.0 + 0.08 * math.sin(now * 6.5)
        halo_r = int(CORE_RADIUS * 1.45 * pulse)
        halo_surf = pygame.Surface((halo_r * 2 + 4, halo_r * 2 + 4), pygame.SRCALPHA)
        center_h = (halo_r + 2, halo_r + 2)
        pygame.draw.circle(halo_surf, (*ring_col, 65), center_h, halo_r)
        pygame.draw.circle(halo_surf, (*heart_fill, 90), center_h, CORE_RADIUS)
        surface.blit(halo_surf, (cx - halo_r - 2, cy - halo_r - 2))

        pygame.draw.circle(surface, ring_col, (cx, cy), CORE_RADIUS, width=3)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), max(6, CORE_RADIUS - 3), width=1)

        self._draw_heart(
            surface,
            cx,
            cy + 1,
            size=22,
            filled=True,
            fill_override=heart_fill,
            edge_override=heart_edge,
        )

        if tracking_source == "memory" and not self.dragging_core:
            badge = self.font_small.render("BUSCANDO SEÑAL...", True, (255, 220, 120))
            b_rect = badge.get_rect(
                center=(cx, min(self.height - 46, cy + CORE_RADIUS + 20))
            ).inflate(16, 6)
            self._draw_panel(surface, b_rect, border_color=(255, 195, 65), alpha=205, radius=8)
            surface.blit(badge, badge.get_rect(center=b_rect.center))

    def _draw_hud(self, surface: pygame.Surface) -> None:
        """Render 3-card HUD on the 1280x720 virtual canvas."""
        lives_card = pygame.Rect(18, 12, 284, 82)
        self._draw_panel(
            surface,
            lives_card,
            border_color=COLOR_HEART if self.lives <= 2 else COLOR_PANEL_BORDER,
            alpha=215,
            radius=12,
        )
        lives_lbl = self.font_small.render(f"VIDAS   {self.lives} / {MAX_LIVES}", True, COLOR_MUTED)
        surface.blit(lives_lbl, (lives_card.left + 16, lives_card.top + 10))
        for i in range(MAX_LIVES):
            hx = lives_card.left + 34 + i * 48
            hy = lives_card.top + 48
            self._draw_heart(surface, hx, hy, size=24, filled=(i < self.lives))

        if self.invuln_timer > 0.0:
            shield_rect = pygame.Rect(lives_card.left + 158, lives_card.top + 8, 112, 20)
            pygame.draw.rect(surface, (65, 52, 18), shield_rect, border_radius=6)
            pygame.draw.rect(surface, COLOR_ORB_GOLD, shield_rect, width=1, border_radius=6)
            inv_txt = self.font_small.render(f"ESCUDO {self.invuln_timer:.1f}s", True, COLOR_ORB_GOLD)
            surface.blit(inv_txt, inv_txt.get_rect(center=shield_rect.center))

        phase = self._get_phase_for_elapsed(self.elapsed)
        center_w = 400
        center_card = pygame.Rect(self.width // 2 - center_w // 2, 12, center_w, 86)
        self._draw_panel(surface, center_card, border_color=phase["color"], alpha=220, radius=12)

        timer_col = COLOR_TEXT if self.time_remaining > 30.0 else (255, 110, 120)
        timer_surf = self.font_timer.render(format_mm_ss(self.time_remaining), True, timer_col)
        surface.blit(timer_surf, timer_surf.get_rect(center=(center_card.centerx, center_card.top + 26)))

        phase_surf = self.font_small.render(phase["name"], True, phase["color"])
        surface.blit(phase_surf, phase_surf.get_rect(center=(center_card.centerx, center_card.top + 56)))

        prog_w = center_w - 40
        prog_h = 8
        prog_x = center_card.centerx - prog_w // 2
        prog_y = center_card.bottom - 16
        pygame.draw.rect(surface, (26, 40, 64), (prog_x, prog_y, prog_w, prog_h), border_radius=4)
        frac = min(1.0, max(0.0, self.elapsed / max(0.01, self.duration)))
        pygame.draw.rect(
            surface, phase["color"], (prog_x, prog_y, int(prog_w * frac), prog_h), border_radius=4
        )
        for mark in (0.20, 0.50, 0.80):
            mx = prog_x + int(prog_w * mark)
            pygame.draw.line(surface, COLOR_TEXT, (mx, prog_y - 2), (mx, prog_y + prog_h + 2), 1)

        score_card = pygame.Rect(self.width - 302, 12, 284, 82)
        self._draw_panel(surface, score_card, border_color=COLOR_ORB_GOLD, alpha=215, radius=12)
        score_lbl = self.font_small.render("PUNTUACIÓN", True, COLOR_MUTED)
        surface.blit(score_lbl, score_lbl.get_rect(topright=(score_card.right - 16, score_card.top + 10)))

        score_surf = self.font_score.render(f"{int(self.score):,}", True, COLOR_ORB_GOLD)
        surface.blit(score_surf, score_surf.get_rect(topright=(score_card.right - 16, score_card.top + 28)))

        mute_tag = "MUTE" if self.sfx.muted else "SFX"
        fps_surf = self.font_small.render(
            f"FPS {self.current_fps:.0f}  ·  {mute_tag}", True, COLOR_MUTED
        )
        surface.blit(fps_surf, fps_surf.get_rect(bottomright=(score_card.right - 16, score_card.bottom - 8)))

        if self.phase_banner_timer > 0.0:
            p = self.phase_banner_phase
            b_rect = pygame.Rect(0, 0, 600, 74)
            b_rect.center = (self.width // 2, 146)
            self._draw_panel(surface, b_rect, border_color=p["color"], alpha=224, radius=12)
            t1 = self.font_h2.render(p["name"], True, p["color"])
            t2 = self.font_body.render(p["subtitle"], True, COLOR_TEXT)
            surface.blit(t1, t1.get_rect(center=(b_rect.centerx, b_rect.top + 24)))
            surface.blit(t2, t2.get_rect(center=(b_rect.centerx, b_rect.top + 52)))

        dbg_key = "[F3]" if self.segmenter.demo_mode else "[D/F3]"
        hint = f"[P/ESC] Pausa   [F11] Pantalla Completa   [B] Fondo   [M] Sonido   {dbg_key} Debug"
        if self.segmenter.demo_mode:
            hint = "DEMO: [Flechas/WASD] Cuerpo · [Mouse] Mano · [Clic/F] Puño · [C] Agachar  |  " + hint
        hint_surf = self.font_small.render(hint, True, COLOR_MUTED)
        surface.blit(hint_surf, hint_surf.get_rect(center=(self.width // 2, self.height - 14)))

    def _draw_menu(
        self,
        surface: pygame.Surface,
        tf: TrackingFrame,
    ) -> None:
        """Render live-demo Main Menu showing silhouette, limbs, hands, and chest core."""
        header_rect = pygame.Rect(140, 14, self.width - 280, 92)
        self._draw_panel(surface, header_rect, border_color=COLOR_SILHOUETTE_CYAN, alpha=215, radius=14)
        title_surf = self.font_title.render("ESQUIVA LA SILUETA", True, COLOR_SILHOUETTE_CYAN)
        sub_surf = self.font_bold.render(
            "Protege el CORAZÓN · Cierra el PUÑO para atraer orbes o mover tu corazón",
            True,
            COLOR_TEXT,
        )
        surface.blit(title_surf, title_surf.get_rect(center=(self.width // 2, header_rect.top + 34)))
        surface.blit(sub_surf, sub_surf.get_rect(center=(self.width // 2, header_rect.top + 72)))

        btn_rect = pygame.Rect(self.width // 2 - 240, 116, 480, 50)
        self._start_btn_rect = btn_rect
        pygame.draw.rect(surface, (0, 145, 195), btn_rect, border_radius=12)
        pygame.draw.rect(surface, COLOR_EDGE_BRIGHT, btn_rect, width=2, border_radius=12)
        btn_txt = self.font_h2.render("PRESIONA ESPACIO O ENTER PARA JUGAR", True, (255, 255, 255))
        surface.blit(btn_txt, btn_txt.get_rect(center=btn_rect.center))

        if not tf.camera_ok and not self.segmenter.demo_mode:
            cal_msg = f"Cámara no disponible: {tf.error_msg}"
            cal_col = COLOR_DANGER
        elif tf.tracking_source == "pose":
            hands_txt = f" · MANOS: {len(tf.hands)}" if tf.hands else ""
            cal_msg = f"● NÚCLEO EN PECHO (POSE · {tf.coverage * 100:.0f}%{hands_txt}) · ¡LISTO!"
            cal_col = COLOR_SAFE
        elif tf.tracking_source == "mask":
            cal_msg = f"● SILUETA DETECTADA ({tf.coverage * 100:.0f}%) · CENTRANDO NÚCLEO..."
            cal_col = (70, 240, 215)
        else:
            cal_msg = "● BUSCANDO SEÑAL: Colócate frente a la cámara"
            cal_col = COLOR_ORB_GOLD

        pill_surf = self.font_bold.render(cal_msg, True, cal_col)
        pill_rect = pill_surf.get_rect(center=(self.width // 2, 190)).inflate(30, 12)
        self._draw_panel(surface, pill_rect, border_color=cal_col, alpha=210, radius=14)
        surface.blit(pill_surf, pill_surf.get_rect(center=pill_rect.center))

        left_card = pygame.Rect(22, 230, 206, 235)
        self._draw_panel(surface, left_card, border_color=COLOR_DANGER, alpha=195, radius=12)
        l_title = self.font_h2.render("ESQUIVA", True, COLOR_DANGER)
        surface.blit(l_title, l_title.get_rect(center=(left_card.centerx, left_card.top + 26)))
        for idx, txt in enumerate([
            "Meteoritos",
            "Láseres (aviso 1.5s)",
            "Paredes rojas",
            "Puño en corazón:",
            "¡Muévelo como escudo!",
        ]):
            col = COLOR_HAND_FIST if idx >= 3 else COLOR_TEXT
            t = self.font_small.render(txt, True, col)
            surface.blit(t, t.get_rect(center=(left_card.centerx, left_card.top + 68 + idx * 32)))

        right_card = pygame.Rect(self.width - 228, 230, 206, 235)
        self._draw_panel(surface, right_card, border_color=COLOR_SAFE, alpha=195, radius=12)
        r_title = self.font_h2.render("ATRAPA", True, COLOR_SAFE)
        surface.blit(r_title, r_title.get_rect(center=(right_card.centerx, right_card.top + 26)))
        for idx, (txt, col) in enumerate([
            ("Orbe Dorado (+250)", COLOR_ORB_GOLD),
            ("Orbe Verde (+1 Vida)", COLOR_ORB_HEAL),
            ("Cierra el PUÑO cerca", COLOR_TEXT),
            ("para atraer el orbe", COLOR_TEXT),
            ("hacia tu corazón", COLOR_HAND_OPEN),
        ]):
            t = self.font_small.render(txt, True, col)
            surface.blit(t, t.get_rect(center=(right_card.centerx, right_card.top + 68 + idx * 32)))

        bar_rect = pygame.Rect(0, self.height - 30, self.width, 30)
        bar_bg = pygame.Surface((bar_rect.w, bar_rect.h), pygame.SRCALPHA)
        bar_bg.fill((6, 10, 20, 200))
        surface.blit(bar_bg, bar_rect.topleft)
        mode_note = (
            "DEMO: [Flechas/WASD] Cuerpo · [Mouse] Mano · [Clic/F] Puño · [C] Agachar  |  "
            if self.segmenter.demo_mode
            else ""
        ) + "[F11] Pantalla Completa   ·   [B] Fondo   ·   [M] Sonido   ·   [ESC] Salir"
        note_surf = self.font_small.render(mode_note, True, COLOR_MUTED)
        surface.blit(note_surf, note_surf.get_rect(center=(self.width // 2, self.height - 15)))

    def _draw_pause_overlay(self, surface: pygame.Surface) -> None:
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((4, 8, 18, 175))
        surface.blit(dim, (0, 0))

        card = pygame.Rect(self.width // 2 - 240, self.height // 2 - 135, 480, 270)
        self._draw_panel(surface, card, border_color=COLOR_SILHOUETTE_CYAN, alpha=235)
        t1 = self.font_h1.render("PAUSA", True, COLOR_SILHOUETTE_CYAN)
        t2 = self.font_body.render("[P] o [ESPACIO]  Reanudar partida", True, COLOR_TEXT)
        t3 = self.font_body.render("[R]  Reiniciar partida", True, COLOR_TEXT)
        t4 = self.font_body.render("[V]  Volver al menú principal", True, COLOR_TEXT)
        t5 = self.font_body.render("[F11]  Pantalla completa / Ventana", True, COLOR_TEXT)
        t6 = self.font_body.render("[ESC]  Salir del juego", True, COLOR_MUTED)
        surface.blit(t1, t1.get_rect(center=(card.centerx, card.top + 42)))
        surface.blit(t2, t2.get_rect(center=(card.centerx, card.top + 96)))
        surface.blit(t3, t3.get_rect(center=(card.centerx, card.top + 130)))
        surface.blit(t4, t4.get_rect(center=(card.centerx, card.top + 164)))
        surface.blit(t5, t5.get_rect(center=(card.centerx, card.top + 198)))
        surface.blit(t6, t6.get_rect(center=(card.centerx, card.top + 232)))

    def _draw_end_screen(self, surface: pygame.Surface, victory: bool) -> None:
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((4, 8, 18, 195))
        surface.blit(dim, (0, 0))

        border_col = COLOR_ORB_GOLD if victory else COLOR_DANGER
        card = pygame.Rect(self.width // 2 - 300, self.height // 2 - 205, 600, 410)
        self._draw_panel(surface, card, border_color=border_col, alpha=242, radius=16)

        heading = "¡VICTORIA · 05:00 COMPLETADOS!" if victory else "GAME OVER"
        sub = (
            "¡Has protegido tu corazón hasta el final!"
            if victory
            else "Tu núcleo vital se quedó sin energía."
        )
        h_surf = self.font_h1.render(heading, True, border_col)
        s_surf = self.font_body.render(sub, True, COLOR_TEXT)
        surface.blit(h_surf, h_surf.get_rect(center=(card.centerx, card.top + 44)))
        surface.blit(s_surf, s_surf.get_rect(center=(card.centerx, card.top + 82)))

        phase_reached = self._get_phase_for_elapsed(self.elapsed)["name"]
        rows = [
            ("Tiempo sobrevivido:", f"{format_mm_ss(self.elapsed)} / {format_mm_ss(self.duration)}"),
            ("Puntaje final:", f"{int(self.score):,} pts"),
            ("Récord de sesión:", f"{self.best_score:,} pts"),
            ("Fase alcanzada:", phase_reached),
            ("Orbes recogidos:", str(self.orbs_collected)),
            ("Golpes recibidos:", str(self.hits_taken)),
        ]
        y = card.top + 128
        for label, val in rows:
            l_surf = self.font_body.render(label, True, COLOR_MUTED)
            v_surf = self.font_bold.render(val, True, COLOR_TEXT)
            surface.blit(l_surf, (card.left + 48, y))
            surface.blit(v_surf, v_surf.get_rect(topright=(card.right - 48, y)))
            y += 32

        btn_r = pygame.Rect(card.left + 48, card.bottom - 72, 236, 46)
        btn_m = pygame.Rect(card.right - 284, card.bottom - 72, 236, 46)
        self._restart_btn_rect = btn_r
        self._menu_btn_rect = btn_m

        pygame.draw.rect(surface, (0, 145, 195), btn_r, border_radius=8)
        pygame.draw.rect(surface, COLOR_EDGE_BRIGHT, btn_r, width=2, border_radius=8)
        r_txt = self.font_bold.render("[R] REINICIAR", True, (255, 255, 255))
        surface.blit(r_txt, r_txt.get_rect(center=btn_r.center))

        pygame.draw.rect(surface, (28, 42, 68), btn_m, border_radius=8)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, btn_m, width=2, border_radius=8)
        m_txt = self.font_bold.render("[V] MENÚ  ·  [ESC] SALIR", True, COLOR_TEXT)
        surface.blit(m_txt, m_txt.get_rect(center=btn_m.center))

    def _draw_debug_overlay(self, surface: pygame.Surface, player_mask: np.ndarray) -> None:
        """Draw miniature binary mask preview and Pose/Hands/Core diagnostics."""
        thumb_w, thumb_h = 176, 99
        small_mask = cv2.resize(player_mask, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        rgb_mask = np.stack([np.zeros_like(small_mask), small_mask, small_mask], axis=2)
        thumb_surf = pygame.image.frombuffer(
            np.ascontiguousarray(rgb_mask), (thumb_w, thumb_h), "RGB"
        )
        box = pygame.Rect(18, self.height - thumb_h - 60, thumb_w + 54, thumb_h + 42)
        self._draw_panel(surface, box, border_color=COLOR_SILHOUETTE_CYAN, alpha=215, radius=8)
        surface.blit(thumb_surf, (box.left + 27, box.top + 6))
        fists = sum(1 for h in self.last_hands if h.is_fist)
        info = self.font_small.render(
            f"SRC:{self.last_tracking_source.upper()} H:{len(self.last_hands)} F:{fists} DRAG:{int(self.dragging_core)}",
            True,
            COLOR_EDGE_BRIGHT,
        )
        surface.blit(info, (box.left + 8, box.bottom - 24))

    def _present_to_screen(self) -> None:
        """Scale and center the 16:9 virtual canvas onto the physical display."""
        disp = pygame.display.get_surface()
        if disp is not None and self.screen is disp:
            sw, sh = disp.get_size()
        else:
            sw, sh = self.screen.get_size()

        if (sw, sh) != self._last_screen_size:
            self._recompute_viewport(sw, sh)

        vr = self._viewport_rect
        if vr.w < sw or vr.h < sh:
            self.screen.fill((0, 0, 0))

        if self._scaled_surface is not None:
            pygame.transform.smoothscale(
                self._virtual_canvas, (vr.w, vr.h), self._scaled_surface
            )
            self.screen.blit(self._scaled_surface, vr.topleft)
        else:
            self.screen.blit(self._virtual_canvas, vr.topleft)

    def step(self, dt: float, flip_display: bool = True) -> None:
        """Execute one game frame on the 1280x720 virtual canvas and present."""
        self.handle_events()
        if not self.running:
            return

        if self.state in (self.STATE_MENU, self.STATE_PLAYING):
            self._update_demo_controls(dt)

        tint_mode = self._current_tint_mode()
        tf = self.segmenter.get_tracking_frame(
            show_camera_bg=self.show_camera_bg,
            tint_mode=tint_mode,
        )
        self.last_coverage = tf.coverage
        self.last_core_pos = tf.core_pos
        self.last_tracking_source = tf.tracking_source
        self.last_hands = tf.hands
        self.last_player_px = int(np.count_nonzero(tf.binary_mask))

        if self.state in (self.STATE_MENU, self.STATE_PLAYING):
            self._update_hand_interactions(dt, tf)

        if self.state == self.STATE_PLAYING:
            self._update_playing(dt, self.active_core_pos)

        frame_surf = pygame.image.frombuffer(
            tf.stylized_rgb, (self.width, self.height), "RGB"
        )
        shaking = self.state == self.STATE_PLAYING and self.screen_shake_timer > 0.0
        canvas = self._shake_surface if shaking else self._virtual_canvas
        canvas.blit(frame_surf, (0, 0))
        canvas.blit(self._grid_surf, (0, 0))

        if self.state in (self.STATE_MENU, self.STATE_PLAYING, self.STATE_PAUSED):
            self._draw_limb_glow(canvas, tf.limb_segments, tf.limb_joints)

        if self.state in (
            self.STATE_PLAYING,
            self.STATE_PAUSED,
            self.STATE_GAME_OVER,
            self.STATE_VICTORY,
        ):
            for obs in self.obstacles:
                obs.draw(canvas)
            self._draw_tractor_beams(canvas, tf.hands)
            for p in self.particles:
                p.draw(canvas)
            for ft in self.floating_texts:
                ft.draw(canvas)

        if self.state in (self.STATE_MENU, self.STATE_PLAYING, self.STATE_PAUSED):
            self._draw_core_tether(canvas, tf.core_pos, self.active_core_pos)
            self._draw_vital_core(
                canvas,
                self.active_core_pos,
                tf.tracking_source,
                list(self._active_core_trail),
            )
            self._draw_hands(canvas, tf.hands)

        if self.state == self.STATE_MENU:
            self._draw_menu(canvas, tf)
        elif self.state == self.STATE_PLAYING:
            self._draw_hud(canvas)
        elif self.state == self.STATE_PAUSED:
            self._draw_hud(canvas)
            self._draw_pause_overlay(canvas)
        elif self.state == self.STATE_GAME_OVER:
            self._draw_hud(canvas)
            self._draw_end_screen(canvas, victory=False)
        elif self.state == self.STATE_VICTORY:
            self._draw_hud(canvas)
            self._draw_end_screen(canvas, victory=True)

        if self.debug_mask:
            self._draw_debug_overlay(canvas, tf.binary_mask)

        if self.hit_flash_timer > 0.0 and self.state == self.STATE_PLAYING:
            flash = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            alpha = int(110 * (self.hit_flash_timer / 0.30))
            flash.fill((255, 30, 75, alpha))
            canvas.blit(flash, (0, 0))

        if shaking:
            self.shake_offset = (self.rng.randint(-8, 8), self.rng.randint(-6, 6))
            self._virtual_canvas.fill(COLOR_BG)
            self._virtual_canvas.blit(self._shake_surface, self.shake_offset)
        else:
            self.shake_offset = (0, 0)

        self._present_to_screen()

        if flip_display:
            pygame.display.flip()

    def run(self) -> None:
        """Run the main real-time game loop until quit."""
        while self.running:
            dt = min(0.05, self.clock.tick(TARGET_FPS) / 1000.0)
            fps = self.clock.get_fps()
            if fps > 0:
                self.current_fps = fps
            self.step(dt, flip_display=True)
