"""Game obstacles, safe-zone walls, collectible orbs, and visual FX."""

from __future__ import annotations

import math
import random
from typing import Optional

import cv2
import numpy as np
import pygame

from config import (
    COLOR_DANGER,
    COLOR_LASER,
    COLOR_LASER_WARN,
    COLOR_ORB_GOLD,
    COLOR_ORB_HEAL,
    COLOR_SAFE,
    COLOR_TEXT,
    LASER_WARNING_DURATION,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

_FONT_CACHE: dict[tuple[int, bool], pygame.font.Font] = {}


def _get_font(size: int, bold: bool = True) -> pygame.font.Font:
    key = (size, bold)
    font = _FONT_CACHE.get(key)
    if font is None:
        font = pygame.font.SysFont("dejavusans,liberation sans,arial", size, bold=bold)
        _FONT_CACHE[key] = font
    return font


class Obstacle:
    """Base interface for all mask-collidable entities."""

    is_hazard: bool = True

    def __init__(self) -> None:
        self.alive: bool = True
        self.just_activated: bool = False

    def is_dangerous_now(self) -> bool:
        return self.is_hazard and self.alive

    def update(self, dt: float) -> None:
        raise NotImplementedError

    def draw(self, surface: pygame.Surface) -> None:
        raise NotImplementedError

    def compute_overlap(
        self, player_mask: np.ndarray, total_player_px: Optional[int] = None
    ) -> int:
        raise NotImplementedError

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        raise NotImplementedError


class Meteor(Obstacle):
    """Falling neon meteorite/crystal obstacle."""

    is_hazard = True

    def __init__(
        self,
        x: float,
        y: float = -45.0,
        radius: int = 32,
        vx: float = 0.0,
        vy: float = 240.0,
        rng: Optional[random.Random] = None,
        screen_size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT),
    ) -> None:
        super().__init__()
        self.x = float(x)
        self.y = float(y)
        self.radius = int(radius)
        self.vx = float(vx)
        self.vy = float(vy)
        self._screen_w, self._screen_h = screen_size
        self.angle = 0.0
        self.rot_speed = (rng.uniform(-2.6, 2.6) if rng else 1.5)
        self.trail: list[tuple[float, float]] = []

        # Build irregular polygon radii for distinctive rocky silhouette
        r_gen = rng if rng is not None else random
        num_verts = 9
        self._vertex_offsets: list[tuple[float, float]] = []
        for i in range(num_verts):
            ang = (2.0 * math.pi * i) / num_verts
            rad = self.radius * r_gen.uniform(0.78, 1.08)
            self._vertex_offsets.append((ang, rad))

        # Pre-render glow sprite once per meteor for fast blitting
        pad = 14
        size = (self.radius + pad) * 2
        self._glow_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size // 2, size // 2)
        pygame.draw.circle(self._glow_surf, (255, 65, 110, 42), center, self.radius + pad)
        pygame.draw.circle(self._glow_surf, (255, 110, 50, 75), center, self.radius + 5)

    def _world_points(self) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for base_ang, rad in self._vertex_offsets:
            ang = base_ang + self.angle
            pts.append((self.x + math.cos(ang) * rad, self.y + math.sin(ang) * rad))
        return pts

    def update(self, dt: float) -> None:
        self.trail.append((self.x, self.y))
        if len(self.trail) > 6:
            self.trail.pop(0)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.rot_speed * dt

        if self.x < self.radius + 8:
            self.x = float(self.radius + 8)
            self.vx = abs(self.vx)
        elif self.x > self._screen_w - self.radius - 8:
            self.x = float(self._screen_w - self.radius - 8)
            self.vx = -abs(self.vx)

        if self.y - self.radius > self._screen_h + 30:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        # Short motion trail
        for idx, (tx, ty) in enumerate(self.trail):
            frac = (idx + 1) / (len(self.trail) + 1)
            r = max(4, int(self.radius * 0.55 * frac))
            pygame.draw.circle(surface, (165, 35, 70), (int(tx), int(ty)), r)

        # Outer aura
        gw, gh = self._glow_surf.get_size()
        surface.blit(self._glow_surf, (int(self.x - gw // 2), int(self.y - gh // 2)))

        # Asteroid body + glowing neon rim
        pts = self._world_points()
        pygame.draw.polygon(surface, (145, 24, 58), pts)
        pygame.draw.polygon(surface, COLOR_DANGER, pts, width=3)

        # Hot inner core
        inner_pts = [
            (
                self.x + (px - self.x) * 0.45,
                self.y + (py - self.y) * 0.45,
            )
            for px, py in pts
        ]
        pygame.draw.polygon(surface, (255, 165, 90), inner_pts)

    def compute_overlap(
        self, player_mask: np.ndarray, total_player_px: Optional[int] = None
    ) -> int:
        if not self.is_dangerous_now():
            return 0
        h, w = player_mask.shape
        margin = int(self.radius * 1.12) + 2
        x1 = max(0, int(self.x) - margin)
        y1 = max(0, int(self.y) - margin)
        x2 = min(w, int(self.x) + margin + 1)
        y2 = min(h, int(self.y) + margin + 1)
        if x1 >= x2 or y1 >= y2:
            return 0

        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_pts = np.array(
            [[int(px - x1), int(py - y1)] for px, py in self._world_points()],
            dtype=np.int32,
        )
        cv2.fillPoly(local_mask, [shifted_pts], 255)
        return int(np.count_nonzero(player_mask[y1:y2, x1:x2] & local_mask))

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        if not self.is_dangerous_now():
            return False
        return math.hypot(cx - self.x, cy - self.y) <= (self.radius * 0.90 + radius)


class Laser(Obstacle):
    """Telegraphed horizontal or vertical laser beam."""

    is_hazard = True

    def __init__(
        self,
        orientation: str,
        pos: int,
        thickness: int = 56,
        warning_duration: float = LASER_WARNING_DURATION,
        active_duration: float = 0.85,
        fade_duration: float = 0.22,
        screen_size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT),
    ) -> None:
        super().__init__()
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"Unsupported laser orientation: {orientation}")
        self.orientation = orientation
        self.pos = int(pos)
        self.thickness = int(thickness)
        self.warning_duration = float(warning_duration)
        self.active_duration = float(active_duration)
        self.fade_duration = float(fade_duration)
        self._screen_w, self._screen_h = screen_size

        self.state = "WARNING"
        self.timer = self.warning_duration
        self._anim_time = 0.0

    def is_dangerous_now(self) -> bool:
        return self.alive and self.state == "ACTIVE"

    def update(self, dt: float) -> None:
        self.just_activated = False
        self._anim_time += dt
        self.timer -= dt

        if self.state == "WARNING" and self.timer <= 0.0:
            self.state = "ACTIVE"
            self.timer = self.active_duration
            self.just_activated = True
        elif self.state == "ACTIVE" and self.timer <= 0.0:
            self.state = "FADE"
            self.timer = self.fade_duration
        elif self.state == "FADE" and self.timer <= 0.0:
            self.alive = False

    def _lane_rect(self, thickness: int) -> pygame.Rect:
        half = thickness // 2
        if self.orientation == "horizontal":
            return pygame.Rect(0, self.pos - half, self._screen_w, thickness)
        return pygame.Rect(self.pos - half, 0, thickness, self._screen_h)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return

        if self.state == "WARNING":
            progress = 1.0 - max(0.0, self.timer / max(0.01, self.warning_duration))
            blink_freq = 5.0 + progress * 16.0
            blink_on = math.sin(self._anim_time * blink_freq * math.pi) > -0.15

            lane = self._lane_rect(self.thickness)
            warn_surf = pygame.Surface((lane.w, lane.h), pygame.SRCALPHA)
            alpha = int(38 + 45 * progress) if blink_on else 18
            warn_surf.fill((255, 65, 55, alpha))
            surface.blit(warn_surf, lane.topleft)

            # Blinking dashed centerline and borders
            line_col = COLOR_LASER_WARN if blink_on else (160, 70, 50)
            if self.orientation == "horizontal":
                pygame.draw.line(surface, line_col, (0, lane.top), (self._screen_w, lane.top), 2)
                pygame.draw.line(surface, line_col, (0, lane.bottom), (self._screen_w, lane.bottom), 2)
                for x in range(0, self._screen_w, 36):
                    pygame.draw.line(
                        surface,
                        (255, 225, 130) if blink_on else (130, 80, 40),
                        (x, self.pos),
                        (x + 20, self.pos),
                        3,
                    )
                # Side emitters
                for ex in (10, self._screen_w - 22):
                    pygame.draw.rect(surface, (40, 20, 30), (ex, lane.top, 12, lane.h), border_radius=4)
                    fill_h = int(lane.h * progress)
                    pygame.draw.rect(
                        surface,
                        COLOR_LASER_WARN,
                        (ex + 2, lane.bottom - fill_h, 8, fill_h),
                        border_radius=3,
                    )
                badge_pos = (self._screen_w // 2, max(106, lane.top - 16))
            else:
                pygame.draw.line(surface, line_col, (lane.left, 0), (lane.left, self._screen_h), 2)
                pygame.draw.line(surface, line_col, (lane.right, 0), (lane.right, self._screen_h), 2)
                for y in range(0, self._screen_h, 36):
                    pygame.draw.line(
                        surface,
                        (255, 225, 130) if blink_on else (130, 80, 40),
                        (self.pos, y),
                        (self.pos, y + 20),
                        3,
                    )
                for ey in (10, self._screen_h - 22):
                    pygame.draw.rect(surface, (40, 20, 30), (lane.left, ey, lane.w, 12), border_radius=4)
                    fill_w = int(lane.w * progress)
                    pygame.draw.rect(
                        surface,
                        COLOR_LASER_WARN,
                        (lane.left, ey + 2, fill_w, 8),
                        border_radius=3,
                    )
                badge_pos = (min(self._screen_w - 65, max(65, self.pos)), 106)

            txt = _get_font(16, bold=True).render(f"LÁSER {max(0.0, self.timer):.1f}s", True, (255, 235, 160))
            txt_rect = txt.get_rect(center=badge_pos)
            bg_rect = txt_rect.inflate(14, 6)
            pygame.draw.rect(surface, (28, 12, 22), bg_rect, border_radius=6)
            pygame.draw.rect(surface, COLOR_LASER_WARN, bg_rect, width=1, border_radius=6)
            surface.blit(txt, txt_rect)

        elif self.state in ("ACTIVE", "FADE"):
            scale = 1.0 if self.state == "ACTIVE" else max(0.0, self.timer / max(0.01, self.fade_duration))
            pulse = 1.0 + 0.08 * math.sin(self._anim_time * 45.0) if self.state == "ACTIVE" else scale
            outer_t = max(6, int(self.thickness * 1.25 * pulse))
            mid_t = max(4, int(self.thickness * pulse))
            core_t = max(2, int(self.thickness * 0.36 * scale))

            outer_lane = self._lane_rect(outer_t)
            glow = pygame.Surface((outer_lane.w, outer_lane.h), pygame.SRCALPHA)
            glow.fill((255, 35, 65, int(115 * scale)))
            surface.blit(glow, outer_lane.topleft)

            mid_lane = self._lane_rect(mid_t)
            pygame.draw.rect(surface, COLOR_LASER, mid_lane)

            core_lane = self._lane_rect(core_t)
            pygame.draw.rect(surface, (255, 245, 245), core_lane)

    def compute_overlap(
        self, player_mask: np.ndarray, total_player_px: Optional[int] = None
    ) -> int:
        if not self.is_dangerous_now():
            return 0
        h, w = player_mask.shape
        half = self.thickness // 2
        if self.orientation == "horizontal":
            y1 = max(0, self.pos - half)
            y2 = min(h, self.pos + half)
            if y1 >= y2:
                return 0
            return int(np.count_nonzero(player_mask[y1:y2, :]))
        x1 = max(0, self.pos - half)
        x2 = min(w, self.pos + half)
        if x1 >= x2:
            return 0
        return int(np.count_nonzero(player_mask[:, x1:x2]))

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        if not self.is_dangerous_now():
            return False
        half = self.thickness * 0.5
        if self.orientation == "horizontal":
            return abs(cy - self.pos) <= (half + radius)
        return abs(cx - self.pos) <= (half + radius)


class SafeZoneWall(Obstacle):
    """'Hole in the Wall' barrier covering the screen except a green safe zone."""

    is_hazard = True

    PATTERN_SPECS = {
        "crouch": {
            "label": "¡AGÁCHATE!",
            "hint": "ZONA SEGURA ABAJO",
            "rect": (0.00, 0.55, 1.00, 0.45),
        },
        "left": {
            "label": "¡A LA IZQUIERDA!",
            "hint": "MUÉVETE A LA IZQUIERDA",
            "rect": (0.00, 0.00, 0.40, 1.00),
        },
        "right": {
            "label": "¡A LA DERECHA!",
            "hint": "MUÉVETE A LA DERECHA",
            "rect": (0.60, 0.00, 0.40, 1.00),
        },
        "center": {
            "label": "¡AL CENTRO!",
            "hint": "MANTENTE EN EL PASILLO CENTRAL",
            "rect": (0.31, 0.00, 0.38, 1.00),
        },
        "bottom_left": {
            "label": "¡AGÁCHATE IZQ.!",
            "hint": "ESQUINA INFERIOR IZQUIERDA",
            "rect": (0.00, 0.48, 0.48, 0.52),
        },
        "bottom_right": {
            "label": "¡AGÁCHATE DER.!",
            "hint": "ESQUINA INFERIOR DERECHA",
            "rect": (0.52, 0.48, 0.48, 0.52),
        },
    }

    def __init__(
        self,
        pattern: str = "crouch",
        warning_duration: float = 2.6,
        active_duration: float = 0.70,
        fade_duration: float = 0.30,
        screen_size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT),
    ) -> None:
        super().__init__()
        if pattern not in self.PATTERN_SPECS:
            raise ValueError(f"Unknown safe zone pattern: {pattern}")
        self.pattern = pattern
        spec = self.PATTERN_SPECS[pattern]
        self.label: str = spec["label"]  # type: ignore[assignment]
        self.hint: str = spec["hint"]  # type: ignore[assignment]
        self._screen_w, self._screen_h = screen_size

        rx, ry, rw, rh = spec["rect"]  # type: ignore[misc]
        sx = int(round(rx * self._screen_w))
        sy = int(round(ry * self._screen_h))
        sw = int(round(rw * self._screen_w))
        sh = int(round(rh * self._screen_h))
        sx = max(0, min(self._screen_w - 1, sx))
        sy = max(0, min(self._screen_h - 1, sy))
        sw = min(self._screen_w - sx, sw)
        sh = min(self._screen_h - sy, sh)
        self.safe_rect = pygame.Rect(sx, sy, sw, sh)
        self._danger_rects = self._compute_complement_rects(
            self.safe_rect, self._screen_w, self._screen_h
        )
        self._overlay = pygame.Surface((self._screen_w, self._screen_h), pygame.SRCALPHA)

        self.warning_duration = float(warning_duration)
        self.active_duration = float(active_duration)
        self.fade_duration = float(fade_duration)
        self.state = "WARNING"
        self.timer = self.warning_duration
        self._anim_time = 0.0
        self.hit_player = False
        self.dodged = False

    @staticmethod
    def _compute_complement_rects(
        safe: pygame.Rect, screen_w: int, screen_h: int
    ) -> list[pygame.Rect]:
        """Partition screen minus `safe` into non-overlapping danger rectangles."""
        rects: list[pygame.Rect] = []
        if safe.top > 0:
            rects.append(pygame.Rect(0, 0, screen_w, safe.top))
        if safe.bottom < screen_h:
            rects.append(pygame.Rect(0, safe.bottom, screen_w, screen_h - safe.bottom))
        if safe.left > 0:
            rects.append(pygame.Rect(0, safe.top, safe.left, safe.height))
        if safe.right < screen_w:
            rects.append(pygame.Rect(safe.right, safe.top, screen_w - safe.right, safe.height))
        return [r for r in rects if r.w > 0 and r.h > 0]

    def is_dangerous_now(self) -> bool:
        return self.alive and self.state == "ACTIVE"

    def update(self, dt: float) -> None:
        self.just_activated = False
        self._anim_time += dt
        self.timer -= dt

        if self.state == "WARNING" and self.timer <= 0.0:
            self.state = "ACTIVE"
            self.timer = self.active_duration
            self.just_activated = True
        elif self.state == "ACTIVE" and self.timer <= 0.0:
            self.state = "FADE"
            self.timer = self.fade_duration
            if not self.hit_player:
                self.dodged = True
        elif self.state == "FADE" and self.timer <= 0.0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return

        if self.state == "WARNING":
            progress = 1.0 - max(0.0, self.timer / max(0.01, self.warning_duration))
            pulse = 0.5 + 0.5 * math.sin(self._anim_time * (6.0 + progress * 10.0))
            danger_alpha = int(42 + 78 * progress + 18 * pulse)
            safe_alpha = int(24 + 28 * pulse)
        elif self.state == "ACTIVE":
            danger_alpha = 185
            safe_alpha = 35
        else:
            fade_frac = max(0.0, self.timer / max(0.01, self.fade_duration))
            danger_alpha = int(120 * fade_frac)
            safe_alpha = int(30 * fade_frac)

        overlay = self._overlay
        overlay.fill((0, 0, 0, 0))
        for rect in self._danger_rects:
            overlay.fill((255, 36, 82, danger_alpha), rect)
            # Subtle hazard hatch lines on danger regions
            if self.state in ("WARNING", "ACTIVE"):
                step = 42
                for offset in range(-rect.h, rect.w, step):
                    x_start = rect.left + offset
                    y_start = rect.top
                    x_end = x_start + rect.h
                    y_end = rect.bottom
                    pygame.draw.line(
                        overlay,
                        (255, 110, 140, min(200, danger_alpha + 25)),
                        (x_start, y_start),
                        (x_end, y_end),
                        2,
                    )

        # Safe zone translucent fill
        overlay.fill((0, 255, 136, safe_alpha), self.safe_rect)
        surface.blit(overlay, (0, 0))

        # Safe zone glowing border
        border_w = 5 if self.state == "WARNING" else 3
        pygame.draw.rect(surface, COLOR_SAFE, self.safe_rect, width=border_w)
        inner = self.safe_rect.inflate(-10, -10)
        if inner.w > 20 and inner.h > 20:
            pygame.draw.rect(surface, (160, 255, 210), inner, width=1)

        # Instruction banner and countdown bar inside the safe zone
        cx, cy = self.safe_rect.center
        label_surf = _get_font(28, bold=True).render(self.label, True, COLOR_SAFE)
        hint_surf = _get_font(15, bold=False).render(self.hint, True, COLOR_TEXT)

        if self.state == "WARNING":
            countdown_str = f"CIERRE EN {max(0.0, self.timer):.1f}s"
            count_col = (255, 235, 130) if self.timer > 0.7 else (255, 110, 110)
        elif self.state == "ACTIVE":
            countdown_str = "¡PARED ACTIVA!"
            count_col = (255, 120, 145)
        else:
            countdown_str = "¡DESPEJADO!"
            count_col = COLOR_SAFE
        count_surf = _get_font(20, bold=True).render(countdown_str, True, count_col)

        box_w = max(label_surf.get_width(), hint_surf.get_width(), 240) + 36
        box_h = 104
        box_rect = pygame.Rect(0, 0, box_w, box_h)
        box_rect.center = (
            max(box_w // 2 + 12, min(self._screen_w - box_w // 2 - 12, cx)),
            max(box_h // 2 + 104, min(self._screen_h - box_h // 2 - 18, cy)),
        )

        card = pygame.Surface((box_rect.w, box_rect.h), pygame.SRCALPHA)
        card.fill((8, 16, 28, 205))
        surface.blit(card, box_rect.topleft)
        pygame.draw.rect(surface, COLOR_SAFE, box_rect, width=2, border_radius=10)

        surface.blit(label_surf, label_surf.get_rect(center=(box_rect.centerx, box_rect.top + 26)))
        surface.blit(hint_surf, hint_surf.get_rect(center=(box_rect.centerx, box_rect.top + 52)))
        surface.blit(count_surf, count_surf.get_rect(center=(box_rect.centerx, box_rect.top + 76)))

        if self.state == "WARNING":
            bar_w = box_rect.w - 28
            frac = max(0.0, self.timer / max(0.01, self.warning_duration))
            bar_bg = pygame.Rect(box_rect.left + 14, box_rect.bottom - 12, bar_w, 6)
            bar_fg = pygame.Rect(box_rect.left + 14, box_rect.bottom - 12, int(bar_w * frac), 6)
            pygame.draw.rect(surface, (35, 55, 70), bar_bg, border_radius=3)
            pygame.draw.rect(surface, COLOR_SAFE, bar_fg, border_radius=3)

    def compute_overlap(
        self, player_mask: np.ndarray, total_player_px: Optional[int] = None
    ) -> int:
        if not self.is_dangerous_now():
            return 0
        total = (
            total_player_px
            if total_player_px is not None
            else int(np.count_nonzero(player_mask))
        )
        if total == 0:
            return 0
        sx, sy, sw, sh = self.safe_rect
        safe_px = int(np.count_nonzero(player_mask[sy : sy + sh, sx : sx + sw]))
        return max(0, total - safe_px)

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        if not self.is_dangerous_now():
            return False
        sr = self.safe_rect
        inside_safe = (
            (sr.left + radius - 2.0) <= cx <= (sr.right - radius + 2.0)
            and (sr.top + radius - 2.0) <= cy <= (sr.bottom - radius + 2.0)
        )
        return not inside_safe


class Orb(Obstacle):
    """Collectible gold score orb or emerald healing orb."""

    is_hazard = False

    def __init__(
        self,
        x: float,
        y: float,
        kind: str = "gold",
        radius: int = 24,
        lifetime: float = 8.0,
    ) -> None:
        super().__init__()
        if kind not in ("gold", "heal"):
            raise ValueError(f"Unknown orb kind: {kind}")
        self.base_x = float(x)
        self.base_y = float(y)
        self.x = float(x)
        self.y = float(y)
        self.kind = kind
        self.radius = int(radius)
        self.lifetime = float(lifetime)
        self.timer = self.lifetime
        self._phase = random.uniform(0.0, math.tau)
        self.grabbed_by: Optional[str] = None

        color = COLOR_ORB_GOLD if kind == "gold" else COLOR_ORB_HEAL
        pad = 16
        size = (self.radius + pad) * 2
        self._sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size // 2, size // 2)
        pygame.draw.circle(self._sprite, (*color, 45), center, self.radius + pad)
        pygame.draw.circle(self._sprite, (*color, 95), center, self.radius + 6)
        pygame.draw.circle(self._sprite, color, center, self.radius)
        pygame.draw.circle(self._sprite, (255, 255, 255), center, self.radius, width=2)

    def pull_toward(self, target_x: float, target_y: float, dt: float) -> None:
        """Magnetically pull the orb toward an active closed fist."""
        blend = min(1.0, max(0.0, dt * 16.0))
        self.x += (target_x - self.x) * blend
        self.y += (target_y - self.y) * blend
        self.base_x = self.x
        self.base_y = self.y

    def release(self) -> None:
        """Release orb from fist grip and resume floating from current coordinates."""
        self.grabbed_by = None
        self.base_x = self.x
        self.base_y = self.y
        self._phase = 0.0

    def update(self, dt: float) -> None:
        if self.grabbed_by is not None:
            # Keep grabbed orb alive while attracted by the player's fist
            self.timer = max(self.timer, 1.5)
            return
        self._phase += dt * 3.4
        self.timer -= dt
        self.x = self.base_x + math.sin(self._phase * 0.8) * 14.0
        self.y = self.base_y + math.cos(self._phase * 1.3) * 10.0
        if self.timer <= 0.0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return
        # Blink during last 2 seconds before despawning
        if self.timer < 2.0 and int(self.timer * 10) % 2 == 0:
            return

        sw, sh = self._sprite.get_size()
        surface.blit(self._sprite, (int(self.x - sw // 2), int(self.y - sh // 2)))

        if self.kind == "heal":
            cx, cy = int(self.x), int(self.y)
            pygame.draw.rect(surface, (255, 255, 255), (cx - 3, cy - 10, 6, 20), border_radius=2)
            pygame.draw.rect(surface, (255, 255, 255), (cx - 10, cy - 3, 20, 6), border_radius=2)
        else:
            cx, cy = int(self.x), int(self.y)
            pts = [
                (cx, cy - 11),
                (cx + 9, cy),
                (cx, cy + 11),
                (cx - 9, cy),
            ]
            pygame.draw.polygon(surface, (255, 252, 220), pts)

    def compute_overlap(
        self, player_mask: np.ndarray, total_player_px: Optional[int] = None
    ) -> int:
        if not self.alive:
            return 0
        h, w = player_mask.shape
        r = self.radius
        x1 = max(0, int(self.x) - r)
        y1 = max(0, int(self.y) - r)
        x2 = min(w, int(self.x) + r + 1)
        y2 = min(h, int(self.y) + r + 1)
        if x1 >= x2 or y1 >= y2:
            return 0

        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(local_mask, (int(self.x) - x1, int(self.y) - y1), r, 255, -1)
        return int(np.count_nonzero(player_mask[y1:y2, x1:x2] & local_mask))

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        if not self.alive:
            return False
        return math.hypot(cx - self.x, cy - self.y) <= (self.radius + radius)


class Particle:
    """Short-lived neon spark particle."""

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        color: tuple[int, int, int],
        life: float = 0.45,
        radius: int = 4,
    ) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life
        self.radius = radius
        self.alive = True

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.94
        self.vy *= 0.94
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        frac = max(0.0, self.life / max(0.01, self.max_life))
        r = max(1, int(self.radius * frac))
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), r)


class FloatingText:
    """Rising status feedback text ('+250', '+1 VIDA', '-1 VIDA', etc.)."""

    def __init__(
        self,
        text: str,
        x: float,
        y: float,
        color: tuple[int, int, int],
        life: float = 1.0,
        size: int = 24,
    ) -> None:
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.alive = True

    def update(self, dt: float) -> None:
        self.y -= 48.0 * dt
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        alpha = int(255 * max(0.0, self.life / max(0.01, self.max_life)))
        surf = _get_font(self.size, bold=True).render(self.text, True, self.color)
        surf.set_alpha(alpha)
        surface.blit(surf, surf.get_rect(center=(int(self.x), int(self.y))))
