"""Procedural retro arcade sound effects synthesized in memory."""

from __future__ import annotations

import numpy as np
import pygame


class SoundFX:
    """Synthesizes and plays lightweight arcade SFX without external audio files."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self.muted = False
        self.available = False
        self._channels = 2
        self._sounds: dict[str, pygame.mixer.Sound] = {}

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=sample_rate, size=-16, channels=2, buffer=512
                )
            init_info = pygame.mixer.get_init()
            if init_info is not None:
                self.sample_rate = int(init_info[0])
                self._channels = int(init_info[2])
                self.available = True
                self._build_bank()
        except Exception:
            self.available = False
            self._sounds = {}

    def _to_sound(self, mono: np.ndarray, volume: float = 0.35) -> pygame.mixer.Sound:
        """Convert float32 mono waveform [-1, 1] to a pygame Sound matching mixer channels."""
        clipped = np.clip(mono * volume, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        if self._channels >= 2:
            stereo = np.column_stack([pcm] * self._channels)
            return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))
        return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))

    def _tone(
        self,
        freq_start: float,
        freq_end: float,
        duration: float,
        wave: str = "sine",
    ) -> np.ndarray:
        """Generate a swept mono waveform with smooth attack/decay envelope."""
        n = max(1, int(self.sample_rate * duration))
        t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
        freqs = np.linspace(freq_start, freq_end, n, dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(freqs) / self.sample_rate

        if wave == "square":
            sig = np.sign(np.sin(phase)).astype(np.float32) * 0.65
        elif wave == "saw":
            sig = (2.0 * ((phase / (2.0 * np.pi)) % 1.0) - 1.0).astype(np.float32) * 0.75
        else:
            sig = np.sin(phase).astype(np.float32)

        attack = min(n, max(1, int(self.sample_rate * 0.008)))
        env = np.linspace(1.0, 0.0, n, dtype=np.float32) ** 1.4
        env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
        return sig * env

    def _sequence(
        self, notes: list[tuple[float, float]], wave: str = "sine"
    ) -> np.ndarray:
        """Concatenate short notes [(freq, dur), ...]."""
        parts = [self._tone(freq, freq, dur, wave=wave) for freq, dur in notes]
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

    def _build_bank(self) -> None:
        rng = np.random.default_rng(42)

        # Impact / life lost
        n_hit = int(self.sample_rate * 0.22)
        noise = rng.uniform(-0.45, 0.45, n_hit).astype(np.float32)
        noise *= np.linspace(1.0, 0.0, n_hit, dtype=np.float32) ** 2.0
        hit_wave = self._tone(190.0, 52.0, 0.22, wave="saw") + noise * 0.6
        self._sounds["hit"] = self._to_sound(hit_wave, volume=0.45)

        # Laser telegraph & beam fire
        self._sounds["laser_warn"] = self._to_sound(
            self._tone(880.0, 960.0, 0.07, wave="sine"), volume=0.20
        )
        self._sounds["laser_fire"] = self._to_sound(
            self._tone(540.0, 170.0, 0.20, wave="saw"), volume=0.32
        )

        # Safe zone wall warning & activation
        self._sounds["wall_warn"] = self._to_sound(
            self._sequence([(440.0, 0.07), (587.0, 0.09)], wave="square"), volume=0.22
        )
        self._sounds["wall_impact"] = self._to_sound(
            self._tone(140.0, 48.0, 0.24, wave="square"), volume=0.34
        )

        # Orbs
        self._sounds["orb_gold"] = self._to_sound(
            self._sequence([(660.0, 0.06), (880.0, 0.07), (1174.0, 0.10)], wave="sine"),
            volume=0.32,
        )
        self._sounds["orb_heal"] = self._to_sound(
            self._sequence([(523.25, 0.07), (659.25, 0.07), (783.99, 0.13)], wave="sine"),
            volume=0.34,
        )

        # Fist magnet / core grab & release
        self._sounds["grab"] = self._to_sound(
            self._tone(360.0, 720.0, 0.08, wave="sine"), volume=0.22
        )
        self._sounds["release"] = self._to_sound(
            self._tone(540.0, 280.0, 0.08, wave="sine"), volume=0.18
        )

        # Phase change, victory, game over
        self._sounds["phase_up"] = self._to_sound(
            self._sequence([(440.0, 0.08), (554.37, 0.08), (659.25, 0.14)], wave="sine"),
            volume=0.32,
        )
        self._sounds["victory"] = self._to_sound(
            self._sequence(
                [(523.25, 0.09), (659.25, 0.09), (783.99, 0.09), (1046.5, 0.25)],
                wave="sine",
            ),
            volume=0.38,
        )
        self._sounds["game_over"] = self._to_sound(
            self._sequence(
                [(392.0, 0.12), (349.23, 0.12), (311.13, 0.14), (246.94, 0.28)],
                wave="saw",
            ),
            volume=0.30,
        )

    def play(self, name: str) -> None:
        """Play a sound effect by key if mixer is available and not muted."""
        if self.muted or not self.available:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            try:
                snd.play()
            except Exception:
                pass

    def toggle_mute(self) -> bool:
        """Toggle mute state and return the new muted status."""
        self.muted = not self.muted
        return self.muted
