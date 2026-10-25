"""Optional sound playback through pygame.mixer, driven by effect hooks and UI events.

pygame is the only optional dependency. Without it (or without an audio
device) the manager logs one info line and every method becomes a no-op, so
the rest of the app behaves identically. Effects never import this module:
they emit named hooks, and ``on_hook`` maps names to sounds.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
import os
import time
from typing import Any
from .config import Config
from .sounds import ensure_sounds

LOGGER = logging.getLogger(__name__)
MAX_INSTANCES = 3  # Copies of one sound playing at once; more would just be noise.


@dataclass(frozen=True)
class Cue:
    """How one hook sounds."""

    sound: str
    volume: float = 1.0
    cooldown_s: float = 0.08  # Repeats sooner than this are dropped (block cascades, slashes).
    loop: bool = False  # Start a loop keyed by the payload's "hand" ...
    stop: bool = False  # ... or stop it.


HOOK_CUES: Mapping[str, Cue] = {
    # Jutsu.
    "fireball_summon": Cue("fire_crackle", 0.75, 0.3),
    "fireball_throw": Cue("whoosh", 0.9, 0.12),
    "fireball_dissipate": Cue("whoosh", 0.35),
    "great_fireball": Cue("fire_crackle", 1.0, 0.5),
    "jutsu_seal": Cue("seal", 0.8, 0.5),
    "chidori_charge": Cue("chidori_charge", 0.7, 0.5),
    "lightning_start": Cue("lightning", 0.55, 0.0, loop=True),
    "lightning_end": Cue("lightning", stop=True),
    "rasenshuriken_form": Cue("wind", 0.8, 0.4),
    "rasenshuriken_throw": Cue("whoosh", 1.0, 0.12),
    "rasenshuriken_dissipate": Cue("whoosh", 0.4),
    "susanoo_summon": Cue("susanoo", 1.0, 1.0),
    "susanoo_dismiss": Cue("dust", 0.6, 1.0),
    # Jujutsu.
    "blue_start": Cue("orb_blue", 0.8, 0.3),
    "red_start": Cue("orb_red", 0.8, 0.3),
    "hollow_purple_form": Cue("purple_charge", 1.0, 0.5),
    "hollow_purple_throw": Cue("purple_fire", 1.0, 0.3),
    "hollow_purple_dissipate": Cue("whoosh", 0.4),
    "domain_expansion": Cue("domain", 1.0, 1.0),
    "domain_end": Cue("reform", 0.6, 0.5),
    "domain_slash": Cue("slash", 0.5, 0.05),
    # The snap.
    "snap": Cue("snap", 1.0, 0.3),
    "snap_dust": Cue("dust", 0.8, 1.0),
    "snap_return": Cue("reform", 0.8, 0.5),
    # Blocks.
    "block_place": Cue("block_place", 0.8, 0.05),
    "block_land": Cue("block_place", 0.45, 0.06),
    "block_break": Cue("block_break", 0.9, 0.03),
    "block_grab": Cue("ui_hover", 0.6),
    "block_select": Cue("ui_click", 0.6),
    "blocks_clear": Cue("block_break", 1.0, 0.5),
    "beam_connect": Cue("beam", 0.7, 0.2),
    "beam_place": Cue("block_place", 0.7, 0.1),
    "beam_fizzle": Cue("whoosh", 0.3),
    "tnt_ignite": Cue("fuse", 0.7, 0.1),
    # App and UI events (emitted on the same hook bus by the app).
    "mode_change": Cue("mode_change", 0.6, 0.2),
    "ui_hover": Cue("ui_hover", 0.5, 0.03),
    "ui_click": Cue("ui_click", 0.6, 0.03),
    "menu_open": Cue("mode_change", 0.5, 0.3),
    "menu_close": Cue("ui_click", 0.5, 0.1),
    "rec_start": Cue("rec", 0.7, 0.2),
    "rec_stop": Cue("rec", 0.7, 0.2),
    "canvas_clear": Cue("whoosh", 0.5, 0.3),
    "drawing_saved": Cue("seal", 0.6, 0.3),
    "screenshot": Cue("ui_click", 0.7, 0.1),
}
# The generic "explosion" hook (every blast) picks its sound by style; the
# per-effect *_explode hooks stay silent so a blast never plays twice.
EXPLOSION_SOUNDS = {"fire": "explosion", "wind": "wind_blast", "purple": "explosion"}


class AudioManager:
    """Plays cues for hooks with cooldowns, a channel limit, master volume, and mute."""

    def __init__(self, config: Config, volume: float = 0.7, muted: bool = False,
                 mixer: Any | None = None, clock: Callable[[], float] = time.perf_counter,
                 enabled: bool = True) -> None:
        """Initialize pygame.mixer (or the given mixer-like object) and load every sound.

        Any failure leaves ``available`` False and the manager silent."""
        self.config, self.clock = config, clock
        self.volume = min(1.0, max(0.0, volume))
        self.muted = muted
        self.available = False
        self._mixer: Any | None = None
        self._sounds: dict[str, Any] = {}
        self._last: dict[str, float] = {}
        self._loops: dict[tuple[str, str], Any] = {}
        if not enabled:
            return
        mixer = mixer if mixer is not None else self._import_mixer()
        if mixer is None:
            return
        try:
            mixer.pre_init(config.sound_sample_rate, -16, 2, 512)  # 512 samples: ~12 ms latency.
            mixer.init()
            mixer.set_num_channels(config.audio_channels)
            self._mixer = mixer
            for name, path in ensure_sounds(config.base_dir / config.sound_folder,
                                            config.sound_sample_rate).items():
                self._sounds[name] = mixer.Sound(str(path))
        except (RuntimeError, OSError, ValueError) as exc:  # pygame.error is a RuntimeError.
            LOGGER.info("Audio unavailable (%s); running silently.", exc)
            self.close()
            return
        self.available = True
        LOGGER.info("Audio ready: %d sounds, %d channels.", len(self._sounds), config.audio_channels)

    @classmethod
    def disabled(cls, config: Config) -> AudioManager:
        """A silent manager (--no-audio, tests)."""
        return cls(config, enabled=False)

    @staticmethod
    def _import_mixer() -> Any | None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # No banner in the console.
        try:
            import pygame  # Optional dependency: pip install pygame (or ar-hands[audio]).
        except ImportError:
            LOGGER.info("pygame is not installed: running without sound (pip install pygame to enable).")
            return None
        return pygame.mixer

    # Playback --------------------------------------------------------------------
    def play(self, name: str, volume: float = 1.0) -> bool:
        """Play a sound once; False if silent, unknown, or too many copies are playing."""
        if not self.available or self.muted:
            return False
        sound = self._sounds.get(name)
        if sound is None or sound.get_num_channels() >= MAX_INSTANCES:
            return False
        channel = sound.play()
        if channel is None:
            return False  # Every channel is busy: drop this sound rather than cut another.
        channel.set_volume(min(1.0, volume * self.volume))
        return True

    def start_loop(self, key: str, name: str, volume: float = 1.0) -> None:
        """Play ``name`` on repeat until stop_loop(key)."""
        if not self.available or self.muted or (name, key) in self._loops:
            return
        sound = self._sounds.get(name)
        channel = sound.play(loops=-1) if sound is not None else None
        if channel is not None:
            channel.set_volume(min(1.0, volume * self.volume))
            self._loops[(name, key)] = channel

    def stop_loop(self, key: str, name: str, fade_ms: int = 250) -> None:
        """Fade out a loop started with start_loop()."""
        channel = self._loops.pop((name, key), None)
        if channel is not None:
            channel.fadeout(fade_ms)

    def on_hook(self, name: str, payload: Mapping[str, Any]) -> None:
        """HookEmitter listener: map an effect or UI event to its cue."""
        if not self.available:
            return
        if name == "explosion":
            power = float(payload.get("power", 1.0))
            sound = EXPLOSION_SOUNDS.get(str(payload.get("style", "fire")), "explosion")
            if self._ready(f"explosion:{sound}", 0.06):
                self.play(sound, min(1.0, 0.6 + 0.3 * power))
            return
        cue = HOOK_CUES.get(name)
        if cue is None:
            return
        key = str(payload.get("hand", ""))
        if cue.stop:
            self.stop_loop(key, cue.sound)
        elif cue.loop:
            self.start_loop(key, cue.sound, cue.volume)
        elif self._ready(name, cue.cooldown_s):
            self.play(cue.sound, cue.volume)

    def _ready(self, key: str, cooldown_s: float) -> bool:
        now = self.clock()
        if now - self._last.get(key, -1e9) < cooldown_s:
            return False
        self._last[key] = now
        return True

    # Controls --------------------------------------------------------------------
    def set_volume(self, volume: float) -> None:
        """Master volume 0..1 (applies to new sounds and running loops)."""
        self.volume = min(1.0, max(0.0, volume))
        for channel in self._loops.values():
            channel.set_volume(self.volume)

    def set_muted(self, muted: bool) -> None:
        """Muting stops everything that is playing."""
        self.muted = muted
        if muted and self._mixer is not None and self.available:
            self._mixer.stop()
            self._loops.clear()

    def toggle_mute(self) -> bool:
        """Flip mute; returns the new state."""
        self.set_muted(not self.muted)
        return self.muted

    def close(self) -> None:
        """Stop and release the audio device (idempotent)."""
        mixer, self._mixer = self._mixer, None
        self.available = False
        self._loops.clear()
        self._sounds.clear()
        if mixer is not None:
            try:
                mixer.stop()
                mixer.quit()
            except (RuntimeError, OSError) as exc:
                LOGGER.warning("Audio cleanup warning: %s", exc)
