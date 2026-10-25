"""Persistent user settings: settings.json in the platform's per-user config folder.

Loading never crashes the app:
  * a missing file is created with defaults;
  * a value with the wrong type or out of range is replaced by its default
    (with a warning naming the field);
  * an unreadable or corrupt file is moved to settings.json.bak and recreated.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

LOGGER = logging.getLogger(__name__)
SETTINGS_VERSION = 1
MODES = ("Draw", "Jutsu", "Blocks", "Jujutsu")
CAMERA_SOURCES = ("Webcam", "Phone")


@dataclass
class Settings:
    """Everything the menu changes, plus first-run and last-session state."""

    version: int = SETTINGS_VERSION
    mode: str = "Draw"
    color_index: int = 0
    thickness: int = 6
    glow: bool = False
    show_skeleton: bool = True
    show_hud: bool = True
    volume: float = 0.7
    muted: bool = False
    sensitivity: float = 0.7  # Hand detection confidence (0.3 = eager, 0.9 = strict).
    fullscreen: bool = False
    record_clean: bool = False
    adaptive_quality: bool = True
    camera_source: str = "Webcam"  # "Phone" streams from the phone page (phone.py).
    max_hands: int = 6  # Hands tracked at once; fewer is a little faster.
    onboarded: bool = False


# Field -> (allowed values) for choices, or (low, high) for numbers.
LIMITS: dict[str, tuple] = {
    "mode": MODES,
    "camera_source": CAMERA_SOURCES,
    "max_hands": (2, 8),
    "color_index": (0, 6),
    "thickness": (2, 30),
    "volume": (0.0, 1.0),
    "sensitivity": (0.3, 0.9),
}


def settings_dir(app_name: str = "ARHands") -> Path:
    """The platform's conventional per-user config folder."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        app_name = app_name.lower()
    return base / app_name


def validate(raw: object) -> tuple[Settings, list[str]]:
    """Build Settings from parsed JSON, replacing bad fields; returns (settings, problems)."""
    defaults = Settings()
    problems: list[str] = []
    if not isinstance(raw, dict):
        return defaults, ["settings file is not a JSON object"]
    values = {}
    for spec in fields(Settings):
        name, default = spec.name, getattr(defaults, spec.name)
        if name not in raw:
            values[name] = default
            continue
        value = raw[name]
        expected = type(default)
        ok = isinstance(value, expected) or (expected is float and isinstance(value, int)
                                             and not isinstance(value, bool))
        if expected is int and isinstance(value, bool):
            ok = False  # JSON true is not a number here.
        if ok and expected is float:
            value = float(value)
        limits = LIMITS.get(name)
        if ok and limits is not None:
            ok = value in limits if isinstance(limits[0], str) else limits[0] <= value <= limits[1]
        if not ok:
            problems.append(f"{name}={value!r} is invalid; using {default!r}")
            value = default
        values[name] = value
    values["version"] = SETTINGS_VERSION
    return Settings(**values), problems


class SettingsStore:
    """Load and save Settings atomically; recover from anything."""

    def __init__(self, path: Path | None = None) -> None:
        """Default location: settings_dir()/settings.json, falling back to the working directory."""
        self.path = path or self._default_path()

    @staticmethod
    def _default_path() -> Path:
        folder = settings_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return folder / "settings.json"
        except OSError:
            LOGGER.warning("Config folder %s is not writable; using the working directory.", folder)
            return Path.cwd() / "settings.json"

    def load(self) -> Settings:
        """Read settings, repairing or recreating the file when needed."""
        if not self.path.exists():
            settings = Settings()
            self.save(settings)
            return settings
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            backup = self.path.with_name(self.path.name + ".bak")
            LOGGER.warning("Settings file is corrupt (%s); moved to %s and recreated.", exc, backup)
            try:
                self.path.replace(backup)
            except OSError as move_error:
                LOGGER.warning("Could not back up the corrupt settings: %s", move_error)
            settings = Settings()
            self.save(settings)
            return settings
        settings, problems = validate(raw)
        for problem in problems:
            LOGGER.warning("Settings: %s", problem)
        if problems:
            self.save(settings)
        return settings

    def save(self, settings: Settings) -> bool:
        """Write atomically (temp file + replace) so a crash never leaves half a file."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as handle:
                json.dump(asdict(settings), handle, indent=2)
                temporary = Path(handle.name)
            temporary.replace(self.path)
            return True
        except OSError as exc:
            LOGGER.warning("Could not save settings to %s: %s", self.path, exc)
            return False

    def reset(self) -> Settings:
        """Restore defaults on disk (keeps the onboarding flag set: the user has seen it)."""
        settings = Settings(onboarded=True)
        self.save(settings)
        return settings
