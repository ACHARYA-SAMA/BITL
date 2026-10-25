"""Command line: flags override saved settings, which override built-in defaults."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from . import __version__
from .config import Config
from .settings import MODES, Settings


@dataclass(frozen=True)
class RunOptions:
    """Flags that are not Config fields."""

    audio: bool = True
    record: bool = False
    debug: bool = False
    reset_settings: bool = False
    selftest: bool = False
    adaptive: bool = True
    mode: str | None = None


def parser() -> argparse.ArgumentParser:
    """The ar-hands argument parser."""
    p = argparse.ArgumentParser(prog="ar-hands", description="Real-time AR hand effects for your webcam.")
    p.add_argument("--camera", type=int, metavar="INDEX", help="camera index to try first (default 0)")
    p.add_argument("--width", type=int, help="requested capture width (default 1280)")
    p.add_argument("--height", type=int, help="requested capture height (default 720)")
    p.add_argument("--fps", type=int, help="requested capture frame rate (default 30)")
    p.add_argument("--no-glow", action="store_true", help="start with skeleton glow off")
    p.add_argument("--no-audio", action="store_true", help="run silently even if pygame is installed")
    p.add_argument("--detect-scale", type=float, metavar="S",
                   help="run hand detection on a frame scaled by S (0.3-1.0) for speed")
    p.add_argument("--mode", choices=[m.lower() for m in MODES], help="start in this mode")
    p.add_argument("--record", action="store_true", help="start recording immediately")
    p.add_argument("--no-adaptive", action="store_true", help="disable automatic quality adjustment")
    p.add_argument("--debug", action="store_true", help="verbose logging and the debug overlay")
    p.add_argument("--reset-settings", action="store_true", help="restore default settings before starting")
    p.add_argument("--selftest", action="store_true", help="run checks without a camera, then exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def build_config(args: argparse.Namespace, settings: Settings, base: Config | None = None) -> tuple[Config, RunOptions]:
    """Merge defaults <- settings <- CLI flags into a Config and run options."""
    config = base or Config()
    # Saved settings first.
    config = replace(config, glow=settings.glow, show_skeleton=settings.show_skeleton,
                     show_hud=settings.show_hud, default_color=settings.color_index,
                     default_thickness=settings.thickness, detection_confidence=settings.sensitivity,
                     presence_confidence=settings.sensitivity, tracking_confidence=settings.sensitivity)
    # Then flags, which always win.
    changes = {}
    for name in ('width', 'height', 'fps'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f'--{name} must be positive')
    if args.camera is not None and args.camera < 0:
        raise ValueError('--camera must be nonnegative')
    if args.camera is not None:
        others = tuple(i for i in config.camera_indices if i != args.camera)
        changes["camera_indices"] = (args.camera, *others)
    if args.width is not None:
        changes["width"] = args.width
    if args.height is not None:
        changes["height"] = args.height
    if args.fps is not None:
        changes["camera_fps"] = args.fps
    if args.no_glow:
        changes["glow"] = False
    if args.detect_scale is not None:
        if not 0.3 <= args.detect_scale <= 1.0:
            raise ValueError("--detect-scale must be between 0.3 and 1.0")
        changes["detect_scale"] = args.detect_scale
    if args.debug:
        changes["show_debug"] = True
    config = replace(config, **changes)
    mode = args.mode.title() if args.mode else settings.mode
    options = RunOptions(audio=not args.no_audio, record=args.record, debug=args.debug,
                         reset_settings=args.reset_settings, selftest=args.selftest,
                         adaptive=settings.adaptive_quality and not args.no_adaptive, mode=mode)
    return config, options
