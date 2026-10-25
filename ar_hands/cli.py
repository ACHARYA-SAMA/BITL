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
    camera_source: str | None = None  # "Phone" or "Webcam" for this run; None = saved setting.


def camera_arg(text: str) -> int | str:
    """A camera index, or a stream URL from a phone webcam app."""
    try:
        return int(text)  # Negative indexes are rejected by build_config with a clear message.
    except ValueError:
        pass
    if text.lower().startswith(("http://", "https://", "rtsp://")):
        return text
    raise argparse.ArgumentTypeError("expected a camera index (0, 1, ...) or a stream URL (http://...)")


def parser() -> argparse.ArgumentParser:
    """The ar-hands argument parser."""
    p = argparse.ArgumentParser(prog="ar-hands", description="Real-time AR hand effects for your webcam.")
    p.add_argument("--camera", type=camera_arg, metavar="INDEX|URL",
                   help="camera index to try first (default 0), or a stream URL from a phone webcam app")
    p.add_argument("--phone", action="store_true",
                   help="use your phone as the camera: scan the QR code shown on start")
    p.add_argument("--webcam", action="store_true", help="use this computer's webcam even if Phone was saved")
    p.add_argument("--phone-port", type=int, metavar="PORT", help="port of the phone camera page (default 8765)")
    p.add_argument("--hands", type=int, metavar="N", help="track up to N hands at once, 2-8 (default 6)")
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
                     presence_confidence=settings.sensitivity, tracking_confidence=settings.sensitivity,
                     num_hands=settings.max_hands)
    # Then flags, which always win.
    changes = {}
    for name in ('width', 'height', 'fps'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f'--{name} must be positive')
    if args.phone and args.webcam:
        raise ValueError('choose either --phone or --webcam')
    if args.phone_port is not None and not 0 <= args.phone_port <= 65535:
        raise ValueError('--phone-port must be between 0 and 65535')
    if isinstance(args.camera, int) and args.camera < 0:
        raise ValueError('--camera must be nonnegative')
    if isinstance(args.camera, str):
        changes["camera_url"] = args.camera
    elif args.camera is not None:
        others = tuple(i for i in config.camera_indices if i != args.camera)
        changes["camera_indices"] = (args.camera, *others)
    if args.phone_port is not None:
        changes["phone_port"] = args.phone_port
    if args.hands is not None:
        if not 2 <= args.hands <= 8:
            raise ValueError('--hands must be between 2 and 8')
        changes["num_hands"] = args.hands
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
    # A stream URL is a webcam-style source, so it also overrides a saved Phone choice.
    source = "Phone" if args.phone else "Webcam" if args.webcam or "camera_url" in changes else None
    options = RunOptions(audio=not args.no_audio, record=args.record, debug=args.debug,
                         reset_settings=args.reset_settings, selftest=args.selftest,
                         adaptive=settings.adaptive_quality and not args.no_adaptive, mode=mode,
                         camera_source=source)
    return config, options
