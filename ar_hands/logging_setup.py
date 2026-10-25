"""Structured logging: concise console output plus a rotating log file."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

CONSOLE_FORMAT = "%(levelname)-7s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(log_dir: Path, debug: bool = False) -> Path | None:
    """Console (INFO, or DEBUG with --debug) and logs/ar_hands.log (1 MB x 3, always DEBUG).

    Returns the log file path, or None if the folder is not writable (the app
    still runs with console logging only).
    """
    root = logging.getLogger()
    for handler in list(root.handlers):  # Idempotent: safe to call again (tests, --selftest).
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)
    path = log_dir / "ar_hands.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning("File logging disabled (%s).", exc)
        return None
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(file_handler)
    # Third-party chatter stays out of the console unless debugging.
    for noisy in ("PIL", "matplotlib", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
