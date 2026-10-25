"""Headless installation checks: no webcam, GUI, settings changes or audio device."""
from __future__ import annotations
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from .config import Config
from .hand_tracker import HandTracker, ensure_model
from .segmentation import PersonSegmenter, ensure_segmenter_model
from .effects.sprites import SpriteLibrary
from .effects.compositing import overlay_rgba
from .settings import Settings, SettingsStore


def run_selftest(config: Config) -> int:
    """Print PASS/FAIL for each independent check and return a process exit code."""
    logger = logging.getLogger(__name__)
    failures = 0

    def tracking() -> None:
        ensure_model(config)
        with HandTracker(config) as tracker:
            assert tracker.detect(np.zeros((240, 320, 3), np.uint8), 1) == []

    def segmentation() -> None:
        ensure_segmenter_model(config)
        segmenter = PersonSegmenter(config)
        try:
            mask = segmenter.mask(np.zeros((240, 320, 3), np.uint8), 1)
            assert mask is not None and mask.shape == (240, 320)
        finally:
            segmenter.close()

    def sprites_and_clipping() -> None:
        sprites = SpriteLibrary(config.effects, config.base_dir / config.sprite_folder)
        sprite = sprites.fire_glow
        for center in ((0, 0), (100, 0), (0, 100), (100, 100), (-1000, -1000)):
            canvas = np.zeros((100, 100, 3), np.uint8)
            overlay_rgba(canvas, sprite, center, .5, 43)
            assert canvas.shape == (100, 100, 3)
        assert sprites.icon('grass', 32).shape == (32, 32, 4)

    def settings_roundtrip() -> None:
        with TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder) / 'settings.json')
            settings = Settings(mode='Jujutsu', volume=.4)
            assert store.save(settings)
            assert store.load() == settings

    for name, check in (('Hand model and blank-frame inference', tracking),
                        ('Person segmentation for domains and dust', segmentation),
                        ('Procedural sprites and edge clipping', sprites_and_clipping),
                        ('Settings round-trip', settings_roundtrip)):
        try:
            check()
            logger.info('PASS %s', name)
        except Exception:
            failures += 1
            logger.exception('FAIL %s', name)
    return 1 if failures else 0
