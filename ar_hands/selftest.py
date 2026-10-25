"""Headless installation checks: no webcam, GUI, settings changes or audio device."""
from __future__ import annotations
from dataclasses import replace
import http.client
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import socket
import ssl
import cv2
import numpy as np
from .config import Config
from .hand_tracker import HandTracker, ensure_model
from .phone import PhoneCamera, available_codecs
from .websocket import BINARY, TEXT, client_frame
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

    def phone_camera() -> None:
        # The real HTTPS page, then one frame over the WebSocket, on this computer's loopback.
        with TemporaryDirectory() as folder:
            with PhoneCamera(replace(config, phone_host='127.0.0.1', phone_port=0),
                             cert_folder=Path(folder)) as phone:
                assert phone.error is None, phone.error
                context = ssl.create_default_context()
                context.check_hostname, context.verify_mode = False, ssl.CERT_NONE
                connection = http.client.HTTPSConnection('127.0.0.1', phone.port, context=context, timeout=5)
                try:
                    connection.request('GET', '/')
                    assert connection.getresponse().read().startswith(b'<!doctype html>')
                finally:
                    connection.close()
                with context.wrap_socket(socket.create_connection(('127.0.0.1', phone.port), 5)) as sock:
                    sock.sendall((f'GET /ws?k={phone.token} HTTP/1.1\r\nHost: bitl\r\nUpgrade: websocket\r\n'
                                  'Connection: Upgrade\r\nSec-WebSocket-Key: c2VsZnRlc3Q=\r\n\r\n').encode())
                    assert b' 101 ' in sock.recv(1024)
                    image = cv2.imencode('.jpg', np.zeros((64, 48, 3), np.uint8))[1].tobytes()
                    for kind, payload in ((TEXT, b'{"t":"hello","client":"selftest","claim":true}'),
                                          (TEXT, b'{"t":"config","codec":"jpeg"}'), (BINARY, image)):
                        sock.sendall(client_frame(kind, payload, b'test'))
                    packet = phone.read(2.)
                assert packet is not None and packet.frame.shape == (64, 48, 3)
                logger.info('Phone video codecs available: %s', ', '.join(available_codecs()))

    for name, check in (('Hand model and blank-frame inference', tracking),
                        ('Person segmentation for domains and dust', segmentation),
                        ('Procedural sprites and edge clipping', sprites_and_clipping),
                        ('Settings round-trip', settings_roundtrip),
                        ('Phone camera page over HTTPS', phone_camera)):
        try:
            check()
            logger.info('PASS %s', name)
        except Exception:
            failures += 1
            logger.exception('FAIL %s', name)
    return 1 if failures else 0
