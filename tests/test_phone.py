"""Phone camera: the WebSocket transport, app wiring, and portrait (phone-shaped) layouts."""
from dataclasses import replace
import fractions
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import ssl
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import ar_hands.app as app_module
from ar_hands.app import App
from ar_hands.camera import CameraState
from ar_hands.cli import build_config, parser
from ar_hands.config import Config
from ar_hands.effects.manager import Mode
from ar_hands.guide import GestureGuide
from ar_hands.hud import HUDRenderer
from ar_hands.menu import ITEMS, Menu
from ar_hands.pairing import PairingScreen
from ar_hands.phone import PhoneCamera, PhoneCommand, PointerEvent, available_codecs, ensure_certificate
from ar_hands.settings import Settings, validate
from ar_hands.websocket import BINARY, CLOSE, TEXT, WebSocket, accept_key, frame, unmask
from ar_hands.window import MouseState

HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None
HAS_AV = "h264" in available_codecs()


def jpeg(width: int = 72, height: int = 128) -> bytes:
    image = np.zeros((height, width, 3), np.uint8)
    image[:, : width // 2] = (0, 0, 255)
    return cv2.imencode(".jpg", image)[1].tobytes()


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class PhonePage:
    """A stand-in for the phone page: the WebSocket protocol, client side."""

    def __init__(self, phone: PhoneCamera, key: str | None = None) -> None:
        raw = socket.create_connection(("127.0.0.1", phone.port), timeout=5)
        if phone.tls:
            context = ssl.create_default_context()
            context.check_hostname, context.verify_mode = False, ssl.CERT_NONE
            raw = context.wrap_socket(raw)
        self.sock = raw
        nonce = "dGhlIHNhbXBsZSBub25jZQ=="
        self.sock.sendall((f"GET /ws?k={phone.token if key is None else key} HTTP/1.1\r\nHost: x\r\n"
                           f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {nonce}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += self.sock.recv(1)
        assert b" 101 " in reply and accept_key(nonce).encode() in reply, reply
        self.buffer = bytearray()

    def send(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes((0x80 | opcode, 0x80 | length))
        elif length < 1 << 16:
            header = bytes((0x80 | opcode, 0x80 | 126)) + length.to_bytes(2, "big")
        else:
            header = bytes((0x80 | opcode, 0x80 | 127)) + length.to_bytes(8, "big")
        self.sock.sendall(header + mask + unmask(payload, mask))

    def send_json(self, message: dict) -> None:
        self.send(TEXT, json.dumps(message).encode())

    def receive(self, timeout: float = 3.0) -> tuple[int, bytes]:
        """The next server message (servers never mask)."""
        end = time.monotonic() + timeout
        while True:
            if len(self.buffer) >= 2:
                length, offset = self.buffer[1] & 0x7F, 2
                if length == 126:
                    length, offset = int.from_bytes(self.buffer[2:4], "big"), 4
                elif length == 127:
                    length, offset = int.from_bytes(self.buffer[2:10], "big"), 10
                if len(self.buffer) >= offset + length:
                    opcode = self.buffer[0] & 0x0F
                    payload = bytes(self.buffer[offset:offset + length])
                    del self.buffer[:offset + length]
                    return opcode, payload
            self.sock.settimeout(max(0.01, end - time.monotonic()))
            try:
                data = self.sock.recv(65536)
            except TimeoutError as exc:
                raise AssertionError("nothing arrived") from exc
            if not data:
                raise ConnectionError("closed")
            self.buffer += data

    def expect(self, kind: str, timeout: float = 3.0) -> dict:
        """Skip messages until a JSON message of type ``kind``."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            opcode, payload = self.receive(end - time.monotonic())
            if opcode == TEXT and json.loads(payload).get("t") == kind:
                return json.loads(payload)
        raise AssertionError(f"no {kind} message")

    def picture(self, timeout: float = 3.0) -> bytes:
        """Skip messages until a finished picture."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            opcode, payload = self.receive(end - time.monotonic())
            if opcode == BINARY:
                return payload
        raise AssertionError("no picture")

    def close(self) -> None:
        self.sock.close()


class WebSocketTests(unittest.TestCase):
    def test_handshake_key_masking_and_fragments(self):
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")
        self.assertEqual(unmask(unmask(b"hello world", b"abcd"), b"abcd"), b"hello world")
        left, right = socket.socketpair()
        try:
            ws = WebSocket(right, max_message=1000)
            mask = b"\x01\x02\x03\x04"
            first = bytes((0x02, 0x80 | 3)) + mask + unmask(b"abc", mask)  # Binary, not final.
            last = bytes((0x80, 0x80 | 3)) + mask + unmask(b"def", mask)  # Continuation, final.
            left.sendall(first + last)
            messages = []
            for _ in range(5):
                messages += ws.poll(0.2)
                if messages:
                    break
            self.assertEqual(messages, [(BINARY, b"abcdef")])
            left.sendall(bytes((0x82, 0x80 | 126)) + (5000).to_bytes(2, "big"))
            with self.assertRaises(Exception):
                for _ in range(5):
                    ws.poll(0.2)
            self.assertEqual(left.recv(16)[:2], frame(CLOSE, (1009).to_bytes(2, "big"))[:2])
        finally:
            left.close()
            right.close()


class PhoneServerTests(unittest.TestCase):
    """The real server on localhost, used the way the phone page uses it."""

    tls = HAS_CRYPTOGRAPHY

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        config = replace(Config(), phone_host="127.0.0.1", phone_port=0)
        self.phone = PhoneCamera(config, cert_folder=Path(self.temp.name), tls=self.tls).start()
        self.assertIsNone(self.phone.error)
        self.pages: list[PhonePage] = []

    def tearDown(self) -> None:
        for page in self.pages:
            page.close()
        self.phone.stop()
        self.temp.cleanup()

    def page(self, client: str = "phone-a", claim: bool = True, key: str | None = None) -> PhonePage:
        page = PhonePage(self.phone, key)
        self.pages.append(page)
        if key is None:
            page.send_json({"t": "hello", "client": client, "claim": claim, "preview": True})
        return page

    def test_page_is_served(self):
        context = None
        if self.tls:
            context = ssl.create_default_context()
            context.check_hostname, context.verify_mode = False, ssl.CERT_NONE
        connection = (http.client.HTTPSConnection("127.0.0.1", self.phone.port, context=context, timeout=5)
                      if self.tls else http.client.HTTPConnection("127.0.0.1", self.phone.port, timeout=5))
        connection.request("GET", "/?k=" + self.phone.token)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b"VideoEncoder", response.read())
        connection.close()

    def test_jpeg_frames_and_finished_pictures_round_trip(self):
        page = self.page()
        self.assertIn("jpeg", page.expect("welcome")["codecs"])
        page.send_json({"t": "config", "codec": "jpeg"})
        self.assertEqual(self.phone.state, CameraState.STARTING)
        page.send(BINARY, jpeg())
        packet = self.phone.read(2.0)
        self.assertEqual(packet.frame.shape, (128, 72, 3))
        self.assertEqual(self.phone.state, CameraState.LIVE)
        self.phone.publish(np.full((128, 72, 3), 200, np.uint8), {"mode": "Jutsu"})
        picture = page.picture()
        self.assertEqual(cv2.imdecode(np.frombuffer(picture, np.uint8), cv2.IMREAD_COLOR).shape, (128, 72, 3))
        self.assertEqual(page.expect("state")["mode"], "Jutsu")
        self.phone.publish(np.full((128, 72, 3), 100, np.uint8), {"mode": "Draw"})
        with self.assertRaises(AssertionError, msg="the next picture waits for the ack"):
            page.picture(timeout=0.3)
        page.send_json({"t": "ack"})
        self.assertTrue(page.picture())

    @unittest.skipUnless(HAS_AV, "H.264 needs PyAV")
    def test_h264_stream_is_decoded(self):
        import av
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width, encoder.height, encoder.pix_fmt = 320, 180, "yuv420p"
        encoder.time_base, encoder.framerate = fractions.Fraction(1, 60), 60
        encoder.options = {"tune": "zerolatency", "preset": "ultrafast"}
        page = self.page()
        self.assertIn("h264", page.expect("welcome")["codecs"])
        page.send_json({"t": "config", "codec": "h264", "width": 320, "height": 180})
        for index in range(5):
            image = np.zeros((180, 320, 3), np.uint8)
            image[:, index * 40:index * 40 + 40] = 255
            video = av.VideoFrame.from_ndarray(image, format="bgr24")
            video.pts = index
            for packet in encoder.encode(video):
                page.send(BINARY, bytes(packet))
        packet = self.phone.read(2.0)
        self.assertEqual(packet.frame.shape, (180, 320, 3))
        page.send(BINARY, b"\x00\x00\x00\x01garbage")
        self.assertEqual(page.expect("keyframe")["t"], "keyframe", "a broken stream asks for a key frame")

    def test_wrong_key_and_second_phone(self):
        stranger = self.page(key="guess")
        self.assertEqual(stranger.expect("error")["code"], 401)
        first = self.page("phone-a")
        first.expect("welcome")
        first.send_json({"t": "config", "codec": "jpeg"})
        first.send(BINARY, jpeg())
        self.assertIsNotNone(self.phone.read(2.0))
        second = self.page("phone-b", claim=False)
        self.assertEqual(second.expect("error")["code"], 409, "a quiet claim does not steal the camera")
        third = self.page("phone-c", claim=True)
        third.expect("welcome")
        first.send(BINARY, jpeg())
        self.assertEqual(first.expect("error")["code"], 409, "the old page is told it was replaced")
        self.assertEqual(self.phone.reconnects, 1)

    def test_commands_and_touches_are_validated_and_queued(self):
        page = self.page()
        page.expect("welcome")
        for message in ({"t": "command", "name": "mode", "value": "Blocks"}, {"t": "command", "name": "quit"},
                        {"t": "command", "name": "mode", "value": 3},
                        {"t": "pointer", "x": 1.5, "y": .25, "down": True},
                        {"t": "pointer", "x": True, "y": .2, "down": True}):
            page.send_json(message)
        page.send(TEXT, b"{broken")
        end = time.monotonic() + 2
        while not self.phone._pointer and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual(self.phone.commands(), [PhoneCommand("mode", "Blocks")])
        self.assertEqual(self.phone.pointer_events(), [PointerEvent(1.0, .25, True)])

    def test_stopped_server_says_so(self):
        page = self.page()
        page.expect("welcome")
        self.phone.stop()
        self.assertEqual(page.expect("error")["code"], 503)

    @unittest.skipUnless(HAS_CRYPTOGRAPHY, "certificates need the cryptography package")
    def test_certificate_is_reused_until_the_address_changes(self):
        folder = Path(self.temp.name) / "certs"
        cert, _ = ensure_certificate(folder, ["192.168.1.20"])
        first = cert.read_bytes()
        self.assertEqual(ensure_certificate(folder, ["192.168.1.20"])[0].read_bytes(), first)
        self.assertNotEqual(ensure_certificate(folder, ["10.0.0.7"])[0].read_bytes(), first)
        from cryptography import x509
        certificate = x509.load_pem_x509_certificate(cert.read_bytes())
        names = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertIn("10.0.0.7", [str(ip) for ip in names.get_values_for_type(x509.IPAddress)])
        self.assertTrue(certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage))
        lifetime = certificate.not_valid_after_utc - certificate.not_valid_before_utc
        self.assertLessEqual(lifetime.days, 825, "iOS refuses longer-lived certificates")


class PlainPhoneServerTests(PhoneServerTests):
    """The same protocol without TLS (how the page is tested in a desktop browser)."""

    tls = False


class PhoneStateTests(unittest.TestCase):
    """The frame slot and connection states, without a server."""

    def test_states_follow_frames_and_silence(self):
        clock = Clock()
        phone = PhoneCamera(Config(), clock=clock)
        self.assertEqual(phone.state, CameraState.STARTING)
        self.assertTrue(phone._claim("a", True))
        self.assertTrue(phone._store("a", np.zeros((128, 72, 3), np.uint8)))
        self.assertEqual(phone.state, CameraState.LIVE)
        clock.now += Config().phone_timeout_s + .1
        self.assertEqual(phone.state, CameraState.LOST)
        self.assertTrue(phone._claim("b", False), "a silent phone can be replaced")
        self.assertFalse(phone._store("a", np.zeros((8, 8, 3), np.uint8)))

    def test_only_the_newest_frame_is_kept(self):
        clock = Clock()
        phone = PhoneCamera(Config(), clock=clock)
        phone._claim("a", True)
        for _ in range(3):
            clock.now += .03
            phone._store("a", np.zeros((8, 8, 3), np.uint8))
        packet = phone.read(.01)
        self.assertEqual(packet.index, 3)
        self.assertIsNone(phone.read(.01))
        self.assertAlmostEqual(phone.fps, 1 / .03, places=3)

    def test_startup_error_is_reported_not_raised(self):
        phone = PhoneCamera(Config())
        with patch("ar_hands.phone.ensure_certificate", side_effect=OSError("disk full")):
            phone.start()
        self.assertFalse(phone.running)
        self.assertEqual(phone.state, CameraState.LOST)
        self.assertIn("disk full", phone.message)


class PhoneAppTests(unittest.TestCase):
    """Phone buttons and touches drive the same paths as keys and the mouse."""

    def test_phone_buttons_reuse_app_actions(self):
        app = App()
        app.phone_command("mode", "Jutsu")
        self.assertIs(app.modes.mode, Mode.JUTSU)
        app.phone_command("mode", "Nonsense")
        self.assertIs(app.modes.mode, Mode.JUTSU)
        app.phone_command("color")
        self.assertEqual(app.color_index, 1)
        app.phone_command("menu")
        self.assertTrue(app.menu.visible)
        app.phone_command("mute")
        self.assertTrue(app.menu.visible, "sound and recording work on top of the menu")
        app.phone_command("undo")
        self.assertFalse(app.menu.visible, "other buttons return to the camera first")
        app.phone_command("guide")
        self.assertTrue(app.guide.visible)
        app.phone_command("guide")
        self.assertFalse(app.guide.visible)

    def test_quick_tap_spans_two_frames(self):
        app = App()
        app.window = MagicMock()
        app.window.mouse = MouseState()
        app._frame_shape = (1280, 720, 3)
        phone = MagicMock(spec=PhoneCamera)
        phone.state, phone.size = CameraState.LIVE, (720, 1280)
        phone.commands.return_value = []
        phone.pointer_events.side_effect = [[PointerEvent(.5, .5, True), PointerEvent(.5, .5, False)], []]
        app._phone_input(phone)
        self.assertTrue(app.window.mouse.pressed)
        self.assertEqual(app.window.mouse.position, (359.5, 639.5))
        app._phone_input(phone)
        self.assertFalse(app.window.mouse.pressed)

    def test_phone_state_for_the_page(self):
        app = App()
        app.open_guide()
        state = app._phone_state()
        self.assertEqual(state["mode"], "Draw")
        self.assertEqual(state["overlay"], "guide")
        self.assertEqual(state["color"], "#00ffff", "BGR cyan becomes CSS #00ffff")

    def test_switching_source_swaps_the_running_stream(self):
        app = App()
        webcam, phone = MagicMock(), MagicMock()
        phone.start.return_value = phone
        app.stream = webcam
        with patch.object(app_module, "PhoneCamera", return_value=phone):
            app.handle_key(ord("o"), None)
        webcam.stop.assert_called_once()
        self.assertIs(app.stream, phone)
        self.assertEqual(app.camera_source, "Phone")
        self.assertEqual(app.settings.camera_source, "Phone")
        self.assertEqual(app.menu_value("camera_source"), "Phone")

    def test_sideways_phone_fills_the_screen_and_upright_restores_it(self):
        app = App()
        app.window = MagicMock()
        app.window.fullscreen = False
        app.window.set_fullscreen.side_effect = lambda on: setattr(app.window, "fullscreen", on)
        phone = MagicMock(spec=PhoneCamera)
        phone.commands.return_value, phone.pointer_events.return_value = [], []
        phone.state, phone.size = CameraState.LIVE, (1280, 720)
        app._phone_input(phone)
        self.assertTrue(app.window.fullscreen, "landscape phone frames go fullscreen")
        app.window.set_fullscreen(False)  # The user leaves fullscreen (Shift+F)...
        app._phone_input(phone)
        self.assertFalse(app.window.fullscreen, "...and is not fought while the phone stays sideways")
        app.window.fullscreen = True
        phone.size = (720, 1280)
        app._phone_input(phone)
        self.assertFalse(app.window.fullscreen, "upright returns to the saved windowed choice")

    def test_waiting_frame_shows_a_scannable_code(self):
        app = App()
        phone = MagicMock(spec=PhoneCamera)
        phone.running, phone.error, phone.state = True, None, CameraState.STARTING
        phone.message = "Waiting for your phone"
        phone.url = "https://192.168.1.20:8765/?k=abcdefghijkl"
        frame = app._waiting_frame(phone)
        self.assertEqual(cv2.QRCodeDetector().detectAndDecode(frame)[0], phone.url)

    def test_pairing_without_server_shows_no_code(self):
        phone = MagicMock(spec=PhoneCamera)
        phone.running, phone.error, phone.state = False, "ports in use", CameraState.LOST
        phone.message = "Phone camera unavailable: ports in use"
        frame = np.zeros((720, 1280, 3), np.uint8)
        PairingScreen(Config()).render(frame, phone, 0.)
        self.assertEqual(cv2.QRCodeDetector().detectAndDecode(frame)[0], "")


class CommandLineTests(unittest.TestCase):
    def test_phone_flags_and_stream_urls(self):
        _, options = build_config(parser().parse_args(["--phone", "--phone-port", "9000"]), Settings())
        self.assertEqual(options.camera_source, "Phone")
        config, options = build_config(parser().parse_args(["--camera", "http://10.0.0.2:8080/video"]),
                                       Settings(camera_source="Phone"))
        self.assertEqual(config.camera_url, "http://10.0.0.2:8080/video")
        self.assertEqual(options.camera_source, "Webcam", "a stream URL is a webcam-style source")
        _, options = build_config(parser().parse_args([]), Settings(camera_source="Phone"))
        self.assertIsNone(options.camera_source, "no flag keeps the saved choice")
        with self.assertRaises(ValueError):
            build_config(parser().parse_args(["--phone", "--webcam"]), Settings())
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parser().parse_args(["--camera", "front"])

    def test_camera_source_setting_is_validated(self):
        self.assertEqual(validate({"camera_source": "Phone"})[0].camera_source, "Phone")
        settings, problems = validate({"camera_source": "Drone"})
        self.assertEqual(settings.camera_source, "Webcam")
        self.assertEqual(len(problems), 1)


class PortraitLayoutTests(unittest.TestCase):
    """Phones film portrait (720x1280): overlays must stay inside and readable."""

    def test_menu_uses_one_column_inside_a_portrait_frame(self):
        menu = Menu(Config(), lambda key: None, lambda key, value: None)
        boxes = menu.layout((720, 1280))
        self.assertEqual(len({box[0] for box in boxes.values()}), 1)
        for x0, y0, x1, y1 in boxes.values():
            self.assertTrue(0 <= x0 < x1 <= 720 and 0 <= y0 < y1 <= 1280)
        ordered = [boxes[item.key] for item in ITEMS]
        self.assertTrue(all(a[3] < b[1] for a, b in zip(ordered, ordered[1:])), "rows do not overlap")
        self.assertGreaterEqual(ordered[0][3] - ordered[0][1], 40, "rows stay finger-sized")
        landscape = Menu(Config(), lambda key: None, lambda key, value: None).layout((1280, 720))
        self.assertEqual(len({box[0] for box in landscape.values()}), 2)

    def test_hud_wraps_long_lines_on_portrait_frames(self):
        hud = HUDRenderer(Config())
        width = 720 - 2 * Config().margin - 16
        self.assertGreater(len(hud._lines(hud.KEY_HINT, width, hud.MIN_FIT_PORTRAIT)), 1)
        self.assertEqual(len(hud._lines(hud.KEY_HINT, 1280 - 40, hud.MIN_FIT)), 1, "landscape keeps one line")
        frame = np.zeros((1280, 720, 3), np.uint8)
        hud.draw(frame, 30., 1, hint=app_module.MODE_HINTS[Mode.DRAW])
        self.assertTrue(frame[:200].any() and frame[-120:].any())
        bottom = hud._panel(np.zeros((1280, 720, 3), np.uint8), hud.KEY_HINT)
        self.assertGreater(bottom, 2 * Config().hud_brush_offset, "wrapped panels are taller")

    def test_guide_renders_portrait_layout(self):
        guide = GestureGuide(Config())
        guide.open("Draw")
        frame = np.full((1280, 720, 3), 90, np.uint8)
        guide.render(frame)
        self.assertEqual(frame.shape, (1280, 720, 3))
        self.assertTrue((frame != 90).any())


if __name__ == "__main__":
    unittest.main()
