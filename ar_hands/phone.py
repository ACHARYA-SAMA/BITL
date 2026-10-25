"""Use a phone as the camera: the app serves a web page that streams the phone's camera.

Many laptop webcams are dim, noisy and capped near 15 FPS, while almost any
recent phone films far better. Nothing is installed on the phone: it scans the
QR code on the pairing screen and opens a page served by this computer. The
page sends its camera frames here and shows the finished AR picture back with
touch controls, so the phone is both the camera and a second display.

Browsers only give camera access to secure (HTTPS) pages, so the server makes
a self-signed certificate on first use; the phone warns about it once. A
random key in the link stops other devices on the network from sending frames
or watching.

Transport is a WebSocket (websocket.py): video streams up, finished pictures
stream down, and neither waits for the other. Video is H.264 from the phone's
hardware encoder when the browser has WebCodecs and PyAV is installed (about
5 Mbps at 720p60), else JPEG. The phone skips frames while its upload is
backed up, and, as in CameraStream, a one-frame slot here drops frames the
app had no time for, so a slow network costs frame rate, never latency.
"""
from __future__ import annotations
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import datetime as dt
from functools import cache
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import ipaddress
import json
import logging
import math
from pathlib import Path
import secrets
import socket
import socketserver
import ssl
import sys
import threading
import time
from types import TracebackType
from urllib.parse import parse_qs, urlsplit
import cv2
import numpy as np
from . import AppError
from .camera import CameraState, FramePacket
from .config import Config
from .settings import settings_dir
from .websocket import BINARY, TEXT, Closed, WebSocket, accept_key

LOGGER = logging.getLogger(__name__)
PAGE = Path(__file__).with_name("web") / "phone.html"
# The phone page's buttons; anything else is rejected before it reaches the app.
COMMANDS = frozenset({"mode", "color", "undo", "redo", "clear", "save", "snap", "record", "mute",
                      "guide", "menu", "background"})
PORT_ATTEMPTS = 10  # A busy port (another app, a second BITL) tries the next ones up.
VIEW_MAX_SIDE = 960  # The finished picture is scaled down to this for the phone screen.
SESSION_POLL_S = 0.004  # How long a connection waits for data before checking for a picture to send.
SESSION_IDLE_S = 10.0  # A page that sent nothing for this long is gone.
STATE_EVERY_S = 0.25  # App state reaches the page at least this often (with no pictures flowing).


@dataclass(frozen=True)
class PhoneCommand:
    """A button pressed on the phone page."""

    name: str
    value: str | None = None


@dataclass(frozen=True)
class PointerEvent:
    """A touch on the phone's view of the menu or guide, in 0..1 frame coordinates."""

    x: float
    y: float
    down: bool


def lan_addresses() -> list[str]:
    """IPv4 addresses other devices may reach this computer at, best guess first."""
    found: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            # Connecting a UDP socket sends nothing; it only picks the interface
            # the default route uses, which is normally the Wi-Fi adapter.
            probe.connect(("8.8.8.8", 80))
            found.append(probe.getsockname()[0])
        except OSError:
            pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    usable: list[str] = []
    for address in found:
        ip = ipaddress.ip_address(address)
        if not (ip.is_loopback or ip.is_link_local or ip.is_unspecified) and address not in usable:
            usable.append(address)
    return usable or ["127.0.0.1"]


def ensure_certificate(folder: Path, hosts: Iterable[str]) -> tuple[Path, Path]:
    """A self-signed certificate for the phone page, made once and then reused.

    Reuse matters: a phone remembers the warning it accepted for a certificate,
    so a new one every run would warn every time. It is remade only when it
    nears expiry or this computer's address changed. iOS rejects certificates
    outright (no warning to accept) without the server-auth usage, the address
    in the alternative names, or with more than 825 days of validity.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise AppError("The phone camera needs the 'cryptography' package: "
                       "pip install cryptography") from exc
    cert_path, key_path = folder / "phone-cert.pem", folder / "phone-key.pem"
    addresses = [ipaddress.ip_address(host) for host in hosts]
    now = dt.datetime.now(dt.timezone.utc)
    try:
        existing = x509.load_pem_x509_certificate(cert_path.read_bytes())
        covered = existing.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress)
        if (key_path.is_file() and existing.not_valid_after_utc - now > dt.timedelta(days=7)
                and all(address in covered for address in addresses)):
            return cert_path, key_path
    except (OSError, ValueError, x509.ExtensionNotFound):
        pass  # Missing, unreadable or outdated: make a new one.
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "BITL phone camera")])
    alt_names = [x509.DNSName("localhost"), *(x509.IPAddress(address) for address in addresses)]
    certificate = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=800))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256()))
    folder.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    if sys.platform != "win32":
        key_path.chmod(0o600)
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    LOGGER.info("Created the phone camera certificate in %s", folder)
    return cert_path, key_path


@cache
def page_bytes() -> bytes:
    """The phone page, read once."""
    return PAGE.read_bytes()


class PhoneCamera:
    """The phone as a frame source, with the same main-thread API as CameraStream.

    HTTP threads call the underscore methods; the main loop calls read(),
    publish(), commands() and pointer_events(). One phone streams at a time:
    a second one takes over when it asks to (its first frame) or when the
    first went quiet, and the first is told so instead of fighting back.
    """

    def __init__(self, config: Config, cert_folder: Path | None = None,
                 clock: Callable[[], float] = time.perf_counter, tls: bool = True) -> None:
        """Nothing listens until start(). ``tls`` False serves plain HTTP: only for
        tests on this computer (localhost counts as secure in browsers)."""
        self.config = config
        self._clock = clock
        self._cert_folder = cert_folder or settings_dir() / "phone"
        self.token = secrets.token_urlsafe(9)
        self.addresses = lan_addresses()
        self.port = 0
        self.tls = tls
        self.error: str | None = None
        self._cond = threading.Condition()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._packet: FramePacket | None = None
        self._published = 0
        self._consumed = 0
        self._client: str | None = None
        self._last_frame_at: float | None = None
        self._view: np.ndarray | None = None
        self._view_version = 0
        self._view_state = "{}"
        self._view_jpeg: tuple[int, bytes] = (0, b"")
        self._viewers = 0  # Connected pages showing the finished picture.
        self._encoder: threading.Thread | None = None
        self._commands: deque[PhoneCommand] = deque(maxlen=64)
        self._pointer: deque[PointerEvent] = deque(maxlen=256)
        self.fps = 0.0
        self.size: tuple[int, int] | None = None
        self.reconnects = 0  # Phones that connected after another, or after going quiet.

    # Lifecycle ---------------------------------------------------------------
    @property
    def url(self) -> str:
        """The link in the QR code, including the key."""
        return f"{'https' if self.tls else 'http'}://{self.addresses[0]}:{self.port}/?k={self.token}"

    @property
    def running(self) -> bool:
        """Whether the page is being served."""
        return self._server is not None

    def start(self) -> PhoneCamera:
        """Serve the page; a failure becomes ``error`` for the pairing screen, never an exception."""
        if self._server is not None:
            return self
        try:
            context = None
            if self.tls:
                cert, key = ensure_certificate(self._cert_folder, self.addresses)
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(cert, key)
            self._server = self._bind(context)
        except (AppError, OSError, ssl.SSLError, ValueError) as exc:
            self.error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            LOGGER.error("Phone camera could not start: %s", exc)
            return self
        self.error = None
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": .25},
                                        name="phone-server", daemon=True)
        self._thread.start()
        self._encoder = threading.Thread(target=self._encode_views, name="phone-view", daemon=True)
        self._encoder.start()
        LOGGER.info("Phone camera page: %s", self.url)
        return self

    def _bind(self, context: ssl.SSLContext | None) -> _Server:
        first = self.config.phone_port
        ports = [first] if first == 0 else range(first, first + PORT_ATTEMPTS)
        error: OSError | None = None
        for port in ports:
            try:
                return _Server((self.config.phone_host, port), self, context)
            except OSError as exc:
                error = exc
        raise OSError(f"ports {first}-{first + PORT_ATTEMPTS - 1} are all in use ({error})")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop serving. Open phone connections are told 'switched off' and closed."""
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        with self._cond:
            self._cond.notify_all()
        for thread in (self._thread, self._encoder):
            if thread is not None:
                thread.join(timeout)
        self._thread = self._encoder = None

    def __enter__(self) -> PhoneCamera:
        """Start on entering a with block."""
        return self.start()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        """Always stop."""
        self.stop()

    # Main-thread API -----------------------------------------------------------
    @property
    def state(self) -> CameraState:
        """STARTING until a phone streams, LIVE while frames arrive, LOST after silence or an error."""
        if self.error is not None:
            return CameraState.LOST
        if self._last_frame_at is None:
            return CameraState.STARTING
        if self._clock() - self._last_frame_at > self.config.phone_timeout_s:
            return CameraState.LOST
        return CameraState.LIVE

    @property
    def message(self) -> str:
        """Human-readable status for the pairing screen."""
        state = self.state
        if self.error is not None:
            return f"Phone camera unavailable: {self.error}"
        if state is CameraState.STARTING:
            return "Waiting for your phone"
        if state is CameraState.LOST:
            return "Phone disconnected - open the page on your phone again"
        width, height = self.size or (0, 0)
        return f"Phone connected ({width}x{height})"

    def read(self, timeout: float = 0.1) -> FramePacket | None:
        """The newest unseen phone frame, waiting up to ``timeout`` seconds; None if none arrived."""
        with self._cond:
            self._cond.wait_for(lambda: self._packet is not None and self._packet.index > self._consumed,
                                timeout)
            packet = self._packet
            if packet is None or packet.index <= self._consumed:
                return None
            self._consumed = packet.index
            self._packet = None  # The HTTP threads never touch a published frame again.
            return packet

    def publish(self, frame: np.ndarray, state: dict[str, object]) -> None:
        """Offer the finished frame (not modified afterwards) and app state to the phone.

        The view thread encodes it, and only while a page shows pictures."""
        text = json.dumps({"t": "state", **state}, separators=(",", ":"))
        with self._cond:
            self._view, self._view_state = frame, text
            self._view_version += 1
            self._cond.notify_all()

    def commands(self) -> list[PhoneCommand]:
        """Buttons pressed since the last call, oldest first."""
        with self._cond:
            items = list(self._commands)
            self._commands.clear()
        return items

    def pointer_events(self) -> list[PointerEvent]:
        """Touches since the last call, oldest first."""
        with self._cond:
            items = list(self._pointer)
            self._pointer.clear()
        return items

    # Connection threads ------------------------------------------------------------
    def _claim(self, client: str, claim: bool) -> bool:
        """May this page stream (and press buttons)? A new page asks (claim) and wins;
        an old page that did not ask only gets the camera back once it went quiet."""
        with self._cond:
            if client == self._client:
                return True
            now = self._clock()
            quiet = self._last_frame_at is None or now - self._last_frame_at > self.config.phone_timeout_s
            if not (claim or quiet):
                return False
            if self._client is not None:
                self.reconnects += 1
            LOGGER.info("Phone %s is now the camera", client[:8])
            self._client = client
            return True

    def _owns(self, client: str) -> bool:
        with self._cond:
            return client == self._client

    def _store(self, client: str, frame: np.ndarray) -> bool:
        """Publish a decoded camera frame; False if another phone took over meanwhile."""
        if not frame.size or max(frame.shape[:2]) > self.config.phone_max_side:
            return True  # Ignore nonsense, keep the connection.
        with self._cond:
            if client != self._client:
                return False
            now = self._clock()
            if self._last_frame_at is not None and now > self._last_frame_at:
                instant = 1.0 / (now - self._last_frame_at)
                self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant
            self._last_frame_at = now
            self._published += 1
            self._packet = FramePacket(frame, self._published, now)
            self.size = (frame.shape[1], frame.shape[0])
            self._cond.notify_all()
        return True

    def _view_after(self, seen: int) -> tuple[int, bytes | None, str]:
        """(version, newest encoded picture or None if nothing newer than ``seen``, state JSON)."""
        with self._cond:
            version, cached = self._view_jpeg
            state = self._view_state
        return (version, cached, state) if version > seen else (seen, None, state)

    def _encode_views(self) -> None:
        """Encode each new finished picture once, off the connection threads: there,
        encoding stalled reading, so frames bunched up and older ones were skipped."""
        def ready() -> bool:
            return not self.running or (self._viewers > 0 and self._view_version > self._view_jpeg[0])

        while self.running:
            with self._cond:
                self._cond.wait_for(ready, 0.5)
                if not self.running:
                    return
                if not (self._viewers > 0 and self._view_version > self._view_jpeg[0]):
                    continue
                version, frame = self._view_version, self._view
            scale = VIEW_MAX_SIDE / max(frame.shape[:2])
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.phone_view_quality])
            if ok:
                with self._cond:
                    self._view_jpeg = (version, encoded.tobytes())

    def _watch(self, session: _Session, watching: bool) -> None:
        """Count pages that show pictures, so none are encoded for nobody."""
        if watching != session.watching:
            session.watching = watching
            with self._cond:
                self._viewers += 1 if watching else -1
                self._cond.notify_all()

    def _command(self, client: str, payload: dict) -> bool:
        name, value = payload.get("name"), payload.get("value")
        if name not in COMMANDS or not (value is None or isinstance(value, str) and len(value) <= 32):
            return False
        with self._cond:
            if client != self._client:
                return False
            self._commands.append(PhoneCommand(name, value))
        return True

    def _touch(self, client: str, payload: dict) -> bool:
        x, y, down = payload.get("x"), payload.get("y"), payload.get("down")
        numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y))
        if not numbers or not isinstance(down, bool):
            return False
        with self._cond:
            if client != self._client:
                return False
            self._pointer.append(PointerEvent(min(1., max(0., x)), min(1., max(0., y)), down))
        return True

    def serve(self, ws: WebSocket, key: str) -> None:
        """One phone's connection, on its own thread, until it closes.

        Incoming: a hello (who is this page), the video codec in use, video
        messages, buttons, touches and acks of shown pictures. Outgoing: the
        finished picture (one at a time: the next waits for the ack, so a slow
        network shows fewer pictures instead of older ones) and app state."""
        if not hmac.compare_digest(key.encode(), self.token.encode()):
            self._send_json(ws, {"t": "error", "code": 401})
            ws.close(4401)
            return
        session = _Session(heard=self._clock())
        try:
            self._converse(ws, session)
        finally:
            self._watch(session, False)

    def _converse(self, ws: WebSocket, session: _Session) -> None:
        while self.running and not ws.closed:
            messages = ws.poll(SESSION_POLL_S)
            video: list[bytes] = []
            for opcode, data in messages:
                if opcode == BINARY:
                    video.append(data)
                    continue
                try:
                    message = json.loads(data)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._handle(ws, session, message)
            if video and session.client is not None:
                self._decode(ws, session, video)
            now = self._clock()
            if messages:
                session.heard = now
            elif now - session.heard > SESSION_IDLE_S:
                break
            self._watch(session, session.client is not None and session.preview)
            if session.watching and not session.view_pending:
                version, jpeg, state = self._view_after(session.seen)
                if jpeg is not None:
                    ws.send(BINARY, jpeg)
                    self._send_text(ws, state)
                    session.seen, session.view_pending = version, True
                    session.state_at = now
            if session.client is not None and now - session.state_at > STATE_EVERY_S:
                with self._cond:
                    state = self._view_state
                self._send_text(ws, state)
                session.state_at = now
        if self.running:
            return
        self._send_json(ws, {"t": "error", "code": 503})
        ws.close(4503)

    def _handle(self, ws: WebSocket, session: _Session, message: dict) -> None:
        kind = message.get("t")
        if kind == "hello":
            client = str(message.get("client", ""))[:64]
            session.preview = bool(message.get("preview", True))
            if not client or not self._claim(client, bool(message.get("claim"))):
                self._send_json(ws, {"t": "error", "code": 409})
                return
            session.client = client
            self._send_json(ws, {"t": "welcome", "codecs": list(available_codecs())})
        elif session.client is None:
            return
        elif kind == "config":
            codec = message.get("codec")
            session.codec = codec if codec in available_codecs() else "jpeg"
            session.decoder = None  # A new stream starts with a key frame.
        elif kind == "ack":
            session.view_pending = False
        elif kind == "preview":
            session.preview, session.seen = bool(message.get("on")), 0
        elif kind == "command":
            self._command(session.client, message)
        elif kind == "pointer":
            self._touch(session.client, message)

    def _decode(self, ws: WebSocket, session: _Session, video: list[bytes]) -> None:
        """Decode what arrived and publish only the newest picture."""
        if not self._owns(session.client):
            self._send_json(ws, {"t": "error", "code": 409})
            session.client = None
            return
        frame = None
        if session.codec == "h264":
            if session.decoder is None:
                session.decoder = H264Decoder()
            for data in video:  # Every packet: later ones are differences from earlier ones.
                try:
                    decoded = session.decoder.decode(data)
                except ValueError as exc:
                    LOGGER.debug("H.264 decode error, asking for a key frame: %s", exc)
                    session.decoder = None
                    self._send_json(ws, {"t": "keyframe"})
                    break
                if decoded is not None:
                    frame = decoded
            if frame is not None:  # Only the newest decoded picture is converted.
                frame = H264Decoder.to_bgr(frame)
        else:
            frame = cv2.imdecode(np.frombuffer(video[-1], np.uint8), cv2.IMREAD_COLOR)
        if frame is not None and not self._store(session.client, frame):
            self._send_json(ws, {"t": "error", "code": 409})
            session.client = None

    @staticmethod
    def _send_json(ws: WebSocket, message: dict) -> None:
        PhoneCamera._send_text(ws, json.dumps(message, separators=(",", ":")))

    @staticmethod
    def _send_text(ws: WebSocket, text: str) -> None:
        ws.send(TEXT, text.encode("utf-8"))


@dataclass
class _Session:
    """Per-connection state of one phone page."""

    client: str | None = None
    codec: str = "jpeg"
    decoder: H264Decoder | None = None
    preview: bool = True
    seen: int = 0  # Newest finished picture sent.
    view_pending: bool = False  # Sent, not yet acknowledged as shown.
    watching: bool = False  # Counted in PhoneCamera._viewers.
    heard: float = 0.0  # When the page last sent anything.
    state_at: float = 0.0


@cache
def available_codecs() -> tuple[str, ...]:
    """H.264 needs PyAV (FFmpeg); JPEG always works."""
    return ("h264", "jpeg") if importlib.util.find_spec("av") is not None else ("jpeg",)


class H264Decoder:
    """Annex-B H.264 from the phone's hardware encoder (WebCodecs) -> BGR frames.

    About 5 Mbps instead of 30 for the same 720p60 picture as JPEG: what lets
    60 FPS fit through ordinary Wi-Fi. Decoding takes about 1-2 ms a frame."""

    def __init__(self) -> None:
        """A fresh decoder waits for a key frame (the phone sends one on 'config')."""
        import av
        self._av = av
        self._codec = av.CodecContext.create("h264", "r")

    def decode(self, data: bytes) -> object | None:
        """Feed one encoded chunk; the newest decoded picture (still YUV), or None.
        Raises ValueError on corrupt data."""
        try:
            frames = self._codec.decode(self._av.Packet(data))
        except self._av.error.FFmpegError as exc:
            raise ValueError(str(exc)) from exc
        return frames[-1] if frames else None

    @staticmethod
    def to_bgr(picture: object) -> np.ndarray:
        """Convert a decoded picture for OpenCV (about 1 ms at 720p)."""
        return picture.to_ndarray(format="bgr24")


class _Server(ThreadingHTTPServer):
    """HTTPS with the TLS handshake on each connection's own thread."""

    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is in use.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address: tuple[str, int], phone: PhoneCamera, context: ssl.SSLContext | None) -> None:
        """Bind immediately so a busy port fails here."""
        self.phone, self.context = phone, context
        super().__init__(address, _Handler)

    def server_bind(self) -> None:
        """Skip HTTPServer's reverse DNS lookup of 0.0.0.0, which can take seconds on Windows."""
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "bitl", self.server_address[1]

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        """Wrap without handshaking: a slow phone must not stall accept() for everyone."""
        sock, address = self.socket.accept()
        if self.context is None:
            return sock, address
        return self.context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), address

    def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
        """A phone rejecting the certificate or walking away is normal: no traceback."""
        LOGGER.debug("Phone connection from %s ended: %s", client_address[0], sys.exc_info()[1])


class _Handler(BaseHTTPRequestHandler):
    """GET / is the page; GET /ws upgrades to the phone's WebSocket."""

    protocol_version = "HTTP/1.1"
    server_version = "BITL"
    sys_version = ""
    timeout = 15  # Seconds an idle connection may stay open (and a dead phone may hang a read).
    server: _Server

    def setup(self) -> None:
        """Handshake under the timeout before reading the request."""
        self.request.settimeout(self.timeout)
        if isinstance(self.request, ssl.SSLSocket):
            self.request.do_handshake()
        super().setup()

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Requests are not worth a log line each."""

    def log_message(self, format: str, *args: object) -> None:
        """Route http.server's messages to the debug log instead of stderr."""
        LOGGER.debug("Phone %s: " + format, self.client_address[0], *args)

    def do_GET(self) -> None:
        """The page (it carries no secrets; the key arrives in its link), or the WebSocket."""
        url = urlsplit(self.path)
        if url.path == "/ws":
            self._websocket(parse_qs(url.query).get("k", [""])[0])
        elif url.path in ("/", "/index.html"):
            self._reply(200, page_bytes(), "text/html; charset=utf-8")
        else:
            self._reply(404, b"Not found")

    def _websocket(self, key: str) -> None:
        if (self.headers.get("Upgrade", "").lower() != "websocket"
                or not self.headers.get("Sec-WebSocket-Key")):
            self._reply(400, b"WebSocket expected")
            return
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key(self.headers["Sec-WebSocket-Key"]))
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True
        phone = self.server.phone
        ws = WebSocket(self.connection, phone.config.phone_max_upload_bytes)
        try:
            phone.serve(ws, key)
        except Closed as exc:
            LOGGER.debug("Phone %s disconnected: %s", self.client_address[0], exc)
        finally:
            ws.close()

    def _reply(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
