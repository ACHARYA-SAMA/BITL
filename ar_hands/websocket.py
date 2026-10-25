"""A small server-side WebSocket (RFC 6455) for the phone page, on one thread per phone.

Why WebSocket: frames stream in one direction and finished pictures in the
other without waiting for each other. With one HTTP request per frame, every
frame waited a full network round trip and the phone managed 4-10 FPS.

Why not a library: the phone needs only binary and text messages, ping and
close. Reading and writing happen on the connection's own thread, never
concurrently, because an SSL socket must not be used from two threads at once:
``poll()`` waits a few milliseconds for incoming data, then the caller sends.
"""
from __future__ import annotations
import base64
import hashlib
import select
import socket
import ssl
import numpy as np

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CONTINUATION, TEXT, BINARY, CLOSE, PING, PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
READ_CHUNK = 256 * 1024


class Closed(Exception):
    """The peer closed the connection (or broke the protocol)."""


def accept_key(client_key: str) -> str:
    """The Sec-WebSocket-Accept answer to a client's Sec-WebSocket-Key."""
    digest = hashlib.sha1((client_key.strip() + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def unmask(payload: bytes | bytearray, mask: bytes) -> bytes:
    """Clients XOR every payload with a 4-byte key; NumPy does 100 KB in microseconds."""
    data = np.frombuffer(bytes(payload), np.uint8)
    key = np.frombuffer(mask * (len(data) // 4 + 1), np.uint8)[:len(data)]
    return (data ^ key).tobytes()


def frame(opcode: int, payload: bytes) -> bytes:
    """One unmasked, unfragmented server frame."""
    length = len(payload)
    if length < 126:
        header = bytes((0x80 | opcode, length))
    elif length < 1 << 16:
        header = bytes((0x80 | opcode, 126)) + length.to_bytes(2, "big")
    else:
        header = bytes((0x80 | opcode, 127)) + length.to_bytes(8, "big")
    return header + payload


def client_frame(opcode: int, payload: bytes, mask: bytes) -> bytes:
    """One masked frame as a browser sends it (for the self-test and tests)."""
    length = len(payload)
    if length < 126:
        header = bytes((0x80 | opcode, 0x80 | length))
    elif length < 1 << 16:
        header = bytes((0x80 | opcode, 0x80 | 126)) + length.to_bytes(2, "big")
    else:
        header = bytes((0x80 | opcode, 0x80 | 127)) + length.to_bytes(8, "big")
    return header + mask + unmask(payload, mask)


class WebSocket:
    """Messages over an accepted (already upgraded) socket."""

    def __init__(self, sock: socket.socket, max_message: int) -> None:
        """``max_message`` bytes is the largest message accepted (a camera frame)."""
        self.sock, self.max_message = sock, max_message
        self._buffer = bytearray()
        self._partial: tuple[int, bytearray] | None = None
        self.closed = False

    def poll(self, timeout: float) -> list[tuple[int, bytes]]:
        """Complete (opcode, payload) messages received within ``timeout`` seconds.

        Pings are answered here; a close is answered and raises Closed."""
        pending = isinstance(self.sock, ssl.SSLSocket) and self.sock.pending() > 0
        if pending or select.select([self.sock], [], [], timeout)[0]:
            try:
                data = self.sock.recv(READ_CHUNK)
            except (TimeoutError, ssl.SSLWantReadError):
                return []
            except OSError as exc:
                raise Closed(str(exc)) from exc
            if not data:
                raise Closed("connection closed")
            self._buffer += data
        return self._parse()

    def _parse(self) -> list[tuple[int, bytes]]:
        messages: list[tuple[int, bytes]] = []
        buffer = self._buffer
        while len(buffer) >= 2:
            fin, opcode = buffer[0] & 0x80, buffer[0] & 0x0F
            masked, length = buffer[1] & 0x80, buffer[1] & 0x7F
            offset = 2
            if length == 126:
                if len(buffer) < 4:
                    break
                length, offset = int.from_bytes(buffer[2:4], "big"), 4
            elif length == 127:
                if len(buffer) < 10:
                    break
                length, offset = int.from_bytes(buffer[2:10], "big"), 10
            if length > self.max_message:
                self.close(1009)
                raise Closed(f"message of {length} bytes is too large")
            mask = b""
            if masked:
                if len(buffer) < offset + 4:
                    break
                mask, offset = bytes(buffer[offset:offset + 4]), offset + 4
            if len(buffer) < offset + length:
                break
            payload = bytes(buffer[offset:offset + length])
            del buffer[:offset + length]
            if mask:
                payload = unmask(payload, mask)
            if opcode == PING:
                self.send(PONG, payload)
            elif opcode == CLOSE:
                self.close(1000)
                raise Closed("closed by the phone")
            elif opcode in (TEXT, BINARY):
                if fin:
                    messages.append((opcode, payload))
                else:
                    self._partial = (opcode, bytearray(payload))
            elif opcode == CONTINUATION and self._partial is not None:
                self._partial[1].extend(payload)
                if len(self._partial[1]) > self.max_message:
                    self.close(1009)
                    raise Closed("fragmented message is too large")
                if fin:
                    messages.append((self._partial[0], bytes(self._partial[1])))
                    self._partial = None
        return messages

    def send(self, opcode: int, payload: bytes) -> None:
        """Send one message; raises Closed if the phone is gone."""
        if self.closed:
            raise Closed("already closed")
        try:
            self.sock.sendall(frame(opcode, payload))
        except OSError as exc:
            self.closed = True
            raise Closed(str(exc)) from exc

    def close(self, code: int = 1000) -> None:
        """Send a close frame once (best effort)."""
        if self.closed:
            return
        try:
            self.sock.sendall(frame(CLOSE, code.to_bytes(2, "big")))
        except OSError:
            pass
        self.closed = True
