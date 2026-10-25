"""Serve the phone page without the full app, to test it in a desktop browser.

    python tools/phone_harness.py [port] [seconds]

Plain HTTP on 127.0.0.1 (browsers treat localhost as secure, so the camera and
WebCodecs work), key "test": open http://localhost:8790/?k=test. Received
frames are echoed back as the finished picture with a frame counter drawn on
them, and receive statistics are written to logs/phone_harness.json.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402
from ar_hands.config import Config  # noqa: E402
from ar_hands.phone import PhoneCamera  # noqa: E402


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 300
    out = Path(__file__).resolve().parent.parent / "logs" / "phone_harness.json"
    out.parent.mkdir(exist_ok=True)
    phone = PhoneCamera(replace(Config(), phone_host="127.0.0.1", phone_port=port), tls=False)
    phone.token = "test"
    phone.start()
    print(f"Open http://localhost:{phone.port}/?k=test", flush=True)
    frames, started, sizes = 0, time.perf_counter(), set()
    window_frames, window_start = 0, time.perf_counter()
    rate = 0.0
    try:
        while time.perf_counter() - started < seconds:
            packet = phone.read(0.5)
            now = time.perf_counter()
            if packet is not None:
                frames += 1
                window_frames += 1
                sizes.add(f"{packet.frame.shape[1]}x{packet.frame.shape[0]}")
                view = packet.frame.copy()
                cv2.putText(view, f"#{frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
                phone.publish(view, {"mode": "Draw", "recording": False, "muted": False, "audio": True,
                                     "overlay": "", "color": "#00ffff", "fps": round(rate)})
            if now - window_start >= 1.0:
                rate = window_frames / (now - window_start)
                window_frames, window_start = 0, now
                out.write_text(json.dumps({"frames": frames, "fps": round(rate, 1), "sizes": sorted(sizes),
                                           "state": phone.state.value}), encoding="utf-8")
    finally:
        phone.stop()


if __name__ == "__main__":
    main()
