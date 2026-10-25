"""Ten-second local hardware smoke check. Does not save camera images or video."""
from pathlib import Path
import sys
import json
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_hands.app import App
from ar_hands.settings import Settings
from ar_hands.effects.manager import Mode
from ar_hands.logging_setup import setup_logging


class SmokeApp(App):
    """Exercise the real UI loop with a bounded runtime and no persistent settings."""

    def __init__(self):
        super().__init__(settings=Settings(onboarded=True))
        self.started = time.perf_counter()
        self.samples = 0

    def process_frame(self, frame, tracker):
        self.samples += 1
        self.switch_mode(list(Mode)[min(3, self.samples // 25)])
        return super().process_frame(frame, tracker)

    def handle_key(self, key, frame):
        return time.perf_counter() - self.started < 10 and super().handle_key(key, frame)


if __name__ == '__main__':
    app = SmokeApp()
    setup_logging(app.config.output_dir / 'logs')
    app.run()
    report = {'frames_processed': app.samples, 'measured_fps': round(app.fps.value, 2),
              'elapsed_seconds': round(time.perf_counter() - app.started, 2),
              'stage_averages_ms': app.profiler.average(), 'captured_media_saved': False}
    path = app.config.output_dir / 'verification' / 'phase5' / 'live-smoke.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if app.samples > 0 else 1)
