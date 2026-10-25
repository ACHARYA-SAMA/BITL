"""Encode and decode synthetic frames to verify real codec playback timing."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_hands.recorder import Recorder


def main() -> None:
    """Save a two-second synthetic codec check; never opens a camera."""
    folder = Path(__file__).resolve().parents[1] / 'verification' / 'phase5'
    recorder = Recorder(folder)
    path = recorder.start((320, 180), 20, 0.)
    # Input throughput deliberately varies; the output still follows elapsed time.
    for timestamp in (0., .05, .10, .30, .70, 1., 1.05, 1.5, 1.9, 2.):
        frame = np.full((180, 320, 3), (40, 60, int(timestamp * 100)), np.uint8)
        cv2.putText(frame, f'Synthetic {timestamp:.2f}s', (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        recorder.write(frame, timestamp)
    recorder.stop()
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while cap.read()[0]:
            count += 1
        assert count == 41, (count, 'expected 41 wall-clock-resampled frames')
        assert abs(fps - 20) < .1, fps
    finally:
        cap.release()
    report = {'codec': recorder.codec, 'frames_decoded': count, 'fps': fps,
              'duration_seconds': count / fps, 'source_span_seconds': 2., 'file': path.name}
    (folder / 'recording-check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
