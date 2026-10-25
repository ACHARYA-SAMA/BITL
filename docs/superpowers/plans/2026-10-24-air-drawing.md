# Air Drawing Implementation Plan

> Execute with the executing-plans workflow, continuing through verification.

**Goal:** Complete the supplied Phase 2 specification while preserving Phase 1.
**Architecture:** Split the existing implementation into the requested ar_hands
package. PenController owns per-hand filters and debounce state; AirCanvas owns
stroke history and cached neon rendering. App coordinates these independent parts.
**Tech Stack:** Existing Python 3.12 environment, OpenCV, MediaPipe Tasks, NumPy.
**Spec:** User attachment a9e87b81-6271-49e6-8d38-5bfcf16f0930/Pasted text.txt.

## Global constraints
- Python 3.10-3.12; no new third-party dependencies or legacy MediaPipe API.
- Preserve Phase 1 keys and camera/tracker cleanup behavior.
- Implement Phase 2 only. Keep is_pen_down(hand) as the single Phase 3 decision hook.
- Output every application file, including the small entry point.

## Tasks
- [x] 1. Refactor Config, camera, tracker, skeleton, HUD and App into ar_hands;
  migrate the 15 existing regression tests and keep them passing.
- [x] 2. Write failing smoothing/canvas tests, then implement timestamp-driven
  OneEuroFilter and independent x/y pair; Stroke/AirCanvas with spline sampling,
  half-resolution halo, cached layers, history, fade, and BGRA export.
- [x] 3. Write failing pen tests; implement wrist-distance pointing detection,
  3/4-frame debounce, per-handedness state, missing-hand grace and immediate
  filter reset, with no reconnecting line across a disappearance.
- [x] 4. Integrate cursor feedback, palette/thickness controls, drawing HUD,
  transparent export, and skeleton visibility into App; run control tests.
- [x] 5. Run complete regression tests, actual model/camera smoke checks,
  render/export visual inspection, and 720p drawing benchmarks; request independent
  code review and fix consequential findings. Update README and full source listing.

## Review focus
- Loss/re-entry before and after grace must not bridge strokes (pen regression).
- Simultaneous hands and duplicate labels must not cross-connect (pen regression).
- Undo/redo after new drawing and clear must have predictable history (canvas test).
- Fade must not erase an actively held pen or incorrectly export opaque black
  pixels; timing and PNG alpha are tested (canvas regression).
- Resize, drawing-free frames, export failures, and original cleanup must remain
  safe (canvas/control/regression tests).

## Execution decisions and progress
- The latest attachment supersedes the interrupted all-five-phases interpretation.
- Continue implementation locally without another approval pause, as requested.
- This folder is not a Git repository; preserve the existing working files and
  use tests and this ledger instead of branch/commit-oriented skill helpers.
- Interface check: pen consumes HandData and writes AirCanvas; App consumes
  Cursor feedback and a BGR canvas layer; HUD receives brush state. No circular
  runtime imports: Config and HandData live in their dedicated modules.

Final verification: 30/30 tests pass; actual camera/model and q exit verified.
Independent review: duplicate-label confidence switching and crossing brightness
changes were reproduced RED, fixed, then verified GREEN with the full suite.
Ruling: ambiguous labels pause drawing rather than risk crossing hand identities.
Physical macOS/Linux testing remains unavailable; measured camera speed is 10 FPS.
