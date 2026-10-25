# BITL portfolio notes

## Resume bullets

- Built a modular Python webcam AR application using MediaPipe Tasks, OpenCV and NumPy, with two-hand tracking, neon drawing and four gesture-controlled modes.
- Designed scale-aware gesture recognition with stabilization, hysteresis, timed holds and an event bus, connecting block physics and procedural anime-inspired effects to hand motion.
- Added latest-frame threaded capture, adaptive quality, stage profiling, optional synthesized audio, time-resampled video recording and automated regression tests for interaction and failure recovery.

Avoid claiming universal 30 FPS or production deployment. The measured webcam rate on the development laptop was approximately 15–20 FPS at 720p (camera-limited; processing about 28 ms per frame). Keep the test count aligned with the latest saved test output.

## 60-second demo script

Prepare the empty background beforehand (B, step out, return), choose a quiet well-lit room, and use an external screen recorder if sound is needed. Disable first-run cards before recording.

- **0–8 s:** Draw mode. Point and draw “BITL”; make peace to change color, draw an underline.
- **8–16 s:** Press 3. Pinch, move and drop two blocks; point to crack one and punch the other.
- **16–24 s:** Press 2. Hold an open palm and flick a fireball; hold OK and throw Rasenshuriken.
- **24–31 s:** Hold both fists to raise Susanoo; pause so the silhouette is visible.
- **31–41 s:** Press 4. Point with one hand and peace with the other, bring Blue and Red together, then flick Hollow Purple.
- **41–49 s:** Cross fingers for Unlimited Void. Briefly hold the pose, then prayer for Malevolent Shrine.
- **49–56 s:** Press 1. Snap to dust. Pause for the crumble and return with N if the second snap is awkward.
- **56–60 s:** Tab to show the settings controls, then ? from the camera view to show the illustrated guide.

Practice holds before the timed take. Use the keyboard mode shortcuts for a reliable one-minute demonstration; hand-only mode switching is available separately.

## Reviewer questions and answers

**Why One Euro instead of a moving average?**

A fixed moving average smooths small tracking errors but also delays intentional fast movement. One Euro adapts its cutoff to estimated speed: it suppresses jitter during pauses and responds more quickly during a stroke. The filters use real timestamps and reset after tracking loss, avoiding a streak from an old fingertip position.

**Why stabilization and hysteresis?**

Landmarks fluctuate near pose boundaries. Stabilization requires a pose to remain consistent before becoming active. Hysteresis uses different entry and exit thresholds so a pinch does not chatter around one cutoff. Timed holds then operate on stable events, and a fresh gesture is required after a focused overlay closes.

**Why a latest-frame capture thread?**

A camera read can block, making the UI unresponsive. Moving it off the UI thread lets the window continue handling input. A queue can accumulate stale images when rendering slows; a single-slot buffer intentionally drops them so the next processed frame is current. Unexpected worker errors surface on the main thread, and connection failures trigger retry logic.

**How does recording stay close to real time?**

Video files use a fixed playback rate while the live app's throughput varies. The writer starts at measured FPS, then schedules frame repeats or skips against elapsed time. A background encoder protects the UI. The bounded queue prioritizes responsiveness, and very long stalls are bounded, so the implementation is suited to short demos rather than lossless archival capture.

**How do you keep effects extensible?**

Tracking produces data, gesture recognition produces events, and modes attach controllers that translate events into effect lifecycles. Effects emit named hooks for optional sound without importing the audio system. The menu suspends gameplay subscriptions to own input explicitly. Compositing, particles, sprites and timing are shared services.

**What remains hard?**

Real hands vary in pose, occlusion and lighting; synthetic tests cannot replace user testing. Segmentation struggles with similar foreground/background colors, and unseen background must be estimated for dust. Full-frame effects have higher costs than small palm effects, so hardware-specific profiling and adaptive quality are necessary. The software reports measured performance rather than assuming the camera's requested rate.
