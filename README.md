# BITL — Bring Ideas to Life

Turn your webcam into a gesture-controlled AR playground: draw neon trails, build blocks, cast jutsu and open cinematic domains.

## Download the Windows app

No Python needed. Get `BITL-v1.0.0-windows-x64.zip` from the [latest release](../../releases/latest), unzip it anywhere, open the `BITL` folder and double-click **BITL.exe**. Keep the folder together; the exe needs the `_internal` folder beside it. Windows SmartScreen may warn because the exe is unsigned: choose **More info → Run anyway**.

## Start here

From source on Windows, double-click **Start BITL.cmd**. Or run from this folder:

```powershell
.\.venv\Scripts\python.exe main.py
```

Press **?** for the illustrated gesture guide or **Tab** for settings. Hold both open palms for one second to open settings with your hands. Menu and guide capture gestures, so you cannot accidentally draw or cast behind them. Release an existing pinch before clicking a newly opened menu.

There are four modes: **1 Draw**, **2 Jutsu**, **3 Blocks**, **4 Jujutsu**. Hold the rock sign (index and pinky extended) for one second to cycle modes.

## Use your phone as the camera

Laptop webcams are often dim, grainy and stuck near 15 FPS. Almost any phone films better, and BITL can use it without installing anything on the phone.

1. Press **O** (or pick **Camera: Phone** in Tab settings, or start with `--phone`). A pairing screen with a QR code appears.
2. Put the phone on the same Wi-Fi as the computer. A phone hotspot works too: connect the computer to it.
3. Scan the QR code with the phone's camera app and open the link.
4. The phone warns that the connection is not private. That warning is about BITL's own certificate, which the computer made for itself: tap **Advanced → Proceed** (iPhone: **Show Details → visit this website**). Phones only allow camera access on secure pages, which is why the page uses HTTPS. You accept the warning once per computer.
5. Choose the selfie or back camera, tap **Start camera** and allow camera access.

The phone sends its video to the computer, which does the tracking and effects as usual. The finished AR picture is shown on the computer **and** on the phone, so the phone works as a remote too. The phone page lets you:

- switch modes, change color, undo, redo, clear, save, snap, record and mute;
- open the gesture guide or the settings menu and tap them directly on the phone screen;
- tap the picture to hide the controls, switch between selfie and back camera, and turn the AR picture off to save battery (the computer keeps processing).

The phone page works in portrait and landscape. Hold the phone sideways and the laptop switches to a landscape picture in fullscreen; turn it upright and the window goes back. This works even when the phone's rotation lock is on: the page reads the phone's motion sensor and turns the picture itself (iPhone asks for motion access on **Start**). If the picture is ever on its side, tap the rotate button. In portrait, BITL's own menu, guide and HUD switch to tall layouts so they stay readable on the phone. Sound still plays on the computer.

The phone streams at up to 60 FPS. Its own hardware encodes the video as H.264, about 5 Mbps at 720p, which ordinary Wi-Fi carries easily; browsers without that support send JPEG instead. When the network falls behind, the phone skips frames rather than letting a delay build up, so the picture stays live. The status pill on the phone shows frames per second and the codec in use. The back camera starts with the AR picture on the phone switched off (its screen faces away), which leaves more Wi-Fi for the camera.

Press **O** again to go back to the webcam. The choice is remembered. If the phone disconnects (screen locked, page closed, Wi-Fi drop), the pairing screen returns; reopen the page and it continues. Each run uses a new link, so an old tab says the link expired: scan the new code.

Prefer a phone webcam app such as IP Webcam or DroidCam? Pass its stream URL: `python main.py --camera http://192.168.1.20:8080/video`. Apps that install a virtual webcam driver on the computer (DroidCam, Iriun, Camo) also work through the normal webcam path with `--camera 1`.

## Play as a group

BITL tracks up to **six hands** by default, so three friends can play together. Each hand keeps its own id from frame to frame, and each person's two hands are paired. Everyone can draw, throw fireballs, cast Chidori, pinch blocks and summon Blue and Red at the same time. Two-hand moves belong to the person who makes them: one friend's prayer seal makes their own Great Fireball, and Blue and Red fuse into Hollow Purple only in the same person's hands. The HUD shows how many people it sees.

Stand side by side with your hands in view, about an arm's length or two from the camera. The phone camera in landscape gives everyone the most room. **Max hands** in Tab settings (2, 4, 6 or 8; `--hands N`) trades crowd size for speed: hand detection costs a little more per hand.

## What's included

- Mirrored camera tracking of up to six hands (three people) using MediaPipe Tasks, with stable per-hand ids and per-person two-hand gestures.
- Phone as the camera over Wi-Fi: QR pairing, no app install, H.264 streaming at up to 60 FPS, auto-rotation, live AR picture and touch controls on the phone.
- Light that behaves like light: fire, lightning, orbs, sparks and neon drawings bloom like a camera lens sees them and light up your face and the room in their color.
- Procedural fire with rising, flickering tongues, embers, smoke and heat haze; billowing fireball explosions with a refracting shockwave; branching, strobing lightning.
- Smoothed neon drawing, seven colors, undo/redo, fading and transparent PNG export.
- Grid-assisted blocks, stacking, grabbing, beams, point-to-break and punch-to-smash.
- Fireballs, Chidori, Rasenshuriken and a violet Susanoo behind your body.
- Snap-to-dust and reappearance with person segmentation and background learning.
- Blue attraction, Red repulsion, Hollow Purple fusion and launch.
- Gojo's space-themed Unlimited Void and Sukuna's Malevolent Shrine.
- Illustrated gesture cards; keyboard, fingertip, dwell, pinch and mouse settings controls.
- 26 synthesized sound cues, mute and master volume; silent fallback without pygame.
- MP4 recording with codec fallback and a clean composition option.
- Latest-frame threaded capture, hand detection pipelined on its own thread, 60 FPS display, camera reconnect, stage timings and adaptive quality.
- Validated per-user settings, rotating logs, headless self-test and a unittest suite.

## Installation on another computer

Use Python **3.10–3.12** and a webcam (or a phone on the same Wi-Fi). A graphical desktop is needed for the live app. The self-test does not need a webcam or a window.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[audio]"
.\.venv\Scripts\python.exe main.py --selftest
.\.venv\Scripts\python.exe main.py
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[audio]'
python main.py --selftest
python main.py
```

Use `pip install -e .` or `pip install -r requirements.txt` to omit audio. The installed console command is `ar-hands`. Models download into `models/` on first run. The hand model is required; missing person segmentation leaves domain overlays usable, but prevents snap-to-dust.

## Gesture cheat sheet

Hold gestures steadily until the progress ring fills. Keep your entire hand in view. Thresholds are scaled to hand size.

| Mode | Gesture | Result |
|---|---|---|
| All | Rock sign, hold 1 s | Next mode |
| All | Two open palms, hold 1 s | Open settings; held spells dissipate |
| Draw | Index extended, other fingers curled | Draw |
| Draw | Fist | Pen up |
| Draw | Peace | Next color |
| Draw | Thumbs-up | Save transparent PNG |
| Draw | Still open palm, hold 1 s | Clear, unless another hand is drawing |
| Draw | Open-palm swipe | Undo / redo |
| Draw | Press thumb to middle finger, then snap | Turn to dust; snap again to return |
| Jutsu | Open palm, hold 0.5 s; flick | Form and throw fireball |
| Jutsu | Fist 0.6 s, then open | Chidori |
| Jutsu | OK sign, hold 0.6 s; flick | Rasenshuriken |
| Jutsu | Both fists, hold 1 s | Toggle Susanoo |
| Jutsu | Prayer 0.8 s, then pull apart | Great Fireball |
| Blocks | Pinch empty space; release | Create and drop a block |
| Blocks | Pinch an existing block or beam end | Move it |
| Blocks | Pinch with both hands, touch, pull apart | Stretch a beam |
| Blocks | Point at a block for 0.35 s | Break it |
| Blocks | Swing a fist through a block | Smash it |
| Blocks | Peace / thumbs-up held 1 s | Cycle type / clear blocks |
| Jujutsu | Point, hold 0.4 s | Blue |
| Jujutsu | Peace, hold 0.4 s | Red |
| Jujutsu | Blue in one hand, Red in the other; bring together | Fuse Hollow Purple; flick to launch |
| Jujutsu | Cross index over middle, hold 1 s | Unlimited Void |
| Jujutsu | Prayer, hold 1 s | Malevolent Shrine |
| Jujutsu | Repeat the domain sign | Close it; otherwise it times out |

**For a convincing dust effect:** keep the camera still, press **B**, then step aside briefly so the camera can learn the empty room. Return and snap. Previously hidden background has to be estimated, so a learned empty scene looks better. **N** is a keyboard fallback for the snap. Good front lighting and a contrasting background improve the person outline.

## Controls

| Key | Action |
|---|---|
| 1 / 2 / 3 / 4 | Draw / Jutsu / Blocks / Jujutsu |
| Tab | Open / close settings |
| O | Phone camera / webcam |
| ? | Illustrated guide |
| Arrows, Enter | Navigate and adjust settings; left/right changes values |
| Left/Right or A/D in guide | Previous / next page |
| Esc | Close overlay; quit when no overlay is open |
| Q | Quit from the camera view |
| R | Start / stop recording |
| M | Mute / unmute |
| Shift+F (capital F) | Fullscreen |
| f (lowercase) | Drawing fade; preserved from earlier phases |
| F1–F7 or Shift+1–7 | Brush color |
| [ / ] | Brush thickness |
| C / U / Y | Clear / undo / redo drawing |
| P / S | Export transparent PNG / save screenshot |
| G / K / H / L | Skeleton glow / skeleton / HUD / landmark labels |
| V / F10 | Gesture debug / timing profiler |
| B / N | Relearn background / snap fallback |

In the menu, point at a control and dwell for 0.8 s or pinch to click. Pinch and move horizontally to drag sliders; sliders ignore dwell, so a resting hand never moves one. A control triggers only once per dwell; move away and back to trigger it again. A pinch click uses where the cursor was just before the fingers closed, so the closing pinch cannot slide it onto the neighbouring button. Mouse controls work through the same letterbox mapping. Disabled audio controls mean the audio device or pygame is unavailable.

The menu includes mode, color, thickness, glow, skeleton, HUD, volume, mute, sensitivity (detection confidence), fullscreen, recording, clean recording, adaptive quality, guide, reset and quit. Sensitivity changes are applied when you close the menu; a failed change keeps the working tracker.

## Recording and output

Press **R**. The blinking REC badge shows elapsed time. Press R again to finish and flush the file. MP4 (`mp4v`) is tried first, followed by XVID and MJPG AVI. The file starts at measured processing FPS and resamples frames against elapsed time when performance changes. Queue overload drops visual detail rather than blocking the UI. Very long stalls are bounded to avoid queuing minutes of repeated frames.

**Recorded video has no audio.** Use a screen recorder such as OBS with desktop audio capture when you want the sound cues included. Clean recording removes HUD, menus and gesture/profiler overlays; skeleton visibility remains your choice.

Source runs save `recordings/`, `screenshots/`, `drawings/` and `logs/` beside the project. The packaged executable saves beside itself. A wheel installed outside the source tree uses `resources/` and `output/` under its per-user ARHands directory. Models and optional sprite overrides are loaded from the resource folder. Logs rotate at 1 MB with three backups.

## Settings and command line

Settings live in `%APPDATA%\ARHands\settings.json` on Windows, `~/Library/Application Support/ARHands/settings.json` on macOS, and `$XDG_CONFIG_HOME/arhands/settings.json` (normally `~/.config/arhands/`) on Linux. A writable working-directory fallback is used if needed. Corrupt JSON is backed up as `settings.json.bak`; invalid individual values revert to defaults. CLI flags override saved settings, which override defaults.

```bash
python main.py --help
python main.py --camera 1 --width 1280 --height 720 --fps 30
python main.py --phone
python main.py --phone --phone-port 9000
python main.py --webcam
python main.py --phone --hands 8
python main.py --camera http://192.168.1.20:8080/video
python main.py --mode jujutsu --detect-scale 0.5 --no-glow
python main.py --no-audio --no-adaptive
python main.py --record --debug
python main.py --reset-settings
python main.py --selftest
python main.py --version
```

`--camera` sets the first candidate; the other configured camera indices remain fallbacks. It also accepts a stream URL. `--phone` and `--webcam` choose the camera source for this run and remember it. The phone page listens on port 8765 (`--phone-port`) and tries the next nine ports if that one is busy. `--hands` tracks up to that many hands (2-8, default 6). `--fps` requests a camera rate; the driver may grant less. `--detect-scale` accepts 0.3–1.0 (default 0.75). F10 exposes the actual bottleneck. Quality lowers glow, then particle count, then detection resolution after sustained overload; it restores detail after sustained headroom. Turn it off in settings or with `--no-adaptive`.

## How it works

The camera worker keeps a single latest frame. Old frames are discarded when the processing loop is busy, preventing a growing delay between your movement and the effect. The frame is mirrored before detection, converted from BGR to RGB, and passed to the modern MediaPipe Tasks HandLandmarker. Its 21 normalized landmarks map back to full webcam resolution even when detection uses a smaller image.

Detection runs on its own thread (`pipeline.py`) while the main thread draws the previous frame, so a frame costs the slower of the two instead of their sum. Between camera frames the last frame is drawn again with effects moved on, so animation stays at 60 FPS even from a 30 FPS camera. On Windows the app also asks for 1 ms timer resolution: by default the `waitKey(1)` every frame needs actually waited about 13 ms. `identity.py` matches each frame's hands to the previous frame's by palm distance, so ids survive classifier Left/Right flips, and pairs each new hand with the nearest person missing that hand.

A One Euro filter smooths slow fingertip jitter while reducing lag during fast movement. Finger bends and distances become gesture candidates; hysteresis and a stabilizer reject brief flickers. A small event bus dispatches stable gestures and timed holds. Each mode subscribes only its own controllers. Opening an overlay detaches gameplay subscriptions and ends drawing strokes; closing resets gesture history before resuming.

Glowing effects and the neon drawing draw into a separate light buffer (`effects/lighting.py`). Its highlights bloom like a lens sees them, and a wide blur of it acts as a light map that brightens the camera picture in the light's color, so fire warms your face and Chidori flashes it blue. Fire is a procedural heat field (`effects/flame.py`): streaky noise scrolled along the flow and colored by temperature, with heat haze and shockwaves that bend the picture itself. Solid blocks use alpha compositing and simple physics. Person segmentation places your body in front of domains and Susanoo. Dust captures a person-shaped ghost and erodes it into particles over an estimated background. These are procedural, stylized effects; no downloaded artwork is required.

```text
CameraStream -> latest frame -> mirror -> HandTracker -> HandData
                                             |              |
                             PersonSegmenter  |      gestures + smoothing
                                    |        |              |
                              background     +------ EventBus
                                    |                  /    \
                                    |           mode controllers  focused UI
                                    |                 |
                                    +-> canvas / effects / particles
                                                      |
                                                skeleton / screen FX
                                                      |
                                              clean recording tap
                                                      |
                                              HUD / menu / guide
                                                      |
                                             recording -> window

Effect hooks -> optional audio       Settings <- menu + CLI
Profiler -> adaptive quality         Logging -> console + rotating file
```

## Verification

```bash
python -m unittest discover -s tests -v
python main.py --selftest
python tools/render_previews.py
python tools/live_smoke.py
```

The last command opens the real camera/window for ten seconds and saves metrics only. Synthetic previews go to `verification/phase5/`; they use a drawn silhouette, not a real camera photograph. The automated tests cover gesture recognition, drawing, blocks/physics, sprite clipping, particle caps, menu focus, new effects, settings, camera threading, recording timing/failures, quality adaptation and profiling.

On 25 October 2026, with a simulated phone streaming 720p H.264 at 60 FPS into the real app on this laptop, the laptop received 59 FPS and displayed **60 FPS**, and sent the finished picture back to the phone at 58 FPS. With a fireball exploding every second (the heaviest effect load), display dropped to about 46 FPS. Hand detection took 12-17 ms on its own thread. Six-hand detection could not be measured without six real hands in view.

On this Windows laptop, the 24 October 2026 hardware smoke check opened camera 0 at 1280×720 and measured roughly **15–20 FPS** (camera-limited) despite a reported camera setting of 30; processing took about 28 ms per frame, so the pipeline itself could sustain about 36 FPS. It initialized 26 sounds and processed all four modes without a frame error. Synthetic 720p effect-only timings were approximately 3–4 ms for orbs/Rasenshuriken, 13–16 ms for domains/Susanoo and 26 ms for dust, excluding tracking and display. These are local measurements, not a cross-device guarantee. First-use scene generation costs more.

Live gesture accuracy, visual likeness and a physical unplug/replug still deserve the manual checks in [docs/PHASE5_QA.md](docs/PHASE5_QA.md). See [docs/PORTFOLIO.md](docs/PORTFOLIO.md) for truthful resume bullets, a demo script and design questions.

## Packaging

BITL is a desktop app, so there is no website to host: `vercel.json` turns off Vercel deployments for git pushes. Without it, a Vercel project linked to this repository sees `requirements.txt` and `main.py`, expects a Python web server, and fails every build.

Build the wheel:

```bash
python -m pip install build
python -m build --wheel
```

Build a standalone folder, on the OS where you will run it:

```bash
python -m pip install -e '.[audio,build]'
python main.py --selftest
python -c "from ar_hands.sounds import ensure_sounds; from ar_hands.config import Config; c=Config(); ensure_sounds(c.base_dir / c.sound_folder, c.sound_sample_rate)"
python -m PyInstaller --noconfirm BITL.spec
```

In Windows PowerShell, use double quotes around `.[audio,build]`. The spec bundles models, available sounds/sprites and MediaPipe runtime resources. Distribute the entire `dist/BITL/` directory, not the executable alone. Run `dist/BITL/BITL.exe --selftest` on Windows or `dist/BITL/BITL --selftest` on macOS/Linux. The source release includes models and generated sounds so the standalone build can work offline after installing build dependencies.

Build separately on Windows, macOS and Linux; this is not a cross-compiler. On macOS allow camera access for the terminal/executable; signing and notarization are separate distribution steps. On Linux use a graphical session with the OpenCV GUI dependencies available. The console build keeps startup and camera errors visible. A standalone build recipe is provided; validation on other operating systems remains outstanding.

## Troubleshooting

- **Phone cannot open the page:** both devices must be on the same network, and guest or campus Wi-Fi often blocks devices from reaching each other; use the phone's hotspot instead. On Windows, allow BITL (or Python) through the firewall when asked; if you clicked Cancel, allow it under Windows Security → Firewall → Allow an app. A VPN on the computer can pick the wrong address: the pairing screen shows the link it uses.
- **Phone says the camera is blocked:** open the page from the QR code (it must start with `https://`), accept the certificate warning, then allow the camera. If you denied it, re-enable camera access for the site in the browser's site settings and reload.
- **Phone picture is choppy:** check the phone's status pill. JPEG instead of H.264 means the browser cannot encode H.264 (update it) or PyAV is missing on the computer (`pip install av`). Move closer to the router (5 GHz Wi-Fi is better), close other heavy Wi-Fi use, or turn off the AR picture on the phone (eye button) so only the upload remains. F10 on the laptop shows whether detection or effects are the bottleneck.
- **Picture on its side:** tap the rotate button on the phone; each tap turns it 90°.
- **A friend's hands are not tracked:** raise Max hands, bring everyone a little closer and into the light, and keep hands from overlapping. Very small, far-away hands are below what the hand model detects.

- **Camera missing/black:** close camera apps, check OS permissions/privacy shutter, then try `--camera 1`. The app retries automatically; Tab and Q stay responsive. An unresponsive native camera driver can outlive shutdown briefly as a daemon thread.
- **Model download fails:** the log gives the model URL and destination. Check connectivity or copy the model to the reported path. Run `--selftest` again.
- **Wrong Python / install failure:** use 3.10–3.12 and the provided dependency constraints. Do not install multiple OpenCV variants into the same environment. Intel macOS uses its own dependency markers.
- **Low FPS:** turn off glow/skeleton, try `--detect-scale 0.5`, use F10 to distinguish capture from processing. Domains and dust do more full-frame work. Requested FPS is not guaranteed camera FPS.
- **A gesture does not fire:** open ? to see the pose; keep fingertips visible, improve lighting, and hold steadily. V shows recognition state. The Sensitivity slider changes model confidence, not gesture hold durations.
- **Block break misses:** keep the index tip over the block until its crack fills, or punch through it. Pinch existing blocks to move them; release to settle into the grid.
- **Dust leaves a ghost:** keep the camera fixed; B then step out briefly before returning. A snap is a quick thumb/middle-finger motion; N can verify the effect independently of recognition.
- **No sound:** install the audio extra and check M, system volume and the output device. Missing pygame or an unavailable device never disables the visual app.
- **Recording fails:** check disk space and `logs/ar_hands.log`; the app tries several codecs. Encoder errors are reported instead of a false success message. Stop recording before moving its output file.
- **Reset a confusing setup:** use `--reset-settings`, or Reset settings in Tab. Capital F is fullscreen; lowercase f remains drawing fade.
