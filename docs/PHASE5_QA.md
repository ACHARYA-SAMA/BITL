# Before a live demo

Run `python main.py --selftest` and `python -m unittest discover -s tests -v` first. Then check these with the actual webcam and your hands. Automated checks cannot certify physical gesture comfort or visual likeness.

1. Start with `Start BITL.cmd`; mirrored camera and hand skeleton appear.
2. First-run cards are readable; any key or a fresh pinch dismisses them.
3. Press 1, 2, 3, 4 and verify each mode name.
4. Hold rock for one second; the mode cycles once.
5. Draw with one hand, then both; curl fingers to stop without stray joins.
6. Change color and thickness; undo, redo and export PNG.
7. Verify lowercase f fades strokes; capital F toggles fullscreen.
8. Pinch empty space for a block, release, and stack several blocks.
9. Pinch an existing block to move it; settle it onto another block.
10. Point to crack/break a block; swing a fist to smash one.
11. Create a two-hand beam and drop blocks onto it.
12. Form and throw a fireball; check throw direction, explosion and sound.
13. Hold fist then open for Chidori; verify its looping sound stops.
14. Hold OK for Rasenshuriken; flick to launch and check wind burst.
15. Hold both fists to summon Susanoo; repeat to dismiss.
16. Learn the empty background, return, snap to dust and snap to reappear.
17. Form Blue and Red in separate hands; bring together for Purple and flick.
18. Cross fingers for Unlimited Void; your body stays in front of space.
19. Prayer opens Malevolent Shrine; verify slashes and repeat to close.
20. Open Tab while drawing; confirm gameplay pauses. An already-held pinch must not click a control.
21. Use dwell, fresh pinch and pinch-drag on menu controls; verify the Sensitivity slider applies after closing.
22. Hold both open palms for one second in every mode; settings opens and deliberately releases held spells.
23. Open ? from every mode; inspect cards, page with arrows, close with a fresh pinch.
24. Resize and fullscreen the window; preserve aspect ratio and correct mouse targets.
25. Mute/unmute and change volume; restart and verify mode/color/settings persist.
26. Record a ten-second clip with R; stop and play it. Check speed, REC badge and clean recording option. Video is silent by design.
27. Unplug/reconnect the webcam or temporarily revoke camera access; UI stays responsive and capture resumes. Repeat during dust/recording if possible.
28. Open F10 and V; inspect timing/gesture panels and adaptive-quality behavior.
29. Quit with Q, Esc outside overlays, the menu Quit action and the window X. Restart to verify the camera/audio devices were released.
30. Test `--no-audio`, corrupt settings recovery (back up your real settings first), and the packaged executable on a clean target machine.

## Evidence recorded on 24 October 2026

- Unit tests and headless tracking/segmentation self-test: see `verification/phase5-tests.log` and `verification/phase5-selftest.log`.
- Synthetic effect/menu/guide previews: `verification/phase5/contact-sheet.jpg`.
- Real camera/audio/window smoke: `verification/phase5/live-smoke.json`; camera media was not saved.
- Wheel build: `verification/phase5-build.log`.
- Physical gesture walkthrough, physical disconnect/reconnect, and macOS/Linux executable checks remain manual items.
