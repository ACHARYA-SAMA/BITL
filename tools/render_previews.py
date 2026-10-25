"""Render repeatable synthetic scenes; no camera or personal images are captured."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_hands.app import App
from ar_hands.effects.domains import UnlimitedVoid, MalevolentShrine
from ar_hands.effects.susanoo import SusanooEffect
from ar_hands.effects.rasenshuriken import Rasenshuriken
from ar_hands.effects.jjk import CursedOrb, HollowPurple
from ar_hands.effects.snap import DustEffect
from ar_hands.handpose import make_hand


def main():
    """Write menu, guide and effect previews plus a measured synthetic rendering report."""
    folder = Path(__file__).resolve().parents[1] / 'verification' / 'phase5'
    folder.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    background = np.zeros((height, width, 3), np.uint8)
    background[:] = (28, 24, 22)
    for y in range(0, height, 60):
        cv2.line(background, (0, y), (width, y), (40, 35, 32), 1)
    mask = np.zeros((height, width), np.float32)
    cv2.circle(mask, (640, 290), 65, 1., -1)
    cv2.ellipse(mask, (640, 560), (130, 220), 0, 0, 360, 1., -1)
    camera = background.copy()
    camera[mask > .5] = (100, 105, 110)
    scenes = {
        'gojo': lambda c: UnlimitedVoid((640, 360), 0),
        'sukuna': lambda c: MalevolentShrine((640, 360), 0),
        'susanoo': lambda c: SusanooEffect(0),
        'rasenshuriken': lambda c: Rasenshuriken('Right', (850, 270), 48, 0),
        'blue': lambda c: CursedOrb('blue', 'Right', (850, 270), 65, 0),
        'red': lambda c: CursedOrb('red', 'Right', (850, 270), 65, 0),
        'purple': lambda c: HollowPurple((850, 270), 70, 0),
        'dust': lambda c: DustEffect(0, c.effects),
    }
    tiles, report = [], []
    for name, create in scenes.items():
        app = App()
        app.background.update(background, np.zeros_like(mask))
        effect = create(app.config)
        app.effects.spawn(effect)
        hands = [make_hand('open_palm')]
        measurements = []
        for i in range(45):
            frame = camera.copy()
            start = time.perf_counter()
            app.effects.update(1/30, hands, [], i / 30, (width, height), camera, mask, app.background)
            app.effects.render(frame)
            measurements.append((time.perf_counter() - start) * 1000)
            if i == (22 if name == 'dust' else 40):
                preview = frame.copy()
        cv2.putText(preview, name.upper() + ' / SYNTHETIC PREVIEW', (26, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imwrite(str(folder / f'{name}.png'), preview)
        tiles.append(cv2.resize(preview, (480, 270)))
        report.append(f'{name}: steady mean {np.mean(measurements[5:]):.2f} ms, first {measurements[0]:.2f} ms')
        app.close_desktop()
    app = App()
    for name in ('menu', 'guide'):
        preview = background.copy()
        if name == 'menu':
            app.open_menu()
            app.menu.render(preview)
        else:
            app.open_guide()
            app.guide.open('Jujutsu')
            app.guide.render(preview)
        cv2.imwrite(str(folder / f'{name}.png'), preview)
        tiles.append(cv2.resize(preview, (480, 270)))
    sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(folder / 'contact-sheet.jpg'), sheet)
    (folder / 'render-timings.txt').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))
    app.close_desktop()


if __name__ == '__main__':
    main()
