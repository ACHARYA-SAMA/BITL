"""Phase 4 tests: compositing, particles, sprites, jutsu, blocks, beams, and modes.

Effects run against the real GestureEngine fed with synthetic hands, so these
tests exercise the same event path the camera does. No webcam needed.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ar_hands.config import EffectsConfig, GestureConfig
from ar_hands.effects.base import EffectState, HookEmitter
from ar_hands.effects.blocks import BlockBuilder, BlockWorld
from ar_hands.effects.compositing import Sprite, overlay_rgba
from ar_hands.effects.fireball import Fireball, FireballJutsu
from ar_hands.effects.lightning import Chidori, ChidoriEffect, midpoint_bolt
from ar_hands.effects.manager import EffectManager, Mode, ModeManager, ModeSwitcher
from ar_hands.effects.particles import Blend, FIRE_RAMP, ParticleSystem
from ar_hands.effects.sprites import SpriteLibrary, iso_cube, block_faces
from ar_hands.gestures.engine import GestureEngine
from ar_hands.gestures.events import BOTH, EventBus
from tests.test_gestures import DT, make_hand, prayer_pair

SIZE = (1280, 720)
EFFECTS = EffectsConfig()
SPRITES = SpriteLibrary(EFFECTS)  # Built once: generation is deterministic.


def solid(size=40, color=(0, 0, 255), alpha=255):
    sprite = np.zeros((size, size, 4), np.uint8)
    sprite[...] = (*color, alpha)
    return sprite


class CompositingTests(unittest.TestCase):
    def test_clips_at_every_edge_and_corner(self):
        width, height = 200, 120
        cases = {(0, 0): 20 * 20, (width, 0): 20 * 20, (0, height): 20 * 20, (width, height): 20 * 20,
                 (100, 0): 40 * 20, (0, 60): 20 * 40, (width, 60): 20 * 40, (100, height): 40 * 20,
                 (100, 60): 40 * 40}
        for center, expected in cases.items():
            with self.subTest(center=center):
                frame = np.zeros((height, width, 3), np.uint8)
                overlay_rgba(frame, solid(), center)
                self.assertEqual(int(np.count_nonzero(frame[..., 2])), expected)

    def test_fully_offscreen_returns_before_resampling(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        with patch.object(cv2, "warpAffine", wraps=cv2.warpAffine) as warp:
            for center in ((-100, 50), (300, 50), (50, -100), (50, 300)):
                overlay_rgba(frame, solid(), center, scale=1.5, angle_deg=30)
            warp.assert_not_called()
        self.assertFalse(frame.any())

    def test_rotation_keeps_corners(self):
        frame = np.zeros((200, 200, 3), np.uint8)
        overlay_rgba(frame, solid(), (100, 100), angle_deg=45)
        columns = np.flatnonzero(frame[..., 2].any(axis=0))
        self.assertGreaterEqual(columns[-1] - columns[0] + 1, 55)  # 40 * sqrt(2) ~ 56.6

    def test_premultiplied_rotation_has_no_dark_fringe(self):
        frame = np.full((120, 120, 3), 255, np.uint8)
        overlay_rgba(frame, solid(color=(255, 255, 255)), (60, 60), scale=1.3, angle_deg=30)
        self.assertGreaterEqual(int(frame.min()), 254)

    def test_blend_modes_and_three_channel_sprites(self):
        base = np.full((40, 40, 3), 100, np.uint8)
        half = solid(color=(200, 200, 200), alpha=128)
        frame = base.copy()
        overlay_rgba(frame, half, (20, 20))
        self.assertAlmostEqual(int(frame[20, 20, 0]), 150, delta=2)  # 200*0.5 + 100*0.5
        frame = base.copy()
        overlay_rgba(frame, solid(color=(200, 200, 200)), (20, 20), mode="add")
        self.assertEqual(int(frame[20, 20, 0]), 255)  # Saturates instead of wrapping.
        frame = base.copy()
        overlay_rgba(frame, solid(color=(200, 200, 200)), (20, 20), mode="screen")
        self.assertAlmostEqual(int(frame[20, 20, 0]), 222, delta=2)  # 100 + 200 * 155/255
        frame = base.copy()
        overlay_rgba(frame, np.full((10, 10, 3), 7, np.uint8), (20, 20), opacity=1.0)
        self.assertEqual(int(frame[20, 20, 0]), 7)  # BGR sprites are opaque.
        with self.assertRaises(ValueError):
            overlay_rgba(frame, half, (20, 20), mode="multiply")

    def test_prepared_sprite_matches_raw_image(self):
        raw = SPRITES.fireball_frames[0].premul  # Any BGRA content works here.
        a = np.full((300, 300, 3), 40, np.uint8)
        b = a.copy()
        image = np.dstack((raw[..., :3], raw[..., 3]))
        overlay_rgba(a, image, (150, 150), 1.2, 20, 0.8, "add")
        overlay_rgba(b, Sprite.from_image(image), (150, 150), 1.2, 20, 0.8, "add")
        np.testing.assert_array_equal(a, b)


class ParticleTests(unittest.TestCase):
    def make(self, capacity=100, **kwargs):
        return ParticleSystem(capacity, Blend.ADD, FIRE_RAMP, seed=1, **kwargs)

    def test_cap_and_recycling_of_the_oldest(self):
        system = self.make(100)
        first = system.emit(60, (10, 10), life=(10, 10))
        second = system.emit(60, (90, 90), life=(20, 20))
        self.assertEqual(system.count, 100)
        recycled = set(first[:20]) & set(second)
        self.assertEqual(len(recycled), 20)  # The 20 oldest slots were reused...
        self.assertTrue(np.all(system.life[first[20:]] == 10))  # ...and only those.
        self.assertEqual(len(system.emit(500, (0, 0))), 100)

    def test_frame_rate_independent_motion(self):
        results = []
        for steps in (30, 60):
            system = self.make(10, drag=1.5, gravity=400)
            system.emit(1, (0, 0), (300, -200), life=(5, 5))
            for _ in range(steps):
                system.update(1.0 / steps)
            results.append(system.pos[system.life > 0][0].copy())
        np.testing.assert_allclose(results[0], results[1], atol=0.05)  # Closed-form: exact.

    def test_dt_is_clamped(self):
        system = self.make(10)
        system.emit(1, (0, 0), (1000, 0), life=(5, 5))
        system.update(1.0)  # A one-second stall must not teleport particles.
        self.assertAlmostEqual(float(system.pos[system.life > 0][0][0]), 50.0, places=3)

    def test_thousand_particles_render_under_four_ms(self):
        system = ParticleSystem(1500, Blend.ADD, FIRE_RAMP, seed=2)
        system.emit(1000, (640, 360), (0, -120), spread=80, radius=40, life=(0.6, 1.0), size=(6, 16))
        frame = np.zeros((720, 1280, 3), np.uint8)
        times = []
        for _ in range(25):
            canvas = frame.copy()
            start = time.perf_counter()
            system.render(canvas)
            times.append(time.perf_counter() - start)
        self.assertLess(sorted(times)[12] * 1000, 4.0)
        self.assertGreater(int(canvas.max()), 200)

    def test_smoke_darkens_a_bright_background(self):
        from ar_hands.effects.particles import SMOKE_RAMP
        smoke = ParticleSystem(300, Blend.ALPHA, SMOKE_RAMP, seed=3)
        smoke.emit(300, (100, 100), radius=20, life=(1, 1), size=(8, 12))
        smoke.update(0.3)
        frame = np.full((200, 200, 3), 240, np.uint8)
        smoke.render(frame)
        self.assertLess(int(frame[100, 100].max()), 200)  # Gray haze, never a white blob.
        self.assertEqual(int(frame[0, 0, 0]), 240)


class SpriteTests(unittest.TestCase):
    def test_procedural_sprites_are_deterministic_bgra(self):
        again = SpriteLibrary(EFFECTS)
        for kind in EFFECTS.block_kinds:
            for stage in range(5):
                a, b = SPRITES.block(kind, 64, stage), again.block(kind, 64, stage)
                np.testing.assert_array_equal(a.premul, b.premul)
                self.assertEqual(a.shape, (64, 64, 4))
                self.assertTrue(a.binary)
        cube = iso_cube(*block_faces("stone"), 64)
        self.assertEqual(int(cube[0, 0, 3]), 0)  # Hexagon corners are transparent.
        self.assertEqual(int(cube[32, 32, 3]), 255)
        np.testing.assert_array_equal(SPRITES.fireball(0).premul, again.fireball(0).premul)

    def test_user_png_overrides_and_bad_files_warn(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            cv2.imwrite(str(path / "block_stone.png"), np.full((16, 16, 3), (0, 255, 0), np.uint8))
            cv2.imwrite(str(path / "fireball.png"), solid(32, (255, 0, 0), 200))
            (path / "block_grass.png").write_bytes(b"not a png")
            with self.assertLogs("ar_hands.effects.sprites", "WARNING") as logs:
                library = SpriteLibrary(EFFECTS, path)
            self.assertIn("block_grass.png", logs.output[0])
            self.assertEqual(len(library.fireball_frames), 1)
            self.assertEqual(library.fireball(3).shape[:2], (32, 32))
            stone = library.block("stone", 48).premul
            self.assertEqual(tuple(int(v) for v in stone[24, 24, :3]), (0, 255, 0))
            grass = library.block("grass", 48).premul  # Fell back to procedural.
            np.testing.assert_array_equal(grass, SPRITES.block("grass", 48).premul)


class Scene:
    """Real engine + bus + effect manager; step() feeds synthetic hands."""

    def __init__(self, config: EffectsConfig = EFFECTS):
        self.config = config
        self.bus = EventBus()
        self.engine = GestureEngine(GestureConfig(), self.bus)
        self.manager = EffectManager(config, self.engine, SPRITES, seed=7)
        self.world = BlockWorld(config, SPRITES, self.manager.particles, self.manager.hooks)
        self.manager.spawn(self.world)
        self.hooks: list[tuple[float, str, dict]] = []
        self.manager.hooks.subscribe(lambda name, payload: self.hooks.append((self.t, name, payload)))
        self.controllers = []
        self.t = 0.0

    def attach(self, *controllers):
        for controller in controllers:
            controller.attach()
        self.controllers.extend(controllers)
        return controllers

    def step(self, *hands, render=False):
        self.t += DT
        events = self.engine.update(list(hands), self.t)
        ctx = self.manager.update(DT, list(hands), events, self.t, SIZE)
        for controller in self.controllers:
            controller.update(ctx)
        if render:
            frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
            self.manager.render(frame)
            for controller in self.controllers:
                controller.render(frame, ctx)
            self.manager.post(frame)
            return frame
        return None

    def run(self, frames, *hands, render=False):
        for _ in range(frames):
            self.step(*hands, render=render)

    def names(self):
        return [name for _, name, _ in self.hooks]

    def first(self, name):
        return next((t for t, hook, _ in self.hooks if hook == name), None)


def jutsu_scene():
    """A scene with the JUTSU controllers attached."""
    scene = Scene()
    chidori = Chidori(scene.bus, scene.engine, scene.manager, EFFECTS)
    scene.attach(FireballJutsu(scene.bus, scene.engine, scene.manager, EFFECTS, chidori.is_busy), chidori)
    return scene


def fireballs(scene):
    return [e for e in scene.manager.effects if isinstance(e, Fireball)]


class JutsuTests(unittest.TestCase):
    def setUp(self):
        self.scene = jutsu_scene()

    def balls(self):
        return fireballs(self.scene)

    def summon(self, scene=None, scale=100, offset=(-200, 50)):
        (scene or self.scene).run(25, make_hand("open_palm", scale=scale, offset=offset), render=True)

    def test_open_palm_hold_summons_after_half_second_sized_by_hand(self):
        self.summon()
        summoned = self.scene.first("fireball_summon")
        self.assertIsNotNone(summoned)
        # STARTED needs 4 frames, then the 0.5 s hold.
        self.assertGreaterEqual(summoned, 4 * DT + EFFECTS.fireball_hold_seconds - 1e-6)
        near = jutsu_scene()  # Same pose, hand twice as large (closer to the camera).
        self.summon(near, scale=200)
        self.assertAlmostEqual(fireballs(near)[0].radius / self.balls()[0].radius, 2.0, delta=0.15)

    def test_cooldown_between_summons(self):
        self.summon()
        self.scene.run(8, make_hand("fist", offset=(-200, 50)))  # Dissipate.
        self.scene.run(25, make_hand("open_palm", offset=(-200, 50)))
        summons = [t for t, name, _ in self.scene.hooks if name == "fireball_summon"]
        self.assertEqual(len(summons), 2)
        self.assertGreaterEqual(summons[1] - summons[0], EFFECTS.fireball_cooldown_s)

    def test_flick_throws_and_it_explodes_off_screen(self):
        self.summon()
        for i in range(6):
            self.scene.step(make_hand("open_palm", offset=(-200 + 60 * i, 50)), render=True)
        self.assertIn("fireball_throw", self.scene.names())
        for _ in range(int(EFFECTS.fireball_max_flight_s / DT) + 5):
            self.scene.step(render=True)
        self.assertIn("fireball_explode", self.scene.names())
        self.assertEqual(self.balls(), [])

    def test_fist_dissipates_with_smoke(self):
        self.summon()
        self.scene.run(8, make_hand("fist", offset=(-200, 50)))
        self.assertEqual([p.get("reason") for _, n, p in self.scene.hooks if n == "fireball_dissipate"],
                         ["fist"])
        self.assertGreater(self.scene.manager.particles.smoke.count, 0)
        self.scene.run(20)
        self.assertEqual(self.balls(), [])

    def test_lost_hand_holds_then_dissipates(self):
        self.summon()
        ball = self.balls()[0]
        self.scene.run(10)  # 0.33 s without hands: holding in place.
        held_at = ball.pos.copy()
        self.assertTrue(ball.held)
        self.scene.run(3)
        np.testing.assert_allclose(ball.pos, held_at)
        self.scene.run(10)  # Past the 0.5 s grace.
        self.assertIs(ball.state, EffectState.DYING)
        self.scene.run(20)
        self.assertEqual(self.balls(), [])

    def test_prayer_then_hands_apart_casts_the_great_fireball(self):
        self.scene.run(36, *prayer_pair())  # Seal held 0.8 s.
        self.assertIn("jutsu_seal", self.scene.names())
        for frame in range(12):
            gap = 40 + 14 * frame
            self.scene.step(make_hand("open_palm", hand="Left", offset=(-gap, 0)),
                            make_hand("open_palm", hand="Right", offset=(gap, 0)))
        self.assertIn("great_fireball", self.scene.names())
        great = [b for b in self.balls() if b.great]
        self.assertEqual(len(great), 1)
        self.assertEqual(great[0].owner, BOTH)

    def test_chidori_charge_release_and_fade(self):
        self.scene.run(30, make_hand("fist", offset=(0, 40)))  # Arm (0.6 s hold).
        self.assertIn("chidori_charge", self.scene.names())
        self.scene.run(25, make_hand("open_palm", offset=(0, 40)), render=True)
        self.assertIn("lightning_start", self.scene.names())
        effects = [e for e in self.scene.manager.effects if isinstance(e, ChidoriEffect)]
        self.assertEqual(len(effects), 1)
        main_bolts = [b for b in effects[0].bolts if len(b) == 2 ** EFFECTS.chidori_depth + 1]
        self.assertTrue(6 <= len(main_bolts) <= 10)
        self.assertNotIn("fireball_summon", self.scene.names())  # The open palm stayed Chidori.
        self.scene.run(10, make_hand("fist", offset=(0, 40)))
        self.assertIn("lightning_end", self.scene.names())
        self.scene.run(10)
        self.assertEqual([e for e in self.scene.manager.effects if isinstance(e, ChidoriEffect)], [])

    def test_open_palm_without_charge_is_a_fireball_not_chidori(self):
        self.summon()
        self.assertNotIn("lightning_start", self.scene.names())

    def test_midpoint_bolt_keeps_its_endpoints(self):
        bolt = midpoint_bolt((0, 0), (100, 50), np.random.default_rng(1), 0.3, 4)
        self.assertEqual(len(bolt), 17)
        np.testing.assert_allclose(bolt[0], (0, 0))
        np.testing.assert_allclose(bolt[-1], (100, 50))


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()
        self.builder = BlockBuilder(self.scene.bus, self.scene.engine, self.scene.manager,
                                    self.scene.world, EFFECTS)
        self.scene.attach(self.builder)
        self.world = self.scene.world
        self.floor = SIZE[1] * EFFECTS.floor_ratio

    def drop(self, x, y, width=80, kind="stone"):
        block = self.world.spawn_block(kind, (x, y), width, self.scene.t)
        self.world.release_block(block)
        return block

    def test_falls_to_floor_and_sleeps_without_jitter(self):
        block = self.drop(400, 100)
        self.scene.run(60)
        self.assertAlmostEqual(block.y, self.floor, places=3)
        self.assertTrue(block.asleep)
        resting = (block.x, block.y)
        self.scene.run(30)
        self.assertEqual((block.x, block.y), resting)

    def test_blocks_stack(self):
        lower = self.drop(400, 300)
        self.scene.run(40)
        upper = self.drop(410, 100)
        self.scene.run(60)
        self.assertAlmostEqual(upper.y, lower.top, places=3)
        self.assertTrue(upper.asleep and lower.asleep)

    def test_removing_a_support_wakes_the_stack(self):
        lower = self.drop(400, 300)
        self.scene.run(40)
        upper = self.drop(400, 100)
        self.scene.run(60)
        self.world.remove_block(lower)
        self.scene.run(40)
        self.assertAlmostEqual(upper.y, self.floor, places=3)

    def test_cap_fades_the_oldest(self):
        first = self.drop(100, 100, 40)
        for index in range(EFFECTS.max_blocks):
            self.drop(100 + (index % 20) * 50, 100, 40)
        self.assertIsNotNone(first.fading_since)
        self.scene.run(int(EFFECTS.fade_out_s / DT) + 2)
        self.assertNotIn(first, self.world.blocks)
        self.assertLessEqual(len(self.world.live_blocks), EFFECTS.max_blocks)

    def test_pinch_spawns_a_block_that_follows_and_drops(self):
        self.scene.run(8, make_hand("pinch", offset=(-200, -100)))
        self.assertIn("block_place", self.scene.names())
        block = self.world.blocks[0]
        self.assertEqual(block.held_by, "Right")
        self.scene.run(15, make_hand("pinch", offset=(-100, -100)))
        pinch = self.scene.engine.position("Right")
        self.assertLess(np.hypot(block.center[0] - pinch[0], block.center[1] - pinch[1]), 6)
        self.scene.run(40, make_hand("open_palm", offset=(-100, -100)))
        self.assertIsNone(block.held_by)
        self.assertAlmostEqual(block.y, self.floor, places=3)
        self.assertLess(abs(block.x - pinch[0]), 15)  # Released in place, not flung.

    def test_pinching_an_existing_block_moves_it_instead_of_spawning(self):
        hand = make_hand("pinch")
        thumb, index = hand.landmarks_px[4], hand.landmarks_px[8]
        tip = ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2)  # Where this pinch lands.
        block = self.drop(tip[0], tip[1])  # Its sprite is centered on that point...
        block.asleep = True  # ...and it stays there (resting on something, say).
        self.scene.run(8, hand)
        self.assertEqual(len(self.world.blocks), 1)
        self.assertEqual(block.held_by, "Right")
        self.assertIn("block_grab", self.scene.names())
        self.scene.run(12, make_hand("pinch", offset=(150, -80)))
        self.assertGreater(block.x, tip[0] + 100)

    def connect_and_stretch(self, pulls=14):
        both = lambda dx: [make_hand("pinch", hand="Left", offset=(-30 - dx, 60)),
                           make_hand("pinch", hand="Right", offset=(0 + dx, 60))]
        self.scene.run(8, *both(0))
        for i in range(pulls):
            self.scene.step(*both(18 * i))
        return both

    def test_two_hand_pinch_touch_and_pull_builds_a_beam(self):
        self.connect_and_stretch()
        self.assertIn("beam_connect", self.scene.names())
        self.assertEqual([b for b in self.world.blocks if b.fresh], [])  # Fresh blocks fused.
        beam = self.world.beams[0]
        self.assertFalse(beam.placed)
        self.assertGreater(beam.length, 300)
        self.scene.run(10, make_hand("open_palm", hand="Left", offset=(-300, 60)),
                       make_hand("open_palm", hand="Right", offset=(270, 60)))
        self.assertTrue(beam.placed)
        self.assertIn("beam_place", self.scene.names())

    def test_short_beam_fizzles(self):
        self.connect_and_stretch(pulls=2)
        self.scene.run(10, make_hand("open_palm", hand="Left", offset=(-30, 60)),
                       make_hand("open_palm", hand="Right", offset=(0, 60)))
        self.assertIn("beam_fizzle", self.scene.names())
        self.assertEqual(self.world.beams, [])

    def test_blocks_land_on_placed_beams(self):
        self.connect_and_stretch()
        self.scene.run(10)
        beam = self.world.beams[0]
        middle = (beam.p1 + beam.p2) / 2
        block = self.drop(float(middle[0]), float(middle[1]) - 200)
        self.scene.run(40)
        self.assertAlmostEqual(block.y, float(middle[1]) - beam.thickness / 2, delta=1.0)

    def test_pointing_cracks_then_breaks_a_block(self):
        point = make_hand("point", offset=(0, 0))
        tip = point.landmarks_px[8]
        block = self.drop(tip[0], tip[1] + 40)
        block.y = tip[1] + 40
        block.asleep = True
        self.scene.run(5, point)  # POINT becomes stable on frame 4.
        self.assertGreater(block.crack, 0)
        self.scene.run(int(EFFECTS.break_seconds / DT) + 2, point)
        self.assertNotIn(block, self.world.blocks)
        self.assertIn("block_break", self.scene.names())
        self.assertGreater(self.scene.manager.particles.chunks.count, 0)

    def test_tnt_ignites_then_chain_explodes(self):
        first = self.drop(300, 600, kind="tnt")
        second = self.drop(370, 600, kind="tnt")
        self.scene.run(30)
        self.world.break_block(first, self.scene.t)
        self.assertIn("tnt_ignite", self.scene.names())
        self.scene.run(int((EFFECTS.tnt_fuse_s + EFFECTS.tnt_chain_fuse_s) / DT) + 6)
        self.assertEqual(self.scene.names().count("tnt_explode"), 2)
        self.assertNotIn(first, self.world.blocks)
        self.assertNotIn(second, self.world.blocks)

    def test_explosion_flings_nearby_blocks(self):
        block = self.drop(500, 600)
        self.scene.run(40)
        self.scene.manager.explode((420, block.center[1]), 150)
        self.scene.step()
        self.assertGreater(block.vx, 100)
        self.assertFalse(block.asleep)

    def test_peace_cycles_type_and_thumbs_up_hold_clears(self):
        self.scene.run(6, make_hand("peace"))
        self.assertEqual(self.builder.kind, EFFECTS.block_kinds[1])
        self.drop(300, 300)
        self.scene.run(50, make_hand("thumbs_up"))
        self.assertIn("blocks_clear", self.scene.names())
        self.assertEqual(self.world.blocks, [])

    def test_hand_lost_while_holding_releases_the_block(self):
        self.scene.run(8, make_hand("pinch", offset=(-200, -100)))
        block = self.world.blocks[0]
        self.scene.run(GestureConfig().lost_grace_frames + 3)
        self.assertIsNone(block.held_by)


class ModeTests(unittest.TestCase):
    def build(self):
        scene = Scene()
        chidori = Chidori(scene.bus, scene.engine, scene.manager, EFFECTS)
        jutsu = FireballJutsu(scene.bus, scene.engine, scene.manager, EFFECTS, chidori.is_busy)
        builder = BlockBuilder(scene.bus, scene.engine, scene.manager, scene.world, EFFECTS)
        switcher = ModeSwitcher(scene.bus, EFFECTS.mode_hold_seconds, lambda now: modes.next(now))
        modes = ModeManager({Mode.JUTSU: (jutsu, chidori), Mode.BLOCKS: (builder,)}, (switcher,),
                            scene.manager.hooks)
        scene.controllers = list(modes.active_controllers)
        return scene, modes

    def test_switching_modes_never_leaks_subscriptions(self):
        scene, modes = self.build()
        counts = {}
        for _ in range(3):
            for mode in Mode:
                modes.switch(mode, scene.t)
                counts.setdefault(mode, set()).add(len(scene.bus))
        self.assertTrue(all(len(values) == 1 for values in counts.values()), counts)
        self.assertEqual(counts[Mode.DRAW], {1})  # Only the global ROCK switch here.
        modes.close()
        self.assertEqual(len(scene.bus), 0)

    def test_rock_hold_cycles_to_the_next_mode(self):
        scene, modes = self.build()
        scene.run(40, make_hand("rock"))
        self.assertIs(modes.mode, Mode.JUTSU)
        self.assertIn("mode_change", scene.names())

    def test_no_orphaned_effects_after_tracking_loss(self):
        scene, modes = self.build()
        modes.switch(Mode.JUTSU, scene.t)
        scene.controllers = list(modes.active_controllers)
        scene.run(25, make_hand("open_palm", offset=(-250, 40)), make_hand("fist", hand="Left", offset=(250, 40)))
        scene.run(10, make_hand("open_palm", offset=(-250, 40)), make_hand("open_palm", hand="Left", offset=(250, 40)))
        names = scene.names()
        self.assertIn("fireball_summon", names)
        self.assertIn("lightning_start", names)
        scene.run(60)  # Hands gone for 2 s.
        live = [e.name for e in scene.manager.effects if not e.persistent]
        self.assertEqual(live, [])

    def test_leaving_jutsu_dissipates_held_fireballs(self):
        scene, modes = self.build()
        modes.switch(Mode.JUTSU, scene.t)
        scene.controllers = list(modes.active_controllers)
        scene.run(25, make_hand("open_palm"))
        modes.switch(Mode.BLOCKS, scene.t)
        scene.controllers = list(modes.active_controllers)
        scene.run(20)
        self.assertEqual(scene.manager.count("fireball"), 0)


class HookAndBudgetTests(unittest.TestCase):
    def test_hook_listener_errors_are_isolated(self):
        hooks, received = HookEmitter(), []

        def broken(name, payload):
            raise RuntimeError("audio device gone")

        hooks.subscribe(broken)
        hooks.subscribe(lambda name, payload: received.append(name))
        with self.assertLogs("ar_hands.effects.base", "ERROR"):
            hooks.emit("block_break", kind="stone")
        self.assertEqual(received, ["block_break"])
        self.assertTrue(hooks.unsubscribe(broken))

    def test_fireball_and_twenty_blocks_fit_the_eight_ms_budget(self):
        scene = Scene()
        for index in range(20):
            block = scene.world.spawn_block(EFFECTS.block_kinds[index % 5], (0, 0), 88, 0.0)
            block.x, block.y = 120 + (index % 10) * 104, 662 - (index // 10) * 44
        ball = scene.manager.spawn(Fireball("Right", (400, 300), 60, 0.0))
        background = np.full((SIZE[1], SIZE[0], 3), 120, np.uint8)
        times = []
        for frame in range(150):
            ball.follow((400 + 200 * np.sin(frame / 15), 300))
            start = time.perf_counter()
            scene.manager.update(DT, [], [], frame * DT, SIZE)
            canvas = background.copy()
            scene.manager.render(canvas)
            scene.manager.post(canvas)
            times.append(time.perf_counter() - start)
        median = sorted(times[30:])[60] * 1000
        self.assertLess(median, 8.0, f"median {median:.2f} ms")


if __name__ == "__main__":
    unittest.main()
