"""Shared immutable application settings: every tunable lives in this file."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import MappingProxyType
from .actions import Action, GestureAction, Trigger
from .gestures.types import Gesture

Color = tuple[int, int, int]


def resource_dir() -> Path:
    """Read-only files (models, sprites, sounds): the project folder, or the
    PyInstaller bundle folder when running as a built executable."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    source = Path(__file__).resolve().parent.parent
    if (source / 'main.py').is_file():
        return source
    # A wheel's site-packages directory may be read-only. Models/synthesized
    # sounds are caches, so an installed app puts them in the user's directory.
    from .settings import settings_dir
    return settings_dir() / 'resources'


def output_root() -> Path | None:
    """Writable output beside the executable when frozen (the bundle folder may
    be read-only); None from source, where outputs go beside the code."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    if (Path(__file__).resolve().parent.parent / 'main.py').is_file():
        return None
    from .settings import settings_dir
    return settings_dir() / 'output'


@dataclass(frozen=True)
class GestureConfig:
    """Gesture thresholds. Distances are ratios of hand scale (wrist to middle
    MCP), so the same numbers hold at arm's length and up close."""

    # Hands smaller than this many pixels are too noisy to classify.
    min_hand_scale_px: float = 12.0
    # Long fingers: openness blends the PIP joint angle with "reach", how much
    # farther the tip is from the wrist than the PIP joint is.
    finger_angle_curled_deg: float = 60.0
    finger_angle_extended_deg: float = 160.0
    finger_reach_curled: float = -0.10
    finger_reach_extended: float = 0.35
    finger_angle_weight: float = 0.5
    # A bone foreshortened below this length (pointing at the camera) has no
    # trustworthy direction, so its joint angle is ignored in favor of reach.
    min_bone_ratio: float = 0.06
    extended_threshold: float = 0.65
    curled_threshold: float = 0.35
    # Thumb: IP joint angle plus tip gaps to the index and pinky knuckles.
    thumb_angle_curled_deg: float = 120.0
    thumb_angle_extended_deg: float = 165.0
    thumb_index_gap_curled: float = 0.35
    thumb_index_gap_extended: float = 0.60
    thumb_pinky_gap_curled: float = 0.70
    thumb_pinky_gap_extended: float = 1.20
    thumb_weights: tuple[float, float, float] = (0.2, 0.4, 0.4)  # angle, index gap, pinky gap
    # Pinch Schmitt trigger and strength ramp (thumb tip to index tip).
    pinch_enter: float = 0.25
    pinch_exit: float = 0.35
    pinch_full: float = 0.10
    pinch_open: float = 0.60
    # The pinch midpoint must be at least this far from the wrist: a fist tucks
    # the thumb and index tips together beside the palm and would otherwise "pinch".
    pinch_min_reach: float = 0.90
    # THUMBS_UP needs the thumb (MCP to tip) within this angle of screen-up.
    thumbs_up_max_tilt_deg: float = 45.0
    # Index tip to the middle finger's last two bones: crossed below, a V above.
    # The gap between the two thresholds is a dead zone so they never flicker.
    crossed_max_gap: float = 0.12
    peace_min_gap: float = 0.22
    # Finger snap: thumb pressed on the middle tip (arm), then within the window
    # the middle finger slams toward the palm and the thumb flies off (fire).
    snap_arm_ratio: float = 0.30
    snap_margin: float = 0.10  # Thumb must be closer to the middle than the index tip.
    snap_release_ratio: float = 0.50
    snap_drop: float = 0.25  # Middle tip moves this much closer to the wrist.
    snap_window_s: float = 0.40
    snap_cooldown_s: float = 1.0
    # Temporal stabilizer, in frames.
    vote_window: int = 5
    enter_frames: int = 4
    exit_grace_frames: int = 3
    lost_grace_frames: int = 4
    # Motion, in hand scales per second.
    motion_history: int = 12
    motion_window_s: float = 0.15
    motion_min_samples: int = 3
    motion_max_speed: float = 40.0  # Faster frame-to-frame jumps are tracking glitches.
    swipe_speed: float = 6.0
    swipe_dominance: float = 2.0
    swipe_cooldown_s: float = 0.6
    swipe_min_samples: int = 4
    # Fast strokes while drawing or dragging are not swipes.
    swipe_suppressed_by: frozenset[Gesture] = frozenset({Gesture.POINT, Gesture.PINCH, Gesture.OK})
    # Swipes need this pose (raw, voted, or active), so quickly repositioning a
    # relaxed hand cannot undo. Use frozenset() to allow any unsuppressed pose.
    swipe_requires: frozenset[Gesture] = frozenset({Gesture.OPEN_PALM})
    # Two hands.
    prayer_palm_distance: float = 0.8
    prayer_tip_distance: float = 0.6
    apart_speed: float = 3.0
    apart_cooldown_s: float = 0.8

    def __post_init__(self) -> None:
        """Fail at startup with a clear message instead of misbehaving mid-demo."""
        if not self.pinch_enter < self.pinch_exit:
            raise ValueError("pinch_enter must be below pinch_exit for hysteresis.")
        if not 0 <= self.curled_threshold < self.extended_threshold <= 1:
            raise ValueError("Need 0 <= curled_threshold < extended_threshold <= 1.")
        if min(self.vote_window, self.enter_frames, self.motion_min_samples) < 1:
            raise ValueError("vote_window, enter_frames and motion_min_samples must be >= 1.")
        if min(self.exit_grace_frames, self.lost_grace_frames) < 0:
            raise ValueError("Grace periods cannot be negative.")
        if abs(sum(self.thumb_weights) - 1.0) > 1e-6:
            raise ValueError("thumb_weights must sum to 1.")


@dataclass(frozen=True)
class EffectsConfig:
    """AR effect tunables. Sizes are ratios of hand scale so effects follow the
    hand's distance from the camera; motion is in pixels and seconds."""

    max_dt: float = 0.05  # Longer frames are clamped so a stall cannot teleport physics.
    # Particles: capacities, low-resolution splat buffer, and per-system motion.
    fire_capacity: int = 1500
    smoke_capacity: int = 500
    spark_capacity: int = 600
    chunk_capacity: int = 400
    spirit_capacity: int = 900
    ember_capacity: int = 700
    dust_capacity: int = 9000
    particle_render_scale: float = 0.5
    particle_buffer_budget: int = 90_000  # Max pixels in a splat buffer.
    particle_max_blur_px: float = 6.0  # Largest blur sigma in buffer pixels.
    particle_size_bins: int = 3
    fire_drag: float = 1.6
    fire_gravity: float = -320.0  # Negative: flames rise.
    fire_turbulence: float = 520.0
    smoke_drag: float = 1.2
    smoke_gravity: float = -90.0
    smoke_turbulence: float = 160.0
    spark_drag: float = 2.2
    spark_gravity: float = 500.0
    chunk_gravity: float = 1900.0
    chunk_drag: float = 0.4
    # Fireball jutsu.
    fireball_hold_seconds: float = 0.5
    fireball_cooldown_s: float = 1.0
    fireball_size: float = 1.15  # Diameter / hand scale.
    fireball_offset: float = 0.95  # Hover height above the palm, in hand scales.
    fireball_spring: float = 140.0  # Follow stiffness (1/s^2), critically damped.
    fireball_throw_speed: float = 5.0  # Palm speed that throws, hand scales/s.
    fireball_launch_gain: float = 1.7
    fireball_min_launch_px: float = 950.0
    fireball_gravity: float = 320.0
    fireball_growth: float = 0.45  # Relative size growth per second in flight (fake depth).
    fireball_max_flight_s: float = 2.5
    fireball_margin_px: int = 60
    fireball_emit_rate: float = 260.0  # Flame particles per second at 50 px radius.
    fireball_lost_hold_s: float = 0.5
    fireball_trail_copies: int = 4
    fireball_dissipate_s: float = 0.45
    seal_hold_seconds: float = 0.8
    # Rasenshuriken: OK sign held, forms above the hand, thrown with a flick.
    rasen_hold_seconds: float = 0.6
    rasen_form_s: float = 0.9  # Core grows, then the blades extend.
    rasen_size: float = 0.85  # Core diameter / hand scale (blades reach 3.2x the core radius).
    rasen_offset: float = 1.9  # Height above the palm, in hand scales.
    rasen_spin_dps: float = 1500.0
    rasen_spring: float = 160.0
    rasen_gravity: float = 40.0
    rasen_growth: float = 0.3
    rasen_max_flight_s: float = 2.2
    rasen_blast_ratio: float = 4.2  # Wind-sphere radius / core radius.
    # Susanoo: both fists held toggles the spirit warrior.
    susanoo_hold_seconds: float = 1.0
    susanoo_rise_s: float = 1.2
    susanoo_fade_s: float = 0.8
    # Jujutsu Kaisen: Blue (point), Red (peace), Hollow Purple (bring them together).
    orb_hold_seconds: float = 0.4
    orb_size: float = 0.5  # Orb diameter / hand scale.
    orb_force: float = 5200.0  # px/s^2 pull (Blue) or push (Red) on nearby blocks.
    purple_fuse_distance: float = 1.4  # Blue and Red this close (hand scales) fuse.
    purple_size: float = 1.6  # Hollow Purple diameter / hand scale while held.
    purple_growth: float = 0.9  # It swells as it travels.
    purple_trail_s: float = 1.2  # How long the erased tunnel lingers.
    # Domain Expansions: crossed fingers (Unlimited Void) or prayer (Malevolent Shrine).
    domain_hold_seconds: float = 1.0
    domain_expand_s: float = 0.9
    domain_collapse_s: float = 0.6
    domain_max_s: float = 15.0
    # The snap (Draw mode): turn to dust and back.
    snap_vanish_s: float = 2.8
    snap_gone_s: float = 8.0
    snap_reappear_s: float = 1.6
    dust_per_frame: int = 1400
    dust_wind: tuple[float, float] = (240.0, -110.0)  # px/s: drifts right and up.
    dust_life: tuple[float, float] = (1.2, 2.6)
    great_fireball_window_s: float = 2.0
    great_fireball_ratio: float = 0.75  # Diameter / wrist distance.
    # Explosions and screen effects.
    explosion_ratio: float = 2.6  # Blast radius / fireball radius.
    explosion_particles: int = 180
    explosion_ring_s: float = 0.45
    flash_frames: int = 6
    flash_strength: float = 0.4
    shake_px: float = 16.0  # At a 60 px blast radius; scales with the blast.
    shake_decay_s: float = 0.22
    # Chidori: FIST held, then opened.
    chidori_arm_seconds: float = 0.6
    chidori_release_window_s: float = 1.0
    chidori_bolts: int = 8
    chidori_refresh_frames: int = 2
    chidori_reach: tuple[float, float] = (1.1, 1.9)  # Random targets, hand scales from the palm.
    chidori_roughness: float = 0.28
    chidori_depth: int = 5
    chidori_fade_s: float = 0.25
    # BGR: halo, mid, and core passes.
    chidori_colors: tuple[tuple[int, int, int], ...] = ((255, 110, 30), (255, 235, 120), (255, 255, 255))
    chidori_widths: tuple[int, int, int] = (9, 4, 1)
    # Blocks.
    block_kinds: tuple[str, ...] = ("grass", "stone", "planks", "tnt", "diamond_ore")
    block_size: float = 0.9  # Width / hand scale at spawn.
    block_size_quantum: int = 4  # Widths snap to this so sprites can be cached.
    block_min_px: int = 28
    floor_ratio: float = 0.92
    gravity: float = 2300.0
    restitution: float = 0.22
    bounce_min_speed: float = 220.0
    ground_friction: float = 9.0
    sleep_speed: float = 14.0
    sleep_frames: int = 8
    max_blocks: int = 40
    fade_out_s: float = 0.4
    grab_spring: float = 420.0
    grab_radius: float = 0.55  # Hand scales; for beam endpoints and bodies.
    block_pick_pad: float = 0.3  # Pinch/point this close (hand scales) to a block counts.
    max_throw_px: float = 2600.0
    snap_to_grid: bool = True  # Released blocks drop into Minecraft-style columns.
    grid_snap_max_speed: float = 700.0  # Faster releases are throws and fly freely.
    break_seconds: float = 0.35
    crack_decay: float = 2.0  # Cracks fade this many times slower than they grow.
    punch_speed: float = 3.0  # A fist moving this fast (hand scales/s) smashes blocks.
    punch_cooldown_s: float = 0.15
    tnt_fuse_s: float = 1.0
    tnt_chain_fuse_s: float = 0.25
    tnt_radius: float = 1.8  # Blast radius / block width.
    clear_hold_seconds: float = 1.0
    # Two-hand beams.
    beam_connect_distance: float = 0.5  # Pinch points this close (hand scales) connect.
    beam_min_length: float = 1.0  # Shorter beams fizzle on release.
    beam_thickness: float = 0.16
    max_beams: int = 12
    beam_platform_max_slope_deg: float = 55.0
    beam_slide_slope_deg: float = 14.0
    beam_color: tuple[int, int, int] = (255, 200, 40)  # BGR electric cyan.
    beam_pulse_speed: float = 0.6  # Beam lengths per second.
    # Modes.
    mode_hold_seconds: float = 1.0  # ROCK held this long cycles the mode.
    mode_banner_s: float = 1.1

    def __post_init__(self) -> None:
        """Catch impossible settings at startup."""
        if self.max_dt <= 0 or self.fire_capacity < 1 or self.max_blocks < 1:
            raise ValueError("max_dt, fire_capacity, and max_blocks must be positive.")
        if not 0 < self.particle_render_scale <= 1:
            raise ValueError("particle_render_scale must be in (0, 1].")
        if not 6 <= self.chidori_bolts <= 10:
            raise ValueError("chidori_bolts must be between 6 and 10.")


def default_gesture_actions() -> Mapping[Gesture, GestureAction]:
    """Return the read-only Draw-mode bindings; Phase 4 modes will supply their own."""
    return MappingProxyType({
        Gesture.POINT: GestureAction(Action.DRAW, Trigger.ACTIVE),
        Gesture.FIST: GestureAction(Action.HOVER, Trigger.ACTIVE),
        Gesture.OPEN_PALM: GestureAction(Action.CLEAR, Trigger.HOLD, 1.0),
        Gesture.PEACE: GestureAction(Action.NEXT_COLOR, Trigger.STARTED),
        Gesture.THUMBS_UP: GestureAction(Action.SAVE_PNG, Trigger.STARTED),
        Gesture.SWIPE_LEFT: GestureAction(Action.UNDO, Trigger.STARTED),
        Gesture.SWIPE_RIGHT: GestureAction(Action.REDO, Trigger.STARTED),
    })


@dataclass(frozen=True)
class MenuTheme:
    """Look and feel of the hand-operated menu (BGR colors, pixels at 1280x720)."""

    backdrop_opacity: float = 0.55  # How much the scene darkens behind the panel.
    panel: Color = (36, 28, 24)
    panel_opacity: float = 0.9
    border: Color = (110, 92, 76)
    item: Color = (56, 46, 40)
    hover: Color = (96, 74, 54)
    accent: Color = (255, 200, 60)  # Selected values, slider fills, dwell rings.
    on: Color = (120, 230, 120)
    text: Color = (242, 242, 242)
    muted: Color = (165, 165, 165)
    disabled: Color = (95, 95, 95)
    danger: Color = (90, 90, 235)
    corner_radius: int = 14
    columns: int = 2
    width: int = 1040
    row_height: int = 60
    gap: int = 12
    title_scale: float = 1.1
    label_scale: float = 0.62
    dwell_seconds: float = 0.8
    cursor_radius: int = 15
    # A pinch drags the index tip toward the thumb, so a click uses the item that
    # was under the cursor this long before the pinch closed.
    pinch_lookback_s: float = 0.12
    cursor_min_cutoff: float = 1.0  # One Euro smoothing of the menu cursor.
    cursor_beta: float = 0.02


@dataclass(frozen=True)
class Config:
    """Central settings; paths stay beside this script regardless of IDE cwd."""

    base_dir: Path = field(default_factory=resource_dir)
    # Where drawings, screenshots, recordings and logs go (None = base_dir).
    data_dir: Path | None = field(default_factory=output_root)
    width: int = 1280
    height: int = 720
    camera_fps: int = 30
    # Uncompressed YUY2 at 720p exceeds what many USB webcams can send at 30 FPS,
    # so they silently drop to 10-15 FPS. Empty string keeps the driver default.
    camera_fourcc: str = "MJPG"
    camera_indices: tuple[int, ...] = (0, 1, 2)
    max_read_failures: int = 30  # Consecutive failed reads before the camera counts as lost.
    retry_delay_ms: int = 30
    # A read that has not returned for this long means the driver is stuck (USB
    # glitch, privacy shutter): the stream starts a fresh capture in a new thread.
    camera_stall_s: float = 2.5
    reconnect_interval_s: float = 1.0
    # Detection on a smaller copy of the frame: the landmark model works on a
    # 224 px crop anyway, so 0.75 loses little accuracy and saves conversion time.
    detect_scale: float = 0.75
    max_frame_failures: int = 30  # Consecutive frames that raised before the app gives up.
    # Profiler (F10) and adaptive quality.
    profiler_history: int = 120
    show_profiler: bool = False
    adaptive_quality: bool = True
    # Achievable FPS (from processing time, not camera rate: a 15 FPS webcam must
    # not make the app throw away quality it can afford).
    quality_target_fps: float = 24.0
    quality_down_after_s: float = 2.0
    quality_up_after_s: float = 5.0
    quality_headroom: float = 1.4  # Step back up only above target x headroom.
    quality_min_detect_scale: float = 0.5
    # Audio (only if pygame is installed) and recording.
    sound_folder: str = "assets/sounds"
    sound_sample_rate: int = 44_100
    audio_channels: int = 12  # Simultaneous sounds; more are dropped, not queued.
    record_folder: str = "recordings"
    record_queue: int = 48  # Frames buffered for the writer thread.
    menu: MenuTheme = field(default_factory=MenuTheme)
    num_hands: int = 2
    detection_confidence: float = 0.7
    presence_confidence: float = 0.7
    tracking_confidence: float = 0.7
    bbox_padding: int = 20
    # Same-label detections closer than this (mean landmark gap / hand scale)
    # are one hand detected twice, not two hands.
    duplicate_hand_ratio: float = 0.25
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    download_timeout: float = 30.0
    download_chunk_size: int = 128 * 1024
    # Person segmentation (domains, Susanoo, snap-to-dust). Runs only when needed.
    segmenter_url: str = (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    )
    segmentation_width: int = 640
    mask_low: float = 0.5  # Confidence below this is background (drops curtain-rod ghosts)...
    mask_high: float = 0.8  # ...above this is fully person; in between stays soft (hair).
    mask_smoothing: float = 0.4  # Temporal EMA weight of the previous mask.
    background_learn_rate: float = 0.12
    background_every_frames: int = 3  # In Draw mode, learn the background on every Nth frame.
    glow: bool = False
    glow_strength: float = 0.65
    glow_kernel: int = 21
    glow_sigma: float = 7.0
    show_indices: bool = False
    show_hud: bool = True
    bone_thickness: int = 3
    joint_radius: int = 5
    tip_radius: int = 8
    joint_outline: int = 2
    fps_smoothing: float = 0.9
    hud_opacity: float = 0.65
    text_scale: float = 0.65
    text_thickness: int = 1
    margin: int = 12
    window_name: str = "AR Hands"
    text_color: tuple[int, int, int] = (245, 245, 245)
    dark_color: tuple[int, int, int] = (18, 18, 24)
    # BGR palettes: thumb, index, middle, ring, pinky, palm.
    right_palette: tuple[tuple[int, int, int], ...] = (
        (40, 150, 255), (70, 255, 100), (255, 220, 50),
        (255, 80, 210), (70, 70, 255), (220, 240, 255),
    )
    left_palette: tuple[tuple[int, int, int], ...] = (
        (90, 190, 255), (160, 255, 80), (255, 160, 80),
        (210, 100, 255), (150, 90, 255), (255, 240, 200),
    )
    # Drawing has its own glow; the original G key still controls skeleton glow.
    palette: tuple[tuple[int, int, int], ...] = (
        (255, 255, 0), (255, 0, 255), (0, 255, 100), (0, 140, 255),
        (0, 255, 255), (147, 20, 255), (255, 255, 255),
    )
    palette_names: tuple[str, ...] = (
        "Cyan", "Magenta", "Lime", "Orange", "Yellow", "Hot pink", "White",
    )
    default_color: int = 0
    default_thickness: int = 6
    min_thickness: int = 2
    max_thickness: int = 30
    thickness_step: int = 1
    drawing_glow_strength: float = 0.85
    halo_scale: float = 0.5
    halo_width: float = 4.0
    halo_sigma: float = 4.0
    halo_padding_sigmas: float = 4.0
    core_width: float = 0.3
    core_white_mix: float = 0.85
    spline_samples: int = 4
    smoothing_min_cutoff: float = 1.2
    smoothing_beta: float = 0.015
    smoothing_d_cutoff: float = 1.0
    # Pen adapter. Pen up/down timing now comes from the gesture stabilizer.
    min_hand_size: float = 8.0
    max_jump_hand_sizes: float = 2.5          # One-frame teleport = label swap: split.
    resume_max_jump_hand_sizes: float = 0.6   # Resuming farther away after a gap: split.
    pen_gap_frames: int = 2                   # Dry frames before resuming counts as a gap.
    pen_backfill_frames: int = 8              # Recover the stroke start the stabilizer delayed.
    fade_seconds: float = 6.0
    fade_enabled: bool = False
    min_point_spacing: float = 2.0
    max_stroke_points: int = 4096
    max_strokes: int = 256
    cursor_radius: int = 9
    cursor_outline: int = 2
    cursor_halo_radius: int = 24
    cursor_halo_sigma: float = 7.0
    cursor_glow_strength: float = 0.6
    show_skeleton: bool = True
    hud_brush_offset: int = 44
    # Gesture engine and its Draw-mode bindings.
    gestures: GestureConfig = field(default_factory=GestureConfig)
    gesture_actions: Mapping[Gesture, GestureAction] = field(
        default_factory=default_gesture_actions, hash=False)
    hold_ring_ratio: float = 1.1    # Hold-progress ring radius / hand scale.
    hold_ring_thickness: int = 4
    toast_seconds: float = 1.6
    show_debug: bool = False
    debug_log_size: int = 8
    debug_arrow_seconds: float = 0.25  # Arrow = where the palm is heading in this time.
    # BGR bar colors for extended, partial, curled fingers in the debug overlay.
    state_colors: tuple[tuple[int, int, int], ...] = ((90, 230, 90), (0, 210, 255), (80, 80, 255))
    # Phase 4 effects and the sprite override folder (assets/sprites/*.png).
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    sprite_folder: str = "assets/sprites"

    def __post_init__(self) -> None:
        """Reject settings that would fail later in a confusing way."""
        if not 0.25 <= self.detect_scale <= 1.0:
            raise ValueError("detect_scale must be between 0.25 and 1.0.")
        if self.max_frame_failures < 1 or self.reconnect_interval_s <= 0:
            raise ValueError("max_frame_failures and reconnect_interval_s must be positive.")

    @property
    def output_dir(self) -> Path:
        """Root folder for drawings, screenshots, recordings and logs."""
        return self.data_dir or self.base_dir
