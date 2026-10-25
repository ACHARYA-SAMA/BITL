"""Procedural sound effects: every sound is synthesized with numpy on first run.

Nothing is downloaded. Each sound is written once as a 16-bit mono WAV in
assets/sounds/ and reused afterwards; a file you drop in with the same name
(.wav, .ogg, .mp3 or .flac) replaces the synthesized one. Generation is
seeded per sound, so a deleted file regenerates identically.

The recipes are classic sound-design building blocks: noise through a moving
band-pass filter (whooshes, wind), decaying sines with falling pitch (thumps,
booms), short noise bursts (crackle, crunch, snaps), and inharmonic partials
(bells, gongs, metal).
"""
from __future__ import annotations
from collections.abc import Callable
import logging
from pathlib import Path
import wave
import zlib
import numpy as np

LOGGER = logging.getLogger(__name__)
OVERRIDE_SUFFIXES = (".wav", ".ogg", ".mp3", ".flac")
LOOPS = frozenset({"lightning"})  # Played on repeat while the effect lasts: no end fades.
Synth = Callable[[np.random.Generator, int], np.ndarray]


# Building blocks -------------------------------------------------------------
def _time(duration: float, sr: int) -> np.ndarray:
    return np.arange(int(duration * sr)) / sr


def _band(x: np.ndarray, sr: int, low: float = 0.0, high: float | None = None, order: int = 4) -> np.ndarray:
    """Smooth FFT band-pass (low = high-pass corner, high = low-pass corner)."""
    spectrum = np.fft.rfft(x)
    freqs = np.maximum(np.fft.rfftfreq(len(x), 1 / sr), 1e-3)
    gain = np.ones_like(freqs)
    if low > 0:
        gain /= 1 + (low / freqs) ** order
    if high:
        gain /= 1 + (freqs / high) ** order
    return np.fft.irfft(spectrum * gain, n=len(x))


def _sweep(x: np.ndarray, sr: int, f0: float, f1: float, width: float = 0.5) -> np.ndarray:
    """Noise through a band-pass whose center slides from f0 to f1 (log scale).

    Short-time FFT with overlap-add: each 1024-sample frame gets a Gaussian
    band (``width`` octaves) around its own center frequency."""
    frame, hop = 1024, 256
    padded = np.concatenate((np.zeros(frame), x, np.zeros(frame)))
    count = 1 + (len(padded) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    window = np.hanning(frame)
    spectra = np.fft.rfft(padded[index] * window, axis=1)
    freqs = np.maximum(np.fft.rfftfreq(frame, 1 / sr), 1.0)
    centers = f0 * (f1 / f0) ** np.linspace(0, 1, count)
    spectra *= np.exp(-0.5 * (np.log2(freqs[None, :] / centers[:, None]) / width) ** 2)
    out = np.zeros(len(padded))
    np.add.at(out, index, np.fft.irfft(spectra, n=frame, axis=1) * window)
    return out[frame:frame + len(x)]


def _tone(freq: np.ndarray | float, t: np.ndarray, sr: int, shape: str = "sine") -> np.ndarray:
    """Oscillator with a (possibly time-varying) frequency in Hz."""
    freq = np.broadcast_to(np.asarray(freq, np.float64), t.shape)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    if shape == "saw":
        return 2 * ((phase / (2 * np.pi)) % 1.0) - 1
    return np.sin(phase)


def _decay(t: np.ndarray, tau: float, attack: float = 0.002) -> np.ndarray:
    """Fast linear attack, exponential decay."""
    return np.minimum(1.0, t / max(attack, 1e-6)) * np.exp(-t / tau)


def _swell(t: np.ndarray, peak: float = 0.4) -> np.ndarray:
    """Rises to a peak at ``peak`` (fraction of the length), then falls to zero."""
    u = t / t[-1]
    a = peak / (1 - peak)
    env = u ** a * (1 - u)
    return env / env.max()


def _bursts(rng: np.random.Generator, n: int, sr: int, count: int, tau: float,
            low: float, high: float, times: np.ndarray | None = None) -> np.ndarray:
    """``count`` short decaying noise bursts (crackle, crunch) at random or given times."""
    out = np.zeros(n)
    length = int(tau * 6 * sr)
    t = np.arange(length) / sr
    starts = (times * sr).astype(int) if times is not None else rng.integers(0, max(1, n - length), count)
    for start in starts:
        grain = rng.standard_normal(length) * np.exp(-t / tau) * rng.uniform(0.3, 1.0)
        end = min(n, start + length)
        out[start:end] += grain[:end - start]
    return _band(out, sr, low, high)


def _finish(x: np.ndarray, sr: int, peak: float = 0.9, fade: bool = True) -> np.ndarray:
    """Remove DC, fade the ends so nothing clicks (not for loops), then normalize."""
    x = x - x.mean()
    fade_in = min(len(x) // 4, int(0.0015 * sr)) if fade else 0
    fade_out = min(len(x) // 4, int(0.006 * sr)) if fade else 0
    if fade_in:
        x[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out:
        x[-fade_out:] *= np.linspace(1, 0, fade_out)
    top = np.abs(x).max()
    return x / top * peak if top > 0 else x


def _loopable(x: np.ndarray, sr: int, overlap_s: float = 0.08) -> np.ndarray:
    """Crossfade the tail into the head so a looped sound has no seam."""
    overlap = int(overlap_s * sr)
    ramp = np.linspace(0, 1, overlap)
    body = x[:-overlap].copy()
    body[:overlap] = body[:overlap] * ramp + x[-overlap:] * (1 - ramp)
    return body


# Sounds ----------------------------------------------------------------------
def whoosh(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Thrown projectile: a noise band sweeping upward, swelling and fading."""
    t = _time(0.55, sr)
    return _sweep(rng.standard_normal(len(t)), sr, 280, 2600, 0.6) * _swell(t, 0.45)


def fire_crackle(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Fireball summon: a low roar under crackling bursts."""
    t = _time(1.1, sr)
    roar = _band(rng.standard_normal(len(t)), sr, 60, 420) * _decay(t, 0.5, 0.15) * 0.8
    crackle = _bursts(rng, len(t), sr, 45, 0.004, 1500, 7000) * np.exp(-t / 0.7) * 3.0
    return roar + crackle


def explosion(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Impact: falling sub boom, a noise blast, and a crackling tail, soft-clipped."""
    t = _time(1.5, sr)
    sub = _tone(110 * (35 / 110) ** (t / t[-1]), t, sr) * _decay(t, 0.38, 0.004)
    blast = _band(rng.standard_normal(len(t)), sr, 30, 1100) * _decay(t, 0.22, 0.002)
    tail = _bursts(rng, len(t), sr, 30, 0.006, 800, 5000) * np.exp(-t / 0.5)
    return np.tanh(2.2 * (sub * 1.1 + blast * 0.9 + tail * 0.8))


def block_place(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Block drop: a short low thump with a soft click."""
    t = _time(0.2, sr)
    thump = _tone(150 * (55 / 150) ** (t / 0.08).clip(0, 1), t, sr) * _decay(t, 0.045)
    click = _band(rng.standard_normal(len(t)), sr, 200, 3000) * _decay(t, 0.004) * 0.5
    return thump + click


def block_break(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Block shatter: a cluster of crunchy noise bursts over a dull thud."""
    t = _time(0.4, sr)
    times = np.array([0.0, 0.028, 0.06, 0.1, 0.15, 0.21])
    crunch = _bursts(rng, len(t), sr, len(times), 0.018, 700, 5200, times) * np.exp(-t / 0.18)
    thud = _tone(95, t, sr) * _decay(t, 0.05) * 0.6
    return crunch * 1.4 + thud


def lightning(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Chidori (looped while active): flickering electric buzz with crackle."""
    t = _time(1.08, sr)
    buzz = _tone(110, t, sr, "saw") * 0.6 + _tone(220.5, t, sr, "saw") * 0.3
    flicker = _band(rng.standard_normal(len(t)), sr, 0, 30)
    flicker = 0.5 + 0.5 * np.clip(flicker / (np.abs(flicker).max() + 1e-9) * 2.5, -1, 1)
    crackle = _bursts(rng, len(t), sr, 70, 0.003, 2000, 9000) * 2.5
    return _loopable(_band(buzz, sr, 80, 4000) * flicker + crackle, sr)


def chidori_charge(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Charging fist: crackle getting denser while a buzz rises in pitch."""
    t = _time(0.7, sr)
    buzz = _tone(70 * 2.2 ** (t / t[-1]), t, sr, "saw") * (t / t[-1]) * 0.5
    density = (t / t[-1]) ** 2
    crackle = _bursts(rng, len(t), sr, 50, 0.003, 2500, 9000) * density * 2.0
    return _band(buzz, sr, 60, 3000) + crackle


def ui_click(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Menu select: a short bright blip."""
    t = _time(0.06, sr)
    return _tone(1900 * (1200 / 1900) ** (t / t[-1]), t, sr) * _decay(t, 0.012, 0.001)


def ui_hover(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Menu hover: a barely-there tick."""
    t = _time(0.03, sr)
    return _tone(3200, t, sr) * _decay(t, 0.004, 0.0005) * 0.4


def mode_change(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Mode switch: a rising two-note chime."""
    t = _time(0.55, sr)
    first = sum(_tone(660 * k, t, sr) / k for k in (1, 2, 3)) * _decay(t, 0.12)
    second_t = np.clip(t - 0.11, 0, None)
    second = sum(_tone(990 * k, second_t, sr) / k for k in (1, 2, 3)) * _decay(second_t, 0.18) * (t > 0.11)
    return first * 0.7 + second


def rec(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Recording start/stop: a clean beep."""
    t = _time(0.18, sr)
    return _tone(880, t, sr) * np.minimum(1, t / 0.005) * np.minimum(1, (t[-1] - t) / 0.03)


def wind(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Rasenshuriken forming: whirring wind pulsing at the blade rate, and a whistle."""
    t = _time(1.4, sr)
    whirl = _sweep(rng.standard_normal(len(t)), sr, 400, 1800, 0.7)
    pulse = 0.6 + 0.4 * np.sin(2 * np.pi * 18 * t)
    whistle = _tone(1250 + 40 * np.sin(2 * np.pi * 6 * t), t, sr) * 0.12
    return (whirl * pulse + whistle) * np.minimum(1, t / 0.6)


def wind_blast(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Rasenshuriken impact: a huge rush of air collapsing downward, with a thump."""
    t = _time(1.7, sr)
    rush = _sweep(rng.standard_normal(len(t)), sr, 2400, 180, 0.9) * _decay(t, 0.55, 0.01)
    thump = _tone(80 * (40 / 80) ** (t / t[-1]), t, sr) * _decay(t, 0.25) * 0.8
    return np.tanh(1.6 * (rush * 1.2 + thump))


def susanoo(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Susanoo rising: a swelling low rumble, a dark pad, and a reverse-cymbal rise."""
    t = _time(2.6, sr)
    rumble = _band(_tone(45, t, sr, "saw") + _tone(67.6, t, sr, "saw") * 0.6, sr, 25, 260)
    vibrato = 1 + 0.004 * np.sin(2 * np.pi * 5 * t)
    pad = sum(_tone(f * vibrato, t, sr) for f in (220, 261.6, 329.6)) * 0.08
    rise = _band(rng.standard_normal(len(t)), sr, 3000) * (t / t[-1]) ** 3 * 0.5
    return (rumble * 0.9 + pad) * _swell(t, 0.55) + rise * (t < 1.6)


def orb_blue(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Blue: suction, a noise band falling inward, over a hum that grows."""
    t = _time(1.0, sr)
    suction = _sweep(rng.standard_normal(len(t)), sr, 3400, 260, 0.6) * (t / t[-1]) ** 1.5
    hum = _tone(70 + 30 * t, t, sr) * (t / t[-1]) * 0.7
    return suction + hum


def orb_red(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Red: a distorted throbbing pulse, pushing outward."""
    t = _time(0.85, sr)
    throb = np.tanh(3 * _tone(92, t, sr) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t)))
    burst = _sweep(rng.standard_normal(len(t)), sr, 300, 2400, 0.6) * _decay(t, 0.25, 0.01) * 0.6
    return throb * _swell(t, 0.3) * 0.7 + burst


def purple_charge(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Blue and Red fusing: beating low tones and a crescendo of noise."""
    t = _time(1.3, sr)
    beat = _tone(110, t, sr) + _tone(111.7, t, sr) + _tone(220.4, t, sr) * 0.5
    swell = _band(rng.standard_normal(len(t)), sr, 500, 6000) * (t / t[-1]) ** 2 * 0.6
    return (beat * 0.5 + swell) * np.minimum(1, t / 0.3)


def purple_fire(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Hollow Purple fired: a rising tearing sweep on top of a deep boom."""
    t = _time(1.8, sr)
    tear = _sweep(rng.standard_normal(len(t)), sr, 180, 5000, 0.8) * _decay(t, 0.6, 0.02)
    boom = _tone(70 * (30 / 70) ** (t / t[-1]), t, sr) * _decay(t, 0.5, 0.005)
    return np.tanh(1.8 * (tear + boom * 1.2))


def domain(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Domain Expansion: a gong hit (inharmonic partials), sub drop, and a long shimmer."""
    t = _time(3.2, sr)
    gong = sum(_tone(98 * ratio, t, sr) * _decay(t, tau, 0.004) / (1 + i)
               for i, (ratio, tau) in enumerate(((1.0, 1.6), (2.76, 1.0), (5.4, 0.6), (8.93, 0.35))))
    drop = _tone(80 * (28 / 80) ** (t / t[-1]), t, sr) * _decay(t, 1.0, 0.01)
    shimmer = _band(rng.standard_normal(len(t)), sr, 4000, 12000) * _decay(t, 1.2, 0.3) * 0.25
    return gong * 0.8 + drop * 0.8 + shimmer


def slash(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Shrine slash: a fast high swish with a metallic ring."""
    t = _time(0.38, sr)
    swish = _sweep(rng.standard_normal(len(t)), sr, 7000, 1600, 0.5) * _decay(t, 0.07, 0.004)
    ring = sum(_tone(f, t, sr) * _decay(t, 0.09) for f in (2630, 3910, 5270)) * 0.15
    return swish + ring


def snap(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Finger snap: a sharp band-limited click with a short resonant body."""
    t = _time(0.12, sr)
    crack = _band(rng.standard_normal(len(t)), sr, 1400, 4500) * _decay(t, 0.008, 0.0005)
    body = _tone(2200, t, sr) * _decay(t, 0.018, 0.0005) * 0.4
    return crack + body


def dust(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Turning to dust: soft granular sand hiss drifting away."""
    t = _time(2.8, sr)
    hiss = _band(rng.standard_normal(len(t)), sr, 2500, 11000)
    grains = _band(np.abs(rng.standard_normal(len(t))) ** 3, sr, 0, 60)
    grains = grains / (grains.max() + 1e-9)
    return hiss * (0.25 + grains) * _swell(t, 0.25)


def reform(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Coming back: the dust sound reversed, with a rising shimmer."""
    t = _time(1.6, sr)
    sand = _band(rng.standard_normal(len(t)), sr, 2500, 11000) * (t / t[-1]) ** 2
    shimmer = _tone(600 * 2 ** (t / t[-1]), t, sr) * (t / t[-1]) ** 3 * 0.25
    return sand + shimmer


def seal(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Hand seal formed: a small bright bell."""
    t = _time(0.8, sr)
    return sum(_tone(f, t, sr) * _decay(t, tau, 0.002) / k
               for k, (f, tau) in enumerate(((1200, 0.5), (2410, 0.3), (3630, 0.2)), start=1))


def fuse(rng: np.random.Generator, sr: int) -> np.ndarray:
    """TNT ignited: a fizzing fuse."""
    t = _time(1.0, sr)
    fizz = _band(rng.standard_normal(len(t)), sr, 3000, 8000) * 0.6
    sparks = _bursts(rng, len(t), sr, 40, 0.002, 4000, 10000) * 2.0
    return fizz + sparks


def beam(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Neon beam connecting: an electric zap chirping upward."""
    t = _time(0.5, sr)
    zap = _band(_tone(200 * 4.5 ** (t / t[-1]), t, sr, "saw"), sr, 100, 5000)
    trem = 0.6 + 0.4 * np.sin(2 * np.pi * 40 * t)
    return zap * trem * _decay(t, 0.25, 0.01) + _bursts(rng, len(t), sr, 12, 0.002, 3000, 9000)


SOUNDS: dict[str, Synth] = {
    "whoosh": whoosh, "fire_crackle": fire_crackle, "explosion": explosion,
    "block_place": block_place, "block_break": block_break, "lightning": lightning,
    "chidori_charge": chidori_charge, "ui_click": ui_click, "ui_hover": ui_hover,
    "mode_change": mode_change, "rec": rec, "wind": wind, "wind_blast": wind_blast,
    "susanoo": susanoo, "orb_blue": orb_blue, "orb_red": orb_red,
    "purple_charge": purple_charge, "purple_fire": purple_fire, "domain": domain,
    "slash": slash, "snap": snap, "dust": dust, "reform": reform, "seal": seal,
    "fuse": fuse, "beam": beam,
}


def synthesize(name: str, sr: int = 44_100) -> np.ndarray:
    """One finished sound as float samples in [-1, 1] (deterministic per name)."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return _finish(SOUNDS[name](rng, sr).astype(np.float64), sr, fade=name not in LOOPS)


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """16-bit mono WAV, written to a temporary name first so a crash leaves no half file."""
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    temporary = path.with_name(path.name + ".part")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(data.tobytes())
    temporary.replace(path)


def find_sound(folder: Path, name: str) -> Path | None:
    """A file for ``name`` in any supported format (user overrides win: same name)."""
    for suffix in OVERRIDE_SUFFIXES:
        path = folder / f"{name}{suffix}"
        if path.is_file() and path.stat().st_size > 44:
            return path
    return None


def ensure_sounds(folder: Path, sr: int = 44_100) -> dict[str, Path]:
    """Path of every sound, synthesizing any that are missing; unwritable folders are tolerated."""
    paths: dict[str, Path] = {}
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Sound folder %s unavailable (%s).", folder, exc)
    created = 0
    for name in SOUNDS:
        existing = find_sound(folder, name)
        if existing is not None:
            paths[name] = existing
            continue
        path = folder / f"{name}.wav"
        try:
            write_wav(path, synthesize(name, sr), sr)
        except OSError as exc:
            LOGGER.warning("Could not write %s: %s", path, exc)
            continue
        paths[name] = path
        created += 1
    if created:
        LOGGER.info("Synthesized %d sound effects into %s", created, folder)
    return paths
