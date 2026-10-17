"""A sung-style rendition: speech retimed and re-pitched onto a melody.

This is Praat's PSOLA (through parselmouth) applied to ordinary text-to-speech audio, so it keeps the
speaker's voice (including a cloned one) but is NOT a trained singing voice. Each line is stretched to
its phrase length and given one pitch plateau per note. Neural singing synthesis would sound better,
but I found none that covers Indian languages.
"""
import io
import wave

import numpy as np
import parselmouth
from parselmouth.praat import call

import prosody


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69.0) / 12.0)


def _trim(x: np.ndarray, rate: int) -> np.ndarray:
    """Drop leading and trailing near-silence so the first syllable lands on the first note."""
    if len(x) == 0:
        return x
    env = np.abs(x)
    live = np.flatnonzero(env > env.max() * 0.02)
    if len(live) == 0:
        return x
    margin = int(0.02 * rate)
    return x[max(0, live[0] - margin): live[-1] + margin]


def speech_median_midi(x: np.ndarray, rate: int) -> float | None:
    pitch = parselmouth.Sound(x.astype(np.float64), sampling_frequency=rate).to_pitch(
        pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(69 + 12 * np.log2(np.median(f0) / 440.0)) if len(f0) >= 5 else None


def sing_line(speech: np.ndarray, rate: int, notes: list[dict], center_midi: float | None = None,
              min_stretch: float = 0.6, max_stretch: float = 2.4) -> np.ndarray:
    """Audio of about sum(note durations) seconds. `center_midi` fixes the key; by default the melody
    is moved to sit just above the speaker's own pitch, which keeps the PSOLA shifts small."""
    speech = _trim(np.asarray(speech, dtype=np.float32), rate)
    d0 = len(speech) / rate
    durations = np.array([n["d"] for n in notes], dtype=float)
    total = float(durations.sum())
    if d0 < 0.15 or total <= 0:
        return np.zeros(int(max(total, 0) * rate), dtype=np.float32)

    midis = np.array([n["p"] for n in notes], dtype=float)
    median = float(np.median(midis))
    squeeze = min(1.0, 12.0 / max(midis.max() - midis.min(), 1.0))  # keep the range singable
    if center_midi is None:
        spoken = speech_median_midi(speech, rate)
        center_midi = spoken + 2.0 if spoken is not None else median
    targets = np.clip(center_midi + (midis - median) * squeeze, 40.0, 79.0)

    factor = float(np.clip(total / d0, min_stretch, max_stretch))
    # Pitch-tier times live on the ORIGINAL timeline (checked against Praat), so note boundaries are
    # spread over the source duration; the duration tier then stretches everything uniformly.
    bounds = np.concatenate(([0.0], np.cumsum(durations))) / total * d0

    sound = parselmouth.Sound(speech.astype(np.float64), sampling_frequency=rate)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    tier = call("Create PitchTier", "melody", 0, d0)
    for i, midi in enumerate(targets):
        a, b = bounds[i], bounds[i + 1]
        margin = min(0.02, (b - a) / 4)
        call(tier, "Add point", a + margin, _hz(midi))
        call(tier, "Add point", b - margin, _hz(midi))
    call([tier, manipulation], "Replace pitch tier")
    stretch = call("Create DurationTier", "stretch", 0, d0)
    call(stretch, "Add point", 0.0, factor)
    call(stretch, "Add point", d0, factor)
    call([stretch, manipulation], "Replace duration tier")
    out = call(manipulation, "Get resynthesis (overlap-add)").values[0].astype(np.float32)

    want = int(total * rate)
    return np.pad(out, (0, want - len(out))) if len(out) < want else out


def render_song(line_wavs: list[bytes], notes_per_line: list[list[dict]], gap_s: float = 0.25) -> bytes:
    """Sing every line and join them with short rests. Returns a 16-bit mono WAV."""
    pieces, rate = [], None
    for wav, notes in zip(line_wavs, notes_per_line):
        speech, r = prosody.read_wav(wav)
        rate = rate or r
        pieces.append(sing_line(speech, r, notes))
        pieces.append(np.zeros(int(gap_s * r), dtype=np.float32))
    audio = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0:
        audio = audio * (0.9 / peak)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate or 24000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()
