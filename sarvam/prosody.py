"""Turn the speaker's delivery into an explicit "tone card".

Hear2Act (arXiv 2608.19515) found that audio LLMs barely use prosody unless it is first written
down as explicit text state. So the delivery is measured here with plain numpy, then passed on as
words: to the translation prompt, to the TTS pace, and to the listener's screen.

The thresholds are uncalibrated heuristics, not a validated emotion classifier.
"""
import io
import wave

import numpy as np

WIN_MS, HOP_MS = 25, 10


def read_wav(wav: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(wav)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"unreadable WAV: {e}") from e
    if width != 2:
        raise ValueError("expected 16-bit PCM WAV")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x, rate


def frame_signal(x: np.ndarray, rate: int, win_ms: int = WIN_MS, hop_ms: int = HOP_MS) -> tuple[np.ndarray, int]:
    """Overlapping frames, shape (n, win), plus the hop in samples."""
    win, hop = int(rate * win_ms / 1000), int(rate * hop_ms / 1000)
    if len(x) < win:
        return np.zeros((0, win), dtype=np.float32), hop
    n = 1 + (len(x) - win) // hop
    return x[np.arange(win)[None, :] + hop * np.arange(n)[:, None]], hop


def frame_db(frames: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def pitch_track(frames: np.ndarray, rate: int, fmin: float = 70.0, fmax: float = 400.0,
                min_strength: float = 0.45) -> np.ndarray:
    """F0 in Hz per frame by FFT autocorrelation with parabolic refinement; NaN where unreliable.

    Shared by the tone card (speech) and the melody extractor (humming and singing).
    """
    if len(frames) == 0:
        return np.zeros(0)
    win = frames.shape[1]
    seg = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(seg, nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :win]
    lo, hi = max(2, int(rate / fmax)), min(int(rate / fmin), win - 2)
    rows = np.arange(len(ac))
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[rows, lag] / (ac[:, 0] + 1e-9)
    y0, y1, y2 = ac[rows, lag - 1], ac[rows, lag], ac[rows, lag + 1]
    denom = y0 - 2 * y1 + y2
    refined = lag + np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0)
    return np.where(strength > min_strength, rate / refined, np.nan)


def analyze_audio(x: np.ndarray, rate: int) -> dict:
    """Pause, pitch-range and loudness-dynamics features from 16-bit mono samples."""
    if len(x) < max(int(rate * WIN_MS / 1000) * 4, int(rate * 0.3)):
        return {"speech_s": 0.0}
    frames, hop = frame_signal(x, rate)
    db = frame_db(frames)
    # Speech threshold: 10 dB over the noise floor, but never above peak - 6 dB, so a clip with no
    # silence at all (a turn cut off mid-sentence) still counts as speech.
    peak = db.max()
    voiced = db > max(min(np.percentile(db, 10) + 10.0, peak - 6.0), peak - 35.0)
    active = np.flatnonzero(voiced)
    if len(active) < 5:
        return {"speech_s": 0.0}

    # Pauses are silent runs of at least 200 ms between the first and last speech frame.
    span = voiced[active[0]: active[-1] + 1]
    pause_frames, run = 0, 0
    for v in span:
        if v:
            pause_frames += run if run >= 20 else 0
            run = 0
        else:
            run += 1
    feats = {
        "speech_s": round((len(span) - pause_frames) * hop / rate, 2),
        "pause_ratio": round(pause_frames / len(span), 3),
        "dyn_db": round(float(np.std(db[voiced])), 2),
    }

    # Pitch on speech frames; range = 10th to 90th percentile in semitones.
    f0 = pitch_track(frames[voiced], rate)
    f0 = f0[~np.isnan(f0)]
    if len(f0) >= 15:
        semitones = 12 * np.log2(f0 / np.median(f0))
        feats["pitch_range_st"] = round(float(np.percentile(semitones, 90) - np.percentile(semitones, 10)), 1)
    return feats


def describe(feats: dict, transcript: str) -> dict:
    """Words describing the delivery, plus a TTS pace multiplier."""
    words = len(transcript.split())
    speech_s = feats.get("speech_s", 0.0)
    if speech_s < 0.4 or not words:
        return {"summary": "", "pace": 1.0, "labels": {}, **feats}
    rate = words / speech_s
    labels = {}
    if rate > 3.2:
        labels["pace"] = "fast"
    elif rate < 1.8:
        labels["pace"] = "slow"
    st = feats.get("pitch_range_st")
    if st is not None:
        if st >= 8:
            labels["pitch"] = "animated pitch"
        elif st <= 4:
            labels["pitch"] = "flat pitch"
    if feats.get("dyn_db", 0) > 9:
        labels["emphasis"] = "emphatic"
    if feats.get("pause_ratio", 0) >= 0.2:
        labels["hesitant"] = "hesitant, many pauses"
    summary = ", ".join(labels.values()) or "steady, even delivery"
    return {
        "summary": summary,
        "labels": labels,
        "rate_wps": round(rate, 2),
        "pace": round(min(1.15, max(0.9, rate / 2.6)), 2),
        **feats,
    }
