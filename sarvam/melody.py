"""Turn hummed or sung audio into notes, and match a hummed tune against known melodies.

Query-by-humming in its classic form: pitch contour, key removed, then dynamic time warping.
Learned hum embeddings (see the README) do better at scale, but need training data this app lacks.
"""
import numpy as np

import prosody

GRID_HZ = 8          # DTW grid: bins per second
MIN_NOTE_S = 0.06
PHRASE_GAP_S = 0.4   # a silence this long ends a phrase (a line of the song)


def contour(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-10 ms MIDI pitch (NaN where unvoiced), dB, and the hop in seconds."""
    frames, hop = prosody.frame_signal(x, rate, win_ms=40, hop_ms=10)
    if len(frames) == 0:
        return np.zeros(0), np.zeros(0), hop / rate
    db = prosody.frame_db(frames)
    gate = max(min(np.percentile(db, 10) + 10.0, db.max() - 6.0), db.max() - 35.0)
    f0 = prosody.pitch_track(frames, rate, fmin=80.0, fmax=800.0, min_strength=0.5)
    f0 = np.where(db > gate, f0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = 69 + 12 * np.log2(f0 / 440.0)
    return _clean(midi), db, hop / rate


def _clean(midi: np.ndarray) -> np.ndarray:
    """Median-smooth, pull octave jumps back toward the local pitch, drop tiny voiced islands."""
    out = midi.copy()
    n = len(midi)
    for i in range(n):
        window = [v for v in midi[max(0, i - 2): i + 3] if v == v]
        out[i] = float(np.median(window)) if len(window) >= 3 and midi[i] == midi[i] else np.nan
    recent: list[float] = []
    for i in range(n):
        v = out[i]
        if v != v:
            continue
        if len(recent) >= 5:
            ref = float(np.median(recent[-20:]))
            if abs(v - ref) > 9:
                best = min((v - 12, v + 12), key=lambda c: abs(c - ref))
                if abs(best - ref) < 5:
                    out[i] = v = best
        recent.append(v)
    i = 0
    while i < n:  # voiced islands shorter than 40 ms are noise
        if out[i] == out[i]:
            j = i
            while j < n and out[j] == out[j]:
                j += 1
            if j - i < 4:
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    return out


def _spans(midi: np.ndarray, step: float) -> list[tuple[int, int]]:
    """Voiced spans, merged across gaps shorter than PHRASE_GAP_S, dropping very short ones."""
    voiced = np.flatnonzero(~np.isnan(midi))
    if len(voiced) == 0:
        return []
    spans, start, prev = [], voiced[0], voiced[0]
    for i in voiced[1:]:
        if (i - prev) * step > PHRASE_GAP_S:
            spans.append((start, prev + 1))
            start = i
        prev = i
    spans.append((start, prev + 1))
    return [(a, b) for a, b in spans if (b - a) * step >= 0.25]


def _notes(midi: np.ndarray, db: np.ndarray, step: float) -> list[dict]:
    """Notes inside one phrase: runs of equal rounded pitch, split again where the loudness dips."""
    q = np.round(midi)
    for i in range(len(q)):  # bridge unvoiced gaps by holding the previous pitch
        if q[i] != q[i]:
            q[i] = q[i - 1] if i else np.nan
    first = next((v for v in q if v == v), None)
    if first is None:
        return []
    q = np.where(np.isnan(q), first, q)

    runs: list[list[float]] = []  # [pitch, start, end)
    for i, v in enumerate(q):
        if runs and runs[-1][0] == v:
            runs[-1][2] = i + 1
        else:
            runs.append([v, i, i + 1])
    merged: list[list[float]] = []
    for r in runs:  # a run under 50 ms is a pitch glide, not a note: absorb it
        if merged and (r[2] - r[1]) * step < 0.05:
            merged[-1][2] = r[2]
        else:
            merged.append(r)

    dbs = np.convolve(db, np.ones(3) / 3, mode="same")
    pieces: list[tuple[float, int, int]] = []
    for pitch, a, b in merged:
        a, b = int(a), int(b)
        cuts = [a]
        for i in range(a + 4, b - 4):  # re-articulation: a dip of 5 dB or more between two louder stretches
            lo, hi = max(a, i - 8), min(b, i + 9)
            if dbs[i] == dbs[lo:hi].min() and min(dbs[lo:i].max(), dbs[i + 1:hi].max()) - dbs[i] >= 5 \
                    and i - cuts[-1] >= 6:
                cuts.append(i)
        cuts.append(b)
        pieces += [(pitch, cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]
    return [{"p": int(p), "d": round((b - a) * step, 3)} for p, a, b in pieces if (b - a) * step >= MIN_NOTE_S]


def _grid(midi: np.ndarray, step: float) -> np.ndarray:
    """Median pitch per 1/GRID_HZ second bin over voiced frames, silent bins dropped."""
    per_bin = max(1, int(round(1 / (GRID_HZ * step))))
    out = []
    for a in range(0, len(midi), per_bin):
        vals = midi[a: a + per_bin]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= max(1, per_bin // 3):
            out.append(float(np.median(vals)))
    return np.array(out)


def analyze(x: np.ndarray, rate: int) -> dict:
    midi, db, step = contour(x, rate)
    phrases = [_notes(midi[a:b], db[a:b], step) for a, b in _spans(midi, step)]
    phrases = [p for p in phrases if p]
    flat = [n["p"] for p in phrases for n in p]
    return {
        "phrases": phrases,
        "note_count": len(flat),
        "duration_s": round(sum(n["d"] for p in phrases for n in p), 2),
        "key_midi": float(np.median(flat)) if flat else None,
        "query": _grid(midi, step),
    }


def notes_to_grid(phrases: list[list[dict]]) -> np.ndarray:
    """Render known notes onto the same grid as a hummed query."""
    out: list[float] = []
    for phrase in phrases:
        for n in phrase:
            out += [float(n["p"])] * max(1, int(round(n["d"] * GRID_HZ)))
    return np.array(out)


def dtw_distance(query: np.ndarray, ref: np.ndarray) -> float:
    """Mean semitone difference of the best alignment of the whole query to any stretch of ref.

    The key is removed by medians (and 5 small offsets), the tempo by warping: each query bin may
    stay on, advance one, or skip one bin of the reference.
    """
    n, m = len(query), len(ref)
    if n < 4 or m < 4:
        return float("inf")
    r = ref - np.median(ref)
    best = float("inf")
    for offset in (-2, -1, 0, 1, 2):
        q = query - np.median(query) + offset
        prev = np.abs(q[0] - r)  # free start: the query may begin anywhere in the reference
        for i in range(1, n):
            shifted1 = np.concatenate(([np.inf], prev[:-1]))
            shifted2 = np.concatenate(([np.inf, np.inf], prev[:-2]))
            prev = np.abs(q[i] - r) + np.minimum(prev, np.minimum(shifted1, shifted2))
        best = min(best, float(prev.min()) / n)  # free end
    return best


def melody_score(dist: float) -> float:
    return 1.0 / (1.0 + dist / 2.0) if dist != float("inf") else 0.0
