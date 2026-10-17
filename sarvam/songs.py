"""The song catalog: a few public-domain songs plus whatever you save, and the matching logic.

Only public-domain songs ship with the app. Anything else you hum is handled from your own input
(the words you sing and the tune you hum), and songs you save stay on your machine in songs/user/.
"""
import json
import re
import uuid
from pathlib import Path

import checks
import lyrics
import melody

SONG_DIR = Path(__file__).parent / "songs"
USER_DIR = SONG_DIR / "user"
_NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)(?::([\d.]+))?$")
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_notes(text: str, bpm: float) -> list[dict]:
    """"C4:1 G4:0.5" -> [{"p": 60, "d": 0.6}, ...]; the number after the colon is beats (default 1)."""
    out = []
    for token in text.split():
        m = _NOTE.match(token)
        if not m:
            raise ValueError(f"bad note {token!r}")
        name, acc, octave, beats = m.groups()
        midi = 12 * (int(octave) + 1) + _SEMITONE[name.upper()] + (1 if acc == "#" else -1 if acc == "b" else 0)
        out.append({"p": midi, "d": round(float(beats or 1) * 60.0 / bpm, 3)})
    return out


def _load_file(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "notes" in raw:  # hand-written format: note strings per line
            phrases = [parse_notes(line, raw.get("tempo_bpm", 100)) for line in raw["notes"]]
        else:               # saved format: phrases of {"p", "d"}
            phrases = raw["phrases"]
        if len(phrases) != len(raw["lines"]):
            raise ValueError("lines and melody phrases differ in count")
        return {"id": raw["id"], "title": raw["title"], "language": raw.get("language", "en-IN"),
                "source": raw.get("source", ""), "lines": raw["lines"], "phrases": phrases,
                "grid": melody.notes_to_grid(phrases)}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None  # a malformed file must not take the catalog down


def load_catalog() -> list[dict]:
    files = sorted(SONG_DIR.glob("*.json")) + (sorted(USER_DIR.glob("*.json")) if USER_DIR.exists() else [])
    return [s for s in (_load_file(p) for p in files) if s]


def lyric_score(transcript: str, song_lines: list[str]) -> float | None:
    """Best trigram similarity between what was sung and any one line or opening stretch of the song."""
    if len(transcript.split()) < 2 or len(transcript) < 8:
        return None  # a hum, or too little text to trust
    best = 0.0
    for k in range(1, min(3, len(song_lines)) + 1):
        best = max(best, checks.similarity(transcript, " ".join(song_lines[:k])))
    return max(best, max(checks.similarity(transcript, line) for line in song_lines))


def rank(analysis: dict, transcript: str, catalog: list[dict], top: int = 5, floor: float = 0.2) -> list[dict]:
    """Candidates ordered by melody match, lyric match, or the average of the two when both exist."""
    query = analysis["query"]
    out = []
    for song in catalog:
        mel = melody.melody_score(min(melody.dtw_distance(query, song["grid"]), 99.0)) if len(query) >= 6 else None
        lyr = lyric_score(transcript, song["lines"])
        parts = [s for s in (mel, lyr) if s is not None]
        if not parts:
            continue
        score = sum(parts) / len(parts)
        if score < floor:
            continue
        why = ", ".join(f"{name} {val:.2f}" for name, val in (("melody", mel), ("lyrics", lyr)) if val is not None)
        out.append({"id": song["id"], "title": song["title"], "language": song["language"],
                    "source": song["source"], "score": round(score, 2), "why": why,
                    "lines": song["lines"], "melody": song["phrases"]})
    return sorted(out, key=lambda c: -c["score"])[:top]


def align_notes(lines: list[str], phrases: list[list[dict]]) -> list[list[dict]]:
    """One note list per lyric line. Equal counts pair up; otherwise notes are split by syllable weight."""
    if len(phrases) == len(lines):
        return phrases
    flat = [n for phrase in phrases for n in phrase]
    if len(flat) < len(lines):
        raise ValueError(f"{len(lines)} lines need at least {len(lines)} notes, but only {len(flat)} were heard")
    weights = [max(1, lyrics.syllables(line)) for line in lines]
    out, start, acc = [], 0, 0
    for i, w in enumerate(weights):
        acc += w
        end = round(acc / sum(weights) * len(flat))
        end = min(max(end, start + 1), len(flat) - (len(lines) - i - 1))
        out.append(flat[start:end])
        start = end
    return out


def save_user_song(title: str, lines: list[str], phrases: list[list[dict]], language: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "song"
    song_id = f"{slug}-{uuid.uuid4().hex[:6]}"
    USER_DIR.mkdir(parents=True, exist_ok=True)
    (USER_DIR / f"{song_id}.json").write_text(json.dumps({
        "id": song_id, "title": title, "language": language, "source": "saved by you",
        "lines": lines, "phrases": phrases,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return song_id
