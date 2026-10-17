"""Deterministic safety checks that wrap the translation: numbers, glossary terms, read-back similarity.

These run without any API call, so they are cheap, repeatable and easy to test. They are
heuristics: digit forms are compared, but numbers spelled out as words are not understood.
"""
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GROUPED = re.compile(r"\d{1,3}(,\d{2,3})+(\.\d+)?")  # 1,000 and Indian-style 1,00,000
_TRIM = ".,!?;:।\"'()[]{}"
SIMILARITY_FLOOR = 0.30  # uncalibrated: below this a round trip looks like a different sentence


def normalize_digits(text: str) -> str:
    """Map every Unicode decimal digit (Devanagari, Tamil, Telugu, ...) to ASCII 0-9."""
    out = []
    for ch in text:
        d = unicodedata.decimal(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


def extract_numbers(text: str) -> list[str]:
    found = []
    for tok in _NUMBER.findall(normalize_digits(text)):
        tok = tok.rstrip(".,")
        found.append(tok.replace(",", "") if _GROUPED.fullmatch(tok) else tok)
    return found


def compare_numbers(source: str, translated: str) -> dict:
    """Do the same numbers appear on both sides? Compared as multisets of digit strings."""
    a, b = extract_numbers(source), extract_numbers(translated)
    if not a and not b:
        return {"status": "none", "source": [], "translated": [], "missing": [], "extra": []}
    only_a, only_b = Counter(a) - Counter(b), Counter(b) - Counter(a)
    return {
        "status": "differs" if only_a or only_b else "ok",
        "source": a, "translated": b,
        "missing": sorted(only_a.elements()), "extra": sorted(only_b.elements()),
    }


def parse_glossary(raw: str) -> list[tuple[str, str | None]]:
    """One term per line; `term => preferred rendering` is optional. Capped to what Saaras accepts."""
    terms = []
    for line in raw.splitlines():
        parts = re.split(r"\s*(?:=>|->)\s*", line.strip(), maxsplit=1)
        term = parts[0].strip()
        if not term or len(term) > 64:
            continue
        target = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        terms.append((term, target))
    return terms[:50]


def match_glossary(transcript: str, glossary: list[tuple[str, str | None]], threshold: float = 0.8):
    """Evidence gate: only terms that actually appear (fuzzily) in this transcript are enforced.

    Alignment-free presence check with fuzzy matching, so a slightly misspelled name still counts.
    """
    words = [w.strip(_TRIM) for w in transcript.split()]
    words = [w for w in words if w]
    folded = transcript.casefold()
    hits = []
    for term, target in glossary:
        needle = " ".join(w.strip(_TRIM) for w in term.split()).casefold()  # same punctuation rules as the words
        n = len(needle.split())
        if not n:
            continue
        if term.casefold() in folded or any(
            SequenceMatcher(None, needle, " ".join(words[i:i + n]).casefold()).ratio() >= threshold
            for i in range(len(words) - n + 1)
        ):
            hits.append((term, target))
    return hits


def _grams(text: str) -> Counter:
    t = re.sub(r"[\s" + re.escape(_TRIM) + r"]+", " ", text.casefold()).strip()
    if len(t) < 3:
        return Counter([t]) if t else Counter()
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def similarity(a: str, b: str) -> float:
    """Character-trigram Dice similarity, 0 to 1. Both texts must be in the same language."""
    ga, gb = _grams(a), _grams(b)
    total = sum(ga.values()) + sum(gb.values())
    return 2 * sum((ga & gb).values()) / total if total else 0.0


def grade(sim: float, numbers: dict, judge: dict | None) -> tuple[str, list[str]]:
    """Combine the signals into "ok" or "check" plus human-readable reasons."""
    reasons = []
    if numbers["status"] == "differs":
        reasons.append(
            f"numbers differ (said {', '.join(numbers['source']) or 'none'}; "
            f"translated {', '.join(numbers['translated']) or 'none'})"
        )
    if sim < SIMILARITY_FLOOR:
        reasons.append(f"read-back differs a lot from what was said (similarity {sim:.2f})")
    if judge and judge.get("score") is not None and judge["score"] <= 4:
        reasons.append(f"judge scored {judge['score']}/5")
        reasons.extend(judge.get("issues", []))
    return ("check" if reasons else "ok"), reasons
