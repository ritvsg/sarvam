"""Singable translation: translate a lyric line so it has as many syllables as the melody has notes.

The loop (give the model the exact syllable target, measure the result, ask again if it is off)
follows the syllable-count prompting and refinement used in recent song-translation work. The syllable
counter here is an approximation, so a result within one syllable is usually fine.
"""
import asyncio
import re

_BASES = {0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00}
_SCHWA_DROP = {0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00}  # Devanagari, Bengali, Gurmukhi, Gujarati, Odia
_TRIM = ".,!?;:।\"'()[]{}-–—…"


def _kind(ch: str) -> tuple[int | None, str | None]:
    """Brahmic scripts share one layout inside their Unicode blocks, so offsets identify the role."""
    cp = ord(ch)
    base, off = cp & ~0x7F, cp & 0x7F
    if base not in _BASES:
        return None, None
    if 0x05 <= off <= 0x14:
        return base, "V"  # independent vowel
    if 0x15 <= off <= 0x39 or 0x58 <= off <= 0x5F:
        return base, "C"  # consonant
    if 0x3E <= off <= 0x4C:
        return base, "S"  # vowel sign
    if off == 0x4D:
        return base, "H"  # virama
    return base, "M"      # anusvara, nukta and other marks


def _indic_word(word: str) -> int:
    kinds = [_kind(c) for c in word]
    base = next((b for b, _ in kinds if b), None)
    sylls = []  # one entry per syllable: True if its vowel is the inherent "a" of a bare consonant
    for i, (_, kind) in enumerate(kinds):
        nxt = kinds[i + 1][1] if i + 1 < len(kinds) else None
        if kind in ("V", "S"):
            sylls.append(False)
        elif kind == "C" and nxt not in ("S", "H"):
            sylls.append(True)
    if base in _SCHWA_DROP and len(sylls) > 1:
        # Hindi-style schwa deletion: the word-final "a" is silent, and so is a medial one
        # that sits before a syllable with an explicit vowel.
        if kinds[-1][1] == "C" and sylls[-1]:
            sylls.pop()
        i, keep = 1, [True] * len(sylls)
        while i < len(sylls) - 1:
            if sylls[i] and not sylls[i + 1] and keep[i - 1]:
                keep[i] = False
            i += 1
        sylls = [s for s, k in zip(sylls, keep) if k]
    return max(1, len(sylls))


def _latin_word(word: str) -> int:
    w = word.lower()
    groups = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not re.search(r"[^aeiouy]le$", w) and groups > 1:
        groups -= 1  # silent final e
    return max(1, groups)


def syllables(text: str) -> int:
    total = 0
    for raw in text.split():
        word = raw.strip(_TRIM)
        if not word or not any(c.isalpha() or _kind(c)[0] for c in word):
            continue
        total += _indic_word(word) if any(_kind(c)[0] for c in word) else _latin_word(word)
    return total


def _clean(text: str) -> str:
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    return line.strip("\"'`“”‘’ ").strip()


async def singable(client, lines: list[str], targets: list[int], src_name: str, tgt_name: str,
                   tries: int = 3) -> list[tuple[str, int]]:
    """Translate each line to fit its note count. Returns (text, syllable count) per line."""

    async def one(line: str, want: int) -> tuple[str, int]:
        system = (
            f"You adapt song lyrics from {src_name} into {tgt_name} so they can be sung to the same melody. "
            "Keep the meaning, use natural singable wording, and match the syllable count exactly. "
            "Reply with the adapted line only: no quotes, notes or explanation."
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Line ({src_name}): {line}\nIt must have exactly {want} syllables."}]
        best, best_gap = "", 10 ** 9
        for _ in range(tries):
            text = _clean(await client.chat(messages, max_tokens=160, temperature=0.3))
            have = syllables(text)
            if text and abs(have - want) < best_gap:
                best, best_gap = text, abs(have - want)
            if best_gap == 0:
                break
            messages += [{"role": "assistant", "content": text or "(empty)"},
                         {"role": "user", "content": f"That has {have} syllables. I need exactly {want}. "
                                                     f"Rewrite it with exactly {want} syllables."}]
        if not best:
            raise ValueError("the model returned no text")
        return best, syllables(best)

    return await asyncio.gather(*(one(line, want) for line, want in zip(lines, targets)))
