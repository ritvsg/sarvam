"""Song endpoints: identify a hummed or sung phrase, ask for a language, sing it back translated."""
import asyncio
import base64
import statistics

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

import checks
import config
import languages
import lyrics
import melody
import prosody
import singer
import songs
from core import Run, client, http_error, peer_of, relay
from sarvam_client import SarvamError

router = APIRouter(prefix="/api")

PROMPTS = {
    "hum": "Hum or sing the first few lines of the song.",
    "language": "Which language should I sing it in?",
}
_prompt_cache: dict[tuple[str, str], tuple[str, bytes]] = {}


class NoteIn(BaseModel):
    p: int
    d: float


class SongIn(BaseModel):
    lines: list[str]
    melody: list[list[NoteIn]]
    src: str
    tgt: str
    title: str = ""
    role: str = "A"
    room: str = ""
    seq: int = 0
    sid: str = ""
    voice_id: str = ""


class SaveIn(BaseModel):
    title: str
    lines: list[str]
    melody: list[list[NoteIn]]
    language: str = "en-IN"


class PromptIn(BaseModel):
    key: str
    lang: str


def _phrases(raw: list[list[NoteIn]]) -> list[list[dict]]:
    return [[{"p": n.p, "d": n.d} for n in phrase] for phrase in raw if phrase]


@router.post("/song/identify")
async def identify(file: UploadFile = File(...)) -> dict:
    """Notes from the hum, a lyric guess from any words, and a ranked list of known songs."""
    try:
        samples, rate = prosody.read_wav(await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    run = Run()

    async def words() -> str:
        try:
            transcript, _, _ = await client.transcribe(_to_wav(samples, rate))
            return transcript
        except SarvamError as e:
            run.warnings.append(f"lyric recognition failed ({e.status}); matched by melody only")
            return ""

    transcript, analysis = await asyncio.gather(
        run.timed("stt", words()), run.timed("melody", asyncio.to_thread(melody.analyze, samples, rate)))
    if not analysis["phrases"]:
        run.warnings.append("I could not hear a melody. Try humming louder and closer to the mic.")
    candidates = songs.rank(analysis, transcript, songs.load_catalog()) if analysis["phrases"] else []
    return {"transcript": transcript, "phrases": analysis["phrases"], "note_count": analysis["note_count"],
            "duration_s": analysis["duration_s"], "candidates": candidates,
            "timings_ms": run.finish(), "warnings": run.warnings}


def _to_wav(samples, rate: int) -> bytes:
    import io
    import wave
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@router.post("/song/language")
async def pick_language(file: UploadFile = File(...)) -> dict:
    """Hear a spoken language name ("Tamil", "तमिल") and return its code, or null if not understood."""
    try:
        transcript, _, _ = await client.transcribe(await file.read())
    except SarvamError as e:
        raise http_error(e) from e
    return {"heard": transcript, "lang": languages.parse(transcript)}


@router.post("/prompt")
async def prompt(body: PromptIn) -> dict:
    """A spoken question for the song flow, in the user's language (translated once, then cached)."""
    if body.key not in PROMPTS or body.lang not in config.LANGUAGES:
        raise HTTPException(400, "unknown prompt or language")
    key = (body.key, body.lang)
    if key not in _prompt_cache:
        try:
            text = PROMPTS[body.key] if body.lang == "en-IN" else await client.translate(PROMPTS[body.key], "en-IN", body.lang)
            audio, _, _ = await client.tts(text, body.lang, "A", 1.0)
        except SarvamError as e:
            raise http_error(e) from e
        _prompt_cache[key] = (text, audio)
    text, audio = _prompt_cache[key]
    return {"text": text, "audio": base64.b64encode(audio).decode()}


@router.post("/song/save")
async def save(body: SaveIn) -> dict:
    title, lines = body.title.strip(), [ln.strip() for ln in body.lines if ln.strip()]
    if not (1 <= len(title) <= 80) or not (1 <= len(lines) <= 12) or body.language not in config.LANGUAGES:
        raise HTTPException(400, "need a title (up to 80 characters), 1 to 12 lines and a known language")
    try:
        per_line = songs.align_notes(lines, _phrases(body.melody))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"id": songs.save_user_song(title, lines, per_line, body.language)}


@router.post("/song/convert")
async def convert(body: SongIn) -> dict:
    """Translate the lyrics to fit the melody, speak each line, then retime and re-pitch it onto the tune."""
    lines = [ln.strip() for ln in body.lines if ln.strip()]
    if not (1 <= len(lines) <= 12) or any(len(ln) > 200 for ln in lines):
        raise HTTPException(400, "send 1 to 12 lines of up to 200 characters")
    if body.src not in config.LANGUAGES or body.tgt not in config.LANGUAGES or body.role not in ("A", "B"):
        raise HTTPException(400, "unknown language or role")
    try:
        per_line = songs.align_notes(lines, _phrases(body.melody))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    peer = peer_of(body.room, body.role)
    tgt = peer["lang"] if peer and peer["lang"] else body.tgt
    src_name, tgt_name = config.LANGUAGES[body.src], config.LANGUAGES[tgt]
    run = Run()

    # 1. Translate each line to the number of notes it must carry.
    try:
        if body.src == tgt:
            fitted = [(ln, lyrics.syllables(ln)) for ln in lines]
        else:
            fitted = await run.timed("translate", lyrics.singable(
                client, lines, [len(n) for n in per_line], src_name, tgt_name))
        translator = "llm"
    except (SarvamError, ValueError) as e:
        run.warnings.append(f"singable translation failed ({e}); used plain MT, not fitted to the melody")
        try:
            plain = await run.timed("translate", asyncio.gather(*(client.translate(ln, body.src, tgt) for ln in lines)))
        except SarvamError as e2:
            raise http_error(e2) from e2
        fitted, translator = [(t, lyrics.syllables(t)) for t in plain], "mt"
    translated = [t for t, _ in fitted]

    # 2. Speak every line (your cloned voice if you made one) while the meaning check runs.
    voice = ""

    async def speak(text: str) -> bytes:
        nonlocal voice
        if body.voice_id and len(text) <= 1000:
            try:
                audio = await client.clone_tts(text, tgt, body.voice_id)
                voice = "your cloned voice"
                return audio
            except SarvamError as e:
                if "voice clone failed" not in " ".join(run.warnings):
                    run.warnings.append(f"voice clone failed ({e.status}); used a stock voice")
        audio, label, warning = await client.tts(text, tgt, body.role, 1.0)
        voice = voice or label
        if warning and warning not in run.warnings:
            run.warnings.append(warning)
        return audio

    async def read_back() -> list[str]:
        try:
            return list(await asyncio.gather(*(client.translate(t, tgt, body.src) for t in translated)))
        except SarvamError:
            return []

    try:
        wavs, back = await asyncio.gather(
            run.timed("tts", asyncio.gather(*(speak(t) for t in translated))), run.timed("check", read_back()))
    except SarvamError as e:
        raise http_error(e) from e

    # 3. Put the melody on the speech.
    sung = await run.timed("sing", asyncio.to_thread(singer.render_song, list(wavs), per_line))

    sims = [checks.similarity(a, b) for a, b in zip(lines, back)] if back else []
    reasons = [f"line {i + 1} reads back differently (similarity {s:.2f})"
               for i, s in enumerate(sims) if s < checks.SIMILARITY_FLOOR]
    check = ({"back_translation": "\n".join(back), "similarity": round(statistics.mean(sims), 2),
              "level": "check" if reasons else "ok", "reasons": reasons} if back else None)
    exact = sum(1 for (_, have), notes in zip(fitted, per_line) if have == len(notes))
    result = {
        "kind": "song", "from": body.role, "seq": body.seq, "sid": body.sid, "src": body.src, "tgt": tgt,
        "title": body.title, "transcript": "\n".join(lines), "detected_lang": None,
        "translation": "\n".join(translated), "translator": translator,
        "tone": {"summary": "sung-style: speech retimed and re-pitched to the melody", "labels": {}},
        "guard": {"status": "none"}, "glossary": [], "voice": voice, "check": check,
        "song": {"exact": exact, "total": len(lines),
                 "lines": [{"tgt": t, "syllables": have, "notes": len(n)} for (t, have), n in zip(fitted, per_line)]},
        "timings_ms": run.finish(), "warnings": run.warnings, "mock": config.MOCK,
        "audio": base64.b64encode(sung).decode(),
    }
    return await relay(peer, result)
