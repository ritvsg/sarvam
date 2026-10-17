"""Live two-way interpreter and song singer: speech -> Saaras -> translation -> Bulbul, between two browsers.

Run:  uvicorn server:app --port 8000
"""
import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import checks
import config
import prosody
import song_routes
from core import Run, announce, client, http_error, peer_of, relay, rooms, skip_peer
from sarvam_client import SarvamError

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await client.aclose()


app = FastAPI(title="Live interpreter", lifespan=lifespan)
app.include_router(song_routes.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


async def _translate_and_speak(run: Run, text: str, src: str, tgt: str, role: str, tone: dict,
                               tone_aware: bool, voice_id: str, glossary: list) -> dict:
    # Evidence gate: only glossary terms that actually occur in this utterance are enforced.
    matched = checks.match_glossary(text, glossary)
    translation, translator = "", "mt"
    if tone_aware or matched:
        try:
            translation = await run.timed("translate", client.llm_translate(
                text, config.LANGUAGES[src], config.LANGUAGES[tgt],
                tone["summary"] or "steady, even delivery", matched))
            translator = "llm"
        except SarvamError as e:
            run.warnings.append(f"LLM translation failed ({e.status}); used plain MT")
    if not translation:
        translator = "mt"
        translation = await run.timed("translate", client.translate(text, src, tgt))

    audio, voice = None, ""
    if voice_id and len(translation) <= 1000:  # the cloned-voice endpoint caps text at 1,000 chars
        try:
            audio = await run.timed("tts", client.clone_tts(translation, tgt, voice_id))
            voice = "your cloned voice"
        except SarvamError as e:
            run.warnings.append(f"voice clone failed ({e.status}); used a stock voice")
    if audio is None:
        audio, voice, warning = await run.timed("tts", client.tts(translation, tgt, role, tone["pace"]))
        if warning:
            run.warnings.append(warning)
    return {"translation": translation, "translator": translator,
            "glossary": [t for t, _ in matched], "audio": audio, "voice": voice}


async def _deliver(run: Run, peer: dict | None, spoken: dict, **fields) -> dict:
    result = {
        **fields,
        "translation": spoken["translation"], "translator": spoken["translator"],
        "glossary": spoken["glossary"], "voice": spoken["voice"],
        "guard": checks.compare_numbers(fields["transcript"], spoken["translation"]),
        "timings_ms": run.finish(), "warnings": run.warnings,
        "mock": config.MOCK, "audio": base64.b64encode(spoken["audio"]).decode(),
    }
    return await relay(peer, result)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "mock": config.MOCK,
        "languages": config.LANGUAGES,
        "models": {"stt": config.STT_MODEL, "mt": config.MT_MODEL, "llm": config.LLM_MODEL, "tts": config.TTS_MODEL},
    }


@app.websocket("/ws/{room}/{role}")
async def room_socket(socket: WebSocket, room: str, role: str) -> None:
    if role not in ("A", "B"):
        await socket.close(code=1008)
        return
    await socket.accept()
    peers = rooms.setdefault(room, {})
    peers[role] = {"ws": socket, "lang": None}
    await announce(room)
    try:
        while True:
            msg = json.loads(await socket.receive_text())
            if msg.get("type") == "hello" and msg.get("lang") in config.LANGUAGES:
                peers[role]["lang"] = msg["lang"]
                await announce(room)
    except (WebSocketDisconnect, json.JSONDecodeError):
        pass
    finally:
        if peers.get(role, {}).get("ws") is socket:  # a refreshed tab may have replaced us
            peers.pop(role, None)
        if peers:
            await announce(room)
        else:
            rooms.pop(room, None)


@app.post("/api/interpret")
async def interpret(
    file: UploadFile = File(...),
    src: str = Form(...),
    tgt: str = Form(...),
    role: str = Form("A"),
    room: str = Form(""),
    seq: int = Form(0),
    sid: str = Form(""),
    tone_aware: bool = Form(False),
    voice_id: str = Form(""),
    glossary: str = Form(""),
) -> dict:
    if src not in config.LANGUAGES or tgt not in config.LANGUAGES or role not in ("A", "B"):
        raise HTTPException(400, "unknown language or role")
    peer = peer_of(room, role)
    if peer and peer["lang"]:
        tgt = peer["lang"]  # the partner's declared language wins over the local dropdown

    wav = await file.read()
    try:
        samples, rate = prosody.read_wav(wav)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    run = Run()
    terms = checks.parse_glossary(glossary)
    try:
        # Speech recognition and tone analysis run side by side.
        (transcript, detected, note), feats = await asyncio.gather(
            run.timed("stt", client.transcribe(wav, [t for t, _ in terms])),
            run.timed("tone", asyncio.to_thread(prosody.analyze_audio, samples, rate)),
        )
        if note:
            run.warnings.append(note)
        if not transcript:
            await skip_peer(peer, seq, sid)
            return {"empty": True, "delivered": bool(peer), "seq": seq}
        tone = prosody.describe(feats, transcript)
        spoken = await _translate_and_speak(run, transcript, src, tgt, role, tone, tone_aware, voice_id, terms)
    except SarvamError as e:
        await skip_peer(peer, seq, sid)
        raise http_error(e) from e

    return await _deliver(run, peer, spoken, **{
        "from": role, "seq": seq, "sid": sid, "src": src, "tgt": tgt,
        "transcript": transcript, "detected_lang": detected, "tone": tone})


@app.post("/api/say")
async def say(
    text: str = Form(...),
    src: str = Form(...),
    tgt: str = Form(...),
    role: str = Form("A"),
    room: str = Form(""),
    seq: int = Form(0),
    sid: str = Form(""),
    voice_id: str = Form(""),
    glossary: str = Form(""),
) -> dict:
    """Typed text through the same translate-and-speak path (also used for "ask them to confirm")."""
    text = text.strip()
    if not text or len(text) > 900 or src not in config.LANGUAGES or tgt not in config.LANGUAGES or role not in ("A", "B"):
        raise HTTPException(400, "empty or too long text, or unknown language or role")
    peer = peer_of(room, role)
    if peer and peer["lang"]:
        tgt = peer["lang"]
    run = Run()
    try:
        spoken = await _translate_and_speak(run, text, src, tgt, role, {"summary": "", "pace": 1.0, "labels": {}},
                                            False, voice_id, checks.parse_glossary(glossary))
    except SarvamError as e:
        await skip_peer(peer, seq, sid)
        raise http_error(e) from e
    return await _deliver(run, peer, spoken, **{
        "from": role, "seq": seq, "sid": sid, "src": src, "tgt": tgt,
        "transcript": text, "detected_lang": None, "tone": {"summary": "", "labels": {}}, "typed": True})


class CheckIn(BaseModel):
    transcript: str
    translation: str
    src: str
    tgt: str
    judge: bool = False


@app.post("/api/check")
async def check(body: CheckIn) -> dict:
    """Read-back: translate the translation back and compare with what was said.

    Called by the speaker's page after the partner already has the audio, so it never delays them.
    """
    if body.src not in config.LANGUAGES or body.tgt not in config.LANGUAGES:
        raise HTTPException(400, "unknown language")
    numbers = checks.compare_numbers(body.transcript, body.translation)
    try:
        back = await client.translate(body.translation, body.tgt, body.src)
        judge = None
        if body.judge:
            try:
                judge = await client.judge(body.transcript, body.translation, back,
                                           config.LANGUAGES[body.src], config.LANGUAGES[body.tgt])
            except SarvamError:
                judge = None  # the judge is optional; the deterministic checks still stand
    except SarvamError as e:
        raise http_error(e) from e
    sim = checks.similarity(body.transcript, back)
    level, reasons = checks.grade(sim, numbers, judge)
    return {"back_translation": back, "similarity": round(sim, 2), "numbers": numbers,
            "judge": judge, "level": level, "reasons": reasons}


@app.post("/api/voice/create")
async def voice_create(file: UploadFile = File(...), name: str = Form(...), language: str = Form(...)) -> dict:
    try:
        data = await client.create_voice(await file.read(), file.filename or "reference.wav", name, language)
    except SarvamError as e:
        raise http_error(e) from e
    voice_id = data.get("voice_id")
    if not voice_id:
        raise HTTPException(502, detail={"stage": "voice-create", "status": 200, "body": json.dumps(data)[:300]})
    return {"voice_id": voice_id}


@app.delete("/api/voice/{voice_id}")
async def voice_delete(voice_id: str) -> dict:
    try:
        await client.delete_voice(voice_id)
    except SarvamError as e:
        raise http_error(e) from e
    return {"deleted": voice_id}
