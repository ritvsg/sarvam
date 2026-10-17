"""Thin async wrappers over the Sarvam REST endpoints used here, plus an offline mock.

Endpoint paths and request fields follow docs.sarvam.ai as read on 2026-10-17. They have not been
run against the live API, so every failure is surfaced verbatim (stage, status, body).
"""
import base64
import io
import json
import math
import re
import struct
import wave

import httpx

import config


class SarvamError(Exception):
    def __init__(self, stage: str, status: int, body: str):
        super().__init__(f"{stage} failed ({status}): {body[:300]}")
        self.stage, self.status, self.body = stage, status, body


def _first_audio(data: dict) -> bytes:
    audio = data.get("audios") or data.get("audio") or data.get("audio_base64")
    if isinstance(audio, list):
        audio = audio[0] if audio else None
    if not audio:
        raise SarvamError("tts", 200, f"no audio in response, keys: {list(data)}")
    return base64.b64decode(audio)


class SarvamClient:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(
            base_url=config.BASE_URL,
            headers={"api-subscription-key": config.API_KEY},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _json(self, stage: str, method: str, path: str, **kwargs) -> dict:
        r = await self.http.request(method, path, **kwargs)
        if r.status_code >= 400:
            raise SarvamError(stage, r.status_code, r.text)
        return r.json()

    async def transcribe(self, wav: bytes, keyterms: list[str] | None = None) -> tuple[str, str | None, str | None]:
        """Returns (transcript, language, note). Keyterms bias recognition (Saaras v4 only)."""
        form = {"model": config.STT_MODEL, "mode": "transcribe"}
        if keyterms and config.STT_MODEL.startswith("saaras:v4"):
            form["keyterms"] = json.dumps(keyterms[:50], ensure_ascii=False)
        files = {"file": ("segment.wav", wav, "audio/wav")}
        note = None
        try:
            data = await self._json("stt", "POST", "/speech-to-text", files=files, data=form)
        except SarvamError as e:
            if "keyterms" not in form:
                raise
            del form["keyterms"]
            note = f"keyterms rejected ({e.status}); transcribed without them"
            data = await self._json("stt", "POST", "/speech-to-text", files=files, data=form)
        return (data.get("transcript") or "").strip(), data.get("language_code"), note

    async def translate(self, text: str, src: str, tgt: str) -> str:
        """Plain machine translation: Mayura first, Sarvam-Translate as the fallback."""
        if src == tgt:
            return text
        last: SarvamError | None = None
        for model, mode in ((config.MT_MODEL, "modern-colloquial"), (config.MT_FALLBACK_MODEL, "formal")):
            try:
                data = await self._json(
                    "translate", "POST", "/translate",
                    json={"input": text, "source_language_code": src, "target_language_code": tgt,
                          "model": model, "mode": mode},
                )
            except SarvamError as e:
                last = e
                continue
            out = data.get("translated_text") or data.get("translatedText")
            if out:
                return out
        raise last or SarvamError("translate", 200, "empty translation")

    async def llm_translate(self, text: str, src_name: str, tgt_name: str, delivery: str, glossary=()) -> str:
        """Tone-aware translation: the measured delivery and any matched glossary terms go in as text."""
        keep = [t for t, rendering in glossary if not rendering]
        mapped = [f"{t} -> {rendering}" for t, rendering in glossary if rendering]
        rules = ""
        if keep:
            rules += f" Keep these terms exactly as written: {', '.join(keep)}."
        if mapped:
            rules += f" Always render these terms as given: {'; '.join(mapped)}."
        system = (
            f"You are a live interpreter. Translate the user's {src_name} utterance into {tgt_name}. "
            "Keep the meaning exact: do not add, explain or omit anything, and keep every number unchanged."
            f"{rules} The speaker's delivery was: {delivery}. Reflect that register and energy through "
            "word choice and punctuation only. Reply with the translation and nothing else."
        )
        data = await self._json(
            "llm-translate", "POST", "/v1/chat/completions",
            json={"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                  "temperature": 0.2, "max_tokens": 512,
                  "reasoning_effort": None},  # null disables thinking, which keeps latency down
        )
        return (data["choices"][0]["message"].get("content") or "").strip()

    async def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.3) -> str:
        data = await self._json(
            "chat", "POST", "/v1/chat/completions",
            json={"model": config.LLM_MODEL, "messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens, "reasoning_effort": None},
        )
        return (data["choices"][0]["message"].get("content") or "").strip()

    async def judge(self, transcript: str, translation: str, back: str, src_name: str, tgt_name: str) -> dict | None:
        """Optional reference-free check by an LLM. Returns {"score": 1-5, "issues": [...]} or None."""
        system = (
            "You audit a live interpreter. You get what was said, the translation, and a back-translation "
            "of the translation. Reply with JSON only: {\"score\": 1-5, \"issues\": [..]}. Score 5 means "
            "faithful. List only real meaning errors (wrong or missing numbers, names, negations, who did "
            "what), at most 3, each under 12 words. Ignore style."
        )
        user = (f"Said ({src_name}): {transcript}\nTranslation ({tgt_name}): {translation}\n"
                f"Back-translation ({src_name}): {back}")
        data = await self._json(
            "judge", "POST", "/v1/chat/completions",
            json={"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  "temperature": 0, "max_tokens": 300, "reasoning_effort": None},
        )
        match = re.search(r"\{.*\}", data["choices"][0]["message"].get("content") or "", re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        score = obj.get("score")
        issues = obj.get("issues") if isinstance(obj.get("issues"), list) else []
        return {"score": int(score) if isinstance(score, (int, float)) else None,
                "issues": [str(i)[:120] for i in issues][:3]}

    async def _tts_once(self, text: str, lang: str, model: str, speaker: str, pace: float) -> bytes:
        data = await self._json(
            "tts", "POST", "/text-to-speech",
            json={"text": text, "language_code": lang, "speaker": speaker, "model": model,
                  "pace": pace, "speech_sample_rate": 24000},
        )
        return _first_audio(data)

    async def tts(self, text: str, lang: str, role: str, pace: float) -> tuple[bytes, str, str | None]:
        """Returns (wav, voice label, warning). Falls back to Bulbul v3 if the first voice is rejected."""
        model, speaker = config.tts_voice(lang, role)
        try:
            return await self._tts_once(text, lang, model, speaker, pace), f"{model}/{speaker}", None
        except SarvamError as e:
            if model == config.TTS_FALLBACK_MODEL:
                raise
            fallback = config.V3_SPEAKERS[role]
            wav = await self._tts_once(text, lang, config.TTS_FALLBACK_MODEL, fallback, pace)
            return wav, f"{config.TTS_FALLBACK_MODEL}/{fallback}", f"{model}/{speaker} failed ({e.status}); used v3"

    async def create_voice(self, audio: bytes, filename: str, name: str, language: str) -> dict:
        return await self._json(
            "voice-create", "POST", "/voices/create",
            files={"file": (filename, audio, "audio/wav")},
            data={"name": name, "language": language},
        )

    async def clone_tts(self, text: str, lang: str, voice_id: str) -> bytes:
        data = await self._json(
            "voice-clone", "POST", "/voices/clone",
            json={"text": text, "language_code": lang, "voice_id": voice_id},
        )
        return _first_audio(data)

    async def delete_voice(self, voice_id: str) -> None:
        await self._json("voice-delete", "DELETE", f"/voices/delete/{voice_id}")


def _tone_wav(seconds: float = 0.6, hz: float = 440.0, rate: int = 24000) -> bytes:
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(9000 * math.sin(2 * math.pi * hz * i / rate) * min(1.0, i / 800, (n - i) / 800)))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


class MockClient:
    """Same interface as SarvamClient with canned output, for testing the flow without a key."""

    async def aclose(self) -> None:
        return None

    async def transcribe(self, wav: bytes, keyterms: list[str] | None = None) -> tuple[str, str | None, str | None]:
        if config.MOCK_TRANSCRIPT:
            return config.MOCK_TRANSCRIPT, "en-IN", None
        return "(mock) नमस्ते, मुझे डॉक्टर से मिलना है, मेरी उम्र ५० साल है।", "hi-IN", None

    async def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.3) -> str:
        """Mock singable translator: misses the syllable target on the first try, hits it after feedback."""
        want = int(re.findall(r"exactly (\d+) syllables", messages[-1]["content"])[-1])
        retried = any(m["role"] == "assistant" for m in messages)
        return " ".join(["la"] * (want if retried else want + 2))

    async def translate(self, text: str, src: str, tgt: str) -> str:
        return text if src == tgt else f"[mock {src}→{tgt}] {text}"

    async def llm_translate(self, text: str, src_name: str, tgt_name: str, delivery: str, glossary=()) -> str:
        terms = "glossary applied" if glossary else "no glossary"  # no digits: they would trip the numbers guard
        return f"[mock tone-aware {src_name}→{tgt_name}, {delivery}, {terms}] {text}"

    async def judge(self, transcript: str, translation: str, back: str, src_name: str, tgt_name: str) -> dict | None:
        return {"score": 5, "issues": []}

    async def tts(self, text: str, lang: str, role: str, pace: float) -> tuple[bytes, str, str | None]:
        return _tone_wav(hz=440 if role == "A" else 330), f"mock/{role}", None

    async def create_voice(self, audio: bytes, filename: str, name: str, language: str) -> dict:
        return {"voice_id": "svc-mock-0000"}

    async def clone_tts(self, text: str, lang: str, voice_id: str) -> bytes:
        return _tone_wav(hz=550)

    async def delete_voice(self, voice_id: str) -> None:
        return None


def get_client():
    return MockClient() if config.MOCK else SarvamClient()
