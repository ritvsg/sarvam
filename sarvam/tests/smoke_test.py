"""End-to-end check in mock mode: starts the real server and drives it over HTTP and WebSocket.

Run from the sarvam folder:  .venv\\Scripts\\python tests\\smoke_test.py
It proves the plumbing (relay, ordering fields, fallbacks, tone analysis). It does not exercise the
live Sarvam API or the browser page.
"""
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import prosody  # noqa: E402

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
RATE = 16000


def make_wav(parts):
    """parts: (seconds, f0_start, f0_end, amplitude); f0_start == 0 means silence."""
    out, phase = [], 0.0
    for secs, f0a, f0b, amp in parts:
        n = int(secs * RATE)
        if f0a == 0:
            out.append(np.zeros(n))
            continue
        f0 = np.linspace(f0a, f0b, n)
        phase_t = phase + np.cumsum(2 * np.pi * f0 / RATE)
        phase = phase_t[-1]
        sig = sum(np.sin(h * phase_t) / h for h in (1, 2, 3))
        out.append(amp * sig / 1.8)
    pcm = (np.concatenate(out) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def check_prosody():
    words = "one two three four five six"
    hesitant = make_wav([(0.6, 150, 150, 0.5), (0.5, 0, 0, 0), (0.6, 150, 150, 0.5), (0.6, 0, 0, 0), (0.6, 150, 150, 0.5)])
    sweep = make_wav([(1.8, 120, 240, 0.5)])
    flat = make_wav([(1.8, 150, 150, 0.5)])
    out = {}
    for label, wav in (("hesitant", hesitant), ("sweep", sweep), ("flat", flat)):
        x, rate = prosody.read_wav(wav)
        out[label] = prosody.describe(prosody.analyze_audio(x, rate), words)
        print(f"  {label:9s} -> {out[label]['summary']!r}  {({k: out[label].get(k) for k in ('pause_ratio', 'pitch_range_st', 'dyn_db', 'rate_wps', 'pace')})}")
    assert "hesitant" in out["hesitant"]["labels"], "long pauses should read as hesitant"
    assert "hesitant" not in out["flat"]["labels"]
    assert out["sweep"]["pitch_range_st"] >= 8, "one-octave sweep should read as a wide pitch range"
    assert out["flat"]["pitch_range_st"] <= 4, "constant pitch should read as flat"
    assert 0.9 <= out["flat"]["pace"] <= 1.15


def check_checks():
    import checks

    assert checks.normalize_digits("५०० ௫௦ ౧౨") == "500 50 12"
    assert checks.extract_numbers("मुझे ५०० रुपये और 1,00,000 चाहिए") == ["500", "100000"]
    assert checks.compare_numbers("५०० रुपये", "500 rupees")["status"] == "ok"
    assert checks.compare_numbers("pay 500", "pay 5000")["status"] == "differs"
    assert checks.compare_numbers("pay 500", "pay 500 and 20")["extra"] == ["20"]
    assert checks.compare_numbers("hello", "namaste")["status"] == "none"
    terms = checks.parse_glossary("Dr. Rao\nmetformin => metformin\n\n  => bad\n")
    assert terms == [("Dr. Rao", None), ("metformin", "metformin")], terms
    assert checks.match_glossary("तो डॉक्टर राव ने कहा", [("डॉक्टर राव", None)])  # exact
    assert checks.match_glossary("the doctor is Dr Rau today", [("Dr. Rao", None)])  # fuzzy
    assert not checks.match_glossary("nothing relevant here", [("Dr. Rao", None)])  # evidence gate
    assert checks.similarity("मुझे डॉक्टर से मिलना है", "मुझे डॉक्टर से मिलना है।") > 0.9
    assert checks.similarity("मुझे डॉक्टर से मिलना है", "आज मौसम बहुत अच्छा है") < 0.3
    level, reasons = checks.grade(0.8, checks.compare_numbers("500", "5000"), None)
    assert level == "check" and "numbers differ" in reasons[0]
    assert checks.grade(0.8, checks.compare_numbers("500", "500"), {"score": 5, "issues": []})[0] == "ok"
    print("  numbers, glossary matching, similarity and grading ok")


def hum_wav(midis_and_secs, gap=0.05):
    """A hummed melody: one harmonic tone per note with a brief dip between notes (like "da da da")."""
    parts = []
    for midi, secs in midis_and_secs:
        f0 = 440 * 2 ** ((midi - 69) / 12)
        parts += [(secs - gap, f0, f0, 0.5), (gap, 0, 0, 0)]
    return make_wav(parts)


TWINKLE_LINE1 = [(60, 0.6), (60, 0.6), (67, 0.6), (67, 0.6), (69, 0.6), (69, 0.6), (67, 1.2)]


def check_song_units():
    import languages
    import lyrics
    import melody
    import singer
    import songs

    assert languages.parse("Tamil") == "ta-IN" and languages.parse("तमिल") == "ta-IN"
    assert languages.parse("please sing it in Telugu") == "te-IN" and languages.parse("banana") is None
    catalog = songs.load_catalog()
    assert {s["id"] for s in catalog} >= {"twinkle", "are-you-sleeping", "row-your-boat"}
    for s in catalog:  # catalog integrity: every line's syllables equal its note count
        for line, phrase in zip(s["lines"], s["phrases"]):
            assert lyrics.syllables(line) == len(phrase), (s["id"], line, lyrics.syllables(line), len(phrase))
    assert lyrics.syllables("कमल") == 2 and lyrics.syllables("कमला") == 2 and lyrics.syllables("தமிழ்") >= 1
    print("  languages, catalog/syllable integrity ok")

    x, rate = prosody.read_wav(hum_wav(TWINKLE_LINE1))
    analysis = melody.analyze(x, rate)
    got = [n["p"] for p in analysis["phrases"] for n in p]
    assert len(got) == 7, got
    steps = [b - a for a, b in zip(got, got[1:])]
    assert steps == [0, 7, 0, 2, 0, -2], steps  # the intervals of the tune, whatever the key
    ranked = songs.rank(analysis, "", catalog)
    assert ranked and ranked[0]["id"] == "twinkle", [(c["id"], c["score"]) for c in ranked]
    # Same tune, hummed a fourth lower and 30% slower: still found.
    slow = hum_wav([(m - 5, s * 1.3) for m, s in TWINKLE_LINE1])
    x2, r2 = prosody.read_wav(slow)
    assert songs.rank(melody.analyze(x2, r2), "", catalog)[0]["id"] == "twinkle"
    print("  hum -> notes -> ranked as Twinkle (also when transposed and slowed) ok")

    # Singer: speech-like tone retimed to 1.5 s and re-pitched to C4, E4, G4.
    rate = 24000
    t = np.arange(int(1.2 * rate)) / rate
    speech = (sum(np.sin(2 * np.pi * 150 * h * t) / h for h in (1, 2, 3)) * 0.3).astype(np.float32)
    notes = [{"p": 60, "d": 0.5}, {"p": 64, "d": 0.5}, {"p": 67, "d": 0.5}]
    out = singer.sing_line(speech, rate, notes, center_midi=64)
    assert abs(len(out) / rate - 1.5) < 0.1, len(out) / rate
    frames, hop = prosody.frame_signal(out, rate, win_ms=40, hop_ms=10)
    f0 = prosody.pitch_track(frames, rate, fmin=80, fmax=800)
    for centre, want in ((0.25, 60), (0.75, 64), (1.25, 67)):
        hz = f0[int(centre * rate / hop)]
        assert hz == hz and abs(69 + 12 * np.log2(hz / 440) - want) < 0.6, (centre, hz)
    print("  singer: duration and note pitches land where the melody says ok")


def wait_for_server():
    for _ in range(60):
        try:
            return httpx.get(f"{BASE}/api/config", timeout=1).json()
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def post(path, wav=None, **fields):
    files = {"file": ("segment.wav", wav, "audio/wav")} if wav is not None else None
    return httpx.post(f"{BASE}{path}", data=fields, files=files, timeout=20)


def recv_until(ws, pred, tries=10):
    for _ in range(tries):
        msg = json.loads(ws.recv(timeout=5))
        if pred(msg):
            return msg
    raise AssertionError("expected message never arrived")


def check_server():
    wav = make_wav([(1.2, 150, 190, 0.5)])
    cfg = wait_for_server()
    assert cfg["mock"] is True and "ta-IN" in cfg["languages"]

    r = post("/api/interpret", wav, src="ta-IN", tgt="hi-IN", role="A", seq=0, sid="s1")
    j = r.json()
    assert r.status_code == 200 and j["delivered"] is False and j["audio"], j
    assert j["translator"] == "mt" and set(j["timings_ms"]) >= {"stt", "tone", "translate", "tts", "total"}
    assert j["guard"]["status"] == "ok" and j["guard"]["source"] == ["50"], j["guard"]  # Devanagari digits
    print("  solo interpret ok:", j["translation"][:60])

    j = post("/api/interpret", wav, src="ta-IN", tgt="hi-IN", role="A", seq=3, sid="s1", glossary="डॉक्टर").json()
    assert j["glossary"] == ["डॉक्टर"] and j["translator"] == "llm", j
    j = post("/api/interpret", wav, src="ta-IN", tgt="hi-IN", role="A", seq=4, sid="s1", glossary="Dr. Rao").json()
    assert j["glossary"] == [] and j["translator"] == "mt", j  # term not said, so it is not enforced
    print("  glossary evidence gate ok")

    j = post("/api/say", None, text="Could you confirm that?", src="en-IN", tgt="hi-IN", role="B", seq=5, sid="s2").json()
    assert j["transcript"] == "Could you confirm that?" and j["typed"] is True and j["audio"], j
    assert post("/api/say", None, text="  ", src="en-IN", tgt="hi-IN").status_code == 400
    print("  typed /api/say ok")

    chk = httpx.post(f"{BASE}/api/check", json={"transcript": "pay 500", "translation": "pay 5000", "src": "en-IN",
                                               "tgt": "hi-IN", "judge": True}, timeout=20).json()
    assert chk["level"] == "check" and chk["numbers"]["status"] == "differs" and chk["judge"]["score"] == 5, chk
    chk = httpx.post(f"{BASE}/api/check", json={"transcript": "pay 500", "translation": "pay 500", "src": "en-IN",
                                               "tgt": "en-IN"}, timeout=20).json()
    assert chk["level"] == "ok" and chk["back_translation"] == "pay 500", chk
    print("  /api/check read-back ok")

    j = post("/api/interpret", wav, src="ta-IN", tgt="hi-IN", role="A", seq=1, sid="s1", tone_aware="true").json()
    assert j["translator"] == "llm", j
    print("  tone-aware path ok")

    assert post("/api/interpret", b"not a wav", src="ta-IN", tgt="hi-IN").status_code == 400
    assert post("/api/interpret", wav, src="xx-XX", tgt="hi-IN").status_code == 400
    print("  bad input rejected ok")

    vid = post("/api/voice/create", wav, name="t", language="ta-IN").json()["voice_id"]
    j = post("/api/interpret", wav, src="ta-IN", tgt="hi-IN", role="A", seq=2, sid="s1", voice_id=vid).json()
    assert j["voice"] == "your cloned voice", j
    assert httpx.delete(f"{BASE}/api/voice/{vid}").json()["deleted"] == vid
    print("  clone create / use / delete ok")

    hum = hum_wav(TWINKLE_LINE1)
    ident = post("/api/song/identify", hum).json()
    top = ident["candidates"][0]
    assert top["id"] == "twinkle" and ident["note_count"] == 7, ident
    song = {"lines": top["lines"], "melody": top["melody"], "src": "en-IN", "tgt": "ta-IN", "title": top["title"],
            "role": "A", "seq": 9, "sid": "s1"}
    j = httpx.post(f"{BASE}/api/song/convert", json=song, timeout=60).json()
    assert j["kind"] == "song" and j["audio"] and j["delivered"] is False, j
    assert j["song"]["total"] == 4 and j["song"]["exact"] == 4, j["song"]  # the mock misses once, then fits
    assert j["check"]["level"] in ("ok", "check") and j["timings_ms"]["sing"] >= 0
    bad = httpx.post(f"{BASE}/api/song/convert", json={**song, "lines": ["a", "b"] * 8}, timeout=20)
    assert bad.status_code == 400
    pr = httpx.post(f"{BASE}/api/prompt", json={"key": "language", "lang": "ta-IN"}, timeout=20).json()
    assert pr["text"] and pr["audio"]
    assert post("/api/song/language", hum).json()["lang"] is None  # the mock hears Hindi text, no language name
    saved = httpx.post(f"{BASE}/api/song/save", json={"title": "My tune", "lines": ["la la la", "la la"],
                                                      "melody": [[{"p": 60, "d": 0.3}] * 3, [{"p": 62, "d": 0.3}] * 2]},
                       timeout=20).json()
    assert saved["id"].startswith("my-tune-")
    (ROOT / "songs" / "user" / f"{saved['id']}.json").unlink()
    print("  song identify / convert / prompt / language / save ok")

    # A whole sung song is over 1 MB of base64, above this client's default frame limit (browsers have none).
    with connect(f"ws://127.0.0.1:{PORT}/ws/room1/A", max_size=None) as a, \
            connect(f"ws://127.0.0.1:{PORT}/ws/room1/B", max_size=None) as b:
        a.send(json.dumps({"type": "hello", "lang": "ta-IN"}))
        b.send(json.dumps({"type": "hello", "lang": "hi-IN"}))
        recv_until(a, lambda m: m["type"] == "peer" and m["connected"] and m["lang"] == "hi-IN")
        recv_until(b, lambda m: m["type"] == "peer" and m["connected"] and m["lang"] == "ta-IN")
        # A's dropdown says English, but the partner's declared language (Hindi) must win.
        j = post("/api/interpret", wav, src="ta-IN", tgt="en-IN", role="A", room="room1", seq=0, sid="sa").json()
        assert j["delivered"] is True and j["audio"] is None and j["tgt"] == "hi-IN", j
        got = recv_until(b, lambda m: m["type"] == "utterance")
        assert got["audio"] and got["from"] == "A" and got["seq"] == 0 and got["sid"] == "sa"
        assert got["guard"]["status"] == "ok", got["guard"]
        # The listener asks the speaker to confirm; it must reach A in A's language.
        j = post("/api/say", None, text="Could you confirm that?", src="en-IN", tgt="hi-IN", role="B",
                 room="room1", seq=0, sid="sb").json()
        assert j["delivered"] is True and j["tgt"] == "ta-IN", j
        back = recv_until(a, lambda m: m["type"] == "utterance")
        assert back["from"] == "B" and back["typed"] is True and back["audio"]
        # A sings a song; the partner (Hindi) must receive it in Hindi, whatever A's target dropdown said.
        j = httpx.post(f"{BASE}/api/song/convert", json={**song, "tgt": "en-IN", "room": "room1", "seq": 3, "sid": "sa"},
                       timeout=60).json()
        assert j["delivered"] is True and j["tgt"] == "hi-IN" and j["audio"] is None, j
        sung = recv_until(b, lambda m: m["type"] == "utterance" and m.get("kind") == "song")
        assert sung["audio"] and sung["tgt"] == "hi-IN"
        post("/api/interpret", b"\x00" * 10, src="ta-IN", tgt="hi-IN", role="A", room="room1", seq=1, sid="sa")
        silent = make_wav([(1.0, 0, 0, 0)])
        j = post("/api/interpret", silent, src="ta-IN", tgt="hi-IN", role="A", room="room1", seq=2, sid="sa").json()
        print("  relay A -> B ok (translation reached the partner in", got["tgt"], ")", "silence ->", j.get("empty", "transcribed (mock)"))
    print("  websocket relay and peer language override ok")


def main():
    print("prosody checks")
    check_prosody()
    print("safety checks")
    check_checks()
    print("song units")
    check_song_units()
    env = {**os.environ, "MOCK": "1"}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    try:
        print("server checks")
        check_server()
    finally:
        server.terminate()
        server.wait(timeout=10)
    print("ALL OK")


if __name__ == "__main__":
    main()
