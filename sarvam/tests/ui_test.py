"""Real-browser check in mock mode: Edge with a fake microphone drives the actual page.

Needs:  pip install -r requirements-dev.txt   (uses the installed Microsoft Edge, no download)
Run:    .venv\\Scripts\\python tests\\ui_test.py

It proves the page's capture, pause-triggered segmentation, WAV encoding, room relay, rendering and
the confirm button. It does not test real speech, real latency or the live Sarvam API.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from smoke_test import make_wav  # noqa: E402

PORT = 8767
BASE = f"http://127.0.0.1:{PORT}"


def fake_mic_wav(path: Path) -> None:
    # Three voiced bursts with short pauses (reads as hesitant, stays one phrase), then a long
    # silence that triggers the pause cut. The browser loops the file.
    path.write_bytes(make_wav([
        (0.3, 0, 0, 0),
        (0.7, 150, 150, 0.5), (0.45, 0, 0, 0), (0.7, 150, 150, 0.5), (0.45, 0, 0, 0), (0.7, 150, 150, 0.5),
        (1.6, 0, 0, 0),
    ]))


def wait_for_server():
    for _ in range(60):
        try:
            return httpx.get(f"{BASE}/api/config", timeout=1).json()
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sarvam-ui-"))
    wav = tmp / "fake_mic.wav"
    fake_mic_wav(wav)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT, env={**os.environ, "MOCK": "1"},
    )
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True, args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required",
            ])
            ctx = browser.new_context(permissions=["microphone"])
            errors = []
            a, b = ctx.new_page(), ctx.new_page()
            for page in (a, b):
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                page.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)

            a.goto(f"{BASE}/?room=uitest&role=A")
            b.goto(f"{BASE}/?room=uitest&role=B")
            assert a.locator("#mockBadge").is_visible()
            assert a.locator("#mine option").count() == 11
            for page in (a, b):
                page.click("#join")
            b.wait_for_function("document.getElementById('peerStatus').textContent.includes('Partner online')")
            a.wait_for_function("document.getElementById('theirs').value === 'hi-IN'")
            print("  join, relay handshake and peer language auto-fill ok")

            a.fill("#glossary", "डॉक्टर")
            a.click("#mic")
            a.wait_for_selector("#mic.live")
            a.wait_for_selector(".entry.me .trans", timeout=20000)  # first segment finished
            a.click("#mic")
            b.wait_for_selector(".entry.them .trans", timeout=15000)
            print("  fake mic -> pause-cut segment -> interpret -> partner received ok")

            mine = a.locator(".entry.me:has(.readback)").last  # oldest checked phrase; newest sit on top
            mine.wait_for(timeout=15000)  # the read-back check lands after the audio is queued
            text = mine.inner_text()
            assert "They heard:" in text and "read-back: ok" in text, text
            assert "numbers match: 50" in text, text
            assert "glossary applied: डॉक्टर" in text, text
            print("  read-back, numbers guard and glossary chips rendered ok")

            theirs = b.locator(".entry.them").last  # the first phrase: the full 3-burst clip
            ttext = theirs.inner_text()
            assert "sounded hesitant" in ttext, ttext  # the 3-burst clip has long internal pauses
            assert "numbers match: 50" in ttext
            theirs.get_by_role("button", name="Ask them to confirm").click()
            a.wait_for_function("document.querySelectorAll('.entry.them').length >= 1", timeout=15000)
            assert "typed" in a.locator(".entry.them").first.inner_text()
            print("  hesitation -> 'Ask them to confirm' -> reached the speaker ok")

            b.fill("#typed", "ठीक है, धन्यवाद")
            b.click("#typedSend")
            a.wait_for_function("document.querySelectorAll('.entry.them').length >= 2", timeout=15000)
            print("  typed message relay ok")

            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    print("UI OK")


if __name__ == "__main__":
    main()
