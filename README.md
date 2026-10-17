# Sarvam Interpreter and Song Singer

**What it is.** A voice-first web app built on Sarvam's Indian-language speech, translation and
text-to-speech models. In *interpreter mode* two people in a shared room each speak their own language
and hear the other in theirs, with delivery cues (hesitation, pace), number checks and a read-back of
what the partner heard. In *song mode* you hum or sing a few lines, it lists the songs it thinks you
mean, asks which language you want (you can just say it), translates the lyrics so they fit the tune,
and sings the result back in a stock or cloned voice, or sends it to your partner in their language.

**Status.** The backend (`tests/smoke_test.py`) and the real page driven by Edge with a fake
microphone (`tests/ui_test.py`) pass in offline mock mode. It has **not** been run against the live
Sarvam API, with real speech, or with real singing, so quality and latency are unmeasured. Request
shapes follow docs.sarvam.ai as read on 2026-10-17; any API error is shown verbatim in the page.

## API key

One key: **`SARVAM_API_KEY`**, created under *API Keys* in the Sarvam dashboard (shown once). Put it
in `sarvam/.env` (copy `.env.example`; `.env` is git-ignored). It covers speech-to-text, translation,
text-to-speech, chat and voice cloning. I found no separate "Indus" key. Voice cloning is beta and may
not be on your account; the app then falls back to stock voices and says so. Early-access model ids or
a different base URL go in `.env` (`SARVAM_STT_MODEL`, `SARVAM_TTS_MODEL`, `SARVAM_BASE_URL`, ...).

## Run

```powershell
cd sarvam
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # then add your key
.venv\Scripts\python -m uvicorn server:app --port 8000
```

Open http://localhost:8000 (two windows for two people: same room, Person A and Person B). With no
key it runs in **mock mode** (canned text, beep audio). Microphone access needs `localhost` or HTTPS.
Tests: `.venv\Scripts\python tests\smoke_test.py`, and after `pip install -r requirements-dev.txt`,
`.venv\Scripts\python tests\ui_test.py`.

## Interpreter mode

| Feature | What it does | Basis |
|---|---|---|
| Tone card | Measures pace, pitch range, pauses; shows "sounded hesitant" to the listener; feeds the translation prompt and TTS pace | Hear2Act |
| Numbers guard | Checks the same numbers were said and translated; native digits (५०, ௫௦) are normalized | Names and numbers are where speech translation fails |
| Read-back | After the partner has the audio, translates it back and shows the speaker "They heard: ..." | Backtranslation helped physicians catch harmful MT errors |
| Evidence-gated glossary | Your terms bias Saaras (`keyterms`) and are enforced only when actually said | EGTA |
| Hesitation-triggered confirm | One tap sends "could you confirm that?" back in the speaker's language | Hear2Act |
| Pause-triggered segments | Each phrase is translated when you pause, not after the whole turn | Simultaneous-translation idea, as a cascade |

## Song mode

1. **Hum or sing** the first few lines (raw microphone: noise suppression would erase a hum).
2. **Identify.** The tune becomes notes; the words (if any) go to Saaras. Known songs are ranked by
   pitch-contour warping (key and tempo removed) averaged with lyric similarity.
3. **Pick** a song, or "none of these" to use your own words and tune. Fix any misheard lyrics.
4. **Language.** Tap one, or say it ("Tamil", "तमிழ்"); "Ask me out loud" speaks the questions.
5. **Sing back.** Each lyric line is translated to the melody's note count (measure, then ask again
   if off), spoken by Bulbul (or your cloned voice), then retimed and re-pitched onto the tune.
   In a room it goes to your partner in their language.

**Songs.** Only public-domain songs ship (`songs/*.json`: three nursery tunes). Anything else is
handled from what you sing, and **Save to my songs** stores your own words and tune in `songs/user/` on
your machine, so it is recognized next time. Add songs by dropping a JSON file in `songs/`.

**Honest limits.** The "singing" is Praat PSOLA on ordinary TTS audio: it keeps the voice but is a
talk-sung rendition, not a trained singing voice. Recognizing sung words is harder than speech. The
tiny catalog means identification is only as wide as what you add. Syllable counting for Indian
scripts is an approximation, and a line within one syllable usually sings fine.

## Newer research used (and how honestly)

| Paper | Link | Use | I opened it |
|---|---|---|---|
| **Hear2Act** (Aug 2026) | https://arxiv.org/abs/2608.19515 | Audio LLMs barely used prosody until it was written down as text; here delivery becomes an explicit tone card | Yes |
| **EGTA** (Jul 2026) | https://arxiv.org/abs/2607.17766 | Inject only terms with evidence in the current stream; the glossary gate is a simple analogue | Abstract |
| **Why We Need Speech to Evaluate Speech Translation** (May 2026) | https://arxiv.org/abs/2605.28227 | Metrics miss prosody and gender agreement, so the speaker sees the actual back-translation | Abstract |
| Physician study on backtranslation (EMNLP 2023, older) | https://arxiv.org/abs/2310.16924 | Backtranslation caught harmful MT errors that quality scores missed | Abstract |
| **MAVL / SylAVL-CoT** (EMNLP 2025) | https://arxiv.org/abs/2505.18614 | Song translation with syllable constraints; the measure-and-retry loop follows this idea (text only, no audio or video) | Abstract (method details from a search summary) |
| **Hum2Song matching** (Zalo challenge, ~94% MRR@10) | https://arxiv.org/abs/2410.20352 | Learned hum embeddings are the upgrade over this app's pitch-contour warping; they need training data this app lacks | Abstract |
| **VocalParse** (May 2026) | https://arxiv.org/abs/2605.04613 | One model for lyrics, melody and word-note alignment from singing; not used, motivates the editable lyrics box | Abstract |
| **SoulX-Singer** (Feb 2026), singing-synthesis review (Jan 2026) | https://arxiv.org/pdf/2602.07803, https://arxiv.org/html/2601.13910v1 | Zero-shot cross-lingual singing synthesis exists but I found none covering Indian languages, so not used | Search summaries only |
| Songs Across Borders, Reffly | https://arxiv.org/pdf/2305.16816, https://arxiv.org/html/2409.00292 | Singable translation as constrained translation; GPT-4 often puts key words on weak notes, which this app does not handle | Search summaries only |
| **AI Twin** (CHI 2026) | https://arxiv.org/abs/2601.11103 | Hearing speech back in your own voice was more motivating; basis for the voice clone | Abstract |
| **IndicFDB**, **SimulU**, **StreamSpeech** | https://arxiv.org/abs/2609.31967, https://arxiv.org/abs/2603.16924, https://arxiv.org/abs/2406.03049 | Barge-in (IndicFDB, abstract) and pause-triggered commit (inspiration only, search summaries) | Mixed |

## Files

`server.py` interpreter endpoints and room relay; `song_routes.py` song endpoints; `core.py` shared
client, rooms and relay; `sarvam_client.py` Sarvam REST calls plus an offline mock; `prosody.py`
pitch and tone; `melody.py` hum to notes and tune matching; `songs.py` catalog and ranking;
`lyrics.py` syllables and singable translation; `singer.py` the PSOLA renderer; `checks.py` numbers,
glossary, read-back similarity; `languages.py` spoken language names; `static/` the page.

## Known limits

- Unverified against the live API: endpoint fields, the voice-clone response shape, whether Bulbul v4 Flash supports cloning, how Saaras handles `keyterms`, and how it transcribes singing.
- Bulbul v4 Flash speaker names are known only for Hindi and English; other languages use Bulbul v3 voices.
- Tone thresholds are uncalibrated heuristics; pace in words per second reads Tamil as "slow" more often than Hindi.
- The numbers guard compares digits only; spelled-out numbers are not understood. A "numbers differ" flag is a prompt to look, not proof.
- Read-back similarity is character trigrams with an uncalibrated floor; round trips paraphrase, so low scores are flags, not verdicts.
- Not true streaming: REST endpoints cap a segment at 30 s (the page cuts at 12 s for speech, 15 s for songs).
- LLM translation can add or drop words. Do not rely on it for medical, legal or financial conversations.
- Audio and transcripts pass through your server and Sarvam. Clone only your own voice.

## Deploying

This is a stateful Python server (WebSocket relay, in-memory rooms, a native Praat library, requests
that run for seconds), so it will **not** run as-is on Vercel's serverless functions. Run it locally,
or host the backend on a long-running host (Render, Railway, Fly.io) and, if you want, serve `static/`
from Vercel with the API base URL pointed at that host (not implemented here).
