"""Settings plus language and voice tables.

Everything model-specific lives here, so early-access model ids or endpoints can be swapped in
through environment variables (see .env.example) without touching the code.
"""
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
MT_MODEL = os.getenv("SARVAM_MT_MODEL", "mayura:v1")
MT_FALLBACK_MODEL = "sarvam-translate:v1"
LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-105b")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")
TTS_FALLBACK_MODEL = "bulbul:v3"

# No key means mock mode, so the whole flow can be exercised offline.
MOCK = os.getenv("MOCK", "").lower() in {"1", "true", "yes"} or not API_KEY

# Test hook: in mock mode, make speech recognition "hear" this text.
MOCK_TRANSCRIPT = os.getenv("MOCK_TRANSCRIPT", "")

LANGUAGES = {
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
    "en-IN": "English",
}

# Role A and role B get different voices so the two people are distinguishable.
# Only Bulbul v4 Flash speakers named in Sarvam's docs are listed; every other language falls
# back to Bulbul v3 voices, which cover all 11 languages.
_V4_SPEAKERS = {
    ("hi-IN", "A"): "ishita_hi_ads",
    ("hi-IN", "B"): "shubh_enhi_ads",
    ("en-IN", "A"): "simran_en_customer",
    ("en-IN", "B"): "shubh_enhi_companion",
}
V3_SPEAKERS = {"A": "ishita", "B": "shubh"}


def tts_voice(lang: str, role: str) -> tuple[str, str]:
    """Return (model, speaker) for a language and role."""
    override = os.getenv(f"SARVAM_TTS_SPEAKER_{role}")
    if override:
        return TTS_MODEL, override
    if TTS_MODEL == "bulbul:v4-flash" and (lang, role) in _V4_SPEAKERS:
        return TTS_MODEL, _V4_SPEAKERS[(lang, role)]
    return TTS_FALLBACK_MODEL, V3_SPEAKERS[role]
