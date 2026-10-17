"""Understand a spoken language name ("Tamil", "तमिल", "தமிழ்") as a language code.

A name table plus fuzzy matching. Speech recognition spells the same name differently depending on
the script it picked, so this is a convenience; the page always has buttons as the fallback.
"""
import re
from difflib import SequenceMatcher

NAMES = {
    "hi-IN": ["hindi", "हिन्दी", "हिंदी", "ஹிந்தி", "హిందీ", "ಹಿಂದಿ", "ഹിന്ദി", "হিন্দি", "હિન્દી", "ਹਿੰਦੀ", "ହିନ୍ଦୀ"],
    "ta-IN": ["tamil", "तमिल", "தமிழ்", "తమిళం", "ತಮಿಳು", "തമിഴ്", "তামিল", "તમિલ", "ਤਮਿਲ", "ତାମିଲ"],
    "te-IN": ["telugu", "तेलुगु", "तेलुगू", "తెలుగు", "தெலுங்கு", "ತೆಲುಗು", "തെലുങ്ക്", "তেলুগু", "તેલુગુ", "ਤੇਲਗੂ"],
    "kn-IN": ["kannada", "कन्नड़", "कन्नड", "ಕನ್ನಡ", "கன்னடம்", "కన్నడ", "കന്നഡ", "কন্নড়", "કન્નડ"],
    "ml-IN": ["malayalam", "मलयालम", "മലയാളം", "மலையாளம்", "మలయాళం", "ಮಲಯಾಳಂ", "মালয়ালম", "મલયાલમ"],
    "mr-IN": ["marathi", "मराठी", "மராத்தி", "మరాఠీ", "ಮರಾಠಿ", "മറാത്തി", "মারাঠি", "મરાઠી", "ਮਰਾਠੀ"],
    "bn-IN": ["bengali", "bangla", "बंगाली", "बांग्ला", "বাংলা", "வங்காளம்", "బెంగాలీ", "ಬಂಗಾಳಿ", "ബംഗാളി", "બંગાળી"],
    "gu-IN": ["gujarati", "गुजराती", "ગુજરાતી", "குஜராத்தி", "గుజరాతీ", "ಗುಜರಾತಿ", "ഗുജറാത്തി", "গুজরাটি"],
    "pa-IN": ["punjabi", "panjabi", "पंजाबी", "ਪੰਜਾਬੀ", "பஞ்சாபி", "పంజాబీ", "ಪಂಜಾಬಿ", "പഞ്ചാബി", "পাঞ্জাবি"],
    "od-IN": ["odia", "oriya", "उड़िया", "ओड़िया", "ଓଡ଼ିଆ", "ஒடியா", "ఒడియా", "ಒಡಿಯಾ", "ഒഡിയ", "ওড়িয়া"],
    "en-IN": ["english", "अंग्रेज़ी", "अंग्रेजी", "इंग्लिश", "ஆங்கிலம்", "ఇంగ్లీష్", "ಇಂಗ್ಲಿಷ್", "ഇംഗ്ലീഷ്", "ইংরেজি", "અંગ્રેજી", "ਅੰਗਰੇਜ਼ੀ"],
}
_TRIM = ".,!?;:।\"'()[]{}"


def parse(text: str, threshold: float = 0.8) -> str | None:
    """Language code for the first language name found in the text, or None."""
    words = [w.strip(_TRIM).casefold() for w in re.split(r"\s+", text) if w.strip(_TRIM)]
    if not words:
        return None
    best, best_score = None, 0.0
    for code, names in NAMES.items():
        for name in names:
            name = name.casefold()
            for w in words:
                score = 1.0 if w == name else SequenceMatcher(None, w, name).ratio()
                if score > best_score:
                    best, best_score = code, score
    return best if best_score >= threshold else None
