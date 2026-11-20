"""
Shared multilingual (English / Tamil / Tanglish) phrase utilities.

Phase 5: Tamil script, Tanglish (Tamil written in Latin script) and
mixed-language messages are now understood alongside English.

Design rules (approved Phase 5 Step 1 audit):
    - Additive only: every existing English behaviour is preserved.
    - No network access, no translation APIs, no extra dependencies.
    - Unicode-aware: NFC normalization, zero-width character removal,
      punctuation treated as word separators, apostrophes inside words
      preserved ("don't want spicy" keeps matching the English
      patterns). Word matching never relies on regex \\b for Tamil
      script (Tamil combining marks make \\b unreliable); padded
      whole-word checks are used instead.
    - Tamil words inflect ("saivam" -> "saiva unavu"), so detection
      uses conservative stem prefixes that cannot create false
      positives (e.g. "meen" is matched as a whole word only, never as
      a stem, so "meenakshi" is untouched).

IMPORTANT: dietary filtering itself remains NAME-BASED on the English
catalogue keywords (see food_search.py) - nothing here claims
ingredient-level dietary safety.
"""

import re
import unicodedata

# Zero-width characters that can sneak in through copy/paste.
_ZERO_WIDTH_CHARS = ("\u200b", "\u200c", "\u200d", "\ufeff")

_IN_WORD_APOSTROPHE = re.compile(r"(?<=\w)'(?=\w)")
# NOTE: Python's \w does NOT match Tamil combining signs (the virama
# U+0BCD and vowel signs are Mn/Mc categories), so the Tamil Unicode
# block (U+0B80-U+0BFF) must be preserved explicitly.
_STRIP_CHARS = re.compile(r"[^\w\s\x01\u0b80-\u0bff]+", re.UNICODE)


def normalize_message(text):
    """
    Unicode-safe normalization used by every language-sensitive
    detector:

        1. NFC normalization (Tamil combining marks become canonical,
           so decomposed keyboard/clipboard input still matches).
        2. Zero-width characters removed.
        3. Curly apostrophes (U+2018/U+2019) become straight ones so
           "don't" keeps matching the English "don't" patterns.
        4. casefold (Latin scripts; Tamil has no case).
        5. Underscores, punctuation and symbols become separators,
           EXCEPT apostrophes inside words ("don't" stays intact).
        6. Whitespace collapsed.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    for zero_width in _ZERO_WIDTH_CHARS:
        text = text.replace(zero_width, "")

    text = text.replace("\u2019", "'").replace("\u2018", "'")

    text = text.casefold()
    text = text.replace("_", " ")

    # Protect apostrophes inside words with a placeholder, turn every
    # other punctuation/symbol into a separator, then restore.
    text = _IN_WORD_APOSTROPHE.sub("\x01", text)
    text = _STRIP_CHARS.sub(" ", text)
    text = text.replace("\x01", "'")

    return " ".join(text.split())


def contains_word(normalized_text, word):
    """
    Unicode-safe whole-word check on an ALREADY-NORMALIZED message.
    The padded comparison is equivalent to a word-boundary check for
    space-separated languages (English and Tamil) and, unlike regex
    \\b, it is reliable for Tamil script.
    """
    return f" {word} " in f" {normalized_text} "


def contains_phrase(normalized_text, phrase):
    """Multi-word phrase check on an already-normalized message."""
    return f" {phrase} " in f" {normalized_text} "


def matches_any_entry(normalized_text, entries):
    """
    entries: list of (word, stem) tuples. A token matches when it IS
    the word, or - when a stem is given - when it starts with the
    stem (Tamil inflections: "saivam" -> "saiva unavu").
    Stems are deliberately conservative so they cannot create false
    positives.
    """
    for word, stem in entries:

        if contains_word(normalized_text, word):
            return True

        if stem and any(
            token.startswith(stem)
            for token in normalized_text.split()
        ):
            return True

    return False


# --------------------------------------------------------------------------
# Preference phrase dictionaries (checked against the normalized message)
# --------------------------------------------------------------------------

# (word, stem) pairs. Meat/chicken words that map directly onto English
# catalogue words (koli -> chicken, meen -> fish, ...) are handled by
# FOOD_WORD_CANONICALIZATION below, so they are NOT repeated here.

VEG_ENTRIES = [
    ("saivam", "saiva"),
    ("sagaharam", "sagahara"),
    ("sakaharam", "sakahara"),
    ("\u0b9a\u0bc8\u0bb5\u0bae\u0bcd", "\u0b9a\u0bc8\u0bb5"),          # saivam
    ("\u0b9a\u0bbe\u0b95\u0bbe\u0bb9\u0bbe\u0bb0\u0bae\u0bcd",
     "\u0b9a\u0bbe\u0b95\u0bbe\u0bb9\u0bbe\u0bb0"),                    # sagaharam
]

NONVEG_ENTRIES = [
    ("asaivam", "asaiva"),
    ("kari", None),                                                   # meat (colloquial)
    ("\u0b85\u0b9a\u0bc8\u0bb5\u0bae\u0bcd", "\u0b85\u0b9a\u0bc8\u0bb5"),  # asaivam
    ("\u0b95\u0bb1\u0bbf", None),                                     # kari
]

SPICY_ENTRIES = [
    ("karam", "karam"),          # karam, karamaana, karamu...
    ("kaaram", "kaaram"),
    ("\u0b95\u0bbe\u0bb0\u0bae\u0bcd", "\u0b95\u0bbe\u0bb0\u0bae"),    # kaaram
]

# Checked BEFORE spicy so negated spice ("karam illa") wins.
MILD_PHRASES = [
    "karam illa", "kaaram illa",
    "karam illama", "kaaram illama",
    "konjam karam", "konjam kaaram",
    "\u0b95\u0bbe\u0bb0\u0bae\u0bcd \u0b87\u0bb2\u0bcd\u0bb2\u0bc8",          # kaaram illai
    "\u0b95\u0bbe\u0bb0\u0bae\u0bcd \u0b87\u0bb2\u0bcd\u0bb2\u0bbe\u0bae",    # kaaram illaama
    "\u0b95\u0bbe\u0bb0\u0bae\u0bcd \u0b95\u0bc1\u0bb1\u0bc8\u0bb5\u0bbe",    # kaaram kuraivaa
]

SWEET_ENTRIES = [
    ("inippu", "inipp"),         # inippu, inippaana, inippaan...
    ("\u0b87\u0ba9\u0bbf\u0baa\u0bcd\u0baa\u0bc1", "\u0b87\u0ba9\u0bbf\u0baa\u0bcd\u0baa"),  # inippu
]

HEALTHY_ENTRIES = [
    ("aarogyamaana", "aarogy"),
    ("aarokkiyamaana", "aarokkiy"),
    ("\u0b86\u0bb0\u0bcb\u0b95\u0bcd\u0b95\u0bbf\u0baf\u0bae\u0bbe\u0ba9",
     "\u0b86\u0bb0\u0bcb\u0b95\u0bcd\u0b95\u0bbf\u0baf"),              # aarokkiyamaana
]

BUDGET_ENTRIES = [
    ("malivaana", "maliva"),
    ("\u0bae\u0bb2\u0bbf\u0bb5\u0bbe\u0ba9", "\u0bae\u0bb2\u0bbf\u0bb5"),  # malivaana
]

BUDGET_PHRASES = [
    "vilai kuraitha", "kammi vilai",
    "\u0bb5\u0bbf\u0bb2\u0bc8 \u0b95\u0bc1\u0bb1\u0bc8\u0ba8\u0bcd\u0ba4",    # vilai kurainthu
    "\u0bb5\u0bbf\u0bb2\u0bc8 \u0b95\u0bc1\u0bb1\u0bc8\u0bb5\u0bbe",          # vilai kuraivaa
]

# --------------------------------------------------------------------------
# Tanglish/Tamil food words -> canonical English catalogue words
# (token-based, order preserved, so it also feeds search keywords)
# --------------------------------------------------------------------------

FOOD_WORD_CANONICALIZATION = {
    # Alternate Latin spellings
    "biriyani": "biryani",
    "biryaani": "biryani",
    "biriani": "biryani",
    "dosai": "dosa",
    "iddli": "idli",
    "chapathi": "chapati",
    "chapatti": "chapati",
    "chappathi": "chapati",
    "chappati": "chapati",
    "pisa": "pizza",
    "pizaa": "pizza",
    "berger": "burger",
    # Tanglish meat/food words -> English catalogue words
    "koli": "chicken",
    "mutta": "egg",
    "meen": "fish",
    "eraal": "prawn",
    # Tamil script -> English catalogue words
    "\u0baa\u0bbf\u0bb0\u0bbf\u0baf\u0bbe\u0ba3\u0bbf": "biryani",     # biriyaani
    "\u0b95\u0bcb\u0bb4\u0bbf": "chicken",                             # koli
    "\u0bae\u0bc1\u0b9f\u0bcd\u0b9f\u0bc8": "egg",                     # mutta
    "\u0bae\u0bc0\u0ba9\u0bcd": "fish",                                # meen
    "\u0b87\u0bb1\u0bbe\u0bb2\u0bcd": "prawn",                         # eraal
    "\u0baa\u0bc0\u0b9f\u0bcd\u0b9a\u0bbe": "pizza",                   # pizza
    "\u0ba4\u0bcb\u0b9a\u0bc8": "dosa",                                # dosai
    "\u0b87\u0b9f\u0bcd\u0bb2\u0bbf": "idli",                          # idli
    "\u0b9a\u0baa\u0bcd\u0baa\u0bbe\u0ba4\u0bcd\u0ba4\u0bbf": "chapati",  # chapathi
}


def canonicalize_food_words(normalized_text):
    """
    Replace Tanglish/Tamil food words with their canonical English
    catalogue words (token-based, order preserved):

        "enakku koli biriyani venum"
            -> "enakku chicken biryani venum"

    The result lets the EXISTING English detection patterns and search
    keywords work unchanged for Tanglish/Tamil food words.
    """
    tokens = normalized_text.split()

    mapped = [
        FOOD_WORD_CANONICALIZATION.get(token, token)
        for token in tokens
    ]

    return " ".join(mapped)


# --------------------------------------------------------------------------
# Filler words, negation words and request-signal words
# --------------------------------------------------------------------------

# Tanglish/Tamil filler words. NEGATION words are deliberately NOT
# fillers ("illa"/"venam" carry meaning: "X venam" = don't want X).
TANGLISH_FILLER_WORDS = frozenset({
    # Tanglish
    "enakku", "enaku", "venum", "venuma", "oru", "konjam",
    "kodunga", "kudunga", "kodukunga", "podu", "pannu", "pannunga",
    "seiyunga", "seidhu", "namma", "unga", "ungal", "ungaluku",
    "ungalukku", "inaiku", "innaiku", "romba", "sollunga", "sollu",
    "kandippa", "vanakkam", "vanakam",
    # Tamil script
    "\u0b8e\u0ba9\u0b95\u0bcd\u0b95\u0bc1",                 # enakku
    "\u0bb5\u0bc7\u0ba3\u0bc1\u0bae\u0bcd",                 # venum
    "\u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd",     # vendum
    "\u0b92\u0bb0\u0bc1",                                   # oru
    "\u0b95\u0bca\u0b9e\u0bcd\u0b9a\u0bae\u0bcd",           # konjam
    "\u0b95\u0bca\u0b9f\u0bc1\u0b99\u0bcd\u0b95",           # kodunga
    "\u0b95\u0bc1\u0b9f\u0bc1\u0b99\u0bcd\u0b95",           # kudunga
    "\u0baa\u0ba3\u0bcd\u0ba3\u0bc1\u0b99\u0bcd\u0b95",     # pannunga
    "\u0ba8\u0bbe\u0ba9\u0bcd",                             # naan (I)
    "\u0bb0\u0bca\u0bae\u0bcd\u0baa",                       # romba
    "\u0b87\u0ba9\u0bcd\u0bb1\u0bc1",                       # indru
    "\u0b87\u0ba9\u0bcd\u0ba9\u0bc8\u0b95\u0bcd\u0b95\u0bc1",  # innaikku
    "\u0b9a\u0bca\u0bb2\u0bcd\u0bb2\u0bc1\u0b99\u0bcd\u0b95",  # sollunga
    "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd",           # vanakkam
})

# Negation words. Kept OUT of the filler list so they can participate
# in negation handling ("enakku biryani venam, dosa venum" -> dosa).
NEGATION_WORDS = frozenset({
    "venam", "vennam", "vendaam", "veendaam",
    "illa", "illai", "illama", "illadha",
    "\u0b87\u0bb2\u0bcd\u0bb2\u0bc8",                       # illai
    "\u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bbe\u0bae\u0bcd",     # vendaam
})

# Tanglish/Tamil broad food-request signal words (gate the general
# fallback the same way "food"/"hungry" do for English).
TANGLISH_REQUEST_SIGNAL_WORDS = frozenset({
    "sapadu", "saapadu", "unavu",
    "pasikuthu", "pasikradhu", "pasikum",
    "saapidalam", "sapadalam",
    "\u0b89\u0ba3\u0bb5\u0bc1",                             # unavu
    "\u0b9a\u0bbe\u0baa\u0bcd\u0baa\u0bbe\u0b9f\u0bc1",     # saapaadu
    "\u0baa\u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bc1\u0ba4\u0bc1",  # pasikuthu
})

# Tanglish continuation requests (ask for MORE/OTHER items). Checked on
# the normalized message next to the English CONTINUATION_PATTERN.
# Kept deliberately specific so food requests like
# "innum biryani venum" (I want biryani again) stay food requests.
TANGLISH_CONTINUATION_PATTERN = re.compile(
    r"\binnum\s+(?:konjam\s+)?(?:kaatunga|kaaminga|kaami|option|options)\b"
    r"|\bvera\s+"
    r"(?:edavadhu|edhavadhu|ethavadhu|ethavathu|edhachum|option|options|sapadu|food)\b"
    r"|\binnoru\s+(?:option|options|food)\b"
)
