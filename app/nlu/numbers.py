"""Spoken Hinglish numbers to exact values.

This module exists because of hard rule 7: the LLM never converts an amount.
It passes the words as spoken (`amount_spoken`), and this code decides what
they mean. A language model that is 99% right about money is wrong about
money, and its mistakes are silent. These are deterministic and testable.

Three things shape the implementation:

**Hindi 1-99 are irregular**, so they are a lookup table rather than a rule.
There is no deriving `unsath` (59) or `chhiyaasi` (86) from parts.

**Romanized spelling is not stable.** Prisma returned `rupay` for "rupaye",
`udhar` for "udhaar" and `du` for "doon" (DECISIONS.md D-012). Romanized Hindi
has no standard orthography, so every lookup goes through `normalise()`, which
folds the common variations, and the tables also list spellings that folding
cannot reach.

**Ambiguity returns None.** The agent then re-asks. Anything else means writing
a number into someone's ledger that nobody verified.

    >>> parse_amount_paise("dedh sau")
    15000
    >>> parse_amount_paise("do hazaar paanch sau")
    250000
    >>> parse_amount_paise("kuch bhi") is None
    True
"""

import re
from fractions import Fraction

__all__ = [
    "heard_values",
    "normalise",
    "parse_amount_paise",
    "parse_number",
    "parse_quantity",
    "parse_signed_quantity",
]

# Guard against a mis-heard word turning into a fortune. A kirana udhaar of over
# a crore is a transcription error, not a transaction.
MAX_RUPEES = 10_000_000

_VOWELS = frozenset("aeiou")
# Digits keep their separators, so "1,500" is one token and not "1" then "500"
# (which would read as 501), and "2.5" is not "2" then "5".
_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")


def normalise(word: str) -> str:
    """Fold the spelling variations romanized Hindi throws around.

    `saadhe`/`sadhe`, `hazaar`/`hajar`, `pandrah`/`pandra` all have to reach the
    same table entry. The trailing `h` is dropped only after a vowel, which keeps
    `saat` (7) and `saath` (60) apart - they would otherwise collide, and a
    customer would be charged 60 rupees instead of 7.
    """
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    word = word.replace("w", "v").replace("z", "j")
    # Long vowels before collapsing doubles, so "ee" becomes "i" rather than "e".
    word = word.replace("ee", "i").replace("oo", "u").replace("aa", "a")
    word = re.sub(r"(.)\1+", r"\1", word)
    if len(word) > 2 and word.endswith("h") and word[-2] in _VOWELS:
        word = word[:-1]
    return word


def _table(pairs: dict[str, int | Fraction]) -> dict[str, int | Fraction]:
    out: dict[str, int | Fraction] = {}
    for spellings, value in pairs.items():
        for spelling in spellings.split():
            key = normalise(spelling)
            if key and key not in out:
                out[key] = value
    return out


# Hindi 0-99. Each key lists the spellings seen in practice; `normalise` catches the rest.
_HINDI_UNITS = {
    "shunya sifar zero": 0,
    "ek": 1, "do": 2, "teen tin": 3, "char chaar": 4, "paanch panch": 5,
    "chhe che chah": 6, "saat sat": 7, "aath ath": 8, "nau no": 9, "das dus": 10,
    "gyarah gyara igyarah": 11, "barah bara": 12, "terah tera": 13, "chaudah chauda": 14,
    "pandrah pandra": 15, "solah sola": 16, "satrah satra": 17, "atharah athara": 18,
    "unnees unnis": 19, "bees bis": 20,
    "ikkees ikkis": 21, "baees bais": 22, "teis teees taees": 23, "chaubees chaubis": 24,
    "pachchees pachees pachis": 25, "chhabbees chhabbis": 26, "sattaees sattais": 27,
    "atthaees atthais": 28, "untees untis": 29, "tees tis": 30,
    "iktees iktis": 31, "battees battis": 32, "taintees taintis": 33, "chauntees chauntis": 34,
    "paintees paintis": 35, "chhattees chhattis": 36, "saintees saintis": 37, "adtees adtis": 38,
    "untaalees untalis": 39, "chaalees chalis": 40,
    "iktaalees iktalis": 41, "bayaalees bayalis": 42, "taintaalees taintalis": 43,
    "chavaalees chavalis chauvalis": 44, "paintaalees paintalis": 45, "chhiyaalees chhiyalis": 46,
    "saintaalees saintalis": 47, "adtaalees adtalis": 48, "unchaas unchas": 49, "pachaas pachas": 50,
    "ikyaavan ikyavan": 51, "baavan bavan": 52, "tirpan": 53, "chauvan": 54, "pachpan": 55,
    "chhappan chhapan": 56, "sattavan satavan": 57, "atthaavan athavan": 58, "unsath": 59,
    "saath": 60,
    "iksath": 61, "baasath basath": 62, "tirsath": 63, "chausath": 64, "painsath": 65,
    "chhiyaasath chhiyasath": 66, "sadsath": 67, "adsath": 68, "unhattar": 69, "sattar satar": 70,
    "ikhattar": 71, "bahattar": 72, "tihattar": 73, "chauhattar": 74, "pachhattar": 75,
    "chhihattar": 76, "sathattar": 77, "athhattar": 78, "unaasi unasi": 79, "assi asi": 80,
    "ikyaasi ikyasi": 81, "bayaasi bayasi": 82, "tirasi": 83, "chaurasi": 84, "pachaasi pachasi": 85,
    "chhiyaasi chhiyasi": 86, "sattasi satasi": 87, "atthasi athasi": 88, "navaasi navasi": 89,
    "nabbe nabe": 90,
    "ikyaanve ikyanve": 91, "bayaanve bayanve": 92, "tiraanve tiranve": 93,
    "chauraanve chauranve": 94, "pachaanve pachanve": 95, "chhiyaanve chhiyanve": 96,
    "sattaanve satanve": 97, "atthaanve athanve": 98, "ninyaanve ninyanve": 99,
}

_ENGLISH_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty fourty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

UNITS = _table({**_HINDI_UNITS, **_ENGLISH_UNITS})
ENGLISH_UNITS = _table(_ENGLISH_UNITS)

# Multipliers that scale what came before them without closing a group.
SCALERS = _table({"sau hundred": 100, "darjan dozen": 12})
# Multipliers that close a group: "do hazaar paanch sau" is 2000 + 500, not 2500 * 1000.
GROUPERS = _table({
    "hazaar hazar hajar thousand": 1_000,
    "lakh lakhs lac": 100_000,
    "crore karod million": 10_000_000,
})

# Whole values in themselves: dedh is one and a half, dhai is two and a half.
FRACTION_VALUES = _table({"dedh derh ded": Fraction(3, 2), "dhai dhaai adhai": Fraction(5, 2)})
# Adjustments to whatever number follows: "saadhe teen sau" is (3 + 1/2) x 100.
FRACTION_DELTAS = _table({
    "sava sawa": Fraction(1, 4),
    "saadhe sadhe sade": Fraction(1, 2),
    "paune pone": Fraction(-1, 4),
})

# Currency and filler words carry no value. Anything else unrecognised is treated as
# ambiguity, because a word we cannot account for might have been the amount.
IGNORED = frozenset(
    normalise(w)
    for w in "rupaye rupay rupaiya rupaya rupee rupees rs inr ka ke ki aur and only total".split()
)
# "paise" would make the unit sub-rupee, and we will not guess which was meant.
REFUSED = frozenset(normalise(w) for w in "paisa paise paisa".split())

# M4 fix 2 (D-042): spellings the real models produced that folding cannot reach,
# kept apart so the eval can score with and without them from one commit.
# Each is a measured case: Evon wrote "ikkes" (ikkees, 21) and "unteen" (untees,
# 29) on text (D-031); Prisma wrote "nao" for nau (9), turning 940 into 140
# (D-038). No blanket e/i folding: "ten" and "teen" must stay apart.
SPELLING_FIXES = _table({"ikkes": 21, "unteen": 29, "nao": 9})
# "aath sau mile" (800 received): Evon carried the verb into amount_spoken.
FILLER_FIXES = frozenset(normalise(w) for w in "mile mila".split())
UNITS_FIXED = {**UNITS, **{k: v for k, v in SPELLING_FIXES.items() if k not in UNITS}}
IGNORED_FIXED = IGNORED | FILLER_FIXES

# M4 fix 3 (D-043): the unit a stock quantity is counted in. Evon passes it along
# ("paanch kilo", "do bori", "paanch jar") and the parser refused every one: 7
# of 30 stock updates on text (D-031). Only quantities accept these; "paanch
# kilo" is never a rupee amount. "darjan"/"dozen" stay multipliers (x12).
UNIT_WORDS = frozenset(
    normalise(w)
    for w in (
        "kilo kilos kg kgs kilogram gram grams gm litre litres liter liters ltr "
        "packet packets pack packs pkt bori bora bag bags jar jars sachet sachets "
        "bottle bottles botal dabba dibba box boxes peti carton cartons tube tubes "
        "piece pieces pcs"
    ).split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _digit_value(token: str) -> Fraction | None:
    """ "1,500" and "1,50,000" (Western or Indian grouping), "350.50", "2.5". None if unclear.

    A comma is only read as a thousands separator when the groups look like one:
    "1,5" could be a European decimal, so it is refused rather than guessed.
    """
    whole, _, decimals = token.partition(".")
    if "." in decimals or "," in decimals:
        return None  # "1.2.3"
    groups = whole.split(",")
    if len(groups) > 1 and (len(groups[-1]) != 3 or not all(1 <= len(g) <= 3 for g in groups[:-1])):
        return None
    return Fraction("".join(groups) + (f".{decimals}" if decimals else ""))


def _can_add(current: Fraction, value: Fraction, *, english: bool) -> bool:
    """Is "current value" one number, or two numbers side by side?

    "do sau paanch" is 205 and "twenty five" is 25, because the second word
    fills a place the first left empty. "do teen" is Hindi for "two or three",
    and "paanch das" for "five or ten": adding them would invent 5 and 15.
    Hindi has a word for every number below a hundred, so only English joins
    tens and units ("twenty five"); Hindi joins only hundreds and below.
    """
    if current.denominator != 1 or value.denominator != 1:
        return False
    if current >= 100 and current % 100 == 0 and value < 100:
        return True
    return english and current >= 20 and current % 10 == 0 and value < 10


def _english_hundreds_idiom(tokens: list[str], ignored: frozenset[str] = IGNORED) -> Fraction | None:
    """"one fifty" is 150, not 51.

    Indian English says an amount as hundreds-then-tens with the "hundred" left
    out. It only applies to a bare unit followed by a bare ten, so "twenty five"
    (25) and "one hundred fifty" are untouched.
    """
    words = [normalise(t) for t in tokens if not t.isdigit()]
    words = [w for w in words if w not in ignored]
    if len(words) != 2 or any(w not in ENGLISH_UNITS for w in words):
        return None
    first, second = ENGLISH_UNITS[words[0]], ENGLISH_UNITS[words[1]]
    if 1 <= first <= 9 and 20 <= second <= 90 and second % 10 == 0:
        return Fraction(first * 100 + second)
    return None


def parse_number(text: str, *, spelling_fixes: bool = True) -> Fraction | None:
    """The numeric value of a spoken phrase, or None if it cannot be read with certainty.

    `spelling_fixes=False` is the parser as it was before M4 fix 2, for the eval's
    "before" (app/agent/fixes.py).
    """
    units = UNITS_FIXED if spelling_fixes else UNITS
    ignored = IGNORED_FIXED if spelling_fixes else IGNORED
    tokens = _tokens(text)
    if not tokens:
        return None
    if (idiom := _english_hundreds_idiom(tokens, ignored)) is not None:
        return idiom

    total = Fraction(0)
    current: Fraction | None = None
    pending: Fraction | None = None  # a sava/saadhe/paune waiting for its number
    seen_value = False
    english = True  # every unit word so far was English, so "twenty five" may join

    def base() -> Fraction:
        """What a multiplier applies to: the running number, or an implied one."""
        start = current if current is not None else Fraction(1)
        return start + pending if pending is not None else start

    for token in tokens:
        if _DIGITS.fullmatch(token):
            value = _digit_value(token)
            if value is None:
                return None
            if pending is not None:
                value, pending = value + pending, None
            if current is not None and not _can_add(current, value, english=False):
                return None
            current = value if current is None else current + value
            english = False
            seen_value = True
            continue

        word = normalise(token)
        if not word or word in ignored:
            continue
        if word in REFUSED:
            return None

        if word in FRACTION_VALUES:
            if pending is not None or current is not None:
                return None  # "dedh sava" or "do dedh" means nothing
            current = Fraction(FRACTION_VALUES[word])
            seen_value = True
        elif word in FRACTION_DELTAS:
            if pending is not None:
                return None  # two fraction words in a row
            pending = Fraction(FRACTION_DELTAS[word])
        elif word in units:
            value = Fraction(units[word])
            if pending is not None:
                value, pending = value + pending, None
            english = english and word in ENGLISH_UNITS
            if current is not None and not _can_add(current, value, english=english):
                return None  # two numbers side by side: "do teen", "teen ke paanch"
            current = value if current is None else current + value
            seen_value = True
        elif word in SCALERS:
            current = base() * SCALERS[word]
            pending = None
            seen_value = True
        elif word in GROUPERS:
            total += base() * GROUPERS[word]
            current, pending = None, None
            seen_value = True
        else:
            return None  # an unknown word might have been the amount

    if pending is not None:
        return None  # "saadhe" with nothing to adjust
    if not seen_value:
        return None
    if current is not None:
        total += current
    return total


def parse_amount_paise(text: str, *, spelling_fixes: bool = True) -> int | None:
    """A spoken amount as integer paise, or None if it is not certain.

    Money is integer paise everywhere (hard rule 6). A value that is not a whole
    number of paise, is not positive, or is implausibly large is refused rather
    than rounded.
    """
    value = parse_number(text, spelling_fixes=spelling_fixes)
    if value is None:
        return None
    paise = value * 100
    if paise.denominator != 1:
        return None
    paise = int(paise)
    if not 0 < paise <= MAX_RUPEES * 100:
        return None
    return paise


def parse_signed_quantity(text: str, *, spelling_fixes: bool = True, unit_words: bool = True) -> Fraction | None:
    """A stock change: "bees" is +20, "minus das" is -10.

    Shared by the executor and the eval, so the eval scores exactly what the
    agent would write rather than a reimplementation of it.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    sign = 1
    for prefix in ("minus", "-"):
        if lowered.startswith(prefix):
            sign, stripped = -1, stripped[len(prefix):].strip()
            break
    magnitude = parse_quantity(stripped, spelling_fixes=spelling_fixes, unit_words=unit_words)
    return None if magnitude is None else sign * magnitude


def parse_quantity(text: str, *, spelling_fixes: bool = True, unit_words: bool = True) -> Fraction | None:
    """A spoken stock quantity, which may be fractional ("saadhe teen kilo").

    Returned as a Fraction rather than an int: whether loose goods are supported
    is the executor's decision, not this module's (risk R6).
    """
    if unit_words:
        text = " ".join(t for t in _tokens(text) if normalise(t) not in UNIT_WORDS)
    value = parse_number(text, spelling_fixes=spelling_fixes)
    if value is None or value <= 0 or value > 1_000_000:
        return None
    return value


def heard_values(text: str, *, spelling_fixes: bool = True) -> set[Fraction]:
    """Every number the words of `text` can be read as, from any run of them.

    M4 fix 4 (D-043) checks Evon's amount against this: the value it proposes
    must be one the owner actually said. Every contiguous run of tokens is
    tried, so fillers inside a number ("teen sau rupaye pachaas") still read as
    350; a run containing any word that is not part of a number reads as nothing.
    """
    tokens = _tokens(text)
    values = set()
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + 12) + 1):
            value = parse_number(" ".join(tokens[start:end]), spelling_fixes=spelling_fixes)
            if value is not None:
                values.add(value)
    return values
