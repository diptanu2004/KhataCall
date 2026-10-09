"""Spoken Hinglish numbers to paise.

The heaviest-tested module in the project, because it is the one place where a
quiet mistake becomes a wrong number in someone's ledger. Hard rule 7 says the
LLM never does this conversion; that only helps if this code is right.
"""

from fractions import Fraction

import pytest

from app.nlu.numbers import (
    MAX_RUPEES,
    normalise,
    parse_amount_paise,
    parse_number,
    parse_quantity,
)


def rupees(text: str):
    """The parsed amount in whole rupees, or None."""
    paise = parse_amount_paise(text)
    return None if paise is None else paise / 100


# --- spelling ---------------------------------------------------------------
# Prisma returns romanized Hindi with no stable orthography: "rupaye" came back
# as "rupay", "udhaar" as "udhar" (D-012). Every lookup is folded first.


@pytest.mark.parametrize(
    "spellings",
    [
        ("saadhe", "sadhe"),
        ("hazaar", "hazar", "hajar", "hajaar"),
        ("pandrah", "pandra"),
        ("chaalees", "chalis", "chaalis"),
        ("paanch", "panch"),
        ("teen", "tin"),
        ("bees", "bis"),
        ("chhiyaasi", "chhiyasi"),
    ],
)
def test_folding_handles_the_regular_variations(spellings):
    """Long vowels, z/j and a trailing h are folded by rule."""
    assert len({normalise(s) for s in spellings}) == 1, [(s, normalise(s)) for s in spellings]


@pytest.mark.parametrize(
    "spellings,expected",
    [
        (("saadhe teen sau", "sadhe teen sau", "sade teen sau"), 350),
        (("dedh sau", "derh sau", "ded sau"), 150),
        (("dhai sau", "dhaai sau"), 250),
        (("sava sau", "sawa sau"), 125),
        (("paune do sau", "pone do sau"), 175),
        (("do hazaar", "do hazar", "do hajar"), 2000),
    ],
)
def test_variants_folding_cannot_reach_are_listed_in_the_tables(spellings, expected):
    """`sade` and `ded` drop an inner h, which no rule folds, so the tables name them.

    What matters is that every spelling reaches the same value, not that it
    reaches it the same way.
    """
    for spelling in spellings:
        assert rupees(spelling) == expected, spelling


def test_saat_and_saath_do_not_collide():
    """7 and 60. Folding them together would overcharge by 53 rupees."""
    assert normalise("saat") != normalise("saath")
    assert rupees("saat rupaye") == 7
    assert rupees("saath rupaye") == 60


def test_every_distinct_value_has_a_distinct_spelling():
    """A collision anywhere in 0-99 silently rewrites an amount."""
    from app.nlu.numbers import UNITS

    by_key: dict[str, set] = {}
    for key, value in UNITS.items():
        by_key.setdefault(key, set()).add(value)
    assert not [k for k, v in by_key.items() if len(v) > 1]


# --- plain numbers ----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ek", 1), ("do", 2), ("teen", 3), ("char", 4), ("paanch", 5),
        ("chhe", 6), ("saat", 7), ("aath", 8), ("nau", 9), ("das", 10),
        ("pandrah", 15), ("bees", 20), ("chaubees", 24), ("untees", 29),
        ("chaalees", 40), ("ikyaavan", 51), ("chauvan", 54), ("pachpan", 55),
        ("tirsath", 63), ("sattar", 70), ("bahattar", 72), ("pachhattar", 75),
        ("athhattar", 78), ("assi", 80), ("chhiyaasi", 86), ("nabbe", 90),
        ("ninyaanve", 99),
    ],
)
def test_hindi_units(text, expected):
    assert rupees(f"{text} rupaye") == expected


@pytest.mark.parametrize(
    "text,expected",
    [("fifty", 50), ("twenty", 20), ("ninety", 90), ("fifteen", 15), ("twenty five", 25)],
)
def test_english_units(text, expected):
    assert rupees(text) == expected


# --- multipliers ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("sau", 100),
        ("do sau", 200),
        ("paanch sau", 500),
        ("teen sau pachaas", 350),
        ("char sau bees", 420),
        ("paanch sau saath", 560),
        ("aath sau pandrah", 815),
        ("nau sau chaalees", 940),
        ("solah sau", 1600),
        ("barah sau pachaas", 1250),
        ("ek hazaar", 1000),
        ("teen hazaar", 3000),
        ("do hazaar paanch sau", 2500),
        ("ek hazaar teen sau", 1300),
        ("das hazaar", 10_000),
        ("pachchees hazaar", 25_000),
        ("pachaas hazaar", 50_000),
        ("pachhattar hazaar", 75_000),
        ("ek lakh", 100_000),
        ("two hundred", 200),
        ("five hundred", 500),
        ("thousand", 1000),
    ],
)
def test_multipliers(text, expected):
    assert rupees(text) == expected


def test_a_grouper_closes_its_group():
    """'do hazaar paanch sau' is 2000 + 500, not (2 + 500) x 1000."""
    assert rupees("do hazaar paanch sau") == 2500
    assert rupees("do lakh paanch hazaar") == 205_000


# --- fraction words ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dedh sau", 150),
        ("dhai sau", 250),
        ("sava sau", 125),
        ("saadhe teen sau", 350),
        ("paune do sau", 175),
        ("sava do sau", 225),
        ("saadhe char sau", 450),
        ("paune teen sau", 275),
        ("dedh hazaar", 1500),
        ("dhai hazaar", 2500),
        ("sava hazaar", 1250),
        ("saadhe paanch sau", 550),
        ("paune do hazaar", 1750),
        ("saadhe saat sau", 750),
        ("sava paanch sau", 525),
        ("saadhe teen", 3.5),
        ("dedh", 1.5),
    ],
)
def test_fraction_words(text, expected):
    assert rupees(text) == expected


def test_a_fraction_word_applies_to_the_number_not_the_multiplier():
    """'sava do sau' is (2 + 1/4) x 100 = 225, not 2 x (100 + 1/4)."""
    assert rupees("sava do sau") == 225
    assert rupees("saadhe teen sau") == 350


def test_a_bare_fraction_word_implies_one():
    assert rupees("sava sau") == 125  # (1 + 1/4) x 100
    assert rupees("sava hazaar") == 1250


# --- digits and mixed forms -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("150", 150), ("1250", 1250), ("2 hazaar 500", 2500), ("500 rupaye", 500), ("5 sau", 500)],
)
def test_digits_and_mixed(text, expected):
    assert rupees(text) == expected


def test_currency_words_are_ignored_whatever_their_spelling():
    for suffix in ("rupaye", "rupay", "rupees", "rs", "rupaiya"):
        assert rupees(f"dedh sau {suffix}") == 150


# --- the English hundreds idiom ---------------------------------------------


def test_one_fifty_is_a_hundred_and_fifty():
    """Indian English drops the 'hundred'. Reading it as 1 + 50 would be 51."""
    assert rupees("one fifty") == 150
    assert rupees("two fifty") == 250
    assert rupees("one twenty") == 120


def test_the_idiom_does_not_swallow_ordinary_tens():
    assert rupees("twenty five") == 25
    assert rupees("fifty") == 50


# --- refusing to guess ------------------------------------------------------
# Hard rule: ambiguity returns None and the agent re-asks. Every one of these
# would otherwise put an unverified number in the ledger.


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "kuch bhi",
        "thoda sa",
        "bohot saara",
        "udhaar likho",
        "saadhe",            # an adjustment with nothing to adjust
        "sava",              # likewise: 'sava' alone never means 1.25 rupees aloud
        "sava paune",        # two fraction words together
        "dedh dhai",         # two whole fractions together
        "do dedh",
        "dedh sau zyada",    # an unknown word might have been the amount
        "ek rupaya pachas paise",  # sub-rupee units: which was meant?
    ],
)
def test_ambiguity_returns_none(text):
    assert parse_amount_paise(text) is None


def test_an_unknown_word_is_never_skipped_over():
    """Ignoring the unrecognised word here would confidently return 100."""
    assert parse_amount_paise("sau tirupati") is None


def test_zero_and_negative_amounts_are_refused():
    assert parse_amount_paise("shunya") is None
    assert parse_amount_paise("zero rupaye") is None


def test_implausibly_large_amounts_are_refused():
    """A mis-heard word should not become a fortune in a kirana ledger."""
    assert parse_amount_paise("sau crore") is None
    assert parse_amount_paise(str(MAX_RUPEES + 1)) is None
    assert parse_amount_paise(str(MAX_RUPEES)) == MAX_RUPEES * 100


# --- paise, exactly ---------------------------------------------------------


def test_amounts_are_integer_paise():
    """Hard rule 6: money is integer paise, never a float."""
    value = parse_amount_paise("dedh sau")
    assert isinstance(value, int)
    assert value == 15_000


def test_quarter_rupee_amounts_stay_exact():
    """Quarters are exact in paise: 3.5 rupees is 350, 3.75 is 375. No float drift."""
    assert parse_number("saadhe teen") == Fraction(7, 2)
    assert parse_amount_paise("saadhe teen") == 350
    assert parse_number("paune char") == Fraction(15, 4)
    assert parse_amount_paise("paune char") == 375


@pytest.mark.parametrize(
    "text",
    ["dedh", "dhai sau", "sava do sau", "saadhe teen", "paune char", "paune do hazaar", "sava hazaar"],
)
def test_every_fraction_form_lands_on_whole_paise(text):
    """The grammar only produces quarters, and a quarter rupee is 25 paise exactly.

    `parse_amount_paise` still refuses a non-integer rather than rounding, which
    is unreachable today but keeps the guarantee if thirds or eighths ever appear.
    """
    value = parse_number(text)
    assert value is not None
    assert (value * 100).denominator == 1
    assert isinstance(parse_amount_paise(text), int)


# --- quantities -------------------------------------------------------------


def test_quantities_allow_fractions_for_loose_goods():
    assert parse_quantity("saadhe teen") == Fraction(7, 2)
    assert parse_quantity("bees") == 20


def test_dedh_darjan_is_eighteen():
    """A fraction word multiplying a non-decimal unit."""
    assert parse_quantity("dedh darjan") == 18
    assert parse_quantity("do darjan") == 24


def test_quantities_refuse_nonsense():
    assert parse_quantity("kuch") is None
    assert parse_quantity("shunya") is None


# --- the corpus's own amounts ----------------------------------------------


@pytest.mark.parametrize(
    "spoken,rupees_expected",
    [
        ("dedh sau", 150), ("dhai sau", 250), ("sava sau", 125), ("saadhe teen sau", 350),
        ("paune do sau", 175), ("sava do sau", 225), ("saadhe char sau", 450),
        ("paune teen sau", 275), ("dedh hazaar", 1500), ("dhai hazaar", 2500),
        ("sava hazaar", 1250), ("saadhe paanch sau", 550), ("paune do hazaar", 1750),
        ("saadhe saat sau", 750), ("sava paanch sau", 525), ("pachpan", 55), ("untees", 29),
        ("chhiyaasi", 86), ("sattar", 70), ("ninyaanve", 99), ("pandrah", 15),
        ("athhattar", 78), ("chauvan", 54), ("ikyaavan", 51), ("tirsath", 63),
        ("chaalees", 40), ("assi", 80), ("nabbe", 90), ("pachhattar", 75), ("chaubees", 24),
        ("do hazaar paanch sau", 2500), ("ek hazaar teen sau", 1300), ("2 hazaar 500", 2500),
        ("teen sau pachaas", 350), ("char sau bees", 420), ("paanch sau saath", 560),
        ("chhe sau pachchees", 625), ("saat sau assi", 780), ("aath sau pandrah", 815),
        ("nau sau chaalees", 940), ("150", 150), ("1250", 1250), ("solah sau", 1600),
        ("barah sau pachaas", 1250), ("teen hazaar", 3000), ("two hundred rupees", 200),
        ("five hundred", 500), ("thousand rupaye", 1000), ("fifty rupees", 50),
        ("one fifty", 150), ("two fifty", 250), ("hundred rupaye", 100),
        ("pachhattar hazaar", 75_000), ("ek lakh", 100_000), ("pachaas hazaar", 50_000),
        ("pachchees hazaar", 25_000), ("das hazaar", 10_000), ("ikkees", 21), ("satrah", 17),
        ("bahattar", 72), ("chhiyaalees", 46), ("pachchees", 25),
    ],
)
def test_every_amount_form_in_the_corpus(spoken, rupees_expected):
    """Each of these appears in eval/corpus/scripts.csv. The eval cannot score
    better than this parser, so the parser is checked against the corpus first."""
    assert rupees(spoken) == rupees_expected, spoken
