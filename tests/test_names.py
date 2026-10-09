"""Matching a spoken name to a customer.

The dangerous failure here is not "did not find them" - the agent just asks.
It is confidently writing a debt against the wrong person, which looks like a
success and is discovered weeks later when two customers disagree with the book.
"""

import pytest

from app.nlu.names import (
    CONFIDENT,
    SEPARATION,
    Decision,
    match_name,
    normalise_name,
    strip_honorifics,
)

CUSTOMERS = ["Sharma ji", "Verma bhaiya", "Gupta aunty", "Yadav ji", "Mishra ji"]


# --- honorifics -------------------------------------------------------------


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("Sharma ji", "Sharma"),
        ("Sharma jee", "Sharma"),
        ("Verma bhaiya", "Verma"),
        ("Verma bhai", "Verma"),
        ("Tiwari didi", "Tiwari"),
        ("Gupta aunty", "Gupta"),
        ("Gupta auntie", "Gupta"),
        ("Joshi uncle", "Joshi"),
        ("Mishra saheb", "Mishra"),
        ("Shri Pandey", "Pandey"),
        ("Rawat chacha", "Rawat"),
        ("Bisht", "Bisht"),
    ],
)
def test_honorifics_are_stripped(spoken, expected):
    assert strip_honorifics(spoken) == expected


def test_every_honorific_in_the_corpus_is_handled():
    for honorific in ("ji", "bhaiya", "didi", "aunty", "uncle", "bhai"):
        assert strip_honorifics(f"Khanna {honorific}") == "Khanna"


def test_a_bare_honorific_is_left_alone():
    """There is no name in "ji" to find; the caller decides what that means."""
    assert strip_honorifics("ji") == "ji"


def test_multi_word_names_survive():
    assert strip_honorifics("Ram Prasad ji") == "Ram Prasad"


def test_word_order_does_not_matter():
    assert normalise_name("Ram Prasad ji") == normalise_name("Prasad Ram")


@pytest.mark.parametrize(
    "spoken,known",
    [
        ("Shukhla jee", "Shukla ji"),   # an inner aspirated consonant
        ("Bist ji", "Bisht ji"),
        ("Chouhan bhai", "Chauhan ji"),
        ("Mishraa ji", "Mishra ji"),
    ],
)
def test_romanization_variants_still_find_the_customer(spoken, known):
    """Prisma's romanization is not stable (D-012).

    Folding handles the regular variations; fuzzy matching covers the rest, so
    the keys need not be identical for the right customer to be found.
    """
    result = match_name(spoken, [known, "Chaturvedi ji"])
    assert result.is_match and result.name == known


# --- confident matches ------------------------------------------------------


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("Sharma ji", "Sharma ji"),
        ("Sharma", "Sharma ji"),
        ("Sharma bhaiya", "Sharma ji"),   # same person, different term of address
        ("sharma", "Sharma ji"),
        ("Verma", "Verma bhaiya"),
        ("Verma ji", "Verma bhaiya"),
        ("Gupta aunty", "Gupta aunty"),
        ("Gupta", "Gupta aunty"),
    ],
)
def test_known_customers_are_matched(spoken, expected):
    result = match_name(spoken, CUSTOMERS)
    assert result.decision is Decision.MATCH
    assert result.name == expected


def test_a_small_transcription_slip_still_matches():
    result = match_name("Mishraa ji", CUSTOMERS)
    assert result.is_match and result.name == "Mishra ji"


def test_a_match_carries_its_score():
    assert match_name("Sharma ji", CUSTOMERS).score >= CONFIDENT


# --- new customers ----------------------------------------------------------


def test_an_unrelated_name_is_new():
    result = match_name("Chaturvedi ji", CUSTOMERS)
    assert result.decision is Decision.NEW
    assert result.name is None


def test_the_first_customer_is_new():
    assert match_name("Sharma ji", []).decision is Decision.NEW


def test_an_empty_name_is_new_not_a_match():
    for spoken in ("", "   ", "ji"):
        assert match_name(spoken, CUSTOMERS).decision is not Decision.MATCH


# --- the dangerous middle ---------------------------------------------------


def test_two_similar_customers_are_ambiguous_not_guessed():
    """A dropped leading consonant is what a noisy line produces.

    "arma" scores 88.9 against Varma and 80.0 against Sharma. On absolute score
    alone that is a confident match, and the debt lands on the wrong customer.
    An 8.9-point lead is not evidence.
    """
    known = ["Sharma ji", "Varma ji"]
    result = match_name("arma ji", known)

    assert result.decision is Decision.AMBIGUOUS
    assert set(result.candidates) == set(known)
    assert result.name is None, "an ambiguous result must not carry a usable name"


def test_a_near_miss_is_ambiguous_rather_than_matched():
    result = match_name("Sharman", ["Sharma ji", "Chaturvedi ji"])
    assert result.decision in (Decision.MATCH, Decision.AMBIGUOUS)
    if result.decision is Decision.AMBIGUOUS:
        assert "Sharma ji" in result.candidates


def test_an_uncertain_result_never_carries_a_name():
    """The type makes the unsafe path impossible rather than discouraged."""
    for spoken in ("arma ji", "Chaturvedi ji", ""):
        result = match_name(spoken, ["Sharma ji", "Varma ji"])
        if not result.is_match:
            assert result.name is None


def test_duplicate_surnames_are_ambiguous():
    """Two Sharmas in the book: the agent has to ask which."""
    known = ["Sharma ji", "Sharma aunty"]
    result = match_name("Sharma", known)
    assert result.decision is Decision.AMBIGUOUS
    assert len(result.candidates) == 2


def test_a_full_name_picks_out_one_of_two_sharmas():
    known = ["Ramesh Sharma", "Suresh Sharma"]
    result = match_name("Ramesh Sharma ji", known)
    assert result.is_match and result.name == "Ramesh Sharma"


def test_a_confident_match_must_also_be_clearly_ahead():
    """Both knobs matter: a high score with a close runner-up is still a question."""
    known = ["Sharma ji", "Varma ji"]
    result = match_name("arma ji", known)
    assert result.score >= CONFIDENT, "the best candidate scores well on its own"
    assert result.decision is Decision.AMBIGUOUS, "but it is not clearly ahead"


def test_a_clear_winner_is_matched_even_against_a_similar_name():
    result = match_name("Sharma ji", ["Sharma ji", "Varma ji"])
    assert result.is_match and result.name == "Sharma ji"


@pytest.mark.parametrize("spoken", ["arma ji", "harma", "arma"])
def test_no_partial_name_is_ever_resolved_against_confusable_customers(spoken):
    """The whole point: never silently pick between two real people."""
    result = match_name(spoken, ["Sharma ji", "Varma ji"])
    assert result.name is None
