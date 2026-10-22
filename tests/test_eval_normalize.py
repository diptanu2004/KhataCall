"""WER normalisation: spelling and number form must not count as recognition errors."""

import pytest

from eval.normalize import devanagari_fraction, normalise_text, raw_text


def test_raw_only_lowercases_and_strips_punctuation():
    assert raw_text("Sharma ji, 150 udhaar likho!") == "sharma ji 150 udhaar likho"


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("Sharma ji ka dedh sau udhaar likho", "sharma ji ka 150 udhar likho"),
        ("do hazaar paanch sau rupaye", "2500 rupaye"),
        ("2 hazaar 500 rupaye", "2500 rupaye"),
        ("paanch kilo cheeni", "5 kilo chini"),
        ("Yadav ji ka saadhe teen sau likho", "yadav ji ka 350 likho"),
        ("paune do hazaar", "1750"),
        ("sava sau diye", "125 diye"),
    ],
)
def test_spoken_and_digit_forms_normalise_alike(spoken, written):
    assert normalise_text(spoken) == normalise_text(written)


def test_spelling_variants_fold_together():
    assert normalise_text("udhaar hazaar") == normalise_text("udhar hajar")


def test_a_lone_do_is_a_verb_not_two():
    assert normalise_text("likh do") == "likh do"
    assert normalise_text("do sau likh do") == "200 likh do"


def test_saat_and_saath_stay_different():
    assert normalise_text("saat") != normalise_text("saath")


def test_a_real_error_still_counts():
    assert normalise_text("dedh sau") != normalise_text("dhai sau")


def test_devanagari_is_measured_not_hidden():
    assert devanagari_fraction("sharma ji") == 0
    assert devanagari_fraction("शर्मा") == 1
