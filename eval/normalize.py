"""Make a reference and a Prisma transcript comparable before counting errors.

Raw WER would charge Prisma for things that are not recognition errors:
"udhaar" vs "udhar" (romanized Hindi has no fixed spelling, D-012), "Sharma
ji," vs "sharma ji", "dedh sau" vs "150". Each of those reads the same to the
agent, because the agent parses through the same folding. So the eval reports
two numbers:

    raw         lowercase and punctuation only
    normalised  also spelling-folded (nlu.numbers.normalise) and every spoken
                number run replaced by its value in digits

The normalised figure is the one that predicts task accuracy; the gap between
the two is how much of Prisma's "error" is spelling.

Limits, stated so nobody over-reads the numbers:
- Spelling folding is the agent's own (doubled vowels, w/v, z/j, trailing h
  after a vowel). Variants outside it ("mein"/"me", "ke"/"ki") still count as
  errors, and some of those are real ones.
- Number runs are found greedily: the longest run of tokens that
  `parse_number` accepts. A transcript that splits a number ("dedh" ... "sau"
  with a word between) yields two values, and the error count shows it.
- Devanagari is not transliterated. Prisma with en-IN has returned only Latin
  script so far (D-012); `devanagari_fraction` is reported so a change would
  show up instead of silently inflating WER.
"""

import re
from fractions import Fraction

from app.nlu.numbers import normalise, parse_number

__all__ = ["devanagari_fraction", "normalise_text", "raw_text"]

_WORD = re.compile(r"[a-z]+|\d+|[ऀ-ॿ]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Number words that are more often something else when they stand alone:
# "likh do" is "write it down", not "write two".
_AMBIGUOUS_ALONE = frozenset({"do"})


def raw_text(text: str) -> str:
    """Lowercase, punctuation stripped, whitespace collapsed."""
    return " ".join(_WORD.findall(text.lower()))


def _is_number_word(token: str) -> bool:
    # Fraction words ("sadhe", "paune", "sava") are not numbers alone, only in front of one.
    # parse_number skips fillers, so "ka sau" is 100 too: the word must change the value.
    if token.isdigit() or parse_number(token, spelling_fixes=False) is not None:
        return True
    value = parse_number(f"{token} sau", spelling_fixes=False)
    return value is not None and value != 100


def _digits(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{float(value):g}"


def normalise_text(text: str) -> str:
    """Spelling-folded, with each spoken number run as one digit token."""
    tokens = raw_text(text).split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if _is_number_word(tokens[i]):
            # Longest run starting here that still parses as one number.
            j = i + 1
            while j < len(tokens) and _is_number_word(tokens[j]):
                j += 1
            for end in range(j, i, -1):
                if end == i + 1 and tokens[i] in _AMBIGUOUS_ALONE:
                    continue
                # Pinned to the pre-M4 parser: WER measures Prisma, and must not
                # move when the agent's spelling fixes change (D-042).
                value = parse_number(" ".join(tokens[i:end]), spelling_fixes=False)
                if value is not None:
                    out.append(_digits(value))
                    i = end
                    break
            else:
                out.append(tokens[i])
                i += 1
            continue
        folded = tokens[i] if _DEVANAGARI.search(tokens[i]) else normalise(tokens[i])
        if folded:
            out.append(folded)
        i += 1
    return " ".join(out)


def devanagari_fraction(text: str) -> float:
    letters = [ch for ch in text if not ch.isspace()]
    return sum(bool(_DEVANAGARI.match(ch)) for ch in letters) / len(letters) if letters else 0.0
