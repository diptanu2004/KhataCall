"""Matching a spoken customer name to one already in the ledger.

Two jobs, and the second is where the risk lives.

**Stripping honorifics.** "Sharma ji", "Sharma bhaiya" and "Sharma" are one
person. The ledger stores the name as first spoken, so lookups compare the
stripped, folded form.

**Deciding whether a near-match is the same person.** Prisma's romanized output
is not stable (D-012), so exact comparison would fail constantly. Fuzzy
matching fixes that and introduces a worse failure: silently writing a debt
against the wrong customer. "Verma" and "Sharma" are one edit apart in sound
and three in spelling.

So matching returns a decision, not a name:

    MATCH      confident, use it
    AMBIGUOUS  two customers score too close together, ask which
    NEW        nothing close, ask before creating

A caller cannot accidentally use an uncertain result, because the uncertain
cases do not carry one.
"""

from dataclasses import dataclass
from enum import StrEnum

from rapidfuzz import fuzz, process

from app.nlu.numbers import normalise

__all__ = ["Decision", "NameMatch", "match_name", "normalise_name", "strip_honorifics"]

# Terms of address, not part of the name. Order matters only for readability.
HONORIFICS = frozenset(
    normalise(w)
    for w in (
        "ji jee saheb sahab sahib shri sri smt srimati madam sir "
        "bhai bhaiya bhaiyya bhaisahab bhayya "
        "didi behen bahen behenji "
        "aunty auntie aunti uncle "
        "chacha chachi mama mami tau taai bua "
        "dada dadi nana nani "
        "miss mrs mr"
    ).split()
)

# Two knobs, doing different jobs. CONFIDENT asks "is the best candidate good enough
# on its own"; SEPARATION asks "is it clearly better than the next one".
#
# Separation is the one that protects against the dangerous failure. Measured:
# "arma" (a dropped leading consonant, which is what a noisy line produces) scores
# 88.9 against "Varma" and 80.0 against "Sharma". On absolute score alone it is a
# confident match for Varma, and the debt lands on the wrong customer. An 8.9-point
# lead is not evidence, so anything under 12 goes back to the shop owner as a question.
#
# With that guard in place CONFIDENT can sit lower, which lets honest romanization
# variants through: bisht/bist score 88.9 and chauhan/chouhan 85.7, and both are the
# same person. Provisional numbers - M4 tunes them against the eval.
CONFIDENT = 85.0
PLAUSIBLE = 72.0
SEPARATION = 12.0


class Decision(StrEnum):
    MATCH = "match"
    AMBIGUOUS = "ambiguous"
    NEW = "new"


@dataclass(frozen=True)
class NameMatch:
    decision: Decision
    name: str | None = None  # set only when the decision is MATCH
    score: float = 0.0
    candidates: tuple[str, ...] = ()  # the names to ask about when AMBIGUOUS

    @property
    def is_match(self) -> bool:
        return self.decision is Decision.MATCH


def strip_honorifics(spoken: str) -> str:
    """"Sharma ji" -> "Sharma". Keeps the original spelling of what remains."""
    words = [w for w in spoken.split() if w.strip()]
    kept = [w for w in words if normalise(w) not in HONORIFICS]
    # An utterance of nothing but honorifics ("ji") leaves the original alone;
    # there is no name in it to find, and the caller decides what that means.
    return " ".join(kept) if kept else spoken.strip()


def normalise_name(spoken: str) -> str:
    """The comparison key: honorifics removed, spelling folded, words sorted.

    Sorting the words means "Sharma ji Ramesh" and "Ramesh Sharma" agree, which
    matters because word order in speech is looser than in a database.
    """
    stripped = strip_honorifics(spoken)
    words = sorted(filter(None, (normalise(w) for w in stripped.split())))
    return " ".join(words)


def match_name(spoken: str, known: list[str]) -> NameMatch:
    """Resolve a spoken name against the customers already in the ledger."""
    key = normalise_name(spoken)
    if not key:
        return NameMatch(Decision.NEW)
    if not known:
        return NameMatch(Decision.NEW)

    keys = {name: normalise_name(name) for name in known}
    # token_sort_ratio so word order does not matter, on keys that are already sorted;
    # it also handles a dropped or added word better than a plain ratio.
    scored = process.extract(
        key, keys, scorer=fuzz.token_sort_ratio, limit=len(keys), processor=None
    )
    # rapidfuzz returns (choice_value, score, choice_key); we want the original name.
    ranked = sorted(((score, name) for _, score, name in scored), reverse=True)
    best_score, best_name = ranked[0]

    if best_score < PLAUSIBLE:
        return NameMatch(Decision.NEW, score=best_score)

    runners_up = [name for score, name in ranked[1:] if best_score - score < SEPARATION]
    if runners_up:
        # Two customers are equally close. Picking one would be a coin flip against
        # somebody's money, so ask which was meant.
        return NameMatch(
            Decision.AMBIGUOUS,
            score=best_score,
            candidates=(best_name, *runners_up),
        )
    if best_score >= CONFIDENT:
        return NameMatch(Decision.MATCH, name=best_name, score=best_score)
    # Close enough to be worth naming, not close enough to assume.
    return NameMatch(Decision.AMBIGUOUS, score=best_score, candidates=(best_name,))
