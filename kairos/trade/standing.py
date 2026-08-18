"""Which proposals a reviewer is asked to decide, and which are only leads.

The reading of an agreement produces two different kinds of thing under one
name. Most proposals carry values: a percentage, a date, a basis, a cap. Some
carry the SHAPE of a term and nothing in it — a discount ladder whose only rung
is zero percent at a threshold of zero, a measurement source whose every field
is unknown, a commitment with no amount. The second kind is not a proposal. It
is the reader saying "something of this sort may live in this clause", which is
a lead worth keeping and a terrible thing to put in a list a person has to
approve line by line.

Mixing them is what makes reviewing an agreement heavy. Measured on the shipped
corpus of eight agreements: of 228 proposals, 16 carry no values, and putting
them in a separate list raises the share of the main list that is correct from
66.7% to 71.7% while moving nothing correct out of it.

WHY EMPTINESS AND NOT CONFIDENCE. The obvious rule is to trust the model's own
confidence, and it is a trap. Measured on the same corpus with the shipped model
routing, withholding every low-confidence proposal would have withheld 68 and
buried 42 CORRECT terms with them, because a smaller model rates itself low on
things it got right. The heavier routing was far better calibrated — it would
have buried only 2 — but a rule whose safety depends on which model answered is
not a rule, it is a coincidence. Emptiness is a property of the ANSWER. It reads
the same whoever wrote it, and it is the property that actually matters to the
person reading: a term with no values in it cannot be checked against the
document, because there is nothing to check.

The two correct terms this does move on the shipped routing both arrived with
``params == {}`` against a truth full of content. Calling those interpretations
is not a loss; it is the honest description of what the reader produced.
"""

from __future__ import annotations

from typing import Any

CONFIDENT = "confident"
INTERPRETIVE = "interpretive"

# At or above this share of empty leaves, a proposal is carrying no answer.
# Measured across 0.4 / 0.6 / 1.0 on both routings: 0.6 is where the split stops
# gaining and starts moving correct terms. It is a threshold, so it is stated
# once, here, and read everywhere.
EMPTY_SHARE = 0.6

# What "no answer" looks like in a parameter leaf. ``unstated`` and ``<UNKNOWN>``
# are the schemas' own words for a field the document did not yield, so they are
# absences the reader declared rather than values it found.
#
# A bare zero counts as an absence too, and that was tested rather than assumed.
# The worry is obvious — a real zero is an answer — so both readings were
# measured on both routings. Counting zeros as answers withholds 10 and lifts
# the main list to 69.7%; counting them as absences withholds 16 and lifts it to
# 71.7%, and BOTH move exactly zero correct terms out. The worry does not
# materialise because a zero in these schemas is how a model fills a required
# number it never found — a discount ladder whose single rung is 0% at a
# threshold of 0 is the shape of the failure — while the genuine zeros in this
# domain (a count of nothing) live on the delivery ledger, not in a contract
# term. If a term ever does carry a meaningful zero, this list is where to say
# so, and the measurement above is the check to re-run.
EMPTY_LEAVES = ("", None, "<UNKNOWN>", "unstated", 0, 0.0, False)


def emptiness(params: Any) -> float:
    """The share of this proposal's leaves that carry no answer, 0.0 to 1.0."""
    leaves = 0
    empty = 0

    def walk(value: Any) -> None:
        nonlocal leaves, empty
        if isinstance(value, dict):
            if not value:
                leaves += 1
                empty += 1
                return
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            if not value:
                leaves += 1
                empty += 1
                return
            for item in value:
                walk(item)
        else:
            leaves += 1
            if value in EMPTY_LEAVES:
                empty += 1

    walk(params if params is not None else {})
    return 1.0 if leaves == 0 else empty / leaves


def standing(instance: Any) -> str:
    """``confident`` when the proposal carries an answer, ``interpretive`` when not.

    Accepts either a TermInstance or the dict shape the store publishes, so the
    engine and the API decide this the same way and cannot drift apart.
    """
    params = instance.get("params") if isinstance(instance, dict) else getattr(instance, "params", None)
    return INTERPRETIVE if emptiness(params) >= EMPTY_SHARE else CONFIDENT


def is_interpretive(instance: Any) -> bool:
    return standing(instance) == INTERPRETIVE


def reason(instance: Any, locale: str = "he") -> str:
    """Why this proposal is only a lead, in the reader's language."""
    if standing(instance) == CONFIDENT:
        return ""
    share = emptiness(instance.get("params") if isinstance(instance, dict)
                      else getattr(instance, "params", None))
    if share >= 1.0:
        return ("הקורא זיהה בסעיף הזה תנאי מסוג כזה ולא חילץ ממנו ולו ערך אחד."
                if locale == "he" else
                "The reader recognised a term of this kind in the clause and extracted no value from it.")
    return ("רוב השדות בתנאי הזה נותרו ריקים, ולכן אין מה להשוות מולו למסמך."
            if locale == "he" else
            "Most of this term's fields came back empty, so there is nothing in it to check against the document.")
