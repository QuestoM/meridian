"""Whether a proposed ordering honours a campaign's paired creatives.

The trade names the constraint plainly: a campaign carries many creatives, and a
common structure is a ten second spot with a six second closer that must air in
the SAME break separated by exactly one or two other advertisements. That is a
hard placement constraint and the optimiser has to honour it.

**Why this judges rather than drops.** Every other rule in
:mod:`kairos.optimize.frequency` answers by removing a spot. A pair cannot be
enforced that way: dropping the closer of a broken pair leaves the campaign with
a lead and no closer, which is worse than the fault it was meant to fix, and
dropping both throws away money the advertiser has already bought. So this
module returns a VERDICT on an ordering, and the surfaces that already carry a
verification list name the fault against the break it happens in.

**The ordering is the input.** ``others_between`` is counted over the order the
caller passes, inside one ``break_id``. That is deliberate: the position column
of a traffic file is the position a campaign CONTRACTED for (1 to 5, plus 99 for
Last and 0 for unrequested), not the order the spots actually air in, so
counting other advertisements from it would count a contract rather than a
broadcast. The caller passes the spots in the order they air, and this counts.

**Three states, never two.** A pair whose second creative is not in the traffic
file at all is UNKNOWN. It is not a violation, because nothing was placed
wrongly, and it is not a pass, because nothing was checked. The same holds when
the spots carry no break identity to judge co-location by. Only a pair whose two
creatives are both present and demonstrably placed wrong is VIOLATED.

**Identity is the house number, and it says when it was not.** A creative's real
filing identity is its house number, which the traffic log carries. A version
name is not an identity: on the shipped file the single house number HGB007510
appears under two different version names. So a pair resolves on house numbers,
and falls back to the version name ONLY when no spot in scope carries a house
number at all. Every verdict records which of the two it matched on, so a reader
never has to assume the stronger one was available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from kairos.optimize._frequency_rules import FrequencyRule

SATISFIED = "satisfied"
VIOLATED = "violated"
UNKNOWN = "unknown"

BY_HOUSE_NUMBER = "house_number"
BY_VERSION_NAME = "version_name"

# The two Hebrew nouns this module needs already exist verbatim on shipped
# surfaces: תשדיר for one spot (kairos_api/campaigns_read_money_reasons.py) and
# ברייק for the break (kairos_api/break_api_pod_math.py, and the product
# vocabulary note that forbids הפסקות for this object). The bidi isolates around
# a figure or a house number are the same ones the money reasons module uses, so
# a Latin house number cannot reorder the Hebrew sentence around it.
_ISOLATE = ("⁦", "⁩")


def _bidi(value: Any) -> str:
    return f"{_ISOLATE[0]}{value}{_ISOLATE[1]}"


def _count(value: float) -> str:
    """A count as a person writes it: 1 rather than 1.0."""
    number = float(value or 0.0)
    return str(int(number)) if number == int(number) else str(round(number, 2))


def _others(count: int) -> tuple[str, str]:
    """A count of intervening advertisements with its noun, in both languages.

    The same singular-and-plural care the money reasons module takes, for the
    same reason: a cap of one is the trade's own common case, so "1 other
    advertisements" and "⁦1⁩ תשדירים" are the sentences most readers would see.
    """
    if count == 1:
        return "1 other advertisement", "תשדיר אחד אחר"
    return f"{count} other advertisements", f"{_bidi(count)} תשדירים אחרים"


@dataclass(frozen=True)
class PairVerdict:
    """One judgement about one authored pair, in one break or across the day.

    ``break_id`` is the break judged, and is empty on the two verdicts that are
    not about a single break: a creative missing from the file entirely, and a
    pair whose two creatives never share a break at all.
    """

    rule_id: str
    advertiser: str
    campaign: str
    lead: str
    closer: str
    state: str
    break_id: str = ""
    others_between: Optional[int] = None
    allowed_min: Optional[float] = None
    allowed_max: Optional[float] = None
    matched_on: str = ""
    lead_key: Any = None
    closer_key: Any = None
    reason: str = ""
    reason_he: str = ""

    @property
    def is_violation(self) -> bool:
        return self.state == VIOLATED


def _in_scope(spot: Any, rule: FrequencyRule) -> bool:
    """True when this spot belongs to the campaign the pair was authored for."""
    if rule.advertiser_id and rule.advertiser_id != spot.advertiser:
        return False
    return rule.campaign == spot.campaign


def _occurrences(scope: list[Any], reference: str) -> tuple[list[Any], str]:
    """Every spot in scope that is this creative, and what identified it.

    Returns an empty list with an empty match kind when the creative is not
    present. The house number wins whenever the scope carries house numbers at
    all, so a file that has them never resolves a pair by a version name.
    """
    by_house = [item for item in scope if getattr(item, "house_number", "") == reference and reference]
    if by_house:
        return by_house, BY_HOUSE_NUMBER
    if any(getattr(item, "house_number", "") for item in scope):
        # The scope files house numbers and none of them is this one, so this
        # creative is genuinely absent rather than merely unidentified.
        return [], ""
    by_name = [item for item in scope if item.ad and item.ad == reference]
    return (by_name, BY_VERSION_NAME) if by_name else ([], "")


def _absent_verdict(rule: FrequencyRule, missing: str, matched_on: str) -> PairVerdict:
    return PairVerdict(
        rule_id=rule.rule_id,
        advertiser=rule.advertiser_id,
        campaign=rule.campaign,
        lead=rule.pair_lead,
        closer=rule.pair_closer,
        state=UNKNOWN,
        matched_on=matched_on,
        allowed_min=rule.value,
        allowed_max=rule.value_max,
        reason=(
            f"The creative {missing} is not in this traffic file, so whether the pair "
            f"{rule.pair_lead} and {rule.pair_closer} aired together cannot be judged."
        ),
        reason_he=(
            f"התשדיר {_bidi(missing)} אינו מופיע בקובץ הטראפיק הזה, ולכן לא ניתן לקבוע אם הצמד "
            f"{_bidi(rule.pair_lead)} ו-{_bidi(rule.pair_closer)} שודר יחד."
        ),
    )


def _never_together(rule: FrequencyRule, matched_on: str) -> PairVerdict:
    return PairVerdict(
        rule_id=rule.rule_id,
        advertiser=rule.advertiser_id,
        campaign=rule.campaign,
        lead=rule.pair_lead,
        closer=rule.pair_closer,
        state=VIOLATED,
        matched_on=matched_on,
        allowed_min=rule.value,
        allowed_max=rule.value_max,
        reason=(
            f"The pair {rule.pair_lead} and {rule.pair_closer} must air in the same break, "
            "and both air here but never in the same one."
        ),
        reason_he=(
            f"הצמד {_bidi(rule.pair_lead)} ו-{_bidi(rule.pair_closer)} חייב לשדר באותו ברייק, "
            "ושניהם משודרים כאן אך לא באותו ברייק."
        ),
    )


def _range_words(rule: FrequencyRule) -> tuple[str, str]:
    """The allowed range as a phrase, with exactly-N kept as one figure."""
    low = _count(rule.value)
    high = _count(rule.value if rule.value_max is None else rule.value_max)
    if low == high:
        return f"exactly {low}", f"בדיוק {_bidi(low)}"
    return f"between {low} and {high}", f"בין {_bidi(low)} ל-{_bidi(high)}"


def _break_verdict(
    rule: FrequencyRule, break_id: str, others: int, matched_on: str,
    lead_key: Any, closer_key: Any,
) -> PairVerdict:
    low = rule.value
    high = rule.value if rule.value_max is None else rule.value_max
    ok = low <= others <= high
    allowed_en, allowed_he = _range_words(rule)
    others_en, others_he = _others(others)
    return PairVerdict(
        rule_id=rule.rule_id,
        advertiser=rule.advertiser_id,
        campaign=rule.campaign,
        lead=rule.pair_lead,
        closer=rule.pair_closer,
        state=SATISFIED if ok else VIOLATED,
        break_id=break_id,
        others_between=others,
        allowed_min=low,
        allowed_max=high,
        matched_on=matched_on,
        lead_key=lead_key,
        closer_key=closer_key,
        reason=(
            f"In break {break_id} the pair {rule.pair_lead} and {rule.pair_closer} has "
            f"{others_en} between them, and the rule allows {allowed_en}."
        ),
        reason_he=(
            f"בברייק {_bidi(break_id)} יש בין הצמד {_bidi(rule.pair_lead)} ו-{_bidi(rule.pair_closer)} "
            f"{others_he}, והכלל מתיר {allowed_he}."
        ),
    )


def _best_in_break(
    rule: FrequencyRule, leads: list[int], closers: list[int]
) -> tuple[int, int, int]:
    """The occurrence pair in one break that comes closest to the allowed range.

    A creative can air twice in one break, which the shipped file really does.
    Judging the pair on its best occurrence is the honest reading of a constraint
    the trade states as a thing the campaign must GET, not a thing every spot must
    independently satisfy. Returns (others_between, lead index, closer index).
    """
    low = rule.value
    high = rule.value if rule.value_max is None else rule.value_max
    best: Optional[tuple[float, int, int, int]] = None
    for lead in leads:
        for closer in closers:
            others = abs(closer - lead) - 1
            distance = 0.0 if low <= others <= high else min(abs(others - low), abs(others - high))
            candidate = (distance, others, lead, closer)
            if best is None or candidate < best:
                best = candidate
    assert best is not None  # a caller only reaches here with both lists non-empty
    return best[1], best[2], best[3]


def pair_verdicts(spots: list[Any], rules: list[FrequencyRule]) -> list[PairVerdict]:
    """Judge every authored pair against the order these spots are given in.

    One verdict per break the pair shares, so a surface can point at the break
    rather than at the campaign. A creative absent from the file yields one
    UNKNOWN verdict; a pair present on both sides that never shares a break
    yields one VIOLATED verdict. With no authored pair this returns an empty
    list, which is the identity case every other rule here also has.
    """
    verdicts: list[PairVerdict] = []
    for rule in rules:
        scope = [item for item in spots if _in_scope(item, rule)]
        leads, lead_kind = _occurrences(scope, rule.pair_lead)
        closers, closer_kind = _occurrences(scope, rule.pair_closer)
        matched_on = lead_kind or closer_kind
        if not leads:
            verdicts.append(_absent_verdict(rule, rule.pair_lead, matched_on))
            continue
        if not closers:
            verdicts.append(_absent_verdict(rule, rule.pair_closer, matched_on))
            continue
        order = {id(item): index for index, item in enumerate(spots)}
        shared = sorted(
            {item.break_id for item in leads if item.break_id}
            & {item.break_id for item in closers if item.break_id}
        )
        if not shared:
            verdicts.append(_never_together(rule, matched_on))
            continue
        for break_id in shared:
            lead_at = [order[id(item)] for item in leads if item.break_id == break_id]
            closer_at = [order[id(item)] for item in closers if item.break_id == break_id]
            others, lead_index, closer_index = _best_in_break(rule, lead_at, closer_at)
            verdicts.append(_break_verdict(
                rule, break_id, others, matched_on,
                spots[lead_index].key, spots[closer_index].key,
            ))
    return verdicts


def pair_counts(verdicts: list[PairVerdict]) -> dict[str, int]:
    """The three states counted, so no surface has to derive them twice."""
    return {
        SATISFIED: sum(1 for item in verdicts if item.state == SATISFIED),
        VIOLATED: sum(1 for item in verdicts if item.state == VIOLATED),
        UNKNOWN: sum(1 for item in verdicts if item.state == UNKNOWN),
    }
