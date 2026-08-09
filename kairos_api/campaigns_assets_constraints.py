"""The two constraints a creative carries, and the measurement behind them.

The trade note names both in one breath. A campaign carries many creatives, and
a common structure is a ten second spot with a six second closer that must air in
the SAME break separated by exactly one or two other advertisements. Each
creative also carries a validity window: until when it may be scheduled. Both are
hard constraints and both were missing from this product entirely.

**Why they are modelled in two different places, and it is not an accident.**

A validity window is a property of ONE tape. It has one subject, one value and no
second party, so it is a column on the asset in ``data/campaign_assets.csv`` and
it is read here.

A pair is a RELATION between two tapes, and it carries its own parameters: which
creative leads, which closes, and how many other advertisements may stand between
them. It is also an AGREEMENT rather than an observation, and the asset ledger is
an observation ledger, rebuilt from the traffic log with ``identity_source`` set
to say so. A pair written as a column would be erased the next time that ledger
is rebuilt, would have to be written on both rows, and could then disagree with
itself. So a pair is a rule row in ``data/frequency_rules.csv`` under the
``pair_separation`` limit type, beside the other placement rules, where the
enforcement already reads.

**What this module does NOT do.** It authors nothing. The shipped rule file holds
no pair, because a pair is a commercial agreement this product has not been told
about, and inventing one would be inventing a constraint an advertiser never
bought. What it offers instead is :func:`candidate_pairs`: the pairs the traffic
file's own shape suggests, named as candidates, so an operator authors from
evidence rather than from memory.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from kairos.optimize._frequency_rules import (
    BETWEEN,
    PAIR_SEPARATION,
    FrequencyRule,
    load_frequency_rules,
    pair_rules,
)
from kairos.optimize._pair_placement import VIOLATED, PairVerdict, pair_counts, pair_verdicts
from kairos.optimize.frequency import SpotView

# The trade's own example is a ten second spot with a six second closer, so the
# closer is the shorter tape and ten seconds is the longest thing the note still
# calls a spot. Both figures are the note's, not this module's, and both are
# named here so a reader can see exactly what "looks like a pair" was measured on.
CLOSER_MAX_SECONDS = 10.0
LEAD_MIN_SECONDS = 10.0

# The trade states the gap as "exactly one or two other advertisements". A
# candidate is judged against that, because it is the constraint the note gives.
TRADE_MIN_BETWEEN = 1.0
TRADE_MAX_BETWEEN = 2.0

# A token shorter than three characters carries no evidence of a shared copy, and
# a bare number is a length or a year rather than a name.
_TOKEN_SPLIT = re.compile(r"[\s\-,/]+")
_TOKEN_STRIP = "\"'.״″׳′()[]"

NO_WINDOW_STATE = "unknown"
WITHIN = "within"
EXPIRED = "expired"
NOT_YET = "not_yet"


def _text(row: Any, column: str) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default="": default
    return str(getter(column, "") or "").strip()


def validity_window(row: Any) -> dict[str, Any]:
    """Until when this creative may be scheduled, or the absence of an answer.

    Two states and never a silent third. A row that declares neither end of the
    window reports unknown with the path that would supply it, because a creative
    with no recorded expiry is not a creative that never expires: it is a creative
    nobody has told this product about.
    """
    from kairos_api.campaigns_assets import UNKNOWN_PATHS

    starts, ends = _text(row, "valid_from"), _text(row, "valid_until")
    if not starts and not ends:
        path_en, path_he = UNKNOWN_PATHS["validity"]
        return {
            "state": NO_WINDOW_STATE,
            "valid_from": None,
            "valid_until": None,
            "source": "",
            "path_en": path_en,
            "path_he": path_he,
        }
    return {
        "state": "real",
        "valid_from": starts or None,
        "valid_until": ends or None,
        "source": _text(row, "validity_source"),
        "path_en": "",
        "path_he": "",
    }


def schedulable_on(window: dict[str, Any], day: str) -> dict[str, Any]:
    """Whether this creative may be scheduled on one broadcast day.

    ISO dates compare correctly as strings, which is why the window is stored in
    that form. An unknown window answers unknown: a tape whose expiry nobody
    recorded must not be reported as clear to schedule, and must not be reported
    as expired either.
    """
    wanted = str(day or "").strip()
    if window.get("state") != "real" or not wanted:
        return {
            "state": NO_WINDOW_STATE,
            "reason": "No validity window is recorded for this creative, so whether it may be scheduled on this day is not known.",
            "reason_he": "לא רשום חלון תוקף לתשדיר הזה, ולכן לא ידוע אם מותר לתזמן אותו ביום הזה.",
        }
    starts, ends = window.get("valid_from"), window.get("valid_until")
    if ends and wanted > str(ends):
        return {
            "state": EXPIRED,
            "reason": f"This creative may be scheduled until {ends}, and {wanted} is after that.",
            "reason_he": f"מותר לתזמן את התשדיר הזה עד ⁦{ends}⁩, ו-⁦{wanted}⁩ מאוחר מכך.",
        }
    if starts and wanted < str(starts):
        return {
            "state": NOT_YET,
            "reason": f"This creative may be scheduled from {starts}, and {wanted} is before that.",
            "reason_he": f"מותר לתזמן את התשדיר הזה מ-⁦{starts}⁩, ו-⁦{wanted}⁩ מוקדם מכך.",
        }
    return {
        "state": WITHIN,
        "reason": "This day is inside the creative's validity window.",
        "reason_he": "היום הזה נמצא בתוך חלון התוקף של התשדיר.",
    }


def authored_pairs() -> list[FrequencyRule]:
    """Every enabled pair rule on disk, or none when the file holds no pair."""
    try:
        return pair_rules(load_frequency_rules().rules)
    except Exception:  # noqa: BLE001 - an unreadable rule file is no pairs, not a crash
        return []


def pairs_for_campaign(campaign: str) -> list[FrequencyRule]:
    """The authored pairs belonging to one campaign, by the campaign's own name."""
    wanted = str(campaign or "").strip()
    return [rule for rule in authored_pairs() if rule.campaign == wanted]


def name_tokens(name: Any) -> set[str]:
    """The words of a version name that could evidence shared copy."""
    tokens = set()
    for part in _TOKEN_SPLIT.split(str(name or "")):
        cleaned = part.strip(_TOKEN_STRIP)
        if len(cleaned) >= 3 and not cleaned.isdigit():
            tokens.add(cleaned)
    return tokens


def _creatives(frame: Any) -> dict[str, list[tuple[str, str, float]]]:
    """One entry per campaign: its distinct creatives as (house, name, seconds)."""
    seen: dict[tuple[str, str], tuple[str, float]] = {}
    for row in frame.itertuples(index=False):
        house = str(getattr(row, "house_number", "") or "").strip()
        campaign = str(getattr(row, "campaign", "") or "").strip()
        if not house or not campaign:
            continue
        seconds = getattr(row, "duration_sec", None)
        seen.setdefault((campaign, house), (str(getattr(row, "creative", "") or "").strip(), float(seconds or 0.0)))
    grouped: dict[str, list[tuple[str, str, float]]] = {}
    for (campaign, house), (name, seconds) in seen.items():
        grouped.setdefault(campaign, []).append((house, name, seconds))
    return grouped


def candidate_pairs(frame: Any, *, require_shared_name: bool = True) -> list[dict[str, Any]]:
    """The pairs a traffic file's own shape suggests, as candidates and not rules.

    A candidate is two distinct creatives of one campaign where one is at least
    ten seconds and the other is at most ten and shorter, which is the trade's own
    description of a spot and its closer. With ``require_shared_name`` the two
    version names must also share a word, which is the second half of "by duration
    and version name" and is what separates a lead with its own closer from two
    unrelated tapes that happen to differ in length.

    Each closer is offered ONE lead, the one sharing the most of its name, because
    a campaign running four creatives would otherwise produce a candidate for
    every combination and none of them would be the pair anybody bought.
    """
    candidates: list[dict[str, Any]] = []
    for campaign, creatives in sorted(_creatives(frame).items()):
        for house, name, seconds in sorted(creatives):
            if seconds > CLOSER_MAX_SECONDS or seconds <= 0:
                continue
            best: Optional[tuple[int, str, str, float]] = None
            for lead_house, lead_name, lead_seconds in sorted(creatives):
                if lead_house == house or lead_seconds <= seconds or lead_seconds < LEAD_MIN_SECONDS:
                    continue
                shared = len(name_tokens(lead_name) & name_tokens(name))
                if require_shared_name and not shared:
                    continue
                ranked = (-shared, lead_house, lead_name, lead_seconds)
                if best is None or ranked < best:
                    best = ranked
            if best is None:
                continue
            candidates.append({
                "campaign": campaign,
                "lead_house_number": best[1],
                "lead_version_name": best[2],
                "lead_seconds": best[3],
                "closer_house_number": house,
                "closer_version_name": name,
                "closer_seconds": seconds,
                "shared_name_words": sorted(name_tokens(best[2]) & name_tokens(name)),
            })
    return candidates


def implied_rule(candidate: dict[str, Any]) -> FrequencyRule:
    """The rule a candidate WOULD be, at the gap the trade note itself states.

    This is how a candidate is measured without being authored. It is never
    written to the rule file, and it is never read from it.
    """
    return FrequencyRule(
        rule_id=f"CANDIDATE_{candidate['lead_house_number']}_{candidate['closer_house_number']}",
        limit_type=PAIR_SEPARATION,
        scope="campaign",
        campaign=candidate["campaign"],
        pair_lead=candidate["lead_house_number"],
        pair_closer=candidate["closer_house_number"],
        value=TRADE_MIN_BETWEEN,
        value_max=TRADE_MAX_BETWEEN,
        unit=BETWEEN,
    )


def spot_views(frame: Any) -> list[SpotView]:
    """A daily traffic frame as the ordered spot views the pair check reads.

    Ordered by break and then by the time each spot actually aired, because the
    count of other advertisements between two spots is a fact about the broadcast
    order and the position column is a fact about the contract.
    """
    ordered = frame.sort_values(["break_start", "spot_time"])
    views = []
    for index, row in enumerate(ordered.itertuples(index=False)):
        views.append(SpotView(
            key=index,
            advertiser=str(getattr(row, "advertiser", "") or "").strip(),
            campaign=str(getattr(row, "campaign", "") or "").strip(),
            ad=str(getattr(row, "creative", "") or "").strip(),
            break_id=str(getattr(row, "break_start", "") or "").strip(),
            position=None,
            minute=None,
            house_number=str(getattr(row, "house_number", "") or "").strip(),
        ))
    return views


def measure_file(frame: Any, *, require_shared_name: bool = True) -> dict[str, Any]:
    """How many candidate pairs a traffic file holds, and how many air correctly.

    The whole point of the piece, stated as one number that can be re-measured:
    of the pairs this file looks like it carries, how many actually air in one
    break with the one or two other advertisements between them that the trade
    says they must. A candidate is counted once, at its best break.
    """
    candidates = candidate_pairs(frame, require_shared_name=require_shared_name)
    views = spot_views(frame)
    rolled: list[PairVerdict] = []
    for candidate in candidates:
        verdicts = pair_verdicts(views, [implied_rule(candidate)])
        worst = next((item for item in verdicts if item.is_violation), None)
        rolled.append(worst or verdicts[0])
    return {
        "campaigns_in_file": int(frame["campaign"].nunique()),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "campaigns_with_a_candidate": len({item["campaign"] for item in candidates}),
        "verdicts": rolled,
        "states": pair_counts(rolled),
    }


def pod_spot_views(spots: list[dict[str, Any]], break_id: str) -> list[SpotView]:
    """One pod's spots as spot views, in the order the pod is currently shown in.

    The pod's order is the proposed ordering: the file's own, or the one an
    operator saved over it. That is exactly what a pair verdict is a statement
    about, so a reordering that breaks a pair is caught by the same check that
    caught the file breaking it.
    """
    return [
        SpotView(
            key=item.get("spot_key"),
            advertiser=(item.get("advertiser") or {}).get("value") or "",
            campaign=(item.get("campaign") or {}).get("value") or "",
            ad=(item.get("creative") or {}).get("value") or "",
            break_id=str(break_id or ""),
            position=None,
            minute=None,
            house_number=(item.get("house_number") or {}).get("value") or "",
        )
        for item in spots
    ]


def pod_pairs(spots: list[dict[str, Any]], break_id: str) -> list[PairVerdict]:
    """Every authored pair judged against this pod's current order."""
    rules = authored_pairs()
    if not rules:
        return []
    return pair_verdicts(pod_spot_views(spots, break_id), rules)


def pod_pair_block(spots: list[dict[str, Any]], break_id: str) -> dict[str, Any]:
    """One pod's whole pair answer: the verdicts, the three states, the errors.

    Shaped here rather than in the pod module so that module stays under the
    450-line cap and stays about reading files and answering routes. ``authored``
    is on the block because a pod showing no pair verdict has two possible
    causes, no pair authored anywhere and no pair touching this break, and a
    surface that could not tell them apart would read the first as the second.
    """
    verdicts = pod_pairs(spots, break_id)
    return {
        "verdicts": [vars(verdict) for verdict in verdicts],
        "states": pair_counts(verdicts),
        "authored": len(authored_pairs()),
        "errors": pod_pair_errors(verdicts),
    }


def pod_pair_errors(verdicts: list[PairVerdict]) -> list[dict[str, Any]]:
    """The violated pairs as verification entries, in the list's own shape.

    Only a violation enters the verification list. An unknown pair is not an
    error: nothing was placed wrongly, and a red mark against a creative that is
    simply not in this file would be a fault this pod does not carry. The unknown
    verdicts still travel on the pod, counted, so they are visible rather than
    quietly dropped.
    """
    errors = []
    for verdict in verdicts:
        if verdict.state != VIOLATED:
            continue
        errors.append({
            "kind": "pair_separation",
            "spot_key": verdict.closer_key or verdict.lead_key or "",
            "detail": verdict.reason,
            "detail_he": verdict.reason_he,
        })
    return errors
