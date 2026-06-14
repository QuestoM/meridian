"""Advertiser agreements: commercial commitments the optimizer must respect.

Some breaks are not purely a revenue-versus-retention trade. The channel sells
parts of its inventory under standing commercial agreements, and those carry
obligations the greedy optimizer does not know about on its own:

  * FIX agreements are fixed-price sponsorships. The advertiser pays an agreed
    sum and the break must air regardless of the rating it delivers, so its
    revenue does not scale with TVR.
  * CPP agreements are priced per rating point, the channel's normal model, and
    behave like ordinary inventory.
  * A must-air commitment promises an advertiser a minimum number of breaks in a
    given daypart (for example, three breaks during prime time). The schedule
    breaches the agreement if it places fewer.
  * An exclusivity commitment promises that no competing advertiser shares the
    same daypart. It is recorded here for completeness; enforcing it needs
    per-break advertiser attribution the optimizer does not yet carry, so it is
    surfaced as a recorded constraint rather than a checked one.

This module loads those agreements from ``data/reference/AdvertiserAgreements.csv``
and checks an :class:`~kairos.optimize.optimizer.OptimizationResult` against the
must-air commitments it can verify. It follows two honesty rules:

  * An empty agreements file (headers only, the current state of the reference
    stub) yields zero agreements and therefore zero constraints. Nothing is
    invented to fill the gap.
  * Only commitments the result actually carries enough information to verify are
    checked. The rest are returned as constraints the caller can see, never as
    silent passes or fabricated violations.

The violation shape mirrors :class:`kairos.optimize.guardrails.Violation` so the
two kinds of breach read the same way to the dashboard and the API.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGREEMENTS_PATH = ROOT / "data" / "reference" / "AdvertiserAgreements.csv"

# Agreement pricing kinds.
FIX = "FIX"
CPP = "CPP"

# Commitment kinds recorded on an agreement.
MUST_AIR = "must_air"
EXCLUSIVITY = "exclusivity"

# Safe column mapping: real header -> canonical field. The reference stub uses
# CamelCase headers; we also accept the lower_snake variants so a hand-edited
# file still loads. Unknown columns are ignored rather than rejected.
_COLUMN_ALIASES = {
    "advertisername": "advertiser",
    "advertiser": "advertiser",
    "advertiser_name": "advertiser",
    "campaignname": "campaign",
    "campaign": "campaign",
    "campaign_name": "campaign",
    "agreementtype": "agreement_type",
    "agreement_type": "agreement_type",
    "type": "agreement_type",
    "programtitle": "program_title",
    "program_title": "program_title",
    "dayofweek": "day_of_week",
    "day_of_week": "day_of_week",
    "positioninbreak": "position_in_break",
    "position_in_break": "position_in_break",
    "daypart": "daypart",
    "channel": "channel",
    "date": "date",
    "date_from": "date_from",
    "date_to": "date_to",
    "commitment": "commitment",
    "commitmenttype": "commitment",
    "minbreaks": "min_breaks",
    "min_breaks": "min_breaks",
    "value1": "value1",
    "value2": "value2",
}


@dataclass(frozen=True)
class AgreementConstraint:
    """One obligation carried by an agreement.

    ``kind`` is :data:`MUST_AIR` or :data:`EXCLUSIVITY`. For a must-air
    commitment ``min_breaks`` is the promised minimum and ``daypart`` (and,
    optionally, ``channel``) scope where those breaks must land. ``checkable`` is
    False for commitments the optimizer cannot yet verify (exclusivity), so the
    caller can tell a recorded constraint from an enforced one.
    """

    kind: str
    daypart: Optional[str] = None
    channel: Optional[str] = None
    min_breaks: int = 0
    checkable: bool = True

    def validate(self) -> None:
        if self.kind not in (MUST_AIR, EXCLUSIVITY):
            raise ValueError(f"unknown constraint kind: {self.kind!r}")
        if self.min_breaks < 0:
            raise ValueError("min_breaks must be non-negative")


@dataclass(frozen=True)
class AdvertiserAgreement:
    """A standing commercial agreement between the channel and an advertiser.

    ``agreement_type`` is :data:`FIX` (fixed sponsorship, must air regardless of
    rating) or :data:`CPP` (priced per rating point). ``constraints`` are the
    obligations the agreement carries. ``raw`` keeps the original parsed row so
    nothing the file said is lost.
    """

    advertiser: str
    agreement_type: str = CPP
    campaign: Optional[str] = None
    channel: Optional[str] = None
    daypart: Optional[str] = None
    program_title: Optional[str] = None
    constraints: tuple[AgreementConstraint, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.advertiser:
            raise ValueError("agreement must name an advertiser")
        if self.agreement_type not in (FIX, CPP):
            raise ValueError(f"unknown agreement_type: {self.agreement_type!r}")
        for constraint in self.constraints:
            constraint.validate()

    @property
    def is_fixed_price(self) -> bool:
        """True when the break must air regardless of the rating it delivers."""
        return self.agreement_type == FIX

    @property
    def must_air_commitments(self) -> tuple[AgreementConstraint, ...]:
        return tuple(c for c in self.constraints if c.kind == MUST_AIR)


@dataclass(frozen=True)
class AgreementViolation:
    """A breached commercial agreement.

    Shaped like :class:`kairos.optimize.guardrails.Violation` so both kinds of
    breach present the same fields to the dashboard and the API.
    """

    code: str
    scope: str
    observed: float
    limit: float
    detail: str


def _normalise_header(header: str) -> str:
    return str(header).strip().lower().replace(" ", "_")


def _map_row(row: dict[str, str]) -> dict[str, str]:
    """Map a raw CSV row onto canonical field names, dropping blanks."""
    mapped: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        canonical = _COLUMN_ALIASES.get(_normalise_header(key))
        if canonical is None:
            continue
        text = (value or "").strip()
        if text:
            mapped[canonical] = text
    return mapped


def _to_int(value: Optional[str]) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _agreement_type(value: Optional[str]) -> str:
    text = (value or "").strip().upper()
    return FIX if text == FIX else CPP


def _constraints_from_row(mapped: dict[str, str]) -> tuple[AgreementConstraint, ...]:
    """Derive the obligations a row carries.

    A row implies a must-air commitment when it names a positive ``min_breaks``
    (or carries a ``must_air`` commitment marker with one). An ``exclusivity``
    commitment marker yields a recorded, non-checkable constraint.
    """
    constraints: list[AgreementConstraint] = []
    daypart = mapped.get("daypart") or mapped.get("day_of_week")
    channel = mapped.get("channel")
    commitment = (mapped.get("commitment") or "").strip().lower()

    min_breaks = _to_int(mapped.get("min_breaks"))
    if min_breaks > 0 or commitment == MUST_AIR:
        if min_breaks > 0:
            constraints.append(AgreementConstraint(
                kind=MUST_AIR, daypart=daypart, channel=channel, min_breaks=min_breaks,
            ))

    if commitment == EXCLUSIVITY:
        constraints.append(AgreementConstraint(
            kind=EXCLUSIVITY, daypart=daypart, channel=channel, checkable=False,
        ))

    return tuple(constraints)


def _agreement_from_row(mapped: dict[str, str]) -> Optional[AdvertiserAgreement]:
    advertiser = mapped.get("advertiser")
    if not advertiser:
        return None
    agreement = AdvertiserAgreement(
        advertiser=advertiser,
        agreement_type=_agreement_type(mapped.get("agreement_type")),
        campaign=mapped.get("campaign"),
        channel=mapped.get("channel"),
        daypart=mapped.get("daypart") or mapped.get("day_of_week"),
        program_title=mapped.get("program_title"),
        constraints=_constraints_from_row(mapped),
        raw=dict(mapped),
    )
    agreement.validate()
    return agreement


def load_agreements(path: str | Path | None = None) -> list[AdvertiserAgreement]:
    """Load advertiser agreements from the reference CSV.

    Returns an empty list honestly when the file is missing or has only headers
    (its current state), so an empty agreements file yields zero constraints and
    never an invented one. Rows without an advertiser name are skipped.
    """
    target = Path(path) if path is not None else DEFAULT_AGREEMENTS_PATH
    if not target.exists():
        return []

    agreements: list[AdvertiserAgreement] = []
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        for row in reader:
            mapped = _map_row(row)
            agreement = _agreement_from_row(mapped)
            if agreement is not None:
                agreements.append(agreement)
    return agreements


def _placement_daypart(placement: Any) -> Optional[str]:
    """The daypart a placement sits in, if it carries one.

    The optimizer's BreakPlacement does not name a daypart directly, so this
    reads any ``daypart`` attribute that may be attached and otherwise returns
    None. A None daypart matches a commitment that does not scope a daypart.
    """
    return getattr(placement, "daypart", None)


def _matches_scope(placement: Any, constraint: AgreementConstraint) -> bool:
    """True when a placement falls inside a commitment's daypart and channel."""
    if constraint.channel is not None:
        if str(getattr(placement, "channel", "")) != str(constraint.channel):
            return False
    if constraint.daypart is not None:
        placement_daypart = _placement_daypart(placement)
        if placement_daypart is None:
            return False
        if str(placement_daypart) != str(constraint.daypart):
            return False
    return True


def _count_matching_breaks(result: Any, constraint: AgreementConstraint) -> int:
    placements = getattr(result, "placements", ()) or ()
    return sum(1 for p in placements if _matches_scope(p, constraint))


def agreement_violations(
    result: Any,
    agreements: Iterable[AdvertiserAgreement],
) -> list[AgreementViolation]:
    """Check a result against the must-air commitments it can verify.

    For each must-air commitment, counts the placed breaks that fall inside its
    scope and reports a violation when the count is below the promised minimum.
    Exclusivity and other non-checkable constraints are skipped here rather than
    guessed at. With no agreements the result is an empty list, the honest answer
    for the current header-only reference stub.
    """
    violations: list[AgreementViolation] = []
    for agreement in agreements:
        for constraint in agreement.must_air_commitments:
            if not constraint.checkable:
                continue
            placed = _count_matching_breaks(result, constraint)
            if placed < constraint.min_breaks:
                scope_daypart = constraint.daypart or "any daypart"
                channel = constraint.channel or "any channel"
                violations.append(AgreementViolation(
                    code="must_air",
                    scope=f"{agreement.advertiser} / {channel} / {scope_daypart}",
                    observed=placed,
                    limit=constraint.min_breaks,
                    detail="fewer breaks placed than the must-air commitment requires",
                ))
    return violations
