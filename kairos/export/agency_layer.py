"""The agency data the daily per-spot pricing path consumes.

Split out of :mod:`kairos.export.spots` to keep that module under the project
line limit. The daily Wally file carries an agency per spot ('משרד / MB' ->
``agency`` in the loader); this layer resolves each spot's agency (the spot's
own column first, the link table as fallback), evaluates agency-scoped
conditions with the SAME rule engine the advertiser conditions use (one
implementation of scope semantics and mode math, keyed by agency_id with no
baselines), and carries each agency's ``rebate_percent`` so the ledger can
report ``net_revenue`` beside gross. Reporting only: gross is unchanged and
nothing is invoiced. An empty layer is the exact identity: premium 1.0, rebate
0, nothing dropped. See docs/agency-layer-design.md for the full contract.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from kairos.optimize.advertiser_rules import AdvertiserRuleEngine, _condition_from_row
from kairos.optimize._rule_helpers import parse_float

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AGENCIES_PATH = _DATA_DIR / "agencies.csv"
AGENCY_LINKS_PATH = _DATA_DIR / "agency_advertisers.csv"
AGENCY_CONDITIONS_PATH = _DATA_DIR / "agency_conditions.csv"


@dataclass(frozen=True)
class AgencyTerms:
    """One agency's pricing-relevant terms, resolved for a spot."""

    agency_id: str
    name: str
    rebate_percent: float = 0.0
    active: bool = True


@dataclass
class AgencyLayer:
    """The agency stores, loaded once per pricing run.

    ``terms`` keys agency_id; ``by_name`` maps the exact Wally string, the
    display name and every alias to an agency_id; ``by_advertiser`` is the link
    table fallback for daily files lacking the agency column (manual links win
    over observed). ``engine`` evaluates agency conditions agency-first on the
    pricing path; a suspended agency's terms and conditions are inert.
    """

    terms: dict[str, AgencyTerms] = field(default_factory=dict)
    by_name: dict[str, str] = field(default_factory=dict)
    by_advertiser: dict[str, str] = field(default_factory=dict)
    engine: AdvertiserRuleEngine = field(
        default_factory=lambda: AdvertiserRuleEngine(baselines={}, conditions={})
    )

    @classmethod
    def from_files(
        cls,
        agencies_path: Path = AGENCIES_PATH,
        links_path: Path = AGENCY_LINKS_PATH,
        conditions_path: Path = AGENCY_CONDITIONS_PATH,
    ) -> "AgencyLayer":
        """Load the shipped agency stores; missing files yield the empty identity."""
        layer = cls()
        for row in _read_csv_rows(agencies_path):
            agency_id = str(row.get("agency_id", "")).strip()
            name = str(row.get("name", "")).strip()
            if not agency_id or not name:
                continue
            active = str(row.get("status", "active")).strip().lower() != "suspended"
            layer.terms[agency_id] = AgencyTerms(
                agency_id=agency_id, name=name, active=active,
                rebate_percent=parse_float(row.get("rebate_percent"), 0.0),
            )
            aliases = str(row.get("aliases", "")).split("|")
            for key in [name, str(row.get("display_name", "")).strip(), *aliases]:
                if key.strip():
                    layer.by_name.setdefault(key.strip(), agency_id)
        for row in sorted(_read_csv_rows(links_path),
                          key=lambda r: str(r.get("source", "")) == "manual"):
            advertiser = str(row.get("advertiser", "")).strip()
            agency_id = str(row.get("agency_id", "")).strip()
            if advertiser and agency_id in layer.terms:
                layer.by_advertiser[advertiser] = agency_id  # manual sorts last, wins
        conditions: dict[str, list] = {}
        for row in _read_csv_rows(conditions_path):
            row = dict(row)
            row["advertiser_id"] = str(row.get("agency_id", "")).strip()
            condition = _condition_from_row(row)
            if condition is not None:
                conditions.setdefault(condition.advertiser_id, []).append(condition)
        layer.engine = AdvertiserRuleEngine(baselines={}, conditions=conditions)
        return layer

    def resolve(self, spot_agency: Any, advertiser: str) -> Optional[AgencyTerms]:
        """The spot's agency terms: the spot's own column first, links fallback."""
        agency_id = self.by_name.get(str(spot_agency or "").strip())
        if agency_id is None:
            agency_id = self.by_advertiser.get(str(advertiser or "").strip())
        return self.terms.get(agency_id) if agency_id else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
