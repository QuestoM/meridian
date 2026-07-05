"""API compliance grading backed by break-level guardrails.

_build_compliance now derives its verdict from the FULL committed plan's break
geometry (_plan_guardrail_items), not from the break-operations display board,
which was truncated to the first programmes per channel and synthesized gold
flags. The grading seam these tests exercise is therefore
_guardrail_compliance_from_breaks over explicit GuardrailBreak items, which is
exactly what the plan geometry feeds it.
"""

from __future__ import annotations

import pandas as pd

from kairos_api.server import (
    GuardrailBreak,
    KairosSettings,
    _build_compliance,
    _guardrail_compliance_from_breaks,
)


def _break(start_seconds: float, *, day: str = "2024-11-04", program_type: str = "Drama",
           retention: float = 0.84, is_gold: bool = False) -> GuardrailBreak:
    return GuardrailBreak(
        channel="Keshet 12",
        day=day,
        hour=int(start_seconds // 3600),
        start_seconds=start_seconds,
        duration_seconds=120,
        program_type=program_type,
        retention=retention,
        is_gold=is_gold,
    )


def test_break_level_compliance_reports_guardrail_violations() -> None:
    # Five News breaks crammed into one hour, low retention, four gold: breaks
    # density, protected load, retention floor and the gold cap all at risk.
    items = [
        _break(21 * 3600 + index * 60, program_type="News", retention=0.65, is_gold=index < 4)
        for index in range(5)
    ]

    payload = _guardrail_compliance_from_breaks(items, KairosSettings())

    assert payload is not None
    assert payload["status"] == "at_risk"
    assert payload["violations"]
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["retention_floor"]["status"] == "at_risk"
    assert checks["break_density"]["status"] == "at_risk"
    assert checks["protected_programs"]["status"] == "at_risk"
    assert checks["gold_breaks"]["status"] == "at_risk"


def test_break_level_compliance_stays_clean_for_safe_items() -> None:
    items = [
        _break(21 * 3600, retention=0.84),
        _break(21 * 3600 + 20 * 60, retention=0.83),
    ]

    payload = _guardrail_compliance_from_breaks(items, KairosSettings())

    assert payload is not None
    assert payload["status"] == "compliant"
    assert payload["violations"] == []
    assert all(check["status"] == "compliant" for check in payload["checks"])


def test_build_compliance_ignores_the_display_board() -> None:
    # The operations argument is signature-compat only: a fabricated board full
    # of violations must not move the verdict, which comes from the committed
    # plan geometry (or the honest schedule-level fallback when absent).
    fabricated = {
        "breaks": [
            {
                "channel": "Keshet 12", "day": "Mon", "start_time": f"21:0{i}",
                "duration_sec": 120, "program_type": "News", "retention": 65,
                "is_gold": True,
            }
            for i in range(5)
        ]
    }
    with_board = _build_compliance(pd.DataFrame(), KairosSettings(), fabricated)
    without_board = _build_compliance(pd.DataFrame(), KairosSettings(), None)
    assert with_board["checks"] == without_board["checks"]
    assert with_board["status"] == without_board["status"]
