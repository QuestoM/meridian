"""API compliance payloads backed by break-level guardrails."""

from __future__ import annotations

import pandas as pd

from kairos_api.server import KairosSettings, _build_compliance


def test_break_level_compliance_reports_guardrail_violations() -> None:
    operations = {
        "breaks": [
            {
                "channel": "Keshet 12",
                "day": "Mon",
                "start_time": f"21:0{index}",
                "duration_sec": 120,
                "program_type": "News",
                "retention": 65,
                "is_gold": index < 4,
            }
            for index in range(5)
        ]
    }

    payload = _build_compliance(pd.DataFrame(), KairosSettings(), operations)

    assert payload["status"] == "at_risk"
    assert payload["violations"]
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["retention_floor"]["status"] == "at_risk"
    assert checks["break_density"]["status"] == "at_risk"
    assert checks["protected_programs"]["status"] == "at_risk"
    assert checks["gold_breaks"]["status"] == "at_risk"


def test_break_level_compliance_stays_clean_for_safe_operations() -> None:
    operations = {
        "breaks": [
            {
                "channel": "Keshet 12",
                "day": "Tue",
                "start_time": "21:00",
                "duration_sec": 120,
                "program_type": "Drama",
                "retention": 84,
                "is_gold": False,
            },
            {
                "channel": "Keshet 12",
                "day": "Tue",
                "start_time": "21:20",
                "duration_sec": 120,
                "program_type": "Drama",
                "retention": 83,
                "is_gold": False,
            },
        ]
    }

    payload = _build_compliance(pd.DataFrame(), KairosSettings(), operations)

    assert payload["status"] == "compliant"
    assert payload["violations"] == []
    assert all(check["status"] == "compliant" for check in payload["checks"])
