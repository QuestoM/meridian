"""P10: the two computed-but-unread findings reach the operator surface."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BREAK = ROOT / "tv-break-dashboard" / "src" / "plan" / "break"


def test_the_preferred_rate_route_has_a_real_frontend_consumer() -> None:
    source = (BREAK / "PreferredPositionRate.jsx").read_text(encoding="utf-8")
    page = (BREAK / "PodPage.jsx").read_text(encoding="utf-8")
    assert "/api/preferred-position-rate?day=" in source
    assert "agency.percent" in source and "channel.percent" in source
    assert "preferred_state === 'real'" in source
    assert "<PreferredPositionRate day={day} locale={locale} />" in page


def test_preference_and_pair_states_are_visible_on_each_pod() -> None:
    board = (BREAK / "PodBoard.jsx").read_text(encoding="utf-8")
    notes = (BREAK / "PodBoardNotes.jsx").read_text(encoding="utf-8")
    assert "<PositionPreferenceNote positions={pod.positions}" in board
    assert "<CreativePairNote pairs={pod.creative_pairs}" in board
    assert "Preferred position status unavailable" in notes
    assert "No lead-and-closer pair rule is configured" in notes


def test_pair_errors_are_recomputed_on_the_order_currently_shown() -> None:
    model = (BREAK / "pod-pair-model.js").read_text(encoding="utf-8")
    assert "export function pairVerificationList" in model
    assert "Math.abs(index.get(verdict.lead_key) - index.get(verdict.closer_key)) - 1" in model
    assert "...pairVerificationList(pod.creative_pairs, spots, locale)" in (
        BREAK / "PodBoard.jsx").read_text(encoding="utf-8")


def test_the_pair_check_changes_when_the_visible_order_changes() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    module = (BREAK / "pod-pair-model.js").as_uri()
    script = f"""
      import {{ pairVerificationList }} from {json.dumps(module)};
      const pairs = {{ verdicts: [{{ rule_id: 'r1', lead_key: 'a', closer_key: 'b', allowed_min: 1, allowed_max: 2 }}] }};
      const spot = (key) => ({{ spot_key: key, advertiser: {{ value: key }} }});
      const good = pairVerificationList(pairs, [spot('a'), spot('x'), spot('b')], 'en');
      const bad = pairVerificationList(pairs, [spot('a'), spot('b'), spot('x')], 'en');
      process.stdout.write(JSON.stringify({{ good: good.length, bad: bad.length, kind: bad[0]?.kind }}));
    """
    done = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"good": 0, "bad": 1, "kind": "pair_separation"}
