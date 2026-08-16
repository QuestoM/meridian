"""Plan step changes acknowledge the new workspace without moving the shell."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src"


def test_plan_steps_use_the_shared_workspace_continuity_and_respect_reduced_motion():
    source = (ROOT / "plan/week/PlanWeek.jsx").read_text(encoding="utf-8")
    assert "queueWorkspaceContinuity" in source
    assert "prefers-reduced-motion: reduce" in source
    assert "behavior: reduced ? 'auto' : 'smooth'" in source

