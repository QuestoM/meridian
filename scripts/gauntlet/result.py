"""The result vocabulary the harness reports in, and how it prints.

Three states, not two. A check that could not run is not a check that passed,
and it is not a failure of the code either. Collapsing "unchecked" into either
one is how a gate starts lying, so it is a first-class state here and it keeps
the exit code non-zero unless the caller explicitly accepts it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any

PASS = "pass"
FAIL = "fail"
UNCHECKED = "unchecked"

_MARK = {PASS: "PASS", FAIL: "FAIL", UNCHECKED: "NOT CHECKED"}


@dataclass
class Result:
    name: str
    title: str
    status: str = UNCHECKED
    summary: str = ""
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    seconds: float | None = None

    def passed(self, summary: str, **measurements: Any) -> "Result":
        self.status = PASS
        self.summary = summary
        self.measurements.update(measurements)
        return self

    def failed(self, summary: str, **measurements: Any) -> "Result":
        self.status = FAIL
        self.summary = summary
        self.measurements.update(measurements)
        return self

    def cannot_check(self, reason: str) -> "Result":
        self.status = UNCHECKED
        self.reason = reason
        return self

    def note(self, line: str) -> None:
        self.detail.append(line)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "reason": self.reason,
            "evidence": self.evidence,
            "detail": self.detail,
            "measurements": self.measurements,
            "seconds": self.seconds,
        }


def _rule(char: str = "-") -> str:
    return char * min(shutil.get_terminal_size((88, 20)).columns, 88)


def render(results: list[Result], reference: str, tree_note: str) -> str:
    """One readable block. Failures first, because that is what a gate is for."""
    lines: list[str] = []
    lines.append(_rule("="))
    lines.append("wave verification against %s" % reference)
    lines.append(tree_note)
    lines.append(_rule("="))

    order = {FAIL: 0, UNCHECKED: 1, PASS: 2}
    for r in sorted(results, key=lambda x: (order.get(x.status, 3), x.name)):
        took = "" if r.seconds is None else "  [%.1fs]" % r.seconds
        lines.append("")
        lines.append("%-12s %s%s" % (_MARK.get(r.status, r.status), r.title, took))
        if r.summary:
            lines.append("  %s" % r.summary)
        if r.reason:
            lines.append("  could not check: %s" % r.reason)
        for line in r.detail:
            lines.append("    %s" % line)
        for ev in r.evidence:
            lines.append("  evidence: %s" % ev)

    counts = {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, UNCHECKED)}
    lines.append("")
    lines.append(_rule("="))
    lines.append(
        "%d passed, %d failed, %d not checked"
        % (counts[PASS], counts[FAIL], counts[UNCHECKED])
    )
    if counts[FAIL]:
        lines.append("VERDICT: the working tree is NOT proven identical to the reference.")
    elif counts[UNCHECKED]:
        lines.append(
            "VERDICT: no failure found, but the proof is incomplete. "
            "Every check above marked not checked is a gap, not a pass."
        )
    else:
        lines.append("VERDICT: the working tree is behaviourally identical to the reference.")
    lines.append(_rule("="))
    return "\n".join(lines)


def exit_code(results: list[Result], allow_unchecked: bool) -> int:
    if any(r.status == FAIL for r in results):
        return 1
    if any(r.status == UNCHECKED for r in results) and not allow_unchecked:
        return 2
    return 0
