"""The model registry at a terminal: compare candidates, and adopt or reject one.

    PYTHONUTF8=1 python scripts/adopt_candidate.py show
    PYTHONUTF8=1 python scripts/adopt_candidate.py rescore
    PYTHONUTF8=1 python scripts/adopt_candidate.py measure <candidate>
    PYTHONUTF8=1 python scripts/adopt_candidate.py checks <candidate>
    PYTHONUTF8=1 python scripts/adopt_candidate.py diff <candidate>
    PYTHONUTF8=1 python scripts/adopt_candidate.py decide <candidate> --decision <ship|no-ship> --actor "<name>" --reason "<sentence>"
    PYTHONUTF8=1 python scripts/adopt_candidate.py adopt <candidate> --adopted-by "<name>" --reason "<sentence>"
    PYTHONUTF8=1 python scripts/adopt_candidate.py revert <adoption id> --reverted-by "<name>" --reason "<sentence>"
    PYTHONUTF8=1 python scripts/adopt_candidate.py report
    PYTHONUTF8=1 python scripts/adopt_candidate.py publish

``--json`` prints the payload instead of the table on ``show``, ``checks``,
``diff``, ``decide``, ``adopt`` and ``revert``, and it is accepted on either
side of the subcommand.

**This is training and it is company staff only.** Its acts write under
``models/``, which is the whole definition of training in section 4.1 of the
specification, so it has no route, no button and no link. It runs here, at a
terminal, in this repository, by the people who own the model. A test asserts
that no endpoint the application publishes and no file the operator's interface
ships can reach it.

``publish`` is the one subcommand that writes outside ``models/``, and by that
same test it is not training: it writes the comparison into this piece's own
frontend row for its own panel to read, moves no model and produces no artifact.
The payload it writes is refused if it names this act at all, because the panel
is behind a route wall and the bundle it travels in is not.

**Nothing expensive happens by accident.** ``show``, ``report`` and ``publish``
read what has been measured and never measure anything. ``rescore`` costs about
ten seconds of data loading. ``measure`` costs about two hundred seconds of
optimizer because it computes the weekly plan twice. ``adopt`` and ``decide``
without ``--perform`` write nothing at all and print the distance to a landing.

**Deciding and adopting are two acts, and the order is fixed.** ``decide``
records a ship or no-ship verdict into the model console's own decision store,
which is the done condition of JS-19 and the only part of it this terminal
previously could not perform. It moves no artifact. ``adopt`` is the separate act
that copies a candidate over the shipped file, and it requires a ship verdict to
already be on record against the model version on disk.

**An adoption that would move a shipped figure stops here.** It is reported as
escalated, with the measured movement in shekels and the scope it was measured
on, and no flag on this command line releases it. Section 9 item 11 of the
specification and JS-19's own target both say so, and the release path is an
owner approval artifact that names the exact figure, described in
``scripts/adopt_candidate_adoption.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import adopt_candidate_adoption as adoption  # noqa: E402
from scripts import adopt_candidate_board as board  # noqa: E402
from scripts import adopt_candidate_cells as cells  # noqa: E402
from scripts import adopt_candidate_decide as verdict  # noqa: E402
from scripts import adopt_candidate_registry as registry  # noqa: E402
from scripts import adopt_candidate_rescore as rescore  # noqa: E402
from scripts import adopt_candidate_words as words  # noqa: E402

REPORT_FILE = "candidate_registry.json"


def _paths() -> rescore.Paths:
    return rescore.Paths()


def _print(lines: list[str]) -> None:
    for line in lines:
        print(line)


def command_show(args: argparse.Namespace) -> int:
    payload = registry.registry(_paths())
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    _print(registry.render(payload))
    return 0


def command_rescore(args: argparse.Namespace) -> int:
    paths = _paths()
    state = rescore.rescore_state(paths)
    if state["state"] == "current" and not args.force:
        print(f"The re-score is already current, measured {state.get('measured_at')}.")
        print("Pass --force to run it again anyway.")
        return 0
    payload = rescore.rescore(paths)
    written = rescore.save_rescore(payload, paths)
    print(f"Re-scored {len(payload['candidates'])} candidates against the shipped model.")
    print(f"Evaluation: {payload['evaluation']['breaks']} breaks, {payload['evaluation']['cells']} cells, {words.window_line(payload['evaluation'])}.")
    print(f"Written to {written.relative_to(paths.root).as_posix()}.")
    print("")
    _print(registry.render(registry.registry(paths)))
    return 0


def command_measure(args: argparse.Namespace) -> int:
    """Measure the money one candidate would move, into the console's own store.

    The measurement and the store are P7's, called rather than reimplemented, so
    the figure a steward reads here and the figure the model console reads are
    the same figure produced by the same code.
    """
    from kairos_api import model_console_candidates as console
    from kairos_api import model_version_store as store

    if console.candidate_path(args.candidate) is None:
        print(f"There is no candidate called {args.candidate}.")
        return 2
    print(f"Computing the weekly plan twice for {args.candidate}. This takes about two hundred seconds.")
    record = console.measure_money_movement(args.candidate)
    store.save_measurement(record)
    delta = record["operator_channel_delta"]["revenue_delta"]
    scope = record["scope"]["operator_channel"]
    print(f"Measured in {record['duration_seconds']} s.")
    print(f"Revenue movement on the operator's own channel: {delta:+,.2f} over {scope['rows']} rows.")
    print(f"Basis: {scope['basis']}.")
    print(f"Whole plan: {record['whole_plan_delta']['revenue_delta']:+,.2f}.")
    return 0


def command_checks(args: argparse.Namespace) -> int:
    state = registry.checks_for(args.candidate, _paths(), adopted_by=args.adopted_by,
                                reason=args.reason)
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=1))
        return 0
    _print(registry.render_checks(state))
    return 0


def command_diff(args: argparse.Namespace) -> int:
    """Every cell one candidate moves, ranked by how much of the movement it carries.

    The score table says a candidate is a thousandth closer. It cannot say
    whether that came from one cell or from thirty-six that moved and cancelled,
    and those are two different artifacts. This reads the stored re-score and
    measures nothing, so it costs a file read.
    """
    row = rescore.candidate_row(args.candidate, _paths())
    if row is None:
        known = ", ".join(sorted(rescore.candidate_id(path)
                                 for path in rescore.candidate_files(_paths())))
        print(f"There is no candidate called {args.candidate}. Known: {known or 'none'}.")
        return 2
    deltas = row.get("cell_deltas") or {}
    if args.json:
        print(json.dumps(deltas, ensure_ascii=False, indent=1))
        return 0
    _print(cells.render_table(args.candidate, deltas, limit=0 if args.all else args.top))
    return 0


def command_decide(args: argparse.Namespace) -> int:
    """Record a ship or no-ship verdict, which is where JS-19 actually ends.

    The record lands in the model console's own store through the console's own
    function, so this terminal and that console cannot hold different verdicts
    about the same candidate.
    """
    decision = verdict.normalise_decision(args.decision)
    result = verdict.decide(args.candidate, decision=decision, actor=args.actor,
                            reason=args.reason, reason_en=args.reason_en,
                            release_note_he=args.release_note_he,
                            release_note_en=args.release_note_en,
                            paths=_paths(), perform=args.perform)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        _print(verdict.render(result))
    return 0 if result["outcome"] in ("recorded", "ready") else 1


def command_adopt(args: argparse.Namespace) -> int:
    plan = adoption.adopt(args.candidate, adopted_by=args.adopted_by, reason=args.reason,
                          release_note_he=args.release_note_he, paths=_paths(),
                          perform=args.perform)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=1))
        return 0 if plan["outcome"] in ("adopted", "ready") else 1
    _print(registry.render_checks(registry.with_origin(plan, _paths())))
    if plan["outcome"] == "adopted":
        record = plan["record"]
        print(f"Adopted {record['candidate_id']} as {record['adoption_id']}.")
        print(f"The shipped artifact is now {record['adopted_sha256'][:12]}, was {record['superseded_sha256'][:12]}.")
        print(f"Undo with: python scripts/adopt_candidate.py revert {record['adoption_id']} --reverted-by \"<name>\" --reason \"<sentence>\"")
        return 0
    if plan["outcome"] == "ready":
        print("Every check passed and nothing has been written.")
        print("Add --perform to adopt.")
        return 0
    return 1


def command_revert(args: argparse.Namespace) -> int:
    result = adoption.revert(args.adoption_id, reverted_by=args.reverted_by,
                             reason=args.reason, paths=_paths(), perform=args.perform)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result["outcome"] in ("reverted", "ready") else 1
    print(f"outcome: {result['outcome']}")
    if result.get("reason_en"):
        print(result["reason_en"])
    if result["outcome"] == "ready":
        print("Add --perform to revert.")
    if result["outcome"] == "reverted":
        record = result["record"]
        print(f"Restored {record['restored_sha256'][:12]}, expected {record['expected_sha256'][:12]}.")
        print(f"Byte-exact: {record['restored_exactly']}.")
    return 0 if result["outcome"] in ("reverted", "ready") else 1


def command_report(args: argparse.Namespace) -> int:
    """Write the whole registry as one payload, for a reader that is not a terminal.

    The model console owns the training-side screens and its routes are frozen,
    so this piece publishes its join as a file rather than reaching into another
    piece's module. When a route serves it, this is the shape it serves.
    """
    paths = _paths()
    payload = registry.registry(paths)
    target = paths.releases_dir / REPORT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Written to {target.relative_to(paths.root).as_posix()}.")
    print(f"{len(payload['candidates'])} candidates, re-score {payload['rescore_state']['state']}.")
    return 0


def command_publish(args: argparse.Namespace) -> int:
    """Publish the comparison to the board this piece's own screen imports.

    Writes one file, into this piece's frontend row, and nothing under
    ``models/``, so publishing is not training by section 4.1's own test: it
    moves no model and produces no artifact. The payload it writes carries no
    command and no name that reaches the training act, and the write refuses if
    one appears, because a browser bundle is not a walled surface.
    """
    paths = _paths()
    payload = board.board(paths)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    written = board.save_board(payload, paths)
    print(f"Written to {written.relative_to(paths.root).as_posix()}.")
    print(f"{len(payload['candidates'])} candidates against the shipped artifact {payload['shipped']['short']}, re-score {(payload['rescore_state'] or {}).get('state')}.")
    print(f"Measured {payload['measured_at']}, published {payload['published_at']}.")
    print("The screen compares these digests with the ones the model console's own route serves, and says stale rather than showing a figure whose subject has moved.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adopt_candidate",
        description="Compare candidate coefficient artifacts and adopt or reject one.")
    parser.add_argument("--json", action="store_true", help="print the payload instead of the table")
    # The same flag on every subcommand, so the natural "show --json" is not an
    # argparse error. SUPPRESS is what makes both placements work: without the
    # flag the subparser leaves the namespace alone, so a --json typed before
    # the subcommand survives instead of being overwritten by a default.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="print the payload instead of the table")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("show", parents=[shared],
                          help="the registry: every artifact, its score, its money, its verdict")

    rescore_parser = subparsers.add_parser("rescore", parents=[shared], help="score every artifact on one common set of breaks")
    rescore_parser.add_argument("--force", action="store_true", help="run again even when the stored score is current")

    measure_parser = subparsers.add_parser("measure", parents=[shared], help="measure the money one candidate would move")
    measure_parser.add_argument("candidate")

    checks_parser = subparsers.add_parser("checks", parents=[shared], help="the adoption checks for one candidate, writing nothing")
    checks_parser.add_argument("candidate")
    checks_parser.add_argument("--adopted-by", default="")
    checks_parser.add_argument("--reason", default="")

    diff_parser = subparsers.add_parser("diff", parents=[shared], help="every coefficient one candidate moves, and what each move bought")
    diff_parser.add_argument("candidate")
    diff_parser.add_argument("--top", type=int, default=12, help="how many cells to show, ranked by the movement they carry")
    diff_parser.add_argument("--all", action="store_true", help="every cell, not only the top ones")

    decide_parser = subparsers.add_parser("decide", parents=[shared], help="record a ship or no-ship verdict against the model version on disk")
    decide_parser.add_argument("candidate")
    decide_parser.add_argument("--decision", choices=verdict.DECISION_CHOICES, required=True,
                               metavar="{ship|no-ship}",
                               help="ship or no-ship. The store's own keys, shipped and not_shipped, are accepted too")
    decide_parser.add_argument("--actor", default="", help="who is taking this verdict")
    decide_parser.add_argument("--reason", default="", help="why, in one sentence, rendered verbatim on a right-to-left card")
    decide_parser.add_argument("--reason-en", default="", help="the same sentence in English, carried in the evidence")
    decide_parser.add_argument("--release-note-he", default="", help="the sentence the operator side reads, required for a ship verdict")
    decide_parser.add_argument("--release-note-en", default="")
    decide_parser.add_argument("--perform", action="store_true", help="record it; without this nothing is written")

    adopt_parser = subparsers.add_parser("adopt", parents=[shared], help="adopt a candidate, or report why it cannot land")
    adopt_parser.add_argument("candidate")
    adopt_parser.add_argument("--adopted-by", default="", help="who is taking this decision")
    adopt_parser.add_argument("--reason", default="", help="why, in one sentence")
    adopt_parser.add_argument("--release-note-he", default="", help="the sentence the operator side reads")
    adopt_parser.add_argument("--perform", action="store_true", help="write it; without this nothing is written")

    revert_parser = subparsers.add_parser("revert", parents=[shared], help="put back the artifact an adoption replaced")
    revert_parser.add_argument("adoption_id")
    revert_parser.add_argument("--reverted-by", default="")
    revert_parser.add_argument("--reason", default="")
    revert_parser.add_argument("--perform", action="store_true", help="write it; without this nothing is written")

    subparsers.add_parser("report", parents=[shared], help="write the whole registry payload to models/releases/")

    subparsers.add_parser("publish", parents=[shared], help="publish the comparison to the board the model steward reads on screen")
    return parser


COMMANDS = {
    "show": command_show,
    "rescore": command_rescore,
    "measure": command_measure,
    "checks": command_checks,
    "diff": command_diff,
    "decide": command_decide,
    "adopt": command_adopt,
    "revert": command_revert,
    "report": command_report,
    "publish": command_publish,
}


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS.get(args.command or "show")
    if handler is None:
        parser.print_help()
        return 2
    if args.command is None:
        args.json = getattr(args, "json", False)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
