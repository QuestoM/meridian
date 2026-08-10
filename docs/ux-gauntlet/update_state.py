#!/usr/bin/env python3
"""Merge one round record into the workbench state.

Reads a round record as JSON on stdin, validates it, and writes state.json
atomically. It only ever sets keys the record actually carries, so a partial
record is safe: nothing it omits is cleared.

The honesty guard is the point of this script. A record that asserts anything
(a verdict, a gap, a measurement, a description of what changed) must carry
evidence, and a measurement whose value is null must say why. A round that
cannot meet that is refused rather than written, because the page it feeds
promises that nothing on it was unmeasured.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE = os.path.join(HERE, "state.json")
DEFAULT_HTML = os.path.join(HERE, "workbench.html")

PIECE_STATUSES = ("waiting", "building", "in_critique", "passed", "blocked", "deferred")

# Fields that assert something about the world. Any of these obliges evidence.
CLAIM_FIELDS = (
    "changed_he",
    "verdict",
    "verdict_he",
    "verdict_detail_he",
    "largest_gap_he",
    "measurements",
    "open_item_updates",
    "decision_updates",
)

# Fields a record may carry onto the piece itself, rather than onto the round.
PIECE_FIELDS = {
    "piece_status": "status",
    "piece_status_detail_he": "status_detail_he",
    "piece_progress": "progress",
}

CAMPAIGN_FIELDS = {"campaign_phase", "campaign_verification"}
TRACKER_FIELDS = {"open_item_updates": "open_items", "decision_updates": "decisions"}

RESERVED = set(PIECE_FIELDS) | CAMPAIGN_FIELDS | set(TRACKER_FIELDS) | {"piece_id"}


class Refused(Exception):
    """The record was not written, and the reason is the message."""


def now_iso():
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def load_state(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise Refused("no state file at %s" % path)
    except json.JSONDecodeError as exc:
        raise Refused("state file is not valid json: %s" % exc)


def read_record(stream):
    raw = stream.read().strip()
    if not raw:
        raise Refused("nothing on stdin. Pipe a round record as json")
    try:
        rec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Refused("stdin is not valid json: %s" % exc)
    if not isinstance(rec, dict):
        raise Refused("the record must be a json object, not %s" % type(rec).__name__)
    return rec


def find_piece(state, piece_id):
    for piece in state.get("pieces", []):
        if piece.get("id") == piece_id:
            return piece
    known = ", ".join(p.get("id", "?") for p in state.get("pieces", []))
    raise Refused("unknown piece %r. The state knows: %s" % (piece_id, known))


def check_evidence(rec):
    """A record that claims something must show where the claim came from."""
    claims = [f for f in CLAIM_FIELDS if rec.get(f) not in (None, "", [], {})]
    if not claims:
        return
    evidence = rec.get("evidence")
    if not evidence:
        raise Refused(
            "this record claims %s but carries no evidence. "
            "Add an evidence list, each entry with a path" % ", ".join(claims)
        )
    if not isinstance(evidence, list):
        raise Refused("evidence must be a list of objects, each with a path")
    for i, item in enumerate(evidence):
        if not isinstance(item, dict) or not item.get("path"):
            raise Refused("evidence[%d] has no path. Every piece of evidence is addressable" % i)


def check_measurements(rec):
    """A number with no value must say why, so the page can render it honestly."""
    for i, m in enumerate(rec.get("measurements") or []):
        if not isinstance(m, dict):
            raise Refused("measurements[%d] must be an object" % i)
        if not m.get("label_he"):
            raise Refused("measurements[%d] has no label_he" % i)
        if m.get("value") is None and not m.get("note_he"):
            raise Refused(
                "measurements[%d] (%s) has no value and no note_he. "
                "An unknown number states its reason, it never renders as zero"
                % (i, m.get("label_he"))
            )


def check_piece_fields(rec):
    status = rec.get("piece_status")
    if status is not None and status not in PIECE_STATUSES:
        raise Refused(
            "piece_status %r is not one of: %s" % (status, ", ".join(PIECE_STATUSES))
        )
    for field in CAMPAIGN_FIELDS:
        if field in rec and not isinstance(rec[field], dict):
            raise Refused("%s must be an object" % field)
    for field in TRACKER_FIELDS:
        if field not in rec:
            continue
        if not isinstance(rec[field], list):
            raise Refused("%s must be a list" % field)
        for index, update in enumerate(rec[field]):
            if not isinstance(update, dict) or not update.get("id"):
                raise Refused("%s[%d] must be an object with an id" % (field, index))


def next_round_number(state, piece_id):
    used = [r.get("round", 0) for r in state.get("rounds", []) if r.get("piece_id") == piece_id]
    return (max(used) + 1) if used else 1


def merge_round(state, rec):
    """Append the round, or merge into the one with the same id, never clobbering."""
    round_fields = {k: v for k, v in rec.items() if k not in RESERVED}
    round_fields.setdefault("piece_id", rec["piece_id"])
    round_fields.setdefault("round", next_round_number(state, rec["piece_id"]))
    round_fields.setdefault("at", now_iso())
    round_fields.setdefault("id", "%s-R%s" % (rec["piece_id"], round_fields["round"]))

    rounds = state.setdefault("rounds", [])
    for existing in rounds:
        if existing.get("id") == round_fields["id"]:
            existing.update(round_fields)
            return round_fields["id"], "merged into"
    rounds.append(round_fields)
    return round_fields["id"], "appended"


def merge_piece(piece, rec, round_id):
    for src, dest in PIECE_FIELDS.items():
        if src in rec:
            piece[dest] = rec[src]
    history = piece.setdefault("rounds", [])
    if round_id not in history:
        history.append(round_id)


def merge_campaign(state, rec):
    """Update campaign-wide facts through the same atomic publish path."""
    campaign = state.setdefault("campaign", {})
    if "campaign_phase" in rec:
        campaign["phase"] = rec["campaign_phase"]
    if "campaign_verification" in rec:
        campaign.setdefault("verification", {}).update(rec["campaign_verification"])


def merge_trackers(state, rec):
    """Update existing decisions/open items without creating parallel truth."""
    for source, target in TRACKER_FIELDS.items():
        if source not in rec:
            continue
        records = state.get(target, [])
        by_id = {item.get("id"): item for item in records}
        for update in rec[source]:
            record_id = update["id"]
            if record_id not in by_id:
                known = ", ".join(sorted(str(key) for key in by_id if key))
                raise Refused("unknown %s id %r. Known: %s" % (target, record_id, known))
            by_id[record_id].update(update)


def write_atomic(path, state):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def refresh_embedded(html_path, state):
    """Re-inline the state so a page opened from disk is not a stale snapshot."""
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    blob = json.dumps(state, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    pattern = re.compile(r'(id="embedded-state">).*?(</script>)', re.S)
    if not pattern.search(html):
        raise Refused("no embedded-state block in %s" % html_path)
    updated = pattern.sub(lambda m: m.group(1) + blob + m.group(2), html, count=1)
    directory = os.path.dirname(html_path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".workbench-", suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(updated)
    os.replace(tmp, html_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge a round record into the workbench state.")
    ap.add_argument("--state", default=DEFAULT_STATE, help="path to state.json")
    ap.add_argument("--html", default=DEFAULT_HTML, help="path to workbench.html")
    ap.add_argument("--embed", action="store_true",
                    help="also refresh the copy embedded in the html, for disk-opened pages")
    ap.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    args = ap.parse_args(argv)

    rec = read_record(sys.stdin)
    if not rec.get("piece_id"):
        raise Refused("the record has no piece_id")

    state = load_state(args.state)
    piece = find_piece(state, rec["piece_id"])

    check_evidence(rec)
    check_measurements(rec)
    check_piece_fields(rec)

    round_id, how = merge_round(state, rec)
    merge_piece(piece, rec, round_id)
    merge_campaign(state, rec)
    merge_trackers(state, rec)
    state.setdefault("meta", {})["updated_at"] = now_iso()

    if args.dry_run:
        print("would be %s: %s on %s (status %s)" % (how, round_id, piece["id"], piece.get("status")))
        return 0

    write_atomic(args.state, state)
    if args.embed:
        refresh_embedded(args.html, state)
    print("%s %s on %s. Piece status: %s. Rounds on this piece: %d"
          % (how, round_id, piece["id"], piece.get("status"), len(piece.get("rounds", []))))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as exc:
        sys.stderr.write("refused: %s\n" % exc)
        sys.exit(2)
