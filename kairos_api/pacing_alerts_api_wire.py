"""Clients, pacing: what the board sends over the wire, once instead of per row.

A helper of :mod:`kairos_api.pacing_alerts_api`, declared under the section 8.2
naming rule.

The board payload was 184,610 bytes for 56 rows and it repeated itself. Measured
on the shipped data before this module existed: of 189,850 bytes of rows, 87,976
were reason and path prose and 20,526 were the same list of unsourced broadcast
dates written three times on every row, once on the flight and once on each of
the two goal lines. Every one of those copies was byte-identical to a string this
package publishes from one constant.

So the prose rides the payload once, keyed by the code or the state that selects
it, and a row carries the key it already carried. Nothing is dropped: a block
whose prose differs from the published one keeps its own, which is the same rule
:func:`pacing_alerts_api_board.collapse_demo` already applies to the demo
marking, and the reason it is safe.

**The shape a surface sees is unchanged.** ``pacing-api.js`` expands the payload
back to the full form the instant it lands, so every component reads exactly the
blocks it read before and the collapse is a wire concern and nothing else.
``tests/test_p11_pacing_board.py`` asserts the round trip is byte-identical
against the real data rather than against a fixture.
"""

from __future__ import annotations

from typing import Any

from kairos_api import pacing_alerts_api_words as words

# The four strings a reason or forward block carries. A block is collapsed only
# when all four match the published ones exactly.
PROSE = ("reason_en", "reason_he", "path_forward_en", "path_forward_he")


def _matches(block: dict[str, Any], published: dict[str, Any]) -> bool:
    return all(str(block.get(key, "")) == str(published.get(key, "")) for key in PROSE)


def _strip(block: dict[str, Any]) -> None:
    for key in PROSE:
        block.pop(key, None)


def _pace_blocks(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Every block on one row that carries a reason code."""
    blocks = [row["headline"]] if isinstance(row.get("headline"), dict) else []
    for key in ("rating", "money"):
        line = row.get(key)
        if isinstance(line, dict):
            if isinstance(line.get("pace"), dict):
                blocks.append(line["pace"])
    return blocks


def collapse(payload: dict[str, Any]) -> dict[str, Any]:
    """Lift every repeated string off the rows and publish it once. Mutates in place.

    Three repetitions are removed and each is replaced by a key the row already
    holds: a pace or headline block keyed by ``code``, a forward block keyed by
    ``state``, and the even-share reference rule, which is one sentence and was
    written on every line of every row.
    """
    reasons: dict[str, dict[str, Any]] = {}
    forwards: dict[str, dict[str, Any]] = {}
    rule = words.reference_rule()
    # Both lookups construct their table again on every call, and this walks four
    # blocks per row over 56 rows. Held here they are built once per read.
    # Measured on the shipped board, the whole collapse costs 0.44 ms.
    known: dict[str, dict[str, Any]] = {}
    known_forward: dict[str, dict[str, Any]] = {}

    for row in payload.get("rows", []):
        for block in _pace_blocks(row):
            code = str(block.get("code") or "")
            if code not in known:
                known[code] = words.reason(code)
            published = known[code]
            if not code:
                # A stated verdict carries four empty strings. They are the
                # payload's way of saying there is no reason to render, and the
                # surface reads them, so the emptiness itself is published once.
                if _matches(block, published):
                    _strip(block)
                continue
            if _matches(block, published):
                reasons[code] = {key: published[key] for key in PROSE}
                _strip(block)
        for key in ("rating", "money"):
            line = row.get(key)
            if not isinstance(line, dict):
                continue
            forward = line.get("forward")
            if isinstance(forward, dict):
                state = str(forward.get("state") or "")
                if state not in known_forward:
                    known_forward[state] = words.forward_reason(state)
                published = known_forward[state]
                if state and _matches(forward, published):
                    forwards[state] = dict(published)
                    _strip(forward)
            reference = line.get("reference")
            if isinstance(reference, dict) and all(
                str(reference.get(name, "")) == rule[name] for name in ("rule_en", "rule_he")
            ):
                reference.pop("rule_en", None)
                reference.pop("rule_he", None)
        _collapse_days(row)

    payload["reasons"] = reasons
    payload["forward_reasons"] = forwards
    payload["reference_rule"] = dict(rule)
    payload["wire"] = {
        "collapsed": True,
        "note_en": "Reason prose rides this payload once, keyed by the code or the state that selects it. A block that carries its own keeps it.",
        "note_he": "נוסח הסיבות נשלח במטען הזה פעם אחת, לפי הקוד או המצב שבוחר אותו. בלוק שנושא נוסח משלו שומר אותו.",
    }
    return payload


# The two day lists a row states, and the block on each goal line that repeats the
# one the flight already carries. This is the one term on the board that grows
# with flight length as well as with the number of campaigns, and it was written
# three times per row.
#
# The collapsed form is an explicit null and never a missing key, because a pace
# block carries an elapsed list only for two of the six reason codes and a reader
# that restored a key by absence would invent a list on the other four.
DAY_LISTS = (("unsourced_remaining_days", "forward"), ("unsourced_elapsed_days", "pace"))
SAME_AS_FLIGHT = None


def _collapse_days(row: dict[str, Any]) -> None:
    flight = row.get("flight")
    if not isinstance(flight, dict):
        return
    for name, holder in DAY_LISTS:
        for key in ("rating", "money"):
            line = row.get(key)
            block = line.get(holder) if isinstance(line, dict) else None
            if isinstance(block, dict) and isinstance(block.get(name), list) and block[name] == flight.get(name):
                block[name] = SAME_AS_FLIGHT


def _expand_days(row: dict[str, Any]) -> None:
    flight = row.get("flight")
    if not isinstance(flight, dict):
        return
    for name, holder in DAY_LISTS:
        for key in ("rating", "money"):
            line = row.get(key)
            block = line.get(holder) if isinstance(line, dict) else None
            if isinstance(block, dict) and name in block and block[name] is SAME_AS_FLIGHT:
                block[name] = list(flight.get(name) or [])


def expand(payload: dict[str, Any]) -> dict[str, Any]:
    """The inverse, for a reader that wants the full shape in Python. Mutates in place.

    The surface does this in ``pacing-api.js``. This exists so a test can prove
    the round trip is lossless against the real board rather than against a
    fixture, which is the only assertion that makes the collapse safe to ship.
    """
    reasons = payload.get("reasons") or {}
    forwards = payload.get("forward_reasons") or {}
    rule = payload.get("reference_rule") or {}
    blank = {key: "" for key in PROSE}
    for row in payload.get("rows", []):
        for block in _pace_blocks(row):
            if any(key in block for key in PROSE):
                continue
            source = reasons.get(str(block.get("code") or ""), blank)
            block.update({key: source.get(key, "") for key in PROSE})
        for key in ("rating", "money"):
            line = row.get(key)
            if not isinstance(line, dict):
                continue
            forward = line.get("forward")
            if isinstance(forward, dict) and not any(name in forward for name in PROSE):
                source = forwards.get(str(forward.get("state") or ""), blank)
                forward.update({name: source.get(name, "") for name in PROSE})
            reference = line.get("reference")
            if isinstance(reference, dict) and "rule_en" not in reference:
                reference.update({"rule_en": rule.get("rule_en", ""), "rule_he": rule.get("rule_he", "")})
        _expand_days(row)
    return payload
