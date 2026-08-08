"""What each artifact's gates decided, and why counting the differences lies.

JS-19's second sentence is "read what its gates decided differently from the
shipped artifact" and its target adds "every gate delta visible with its held-out
figure". Both were reachable only from the ``checks`` command, which is the last
thing a steward runs, and neither was on the board this piece publishes. That is
the gap this module closes, and closing it surfaced a reading defect underneath
it that was already on the terminal.

**A difference is not a disagreement.** ``model_console_candidates.gate_deltas``
returns every gate key on which the two artifacts do not hold the same value, and
a key the candidate does not carry at all comes back as a difference with
``candidate_absent`` set. Measured on this tree, three of the five candidates
return **ten** such rows and every one of them is an absence: those artifacts
record no gate decision whatsoever. A count of ten reads as ten gates decided the
other way, which is the opposite of what is true. So the rows are separated here
into the keys where both artifacts hold a value and the values differ, and the
keys one side does not carry, and the sentence that follows is chosen from that.

Measured on this tree, over all five candidates: exactly **one** key on one
candidate is a real disagreement, ``placebo_correction_active`` on ``calibrated``,
true on the shipped artifact and false on the candidate. Every other row on every
other candidate is an absence.

**The held-out sizes are the argument for this whole piece.** Each artifact
reports the amount its own gates were decided on, and those amounts do not agree:
the shipped artifact decided its series gate on 2,532 breaks and
``placebo_corrected`` decided the same gate on 506. Two figures taken on
different amounts are not comparable, so the sizes travel side by side with the
sentence that says so, rather than the gate figures alone.

The gate-shaped strings live here with the gate-shaped arithmetic rather than in
``adopt_candidate_words.py``, and the bilingual walker discovers every module of
this piece rather than that one file, so the law that an authored string is two
strings still reaches them.
"""

from __future__ import annotations

from typing import Any

# What a gate cell says when one artifact does not carry the key at all. A
# different fact from a key it carries with a different value, so it is a
# sentence and never a blank.
GATE_ABSENT: dict[str, str] = {
    "en": "does not carry it",
    "he": "אינו נושא אותו",
}


# What each held-out block counted. Three gates on this tree record their size
# under three different key names and they are three different things, so the
# unit travels with the figure and 34,560 is never read as 34,560 breaks.
HELD_OUT_UNITS: dict[str, str] = {
    "n_test": "breaks",
    "n_test_minutes": "minutes",
    "n_test_days": "days",
}

HELD_OUT_UNITS_HE: dict[str, str] = {
    "n_test": "ברייקים",
    "n_test_minutes": "דקות",
    "n_test_days": "ימים",
}


# The sentence the gate rows amount to, chosen by the measurement rather than by
# the row count. ``differing`` is the number of keys both artifacts carry with
# different values, which is the only one of these numbers that means a gate
# decided differently.
#
# **Nothing here states a share of all gate keys**, and the wording is careful
# about it. The list of keys the console compares is a constant inside a frozen
# module, this piece reads the comparison rather than the list, and a key both
# artifacts leave out is indistinguishable from a key both artifacts agree on
# from the comparison alone. So every count here is a count of the keys the two
# artifacts differ on, which is exactly what was measured, and the sentence never
# says "records no gate at all" about an artifact whose other keys it cannot see.
GATE_READING: dict[str, dict[str, str]] = {
    "same": {
        "en": "These two artifacts hold the same value in every gate key the comparison looks at. Nothing here decided differently.",
        "he": "שני הקבצים האלה מחזיקים אותו ערך בכל מפתח שער שההשוואה בוחנת. שום דבר כאן לא הוכרע אחרת.",
    },
    "absent_only_candidate": {
        "en": "No gate key that both artifacts carry holds a different value. All {not_identical} keys they differ on are keys the candidate does not carry at all, so it records no decision on any of them.",
        "he": "אין מפתח שער ששני הקבצים נושאים ובו ערך שונה. כל {not_identical} המפתחות שהם נבדלים בהם הם מפתחות שהמועמד אינו נושא כלל, ולכן הוא אינו רושם הכרעה באף אחד מהם.",
    },
    "absent_only_candidate_one": {
        "en": "No gate key that both artifacts carry holds a different value. The one key they differ on is a key the candidate does not carry at all, so it records no decision on it.",
        "he": "אין מפתח שער ששני הקבצים נושאים ובו ערך שונה. המפתח היחיד שהם נבדלים בו הוא מפתח שהמועמד אינו נושא כלל, ולכן הוא אינו רושם בו הכרעה.",
    },
    "absent_only_mixed": {
        "en": "No gate key that both artifacts carry holds a different value. Of the {not_identical} keys they differ on, {absent} are absent from the candidate and {absent_on_shipped} are absent from the shipped artifact, and an absence is not a decision.",
        "he": "אין מפתח שער ששני הקבצים נושאים ובו ערך שונה. מתוך {not_identical} המפתחות שהם נבדלים בהם, {absent} חסרים במועמד ו {absent_on_shipped} חסרים בקובץ המשודר, והיעדרות אינה הכרעה.",
    },
    "decides_differently": {
        "en": "{differing} gate keys are carried by both artifacts and hold a different value, which is what a gate decided differently means here.",
        "he": "{differing} מפתחות שער נישאים בשני הקבצים ומחזיקים ערך שונה, וזו המשמעות של שער שהוכרע אחרת.",
    },
    "decides_differently_one": {
        "en": "One gate key is carried by both artifacts and holds a different value, which is what a gate decided differently means here.",
        "he": "מפתח שער אחד נישא בשני הקבצים ומחזיק ערך שונה, וזו המשמעות של שער שהוכרע אחרת.",
    },
}


# Appended to a differing reading when there are absences beside the difference,
# so a row never states a difference count and leaves the absences unsaid.
GATE_ALSO_ABSENT: dict[str, str] = {
    "en": "A further {absent} keys the candidate does not carry at all.",
    "he": "עוד {absent} מפתחות שהמועמד אינו נושא כלל.",
}


# Why the figures each artifact reports about its own gates may not be read
# against each other. This is the argument the whole piece rests on, said where
# the two sizes are printed side by side. It is unconditionally true and says
# nothing about any particular pair.
HELD_OUT_RULE: dict[str, str] = {
    "en": "Each artifact reports the amount its own gates were decided on, and two figures taken on different amounts are not comparable. That is why the amounts are printed beside the gates rather than the gate figures alone, and why every artifact is scored again on one identical set of breaks.",
    "he": "כל קובץ מדווח על הכמות שהשערים שלו הוכרעו עליה, ושני מספרים שנלקחו על כמויות שונות אינם ברי השוואה. זו הסיבה שהכמויות מודפסות לצד השערים ולא רק מספרי השערים, ומכאן שכל קובץ נמדד שוב על אותה קבוצת ברייקים בדיוק.",
}


# And whether it bit on THIS pair, which is a measurement and not a constant. The
# sentence used to assert unevenness of every pair, and on ``calibrated`` every
# held-out block agrees, so the surface was stating a confound that pair does not
# carry. Three states, and the even one says what it is not evidence of.
HELD_OUT_STATE: dict[str, dict[str, str]] = {
    "even": {
        "en": "Every held-out block both artifacts report was taken on the same amount, so on this pair those figures are comparable with each other. That is a property of this pair and not of the shelf.",
        "he": "כל בלוק שנשמר בצד ששני הקבצים מדווחים עליו נלקח על אותה כמות, ולכן בזוג הזה המספרים האלה ברי השוואה זה לזה. זו תכונה של הזוג הזה ולא של המדף.",
    },
    "uneven": {
        "en": "These amounts do not agree, so the gate figures taken on them are not comparable with each other.",
        "he": "הכמויות האלה אינן תואמות, ולכן מספרי השערים שנלקחו עליהן אינם ברי השוואה זה לזה.",
    },
    "none": {
        "en": "Neither artifact reports how much its gates were decided on, so whether they are comparable is unknown rather than established.",
        "he": "אף אחד מהקבצים אינו מדווח על הכמות שהשערים שלו הוכרעו עליה, ולכן השאלה אם הם ברי השוואה אינה ידועה ולא הוכחה.",
    },
}


def gate_cell(value: Any, absent: bool, language: str = "en") -> str:
    """One side of a gate row: its value, or that this side has no such key.

    A boolean is lowered and a count is grouped, because ``True`` and ``2532``
    are how Python prints them and not how a person reads them, and a float is
    cut at six decimals so a p-value does not arrive with seventeen.
    """
    if absent:
        return GATE_ABSENT[language]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    return str(round(value, 6)) if isinstance(value, float) else str(value)


def size_cell(size: Any, unit: str, absent: bool, language: str = "en") -> str:
    """One side of a held-out row: how much it was measured on, and of what."""
    if absent:
        return GATE_ABSENT[language]
    table = HELD_OUT_UNITS if language == "en" else HELD_OUT_UNITS_HE
    return f"{gate_cell(size, False)} {table.get(unit, unit or '')}".strip()


def _reading(key: str, counts: dict[str, int]) -> dict[str, str]:
    """The sentence, with the absence clause added only when there is one."""
    entry = GATE_READING[key]
    reading = {"reading_en": entry["en"].format(**counts),
               "reading_he": entry["he"].format(**counts)}
    if key.startswith("decides_differently") and counts["absent"]:
        for half in ("en", "he"):
            reading[f"reading_{half}"] += " " + GATE_ALSO_ABSENT[half].format(**counts)
    return reading


def _reading_key(state: str, counts: dict[str, int]) -> str:
    """Which of the sentences this row takes, including its singular form."""
    if state == "decides_differently":
        return "decides_differently_one" if counts["differing"] == 1 else state
    if state == "absent_only_candidate":
        return "absent_only_candidate_one" if counts["not_identical"] == 1 else state
    return state


def _gate_row(row: dict[str, Any]) -> dict[str, Any]:
    """One gate row with each side's value already rendered, once, here.

    A stored ``1.0`` is a float this terminal prints as 1.0 and a browser reading
    the same JSON prints as 1, because JavaScript has one number type and cannot
    see the difference. Two surfaces of one piece printing one stored value two
    ways is a divergence a steward walks into inside a session, so the rendering
    happens once, on the side that still knows what the value is.
    """
    out = dict(row)
    for side in ("shipped", "candidate"):
        out[f"{side}_text"] = (None if row.get(f"{side}_absent")
                               else gate_cell(row.get(side), False))
    return out


def _held_out_row(row: dict[str, Any]) -> dict[str, Any]:
    """One held-out row with its unit as a word in both halves, not as a key.

    ``n_test_minutes`` is a key in the artifact and it is not a word anybody
    reads. The screen would otherwise have to hold its own map from that key to
    a noun, which is a second vocabulary for one fact, so the noun travels with
    the figure from the one table that holds it.
    """
    out = dict(row)
    for side in ("shipped", "candidate"):
        unit = str(row.get(f"{side}_unit") or "")
        out[f"{side}_unit_en"] = HELD_OUT_UNITS.get(unit, unit)
        out[f"{side}_unit_he"] = HELD_OUT_UNITS_HE.get(unit, unit)
    return out


def gate_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    """The gate rows as a reading, with the count that means what it says.

    ``not_identical`` is the whole set of keys the two artifacts do not agree on,
    which is what the console's own comparison returns. It is split rather than
    reported, because only ``differing`` is a gate that decided differently and
    the other two are keys somebody did not write down.

    There is no denominator over all gate keys here on purpose. The list of keys
    the console compares is a constant inside a frozen module, and this piece
    reads the comparison rather than the list, so it states the count it actually
    measured and never a share of a total it did not read.
    """
    rows = [_gate_row(row) for row in (evidence or {}).get("verdicts") or []]
    differing = [row for row in rows
                 if not row.get("shipped_absent") and not row.get("candidate_absent")]
    absent_on_candidate = [row for row in rows if row.get("candidate_absent")]
    absent_on_shipped = [row for row in rows if row.get("shipped_absent")]
    if not rows:
        state = "same"
    elif differing:
        state = "decides_differently"
    elif len(absent_on_candidate) == len(rows):
        state = "absent_only_candidate"
    else:
        state = "absent_only_mixed"
    counts = {"not_identical": len(rows), "differing": len(differing),
              "absent": len(absent_on_candidate),
              "absent_on_shipped": len(absent_on_shipped)}
    held_out = [_held_out_row(row) for row in (evidence or {}).get("held_out") or []]
    # A block both artifacts report and report the same size for. Anything else
    # is either uneven or one sided, and neither is comparable.
    comparable = sum(1 for row in held_out if row.get("comparable"))
    uneven = [str(row.get("block")) for row in held_out if not row.get("comparable")
              and not row.get("shipped_absent") and not row.get("candidate_absent")]
    one_sided = [str(row.get("block")) for row in held_out
                 if row.get("shipped_absent") or row.get("candidate_absent")]
    held_out_state = ("none" if not held_out else
                      "even" if comparable == len(held_out) else "uneven")
    return {
        **counts,
        "state": state,
        **_reading(_reading_key(state, counts), counts),
        "rows": rows,
        "differing_keys": [str(row.get("key")) for row in differing],
        "held_out": held_out,
        "held_out_blocks": len(held_out),
        "held_out_comparable": comparable,
        "held_out_uneven": uneven,
        "held_out_one_sided": one_sided,
        "held_out_state": held_out_state,
        "held_out_rule_en": HELD_OUT_RULE["en"],
        "held_out_rule_he": HELD_OUT_RULE["he"],
        "held_out_basis_en": HELD_OUT_STATE[held_out_state]["en"],
        "held_out_basis_he": HELD_OUT_STATE[held_out_state]["he"],
    }


def render_summary(summary: dict[str, Any]) -> list[str]:
    """The gate reading at a terminal, under the table that ranks the artifacts."""
    if not summary:
        return []
    lines = [f"  {'':20s} {summary['reading_en']}"]
    for row in summary.get("rows") or []:
        if row.get("shipped_absent") or row.get("candidate_absent"):
            continue
        lines.append(f"  {'':20s} {row['key']}: shipped "
                     f"{gate_cell(row['shipped'], False)}, candidate "
                     f"{gate_cell(row['candidate'], False)}")
    uneven = (summary.get("held_out_uneven") or []) + (summary.get("held_out_one_sided") or [])
    if uneven:
        lines.append(f"  {'':20s} held-out amounts not comparable on: {', '.join(uneven)}")
    return lines
