"""A file that carries its header and no data rows, per kind.

Split out of ``uploads_validate.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule the package already follows.

**The measured gap this module exists for.** A CSV carrying the real header and
zero data rows was accepted by ``POST /api/uploads/{kind}/check`` for six of the
seven kinds, with no finding, no warning and ``rows: 0``, and for three of those
six the same body said ``will_be_read: true`` with the consequence
``replaces_live_input``. So the card printed a green tick, ``0 שורות`` and
"this is the live input" over a commit button, and committing it really did
empty the live daily input: the state went from ``in_use`` with 175 rows to
``empty`` with none, and the only screen that said so said it after the click.
Only the dayparts kind refused, because it was the only one anybody had written
the rule for.

**Zero rows is not one fact, so it does not get one answer.** For a lineup, a
spot history, a table of advertiser rules or a rate card, a file with no rows is
an export that went wrong: there is no world in which the operator meant to
publish an empty one, so it is refused at the door with the reason. For the
daily log and the campaign flights, an empty file is a state that really occurs,
a broadcast day with nothing booked on it and a flight list nobody has filled in
yet, so the file is accepted and what it will do is named instead of assumed:
a warn-severity finding, and the consequence
``replaces_live_input_with_no_rows`` in place of the green one.

Nothing here writes a path, and nothing here authors a sentence or adds it to a
report on its own. The function that records a finding in both languages stays
in :mod:`kairos_api.uploads_validate` and is handed in, so one module keeps the
authoring and this one keeps the rule, and neither has to import the other.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from kairos.data.loaders import CHANNELS
from kairos_api import uploads_channels, uploads_messages, uploads_replay

# What a file with no rows means, one kind at a time. Every kind the door
# accepts is declared here and the door's own test sweeps that, so a kind added
# later cannot quietly inherit the hole this module closed.
SEVERITY: dict[str, str] = {
    "programmes": "error",
    "spots": "error",
    "dayparts": "error",
    "advertiser_rules": "error",
    "rate_card": "error",
    "daily": "warning",
    "campaign_flights": "warning",
}

# What one row of each kind IS, so the reason names the thing that is missing
# rather than saying "rows" at a person who knows perfectly well the file is
# empty. The dayparts kind is absent on purpose: its own sentence says what its
# rows are and why they came to zero, and that sentence is older than this one.
ROW_NOUN: dict[str, str] = {
    "programmes": "rows_of_programmes",
    "spots": "rows_of_spots",
    "advertiser_rules": "rows_of_advertiser_rules",
    "rate_card": "rows_of_rate_card",
    "daily": "rows_of_daily",
    "campaign_flights": "rows_of_campaign_flights",
}

# The copy table entry for each severity: the same code, said twice, because the
# outcome is not the same outcome. One says nothing can be computed from this
# input at all, the other says every figure from it will be empty.
KEYS = {"error": "no_data_rows_refused", "warning": "no_data_rows_accepted"}


def add_when_empty(
    add: Callable[..., None],
    kind: str,
    rows: int,
    raw_frame: pd.DataFrame,
    report: Any,
    authored: dict[tuple[str, str], dict[str, Any]],
    owned: str | None = None,
) -> None:
    """Raise this kind's no-rows finding when the engine gets nothing from the file.

    ``rows`` is what the engine really reads out of the file: the loader's own
    count for a kind that has one, and the CSV's data rows for a kind that has
    none. ``add`` is :func:`kairos_api.uploads_validate.add_finding`, handed in
    so the authoring of a finding stays in one module.
    """
    if rows > 0 or kind not in SEVERITY:
        return
    if kind == "dayparts":
        dayparts_empty_finding(add, raw_frame, report, authored, owned)
        return
    severity = SEVERITY[kind]
    add(
        report,
        authored,
        "",
        "no_data_rows",
        severity,
        KEYS[severity],
        scope="frame",
        rows_are=uploads_messages.say(ROW_NOUN[kind]),
    )


def dayparts_empty_finding(
    add: Callable[..., None],
    raw_frame: pd.DataFrame,
    report: Any,
    authored: dict[tuple[str, str], dict[str, Any]],
    owned: str | None = None,
) -> None:
    """Explain, in headers, why a dayparts upload melts to zero audience rows.

    The header gate only requires Dates+Timebands, but the loader melts ONLY the
    known channel columns; a file with renamed channel columns validates green
    yet yields nothing. Name the operator's own channel and the unrecognized
    headers, so the export can be fixed instead of a silent empty model chased.

    **Only the operator's own channel is named.** Measured before that
    correction, with the operator channel set to ``רשת 13``: this message listed
    all four channels the loader knows, and the card renders a finding's message
    verbatim in its red refusal panel, so a plausibly re-exported dayparts file
    put three rival channel names on the operator's own screen. The same
    sentence reaches the assistant, which reads the stored validation report
    through ``get_upload_status``. The unrecognized headers are still listed,
    minus any that carry a channel this account does not own, and that count is
    stated instead.

    **How many names the loader knows is stated.** Withholding a name is not a
    licence to withhold the contract: a person cannot fix a refused export if the
    product will not say what it accepts, and this refusal only fires when the
    operator's own column was renamed too, so the accepted shape is the
    actionable part. A count names nobody, which is the same disclosure the
    withheld column names already take.

    **The sentence is not what is stored**: the headers this refusal may list and
    the count of the ones it may not go to disk as :func:`uploads_replay.boundary`'s
    record, and both the wording and the arithmetic are derived again on every
    read against the channel the account reading it owns then.
    """
    if [c for c in CHANNELS if c in raw_frame.columns]:
        add(report, authored, "", "no_data_rows", "error", scope="frame")
        return
    owned = uploads_channels.owned_channel() if owned is None else str(owned or "").strip()
    bound = uploads_replay.boundary(
        [c for c in raw_frame.columns if str(c) not in CHANNELS and str(c) not in ("Dates", "Timebands")],
        owned,
    )
    key = uploads_replay.channel_key(owned)
    add(report, authored, "", "no_recognized_channel_columns", "error", key, bound, "header", **uploads_replay.channel_fields(bound, owned))
