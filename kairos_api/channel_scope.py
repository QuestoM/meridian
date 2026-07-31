"""The competitor boundary: the operator owns exactly one channel.

The saved weekly plan and the EPG both carry every channel, because the
retention model is measured against the competitive lineup. Nothing that
reaches an operator surface may. This module is the one place that turns "the
whole market" into "the channel this operator owns", so a route adopts the
boundary in one line instead of re-deriving it.

Measured on the reference data before this module existed, with the operator
channel set to ``רשת 13``: ``GET /api/schedule`` returned a 200-row
``break_schedule`` of which 3 rows were the operator's own and 197 belonged to
three competitors (96, 73 and 28), plus a ``rows`` canvas of 1,852 programmes
of which 1,328 were competitors'. ``GET /api/break-operations`` returned 12
programmes for each of the four channels. Those routes belong to other pieces;
this module is what they call.

Two forms, because there are two honest answers:

- ``scope_frame`` and ``scope_records`` drop every row that is not the
  operator's. This is the form every operator surface uses.
- ``competitor_aggregate`` keeps the competitive fact and destroys the
  identity: a count of channels and rows, and sums of whichever numeric
  columns the caller asks for, with no channel name anywhere in the result.
  This is the only form in which a competitor may reach a payload, and it is
  what the model's competitor factor needs.

Every scoping call also returns a note. The note is the disclosure the surface
prints beside its numbers: which channel was summed, how many rows went in and
came out, and, when no channel is configured yet, the reason the scope could
not be applied. A surface that cannot scope says so; it never quietly serves
the market total as if it were the operator's.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "NO_OPERATOR_CHANNEL_REASON",
    "competitor_aggregate",
    "operator_channel",
    "scope_frame",
    "scope_note",
    "scope_records",
]

NO_OPERATOR_CHANNEL_REASON = "operator channel is not configured in settings"

# The column and key names the plan, the EPG and the payloads use for a channel.
FRAME_CHANNEL_COLUMN = "channel"
EPG_CHANNEL_COLUMN = "Channel"


def operator_channel(settings: Any = None) -> str:
    """The one channel the operator owns, or an empty string when unset.

    Reads ``operator_channel`` off the saved settings by default, and off the
    settings object or mapping the caller already loaded when it passes one, so
    a route that has settings in hand does not read the file twice.
    """
    if settings is None:
        from kairos_api.core import _load_settings

        settings = _load_settings()
    if isinstance(settings, Mapping):
        value = settings.get("operator_channel", "")
    else:
        value = getattr(settings, "operator_channel", "")
    return str(value or "").strip()


def _texts(values: Iterable[Any]) -> list[str]:
    return [str(value or "").strip() for value in values]


def scope_note(
    channel: str,
    rows_in: int,
    rows_out: int,
    channels_in: int,
    scoped: bool,
) -> dict[str, Any]:
    """The disclosure block that travels with every scoped payload.

    The two excluded counts are what the scope actually removed, so they are
    both zero on the pass-through path: nothing was dropped there, and the
    reason field is what says the boundary could not be applied.
    """
    owned_present = 1 if (scoped and rows_out) else 0
    return {
        "scope_channel": channel or None,
        "scoped": scoped,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "channels_in": int(channels_in),
        "competitor_rows_excluded": int(rows_in - rows_out) if scoped else 0,
        "competitor_channels_excluded": max(0, int(channels_in) - owned_present) if scoped else 0,
        "reason": None if scoped else NO_OPERATOR_CHANNEL_REASON,
    }


def scope_records(
    records: Sequence[Mapping[str, Any]],
    key: str = FRAME_CHANNEL_COLUMN,
    channel: Optional[str] = None,
    settings: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only the operator's rows in a list of payload records.

    Returns ``(rows, note)``. With no configured channel the rows pass through
    unchanged and the note says why, because inventing a scope would be a
    guess; the caller decides whether to serve or to refuse, and the note makes
    either one honest.
    """
    owned = operator_channel(settings) if channel is None else str(channel or "").strip()
    values = _texts(record.get(key) for record in records)
    channels_in = len({value for value in values if value})
    if not owned:
        return (
            [dict(record) for record in records],
            scope_note("", len(records), len(records), channels_in, scoped=False),
        )
    kept = [dict(record) for record, value in zip(records, values) if value == owned]
    return kept, scope_note(owned, len(records), len(kept), channels_in, scoped=True)


def scope_frame(
    frame: Any,
    column: str = FRAME_CHANNEL_COLUMN,
    channel: Optional[str] = None,
    settings: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Keep only the operator's rows in a pandas frame.

    Returns ``(frame, note)``. An empty frame, a missing channel column or an
    unconfigured operator channel all pass the frame through with a note that
    names what stopped the scope. The frame is never copied when nothing was
    dropped, so this is free on the pass-through path.
    """
    owned = operator_channel(settings) if channel is None else str(channel or "").strip()
    rows_in = 0 if frame is None else int(len(frame))
    columns = list(getattr(frame, "columns", []))
    if frame is None or rows_in == 0:
        return frame, scope_note(owned, rows_in, rows_in, 0, scoped=bool(owned))
    if column not in columns:
        note = scope_note("", rows_in, rows_in, 0, scoped=False)
        note["reason"] = f"the frame has no {column} column"
        return frame, note
    values = frame[column].astype(str).str.strip()
    channels_in = int(values[values.ne("")].nunique())
    if not owned:
        return frame, scope_note("", rows_in, rows_in, channels_in, scoped=False)
    scoped = frame[values == owned]
    return scoped, scope_note(owned, rows_in, int(len(scoped)), channels_in, scoped=True)


def competitor_aggregate(
    records: Sequence[Mapping[str, Any]],
    key: str = FRAME_CHANNEL_COLUMN,
    sum_fields: Sequence[str] = (),
    channel: Optional[str] = None,
    settings: Any = None,
) -> dict[str, Any]:
    """The unnamed aggregate: the competitive fact without the identity.

    ``{channels, rows, totals}`` over every row that is not the operator's, and
    nothing else. No channel name, no per-channel breakdown and no ranking, so
    a single-competitor market cannot be de-anonymised by subtraction of a
    named part. This is the only shape in which a competitor number may reach
    a payload, and it exists because the retention model genuinely needs the
    lineup as a factor.
    """
    owned = operator_channel(settings) if channel is None else str(channel or "").strip()
    values = _texts(record.get(key) for record in records)
    rival_rows = [
        record for record, value in zip(records, values) if value and value != owned
    ]
    rival_channels = len({value for value in values if value and value != owned})
    totals: dict[str, float] = {}
    for name in sum_fields:
        total = 0.0
        for record in rival_rows:
            try:
                total += float(record.get(name) or 0)
            except (TypeError, ValueError):
                continue
        totals[name] = total
    return {"channels": rival_channels, "rows": len(rival_rows), "totals": totals}
