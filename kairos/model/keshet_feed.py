"""One call, run daily: the rival's week, as this engine will actually read it.

This is the seam where the three pieces meet — the session that renews itself
(:mod:`kairos.model.kway_session`), the converter that speaks the existing
competitor contract (:mod:`kairos.model.keshet_epg`), and the refresh that
reports what moved and refuses to overwrite a schedule with a failure
(:mod:`kairos.model.keshet_refresh`). Nothing new is invented here; this is the
wiring, and the wiring is where the honest failures have to live.

WHAT IT REFUSES BEFORE IT STARTS
--------------------------------
The channel name is settled first, against the engine's own registry, and a
name that matches nothing stops the pull. That order is deliberate. A schedule
written under a name history does not use is not a small error that surfaces
later — it is a file that loads cleanly and contributes exactly zero to every
decision downstream, permanently and without a word. The same is true, for the
same reason, of filing the rival's schedule under the operator's OWN channel:
the counter-programming features drop the operator's channel from the rival
list, so a schedule stamped with it leaves no rivals at all and the adjustment
quietly becomes nothing.

WHAT IT DOES WHEN IT CANNOT
---------------------------
It says so, and keeps yesterday's schedule with its age. There is no state in
which this returns a plan-shaped success it did not earn: no session, a network
that is down, a shape that changed, a publication that came back empty — each
of them ends as ``refreshed: False`` with a reason and, when a person is needed,
the one step for them to take.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kairos.model import keshet_epg, keshet_refresh, kway_session

# Where the competitor contract lives, which is where the loader already looks.
DEFAULT_TARGET = Path("data/reference/CompetitorProgrammes.csv")

# The rival this feeds. Keshet is Reshet's largest competitor, and the name is
# resolved against the engine's registry rather than trusted as spelled here.
DEFAULT_CHANNEL = "קשת 12"


def known_channels() -> tuple[str, ...]:
    """The channel names this engine's own reference data uses."""
    from kairos.data.loaders import CHANNELS

    return tuple(CHANNELS)


def configured_operator_channel() -> str:
    """The channel this operator owns, as the rest of the product reads it.

    Asked of the settings rather than passed on a command line, so the check
    that a rival is not filed under the operator's own channel keeps working
    after somebody changes that setting and forgets this job exists. Optional:
    the model layer must stay importable without the API around it, so an
    absent settings store means the check is simply not made, and says so.
    """
    try:
        from kairos_api import channel_scope

        return channel_scope.operator_channel()
    except Exception:  # noqa: BLE001 - a missing settings store is not a failure here
        return ""


def pull(
    *,
    channel: str = DEFAULT_CHANNEL,
    operator_channel: Optional[str] = None,
    target: str | Path = DEFAULT_TARGET,
    history_dir: Optional[str | Path] = None,
    known: Optional[Iterable[str]] = None,
    allow_renew: bool = True,
) -> dict[str, Any]:
    """Refresh the rival's schedule, and report honestly either way.

    Returns the refresh result with the session's state folded in, so a caller
    that only reads ``refreshed`` still cannot mistake "we could not sign in"
    for "the rival changed nothing".
    """
    registry = tuple(known) if known is not None else known_channels()
    own_name = configured_operator_channel() if operator_channel is None else operator_channel
    try:
        resolved = keshet_epg.resolve_channel(channel, registry)
    except keshet_epg.UnknownChannel as exc:
        return {
            "refreshed": False,
            "reason": str(exc),
            "needs_human": True,
            "do_this": f"Name the competitor channel as one of: {', '.join(registry)}",
        }
    if own_name:
        try:
            own = keshet_epg.resolve_channel(own_name, registry)
        except keshet_epg.UnknownChannel:
            own = own_name
        if own == resolved:
            return {
                "refreshed": False,
                "reason": (
                    f"'{resolved}' is this operator's own channel, not a rival. A "
                    f"competitor schedule filed under it leaves no rivals at all, "
                    f"and every counter-programming adjustment silently becomes zero."
                ),
                "needs_human": True,
                "do_this": "Name the rival's channel, not the operator's own",
            }

    session, status = kway_session.current(allow_renew=allow_renew)
    if session is None:
        result = keshet_refresh.refresh(fetch=None, channel=resolved, target=target)
        result["reason"] = f"{status.get('reason', 'no session')} ({result['reason']})"
        result["session"] = status
        if status.get("do_this"):
            result["needs_human"] = True
            result["do_this"] = status["do_this"]
        return result

    result = keshet_refresh.refresh(
        fetch=lambda: kway_session.fetch_epg(session),
        channel=resolved,
        target=target,
        history_dir=history_dir,
    )
    result["session"] = status
    result["channel"] = resolved
    return result


def headline(result: dict[str, Any], locale: str = "he") -> str:
    """The line an operator reads before a run, including who is needed."""
    line = keshet_refresh.headline(result, locale)
    if result.get("needs_human") and result.get("do_this"):
        step = result["do_this"].strip().splitlines()[0]
        line += (f" נדרשת פעולה חד־פעמית: {step}" if locale == "he"
                 else f" One step is needed: {step}")
    return line


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--operator-channel", default=None,
                        help="defaults to the operator channel saved in settings")
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--history-dir", default=None)
    parser.add_argument("--no-renew", action="store_true",
                        help="use the stored session only; never open a browser")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = pull(
        channel=args.channel,
        operator_channel=args.operator_channel,
        target=args.target,
        history_dir=args.history_dir,
        allow_renew=not args.no_renew,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    else:
        print(headline(result))
    return 0 if result.get("refreshed") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
