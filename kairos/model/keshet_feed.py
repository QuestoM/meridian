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
# Absolute, derived from this file. A relative default works only when the
# process happens to start in the repository root, which a daily job scheduled
# by the operating system has no reason to do — and the failure would be a new
# empty schedule written somewhere nobody reads, while the real one quietly aged.
DEFAULT_TARGET = Path(__file__).resolve().parents[2] / "data" / "reference" / "CompetitorProgrammes.csv"

# The rival this feeds by default. Keshet is Reshet's largest competitor, and the
# name is resolved against the engine's registry rather than trusted as spelled
# here.
DEFAULT_CHANNEL = "קשת 12"

# Which channels this engine can actually pull, and from where.
#
# The optimizer reads the whole competitive lineup out of one file, so a rival
# with no source is not a different feature — it is a missing row in this table.
# A channel with no entry is refused BY NAME rather than pulled as an empty
# schedule, because an empty schedule for a channel that is broadcasting is the
# most expensive lie a plan can be told.
#
# Keshet stays on the licensed publication rather than moving to the free one.
# Measured, mako.co.il/AjaxPage?jspName=EPGResponse.jsp answers 200 with no
# account at all and returns the SAME 300 programmes in the SAME shape — the
# existing converter reads it unchanged, 300 in and 300 out. So the credential
# is not technically required. It is kept because the subscription is paid for,
# it carries fields the free feed does not, and swapping a licensed data path
# for the competitor's own website is a commercial decision and not a
# refactoring. The alternative is recorded so the choice can be made rather
# than discovered.
SOURCES = {
    "קשת 12": "kway",
    "כאן 11": "freetv",
    "רשת 13": "freetv",
    "עכשיו 14": "freetv",
}


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
    days: int = 8,
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

    source = SOURCES.get(resolved)
    if source is None:
        # An honest gap, named. Every rival matters to the optimizer and only one
        # of them publishes somewhere this engine can reach today; saying so is
        # the difference between a feature that is unfinished and a schedule
        # that is quietly missing a channel nobody thinks to look for.
        return {
            "refreshed": False,
            "reason": (
                f"no schedule source is wired for {resolved}. This engine can pull "
                f"{', '.join(sorted(SOURCES))} and no other channel yet."
            ),
            "reason_he": (
                f"אין מקור לוח שידורים מחובר עבור {resolved}. המערכת יודעת למשוך "
                f"את {', '.join(sorted(SOURCES))} בלבד."
            ),
            "kept_rows": len(keshet_refresh._read_rows(Path(target))),
            "channel": resolved,
            "sources_available": sorted(SOURCES),
        }

    if source == "freetv":
        from kairos.model import freetv_epg

        result = keshet_refresh.refresh(
            fetch=lambda: freetv_epg.fetch(resolved, days=days),
            convert=lambda records, *, channel: freetv_epg.to_contract_rows(
                records, channel=channel),
            channel=resolved,
            target=target,
            history_dir=history_dir,
        )
        result["channel"] = resolved
        result["source"] = "freetv"
        return result

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
        fetch=lambda: kway_session.fetch_epg(session),   # SOURCES[resolved] == "kway"
        channel=resolved,
        target=target,
        history_dir=history_dir,
    )
    result["session"] = status
    result["channel"] = resolved
    result["source"] = "kway"
    return result


def headline(result: dict[str, Any], locale: str = "he") -> str:
    """The line an operator reads before a run, including who is needed."""
    if result.get("sources_available") is not None:
        # A channel with no source at all is not a stale schedule, and calling it
        # one would send somebody looking for a network fault that is not there.
        key = "reason_he" if locale == "he" else "reason"
        return str(result.get(key) or result.get("reason") or "")
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
    parser.add_argument("--days", type=int, default=8,
                        help="how many broadcast days to pull, where the source is asked per day")
    parser.add_argument("--all", action="store_true",
                        help="pull every channel that has a source, skipping the operator's own")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    def one(channel: str) -> dict[str, Any]:
        return pull(
            channel=channel,
            operator_channel=args.operator_channel,
            target=args.target,
            history_dir=args.history_dir,
            allow_renew=not args.no_renew,
            days=args.days,
        )

    # Every rival, or one. The operator's own channel is skipped rather than
    # refused loudly here: asking for "every channel with a source" and being
    # told off about the one you own is noise, and the guard inside pull()
    # still refuses it if somebody names it directly.
    if args.all:
        own = (configured_operator_channel() if args.operator_channel is None
               else args.operator_channel)
        results = {channel: one(channel) for channel in sorted(SOURCES)
                   if channel != own}
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
        else:
            for channel, result in results.items():
                print(f"{channel}: {headline(result)}")
        return 0 if all(r.get("refreshed") for r in results.values()) else 1

    result = one(args.channel)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    else:
        print(headline(result))
    return 0 if result.get("refreshed") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
