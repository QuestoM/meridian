"""The competitor boundary applied to the NAMES this destination echoes back.

Split out of ``uploads.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule. :mod:`kairos_api.channel_scope` is the
boundary for rows, ``uploads_preview._hide_rival_columns`` applies it to the
columns of a preview, and this is the same boundary for every other place this
destination hands a file's own names back to the operator.

Three of those places, and all three were measured leaking with the operator
channel set to ``רשת 13``:

- the refusal a dayparts file with no recognisable channel column receives. It
  named the four channels the loader knows, and ``SourceCard`` renders a
  finding's message verbatim in its red panel, so a plausibly re-exported file
  put three rival channel names on the operator's screen.
- the header list on ``GET /api/uploads/status``. The dayparts input carried
  its real header, which is one column per channel, so three names this account
  does not own were in the payload the browser and the assistant's own read
  tool both receive.
- the header list the door and the commit answer with, which is the header of
  the file the operator just handed over.

One rule: a name that carries a channel this account may not own is withheld
and counted, never printed. With no operator channel configured every channel
is a name this account may not own, which is the stance the preview already
takes for rows: the boundary is not "hide the three rivals", it is "never
render a channel this account may not own".

The count is the disclosure, and it is what keeps the arithmetic honest. A
withheld name subtracts from nothing: the count travels beside the names in the
same record, and the card adds the two together to print how many columns the
file really has, so hiding a name never moves a number.
"""

from __future__ import annotations

from typing import Any, Iterable

from kairos.data.loaders import CHANNELS
from kairos_api import channel_scope

__all__ = ["columns_record", "owned_channel", "rivals", "withhold"]


def owned_channel(settings: Any = None) -> str:
    """The one channel this operator owns, or an empty string when unset."""
    return channel_scope.operator_channel(settings)


def rivals(owned: str) -> list[str]:
    """Every channel name that may not reach this operator's screen.

    With no configured channel that is all of them, because there is then no
    way to tell which of the names is this account's own.
    """
    return [name for name in CHANNELS if name != str(owned or "").strip()]


def withhold(names: Iterable[Any], owned: str) -> tuple[list[str], int]:
    """``(the names the operator may read, how many were withheld)``.

    A name is withheld when a rival channel's name is anywhere inside it, not
    only when it equals one: a re-export that renamed a column to a channel
    name with a stray space around it would otherwise carry that name through.
    """
    hidden = rivals(owned)
    given = [str(name) for name in names]
    shown = [name for name in given if not any(rival in name for rival in hidden)]
    return shown, len(given) - len(shown)


def columns_record(names: Iterable[Any], owned: str | None = None) -> dict[str, Any]:
    """The two column keys every payload here carries, as one record.

    ``owned`` is passed in by a caller that already resolved it, so a status
    that reports seven inputs reads the settings once rather than seven times.
    """
    shown, hidden = withhold(names, owned_channel() if owned is None else owned)
    return {"columns": shown, "columns_withheld": hidden}
