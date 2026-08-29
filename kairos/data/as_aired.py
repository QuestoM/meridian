"""The commercial pods the channel actually aired, as the channel itself counted them.

Everything else in this engine infers where one ad break ends and the next
begins, by grouping spots that fall within some seconds of each other. That
inference is a guess, and it is a load-bearing one: change the gap threshold and
the answer to "how many breaks were there" moves enough to move the revenue
figure by millions.

The guess is unnecessary. The as-run log already carries the broadcaster's own
numbering, in two columns nothing in this codebase read before: ``Pos. Block 1``
is a spot's position inside its block, and ``Spots Block 1`` is how many spots
that block holds. Restarting the counter reconstructs the blocks exactly --
measured on one month of the operator channel, 829 pods, agreeing with the
channel's own reported block size on 100 percent of them.

Two facts fall straight out of that, and both matter more than the reconstruction:

- Only rows whose ``Spot type`` is פרסומת carry a block number at all. Promos,
  sponsorships and public-service announcements never do. The channel is telling
  us directly which airtime it sold and which it did not, so an inferred break
  that sweeps in promos is not a break at all.
- Restricted to those commercial rows, the gap threshold barely matters: five
  seconds and five hundred seconds land within one percent of each other, and
  both within about five percent of this ground truth. The large sensitivity
  people find when they probe the threshold comes from reconstructing over ALL
  spot rows, which counts unsold airtime as sold.

This module reads the ground truth and nothing else. It makes no pricing
decision and no planning decision; it is the measuring stick the rest of the
engine is checked against.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# The channel numbers a spot's place inside its block, and states the block's
# size. A block starts wherever that position counter restarts.
POSITION_COLUMN = "Pos. Block 1"
BLOCK_SIZE_COLUMN = "Spots Block 1"

# The spot type the channel gives a block number to. Sponsorships (חסות),
# promos (פרומו) and PSAs (תשדיר שרות) never carry one, which is the channel
# saying they are not part of a sold commercial pod.
COMMERCIAL_SPOT_TYPE = "פרסומת"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce")


def commercial_spots(spots: pd.DataFrame) -> pd.DataFrame:
    """The rows the channel itself counted into a commercial block.

    Selection is on the block number rather than on ``Spot type``, so a row the
    channel numbered is kept even if its type label is unexpected, and a row it
    did not number is dropped even if the label looks commercial. The channel's
    own bookkeeping is the authority here, not our reading of a text field.
    """
    if spots.empty or BLOCK_SIZE_COLUMN not in spots.columns:
        return spots.iloc[0:0].copy()
    size = _numeric(spots, BLOCK_SIZE_COLUMN)
    return spots[size.fillna(0) > 0].copy()


def identify_aired_pods(spots: pd.DataFrame, *, channel: Optional[str] = None) -> pd.DataFrame:
    """One row per commercial pod the channel aired, from its own numbering.

    ``spots`` is the as-run frame from :func:`kairos.data.loaders.load_spots`,
    which must carry ``air_dt``. Pass ``channel`` to restrict to one broadcaster.

    Returns columns ``channel, day, hour, start, seconds, spots, tvr``, where
    ``seconds`` is the pod's total spot duration and ``tvr`` the mean rating
    across its spots. An empty or unnumbered input returns an empty frame with
    those columns rather than raising, because a source without the block
    columns is a real state (an older export) and not a programming error.
    """
    empty = pd.DataFrame(columns=["channel", "day", "hour", "start", "seconds", "spots", "tvr"])
    if spots is None or spots.empty:
        return empty
    frame = spots
    if channel is not None:
        frame = frame[frame["Channel"].astype(str).str.strip() == str(channel).strip()]
    frame = commercial_spots(frame)
    if frame.empty or "air_dt" not in frame.columns:
        return empty

    # Sort WITHIN a channel before reading the counter. Channels broadcast
    # concurrently, so a frame ordered by time alone interleaves their rows and
    # the position counter restarts constantly - measured on the real month,
    # that mistake reported 1,387 pods on the operator channel instead of 829,
    # with a median of 84 seconds instead of 181. Every pod would have been
    # shorter and more numerous than reality, which is the exact error this
    # module exists to correct.
    frame = frame.sort_values(["Channel", "air_dt"]).reset_index(drop=True)
    position = _numeric(frame, POSITION_COLUMN)
    # A pod begins at position 1, and also wherever the counter goes backwards:
    # two adjacent pods whose numbering happens not to restart at 1 (a log that
    # begins mid-pod) must still be separated rather than silently merged.
    starts = (position == 1) | (position < position.shift(1).fillna(position.max() + 1))
    # A change of channel always begins a new pod, whatever the counter says.
    starts = starts | (frame["Channel"].astype(str) != frame["Channel"].astype(str).shift(1))
    frame["_pod"] = starts.cumsum()

    duration = _numeric(frame, "Duration")
    rating = _numeric(frame, "TVR")
    grouped = frame.assign(_duration=duration, _tvr=rating).groupby("_pod", sort=True)
    pods = pd.DataFrame({
        "channel": grouped["Channel"].first().astype(str).str.strip(),
        "start": grouped["air_dt"].min(),
        "seconds": grouped["_duration"].sum(),
        "spots": grouped.size(),
        "tvr": grouped["_tvr"].mean(),
    }).reset_index(drop=True)
    pods["day"] = pods["start"].dt.strftime("%Y-%m-%d")
    pods["hour"] = pods["start"].dt.hour
    return pods[["channel", "day", "hour", "start", "seconds", "spots", "tvr"]]


def reconstruction_agreement(spots: pd.DataFrame, *, channel: Optional[str] = None) -> dict:
    """How far the reconstruction agrees with the size the channel reported.

    The channel states each block's size on every row of that block. Grouping by
    the position counter and counting the rows must reproduce it. This returns
    the share that agree, so the ground truth is checked rather than trusted --
    a source whose numbering does not round-trip is a source to stop using, not
    one to quietly build a revenue figure on.
    """
    frame = spots
    if channel is not None:
        frame = frame[frame["Channel"].astype(str).str.strip() == str(channel).strip()]
    frame = commercial_spots(frame)
    if frame.empty:
        return {"pods": 0, "agreeing": 0, "agreement": None}
    frame = frame.sort_values(["Channel", "air_dt"]).reset_index(drop=True)
    position = _numeric(frame, POSITION_COLUMN)
    starts = (position == 1) | (position < position.shift(1).fillna(position.max() + 1))
    starts = starts | (frame["Channel"].astype(str) != frame["Channel"].astype(str).shift(1))
    frame["_pod"] = starts.cumsum()
    grouped = frame.groupby("_pod")
    counted = grouped.size()
    reported = grouped[BLOCK_SIZE_COLUMN].first().astype(float)
    agreeing = int((counted == reported).sum())
    return {
        "pods": int(len(counted)),
        "agreeing": agreeing,
        "agreement": round(agreeing / len(counted), 4) if len(counted) else None,
    }


def hourly_ad_load(pods: pd.DataFrame) -> pd.DataFrame:
    """Advertising seconds and pod count per clock hour the channel actually aired.

    This is what a regulatory cap is measured against, and what any comparison
    between a plan and reality has to hold fixed. A plan carrying more airtime
    than this is not a better plan on the same inventory; it is a different and
    larger inventory, and saying so is the difference between a real result and
    a arithmetic artefact.
    """
    if pods.empty:
        return pd.DataFrame(columns=["channel", "day", "hour", "seconds", "pods"])
    grouped = pods.groupby(["channel", "day", "hour"], sort=True)
    out = pd.DataFrame({
        "seconds": grouped["seconds"].sum(),
        "pods": grouped.size(),
    }).reset_index()
    return out
