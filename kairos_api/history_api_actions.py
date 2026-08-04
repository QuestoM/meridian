"""What a recorded request was: its act, its kind, and where its output lands.

Split out of ``history_api_timeline.py`` to keep that module under the
file-size cap, and it earns its own file for a second reason: this is the one
place the product decides what a mutating request actually did, and both
History and the settings activity log read this same decision rather than each
matching on paths of their own.

Four questions are answered here and nothing else is.

**What act was it.** The recorder stores the concrete path, so the map matches
on a prefix and an optional suffix rather than on a route template.

**Did it save anything.** A large share of the recorded verbs compute an answer
and write nothing, and calling those changes would be wrong about the one thing
History exists to answer.

**Did it land.** The recorder has stored the answer's status code since the log
existed and nothing read it, so a write the server refused was printed with the
same sentence as one it carried out.

**Where did its output land.** That is the training test of specification
section 4.1 as a data field: ``models`` means training, and training is
company-only.
"""

from __future__ import annotations

from typing import Any, Optional


# The closed vocabulary of what can appear on the timeline. A surface renders a
# word per kind and a critic can enumerate the set, which is the Google Ads
# device: a small status vocabulary, every member actionable.
#
# ``preview`` is a kind rather than a footnote because the request recorder
# records every mutating verb, and a large share of them save nothing. Measured
# on this deployment while wave one was building: of 345 recorded requests in
# the newest 500 entries, 57 were the day board scoring a placement nobody had
# saved yet. A timeline that calls those changes is wrong about the one thing it
# exists to answer.
KINDS = ("change", "preview", "run", "restore_point", "restore", "sign_in")

# Where an act's output lands. This is the training test of section 4.1 as a
# data field: models means training, and training is company-only.
ARTIFACT_ROOTS = ("data", "output", "models")

# Paths whose writes produce a model artifact. The model console publishes four
# of them today (measured on the live surface: /api/model/training,
# /api/model/versions, /api/model/decisions and
# /api/model/candidates/{id}/measure), and every one is filtered out of a
# channel account's timeline by :func:`visible`.
_MODEL_PATH_PREFIXES = ("/api/model", "/api/training", "/api/candidates")

# A mutating request, in the operator's own terms: (method, prefix, suffix,
# code). ``*`` matches any method and an empty suffix matches any path. The
# first match wins, so a specific row sits above the family it belongs to, which
# is why /api/uploads/{kind}/check precedes /api/uploads.
#
# The map was written against the 87 write operations the assembled product
# publishes, enumerated from its own openapi.json rather than guessed, and
# ``tests/test_p8_history.py`` pins every one of them so a route that nobody
# classified shows up as a failing test rather than as the word "Change" over
# something that changed nothing.
_ACTIONS: tuple[tuple[str, str, str, str], ...] = (
    # Computed an answer, saved nothing.
    ("POST", "/api/scenario", "", "preview"),
    ("POST", "/api/scenario-compare", "", "preview"),
    ("POST", "/api/optimizer-plan", "", "preview"),
    ("POST", "/api/optimal-plan", "", "preview"),
    ("POST", "/api/plan/day/score", "", "placement_preview"),
    ("POST", "/api/pricing/effect", "", "price_preview"),
    ("POST", "/api/pricing/price-slot", "", "price_test"),
    ("POST", "/api/constraints/restrictions/preview", "", "restriction_preview"),
    ("POST", "/api/uploads", "/check", "source_check"),
    ("POST", "/api/assistant/context/warm", "", "assistant_context"),
    ("POST", "/api/assistant/ask", "", "assistant_ask"),
    # Saved something.
    ("PUT", "/api/settings", "", "settings_change"),
    ("PUT", "/api/pricing", "", "pricing_change"),
    ("POST", "/api/recompute-schedule", "", "plan_run"),
    ("POST", "/api/jobs/recompute", "", "plan_run"),
    ("*", "/api/plan-target", "", "target_change"),
    ("POST", "/api/plan-versions", "/restore", "plan_restore"),
    ("*", "/api/plan-versions", "", "plan_publish"),
    ("*", "/api/breaks", "/gold", "gold_change"),
    ("*", "/api/breaks", "/placement", "placement_change"),
    ("*", "/api/breaks", "", "break_change"),
    ("*", "/api/constraints", "", "restriction_change"),
    ("*", "/api/overrides", "", "override_change"),
    ("*", "/api/rules/guardrails", "", "guardrail_change"),
    ("PUT", "/api/rules/model-activation", "", "model_activation_change"),
    ("PUT", "/api/rules/operator-channel", "", "channel_change"),
    ("POST", "/api/clients/onboarding", "", "client_onboarding"),
    ("*", "/api/clients/campaigns", "", "campaign_change"),
    ("*", "/api/advertisers", "", "client_change"),
    ("*", "/api/advertiser-conditions", "", "client_change"),
    ("*", "/api/agencies", "", "client_change"),
    ("*", "/api/agency-advertisers", "", "client_change"),
    ("*", "/api/agency-conditions", "", "client_change"),
    ("*", "/api/events", "", "calendar_change"),
    ("*", "/api/uploads", "", "source_upload"),
    ("POST", "/api/versions/snapshot", "", "restore_point_saved"),
    # The rename control this destination itself ships is a PATCH on a version,
    # and it sat under the family row below, so it was recorded and printed as a
    # restore. Measured live on 2026-08-01: renaming a point at 23:24 put
    # "Restore applied | PATCH /api/versions/a3bd7ff7f743 | 200" on the Change
    # tab, and a compliance owner reading that tab counted a restore that never
    # happened. Naming a point and putting one back are different acts.
    ("PATCH", "/api/versions", "", "restore_point_renamed"),
    ("*", "/api/versions", "", "restore"),
    ("*", "/api/assistant/proposals", "", "assistant_action"),
    ("POST", "/api/assistant/restore", "", "assistant_undo"),
    ("*", "/api/assistant/conversations", "", "conversation_change"),
    ("*", "/api/assistant/thread", "", "conversation_change"),
    ("*", "/api/assistant/upload", "", "assistant_upload"),
    ("*", "/api/assistant/uploads", "", "assistant_upload"),
    ("PUT", "/api/auth/job", "", "job_change"),
    ("POST", "/api/auth/change-password", "", "password_change"),
    ("*", "/api/auth", "", "account_change"),
    ("POST", "/api/break-decisions", "", "decision"),
    # Training. Every one of these is filtered out of a channel account's
    # timeline by the artifact root, so these words never reach the other side.
    ("POST", "/api/model/training", "", "model_training"),
    ("POST", "/api/model/versions", "", "model_version"),
    ("POST", "/api/model/decisions", "", "model_decision"),
    ("POST", "/api/model/candidates", "/measure", "candidate_measure"),
)

# The acts that compute an answer and write nothing. They are recorded because
# the recorder records every mutating verb, and they are told apart here because
# "nothing was saved" is a different piece of news from "something was saved".
PREVIEW_ACTIONS = frozenset({
    "preview",
    "placement_preview",
    "price_preview",
    "price_test",
    "restriction_preview",
    "source_check",
    "assistant_context",
    "assistant_ask",
})


def artifact_root(path: Optional[str]) -> str:
    """Where a request's write lands: ``data`` or ``models``.

    The training test, applied to an HTTP path. Everything the product publishes
    today writes ``data/`` or ``output/``; the model prefixes are the door the
    first training route will come through, and it will be walled the day it
    does.
    """
    text = str(path or "")
    if any(text.startswith(prefix) for prefix in _MODEL_PATH_PREFIXES):
        return "models"
    return "data"


def action_for(method: Optional[str], path: Optional[str]) -> str:
    """The closed action code for one mutating request, or ``other``.

    The recorder stores the concrete path, so this matches on a prefix and an
    optional suffix rather than on a route template: a placement act arrives as
    ``/api/breaks/2024-11-01|<channel>|000~1/placement`` and has to be told from
    a gold act on the same break.
    """
    verb = str(method or "").upper()
    text = str(path or "")
    for wanted_method, prefix, suffix, code in _ACTIONS:
        if wanted_method not in ("*", verb):
            continue
        if not (text == prefix or text.startswith(prefix + "/")):
            continue
        if suffix and not text.endswith(suffix):
            continue
        return code
    return "other"


def kind_for(action: str) -> str:
    """Which timeline kind an action code belongs to.

    An act that saved nothing is a ``preview``, whatever verb it arrived on.
    Everything else the recorder holds is a ``change``, including a refused one:
    a write that was attempted and answered 403 is exactly what somebody reading
    this surface needs to see.

    The kind says what was attempted. Whether it landed is :func:`outcome_for`,
    and that is the half no count may skip: a refused attempt belongs on the
    list and belongs in no figure that attests.
    """
    return "preview" if action in PREVIEW_ACTIONS else "change"


# Whether the write actually happened, from the status code the recorder has
# stored on every line since this log existed.
#
# **This is the half nothing read.** The act was derived from the method and the
# path alone, so a refused write carried the sentence of an accomplished one:
# measured by a blind critic on 2026-08-02, 680 of the 2,264 change entries on
# the record (30.0 percent) answered 400 or more, four consecutive rows read
# "the regulatory limit was saved" at the same minute with two of them refused,
# and the compliance strip counted every one of them as a change. Re-measured on
# this repository's own recorder at 00:33 on 2026-08-04: 3,263 recorded
# requests, 811 of them refused (24.9 percent), 528 of those 403, and not one
# line without a status.
#
# Tri-state, because "the server refused it" and "nobody recorded what happened"
# are different pieces of news and only one of them can be attested to.
OUTCOMES = ("applied", "refused", "unknown")
APPLIED, REFUSED, OUTCOME_UNKNOWN = OUTCOMES

# The line between an answer that could have written and one that cannot have.
# A 4xx is the server declining before it wrote: 401 and 403 are the wall, 404
# and 405 mean the route does not exist, 409 and 422 mean the request could not
# be carried out. Not one of them wrote a byte.
REFUSED_FROM = 400

# And the line past which the product stops claiming anything. A 5xx is the
# server failing rather than declining, and a failure can land after a write has
# begun, so calling it a refusal would be the same certainty this module exists
# to remove, pointed the other way. In the same 3,263 recorded requests there is
# not one 5xx and not one line without a status, so this names a state the store
# does not hold today rather than guessing at it the day it appears.
SERVER_ERROR_FROM = 500


def outcome_for(status: Any) -> str:
    """What one recorded request did: ``applied``, ``refused`` or ``unknown``.

    A status the recorder never wrote, one outside the HTTP range, and a server
    failure are all unknown rather than either of the answers a reader could act
    on. The store holds no line of any of the three today, which is exactly why
    the state is carried: the day one appears it must not read as a change that
    happened, and it must not read as a refusal either.
    """
    try:
        code = int(status)
    except (TypeError, ValueError):
        return OUTCOME_UNKNOWN
    if code >= SERVER_ERROR_FROM:
        return OUTCOME_UNKNOWN
    if code >= REFUSED_FROM:
        return REFUSED
    return APPLIED if code >= 100 else OUTCOME_UNKNOWN
