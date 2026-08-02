"""What a recorded request was: its act, its kind, and where its output lands.

Split out of ``history_api_timeline.py`` to keep that module under the
file-size cap, and it earns its own file for a second reason: this is the one
place the product decides what a mutating request actually did, and both
History and the settings activity log read this same decision rather than each
matching on paths of their own.

Three questions are answered here and nothing else is.

**What act was it.** The recorder stores the concrete path, so the map matches
on a prefix and an optional suffix rather than on a route template.

**Did it save anything.** A large share of the recorded verbs compute an answer
and write nothing, and calling those changes would be wrong about the one thing
History exists to answer.

**Where did its output land.** That is the training test of specification
section 4.1 as a data field: ``models`` means training, and training is
company-only.
"""

from __future__ import annotations

from typing import Optional


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
    """
    return "preview" if action in PREVIEW_ACTIONS else "change"
