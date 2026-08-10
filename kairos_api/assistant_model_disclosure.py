"""The model-disclosure wall for Kai, applied per acting account.

Kai is a run surface. A run surface may never carry training content to a
channel-affiliated account: no gate verdict, no held-out delta, no drift
measurement, no coefficient. What a channel account gets instead is the release
note and the model version, which is the only training-authored text allowed to
cross the line, because the alternative is money moving with no legible cause.

Four read tools and three grounding sections carry training content today, so
they are the whole surface this module walls:

* ``get_audience_model`` and the ``audience_model`` section, whose gates carry
  per-family held-out deltas
* ``get_audience_stability``, whose payload is the coefficient level-drift
  monitor
* ``get_event_pipeline`` and the ``event_pipeline`` section, whose fourth stage
  is the event gate verdict
* ``get_model_adoption``, whose candidate shelf carries gate, held-out and
  coefficient deltas and recorded company adoption decisions
* ``model_state``, whose grounding snapshot carries the same company-side model
  evidence

The wall is applied at one chokepoint per surface: ``execute_read_tool`` for
tools and ``extend_with_keyword_sections`` for sections. Affiliation is the only
input, resolved through the same ``actor_is_company`` helper the event write
gate already uses, so there is one answer to "is this account company" in the
process.

The release note is authored on the training side. Until that store exists this
module reports it as unknown with the path that would supply it, which is the
tri-state rule: real, unavailable, unknown, never a confident guess.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The lexicon a critic greps a run surface's responses for, from section 4.2 of
# the rebuild specification.
TRAINING_LEXICON = (
    "gate",
    "held_out",
    "held-out",
    "tau",
    "drift",
    "coefficient",
    "pooling",
    "p_value",
    "training_window",
    "wartime",
)

# The subset that is a model internal in itself, which is the line this wall
# actually enforces. The difference is one word and it is deliberate: a payload
# may still name "coefficients" as the NAME of a changed input group, because
# the operator's own staleness banner already prints exactly that
# (shell/ScheduleStalenessBanner.jsx:39, "the model's learned values
# (coefficients)"), and P1's Bar 3 floor requires that banner to keep naming
# what changed. A negative statement, "the price layer does not touch the
# coefficients", is likewise a disclosure of nothing. What may never cross is a
# gate verdict, a held-out delta, a drift measurement or a coefficient value.
MODEL_INTERNALS_LEXICON = tuple(word for word in TRAINING_LEXICON if word != "coefficient")

# The read tools whose payload is training content in whole or in part.
WALLED_READ_TOOLS = frozenset({
    "get_audience_model",
    "get_audience_stability",
    "get_event_pipeline",
    "get_model_adoption",
})

# The grounding sections with the same content.
WALLED_SECTIONS = frozenset({"audience_model", "event_pipeline", "model_state"})

RELEASE_NOTE_PATH = "models/releases/"

WITHHELD_REASON = (
    "Model internals are company-side and are not shown on an operator surface. "
    "What an operator sees instead is the release note and the model version."
)
WITHHELD_REASON_HE = (
    "פנימיות המודל שמורות לצוות החברה ואינן מוצגות במסך תפעולי. "
    "מה שהמפעיל רואה במקומן הוא הודעת הגרסה ומספר הגרסה."
)

# The operator-safe replacement for the event pipeline's fourth stage. It states
# the same operational truth (recording an event changes no number by itself,
# and nobody may assert a retention effect) without naming a verdict.
EVENT_STAGE_FOUR_OPERATOR_HE = (
    "שלב 4: השפעת האירוע על השימור נמדדת בצד החברה ואינה נקבעת ידנית; עד שהמדידה קובעת אחרת, רישום האירוע ותמחורו אינם משנים את השימור החזוי"
)


def actor_is_company(user: "str | None") -> bool:
    """Whether the acting account may see model internals.

    Delegates to the event pipeline's affiliation helper, which is the single
    implementation in the process: company when the affiliation field says so
    or when authentication is disabled, and never on a broken checker.
    """
    from kairos_api.assistant_event_pipeline import actor_is_company as _is_company

    return _is_company(user)


def _model_version() -> dict[str, Any]:
    """The shipped model version, tri-state.

    ``real`` when the audience artifact carries a computed_at stamp, ``unknown``
    with the reason when it does not. Never a guess.
    """
    try:
        from kairos_api.assistant_audience_model import audience_model_summary

        summary = audience_model_summary()
    except Exception:  # noqa: BLE001 - an unreadable artifact is an honest unknown
        logger.exception("model version lookup failed")
        return {"state": "unknown", "reason": "the model artifact could not be read"}
    computed_at = summary.get("computed_at")
    if not summary.get("available") or not computed_at:
        return {"state": "unknown",
                "reason": str(summary.get("reason") or "the model artifact has not been built")}
    return {"state": "real", "computed_at": str(computed_at)}


def _release_note() -> dict[str, Any]:
    """The release note for the shipped model version, tri-state.

    The note is authored on the training side when a version ships. This reads
    the store when it exists and reports an honest unknown, naming the path that
    would supply it, when it does not. Nothing is invented.
    """
    try:
        from kairos_api import model_version_store  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - the store is a separate piece and may not exist yet
        return {
            "state": "unknown",
            "reason": "no release note has been published for the shipped model version",
            "supplied_by": RELEASE_NOTE_PATH,
        }
    reader = getattr(model_version_store, "current_release_note", None)
    if not callable(reader):
        return {
            "state": "unknown",
            "reason": "the release-note store carries no reader for the current version",
            "supplied_by": RELEASE_NOTE_PATH,
        }
    try:
        note = reader()
    except Exception:  # noqa: BLE001 - a failing store is unavailable, never a guess
        logger.exception("release note lookup failed")
        return {"state": "unavailable", "reason": "the release-note store could not be read",
                "supplied_by": RELEASE_NOTE_PATH}
    text = str(note or "").strip()
    if not text:
        return {
            "state": "unknown",
            "reason": "no release note has been published for the shipped model version",
            "supplied_by": RELEASE_NOTE_PATH,
        }
    return {"state": "real", "text": text}


def operator_model_note() -> dict[str, Any]:
    """What an operator surface may say about the model: version plus release
    note, each tri-state, plus the reason the rest is withheld.

    The provenance source rides along already set, because the caller stamps a
    default one and every default for a walled tool names the artifact the wall
    just refused to open.
    """
    return {
        "model_version": _model_version(),
        "release_note": _release_note(),
        "withheld": WITHHELD_REASON,
        "withheld_he": WITHHELD_REASON_HE,
        "source": "the shipped model version and its release note",
    }


def _walled_event_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """The event pipeline with its measured-gate stage replaced.

    Every other stage is run-side and stays verbatim: the events store, the
    operator-asserted pricing layer, plan freshness and the account's own write
    permission. Only the measured verdict and the training step of the
    playbook are replaced.
    """
    safe = {key: value for key, value in payload.items()
            if key not in {"training_gate", "operational_order"}}
    order = payload.get("operational_order")
    if isinstance(order, (list, tuple)) and order:
        safe["operational_order"] = [*[str(step) for step in order[:3]],
                                     EVENT_STAGE_FOUR_OPERATOR_HE]
    safe["model_effect"] = operator_model_note()
    # The default provenance stamp names the stage this wall just removed, so
    # the walled payload carries its own.
    safe["source"] = "event pipeline snapshot (events store, pricing layer, schedule freshness)"
    return safe


def wall_read_tool(name: str, payload: Any, user: "str | None") -> Any:
    """Apply the wall to one read-tool result.

    Returns the payload unchanged for a company account and for a tool that
    carries no training content. Otherwise returns the operator-safe payload,
    which names what is withheld and why rather than pretending the tool failed.
    """
    if name not in WALLED_READ_TOOLS or not isinstance(payload, dict):
        return payload
    if actor_is_company(user):
        return payload
    if name == "get_event_pipeline":
        return _walled_event_pipeline(payload)
    walled = operator_model_note()
    walled["available"] = False
    walled["reason"] = WITHHELD_REASON
    if name == "get_audience_model":
        activation = payload.get("activation")
        if isinstance(activation, dict):
            walled["activation"] = {"flag": activation.get("flag"),
                                    "enabled": bool(activation.get("enabled"))}
    return walled


def wall_section(name: str, payload: Any, user: "str | None") -> Any:
    """Apply the same wall to one keyword grounding section."""
    if name not in WALLED_SECTIONS or not isinstance(payload, dict):
        return payload
    if actor_is_company(user):
        return payload
    if name == "event_pipeline":
        return _walled_event_pipeline(payload)
    walled = operator_model_note()
    walled["available"] = False
    walled["reason"] = WITHHELD_REASON
    return walled


def _coefficient_state() -> dict[str, Any]:
    """The measured retention-coefficient state, for a company account only.

    Read straight from the artifact's own metadata plus the level-drift monitor,
    so nothing here is asserted. An unreadable artifact is an honest unavailable.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "models" / "tv_break_coefficients.json"
    try:
        meta = dict((json.loads(path.read_text(encoding="utf-8")).get("metadata") or {}))
    except Exception:  # noqa: BLE001 - an unreadable artifact is unavailable, never a guess
        return {"available": False, "reason": f"{path.name} could not be read"}
    keys = ("computed_at", "pooling_method", "detrended", "first_break_active",
            "first_break_multiplier", "first_break_p_value", "event_layer_gate")
    state: dict[str, Any] = {"available": True, "artifact": "models/tv_break_coefficients.json"}
    for key in keys:
        if key in meta:
            state[key] = meta[key]
    fingerprints = meta.get("source_fingerprints")
    if isinstance(fingerprints, dict):
        state["source_files"] = sorted(fingerprints)
    return state


def model_state_section(user: "str | None" = None) -> dict[str, Any]:
    """The ``model_state`` grounding section: why the plan's numbers are what
    they are. Company accounts get the measured coefficient state, the drift
    monitor's verdict and the plan's freshness; every account gets the model
    version and the release note. Nothing is invented on either side.
    """
    from pathlib import Path

    section: dict[str, Any] = {
        "basis": ("the retention coefficients are measured on real breaks and rebuilt on the "
                  "company side; an operator never sets one"),
        **operator_model_note(),
    }
    try:
        from kairos.export.schedule_freshness import schedule_freshness

        freshness = dict(schedule_freshness(Path(__file__).resolve().parents[1]))
        section["plan_freshness"] = {key: freshness.get(key)
                                     for key in ("status", "computed_at", "changed")}
    except Exception:  # noqa: BLE001 - an absent freshness verdict is omitted, never faked
        logger.exception("plan freshness lookup failed for the model_state section")
    if not actor_is_company(user):
        return section
    section["coefficients"] = _coefficient_state()
    try:
        from kairos_api.catalog_api import impact

        measured = impact().get("drift")
        if isinstance(measured, dict) and measured:
            section["level_drift"] = {key: value for key, value in measured.items()
                                      if key != "weekly_levels"}
    except Exception:  # noqa: BLE001 - an absent drift monitor is omitted, never faked
        logger.exception("level-drift lookup failed for the model_state section")
    return section


def lexicon_hits(blob: str) -> list[str]:
    """Which of section 4.2's training words a serialized payload carries. The
    critic's own grep, available in process so a test can assert it."""
    lowered = blob.lower()
    return [word for word in TRAINING_LEXICON if word in lowered]


def internals_hits(blob: str) -> list[str]:
    """Which model internals a serialized payload carries. This is the list the
    wall enforces, and the difference from lexicon_hits is documented above it."""
    lowered = blob.lower()
    return [word for word in MODEL_INTERNALS_LEXICON if word in lowered]
