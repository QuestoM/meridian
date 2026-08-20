"""Category, season and episode for a competitor's programmes — built to last.

The schedule arrives with a name, a description and nothing else: ``Season`` and
``Episode`` are fields the publication carries and leaves EMPTY on every record,
and only 5 descriptions in 127 mention them in prose. So the engine's own
categories have to be inferred, and the inference has to keep working next year
without anyone maintaining it.

Three things break a naive version, and all three were measured on a real week
before this file was written.

1. THE IDENTITY TO REMEMBER IS THE SERIES, NOT THE BROADCAST.
   ``ProgramCode`` looks like a programme id and is not one: 11 codes are used
   for more than one name, and "מבזק חדשות" carries four. The full title is not
   it either — five series appear under a different title every episode
   ("רוקדים עם כוכבים – ששת הגדולים", "רוקדים עם כוכבים – אתגר הרביעיות").
   Remembering by full title means paying for every episode forever, which is
   the exact failure this file exists to avoid. The series base — the title
   before its episode separator — is stable, and it is what a category belongs
   to: a dance contest is Reality whatever this week's episode is called.

2. THE EPISODE NAME IS ALREADY IN THE TITLE, SO IT IS NEVER PAID FOR.
   The same separator that gives the series gives the episode name on its other
   side. Only the NUMBER is genuinely absent, and a number that no source states
   is returned as null rather than counted from the order of broadcasts — the
   third episode aired this week is not episode 3.

3. MEANING DRIFTS UNDER A STABLE NAME.
   Six titles in one week carried more than one description: "כותרות הבוקר" ran
   with two different presenters. A cache keyed on the name alone would answer
   from a stale reading forever. So a remembered decision is stamped with the
   description it was made from, and a materially different description asks
   again. That is what makes this correct in a year rather than merely correct
   today.

The category vocabulary is CLOSED and comes from the engine's own taxonomy,
because the optimizer, the dayparts and the forecast all key off it. The model
picks from that list or answers ``unfittable`` with a reason — it may never
invent a sixteenth category. An unfittable answer is not a failure to hide: it
is the signal that the taxonomy needs a human's attention, and it is reported.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

# The separators Keshet uses between a series and its episode. Both dashes are
# real in the captured week: an ASCII hyphen and an en dash.
EPISODE_SEPARATORS = (" – ", " - ", " — ")

UNFITTABLE = "unfittable"

# The normalisation rule a remembered decision was stamped under. It is stored
# with every entry because CHANGING THIS RULE SILENTLY RE-BUYS ANSWERS: when the
# trailing-space bug in the normaliser was fixed, four of twenty-six remembered
# series stopped matching their own descriptions and would have been re-asked
# with no explanation. A version makes that visible — an entry from an older
# rule is reported as such, not confused with a broadcaster changing a synopsis.
FINGERPRINT_VERSION = 2

# Below this, the keyword classifier has not really decided and the model is
# asked. Measured on the real week: at 0.6, 38 of 70 titles are settled for free
# and 32 go to the model, of which 27 had no answer at all.
SETTLED_CONFIDENCE = 0.6


def series_of(title: str) -> tuple[str, str]:
    """``(series, episode_name)`` — the split the titles already carry.

    Falls back to the whole title as the series with no episode name, which is
    the right answer for a programme that simply has no episodes.
    """
    text = str(title or "").strip()
    for separator in EPISODE_SEPARATORS:
        if separator in text:
            head, _, tail = text.partition(separator)
            head, tail = head.strip(), tail.strip()
            if head and tail:
                return head, tail
    return text, ""


def _fingerprint(text: str) -> str:
    """What a decision was made from, so drift under a stable name is visible.

    Normalised before hashing: whitespace and punctuation churn in a published
    description is not a change of meaning, and re-asking on it would pay for
    the same answer repeatedly.
    """
    normalised = re.sub(r"[^\w\s]", "", str(text or ""))
    # Collapse and strip AFTER the punctuation is gone, not before: removing a
    # trailing "!!" leaves the space it was attached to, and a hash taken then
    # differs from the hash of the same sentence without it — which would re-buy
    # the same answer every time a publisher touched their punctuation.
    normalised = re.sub(r"[\s‏‎]+", " ", normalised).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def enrichment_schema(categories: Iterable[str]) -> dict[str, Any]:
    """The strict shape the model must answer in.

    ``category`` is an enum of the engine's own categories plus ``unfittable``.
    An enum is the whole point: a free-text category would reach the optimizer
    as a value nothing downstream knows, and would do it silently.
    """
    allowed = list(dict.fromkeys(categories)) + [UNFITTABLE]
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": allowed,
                "description": (
                    "The single best fit from this closed list. Answer "
                    f"'{UNFITTABLE}' only when no listed category could "
                    "reasonably describe the programme."
                ),
            },
            "category_reason": {
                "type": "string",
                "maxLength": 220,
                "description": "One short sentence, in Hebrew, saying what in the name or description decided it.",
            },
            "is_episodic": {
                "type": "boolean",
                "description": "True when this is a series with distinct episodes, false for a recurring strip such as a news bulletin.",
            },
            "season": {
                "type": ["integer", "null"],
                "description": "The season number ONLY if the name or description states or clearly implies it. Never inferred from broadcast order. Null otherwise.",
            },
            "episode": {
                "type": ["integer", "null"],
                "description": "The episode number ONLY if stated or clearly implied. Never counted from the schedule. Null otherwise.",
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["category", "category_reason", "is_episodic", "season", "episode", "confidence"],
        "additionalProperties": False,
    }


SYSTEM = """אתה מסווג תוכניות טלוויזיה ישראליות עבור מנוע תכנון שידורים.

עליך לבחור קטגוריה אחת מתוך רשימה סגורה בלבד. הרשימה משמשת את המנוע לתמחור,
לחלוקת רצועות ולתחזית רייטינג, ולכן קטגוריה שאינה ברשימה תישבר במורד הזרם.
אם באמת שום קטגוריה ברשימה אינה מתאימה — ענה unfittable והסבר. אל תמציא קטגוריה.

לגבי עונה ופרק: החזר מספר רק אם השם או התיאור נוקבים בו או רומזים עליו בבירור
(למשל "ערב ההשקה השני", "סוגר עונה"). אל תסיק מספר פרק מסדר השידורים — הפרק
השלישי שמשודר השבוע אינו בהכרח פרק 3. אם אין בסיס, החזר null. null הוא תשובה
נכונה ומכובדת כאן; מספר שהומצא הוא שגיאה שתיכנס למנוע."""


def build_prompt(series: str, examples: list[Mapping[str, Any]]) -> str:
    """Everything known about one series, so the model decides once, well."""
    lines = [f"שם הסדרה: {series}", ""]
    for example in examples[:4]:
        title = str(example.get("Title") or "")
        _, episode_name = series_of(title)
        lines.append(f"— שידור: {title}")
        if episode_name:
            lines.append(f"  שם הפרק: {episode_name}")
        description = str(example.get("Description") or "").strip()
        if description:
            lines.append(f"  תיאור: {description}")
        flags = []
        if example.get("Live"):
            flags.append("שידור חי")
        if example.get("Rerun"):
            flags.append("שידור חוזר")
        if flags:
            lines.append(f"  {', '.join(flags)}")
        duration = example.get("Duration")
        if duration:
            lines.append(f"  משך: {int(duration) // 60} דקות")
        lines.append("")
    return "\n".join(lines)


class SeriesMemory:
    """What has already been decided, and what has since drifted.

    Keyed by series, stamped with the description it was decided from. A series
    whose description has materially changed is reported as STALE rather than
    answered from memory — the presenter changed, and the reading may have to.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - an unreadable memory is an empty one
                self._entries = {}

    def get(self, series: str, description: str) -> tuple[Optional[dict[str, Any]], str]:
        """``(entry, state)``: ``fresh``, ``stale``, ``restamped`` or ``absent``.

        ``restamped`` is an entry written under an older normalisation rule. It
        is a fact about OUR code, never about the broadcaster, and it is kept
        apart so a normaliser change cannot masquerade as a schedule change.
        """
        entry = self._entries.get(series)
        if entry is None:
            return None, "absent"
        if int(entry.get("fingerprint_version") or 1) != FINGERPRINT_VERSION:
            return entry, "restamped"
        if entry.get("fingerprint") != _fingerprint(description):
            return entry, "stale"
        return entry, "fresh"

    def put(self, series: str, description: str, decision: Mapping[str, Any]) -> None:
        self._entries[series] = {
            **dict(decision),
            "fingerprint": _fingerprint(description),
            "fingerprint_version": FINGERPRINT_VERSION,
            "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        return self.path

    def __len__(self) -> int:
        return len(self._entries)


# Where the resolved series live. Written by :func:`enrich` and, until this was
# wired, read by nothing at all: 26 series carried a category, a reason and a
# confidence each, and no module and no screen ever asked.
MEMORY_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "keshet-series-memory.json"

_MEMORY_CACHE: dict[str, "SeriesMemory"] = {}


def shipped_memory(path: str | Path = MEMORY_PATH) -> "SeriesMemory":
    """The resolved-series memory on disk, loaded once per path.

    An absent or unreadable file yields an EMPTY memory rather than raising, so
    a machine without it simply gets the classifier's own answers.
    """
    key = str(path)
    if key not in _MEMORY_CACHE:
        _MEMORY_CACHE[key] = SeriesMemory(path)
    return _MEMORY_CACHE[key]


def remembered_category(
    title: Any, description: Any = None, *, memory: Optional["SeriesMemory"] = None,
) -> tuple[str, str]:
    """``(category, state)`` for a title whose genre nothing else could read.

    LAST in the ladder, and deliberately so. These categories were decided by a
    MODEL from a synopsis, which is weaker evidence than a taxonomy keyword and
    must never overrule one. MEASURED on the pulled fortnight: the memory
    disagrees with the taxonomy on 48 broadcasts it can already place -- it calls
    "חדשות הבוקר עם יואב לימור" a Morning Program where the keywords say News,
    and both are real categories in the taxonomy. Those 48 keep the taxonomy's
    answer, and the disagreement is reported rather than acted on, because which
    of the two is right is a decision for a person.

    Only a FRESH entry answers. The memory stamps every decision with the
    description it was made from precisely so a broadcaster who changes a
    synopsis gets re-asked instead of answered from a stale reading, and using a
    stale entry here would quietly undo the whole point of that design. A stale
    or restamped entry is returned with its state so a caller can report the
    enrichment work that is pending.

    Returns ``("", state)`` when there is nothing to say.
    """
    series, _ = series_of(str(title or ""))
    if not series:
        return "", "absent"
    store = shipped_memory() if memory is None else memory
    entry, state = store.get(series, str(description or ""))
    if state != "fresh" or not entry:
        return "", state
    category = str(entry.get("category") or "")
    if not category or category == UNFITTABLE:
        # unfittable is the module's own honest refusal: the taxonomy has no home
        # for this series. It is never a category, and never becomes one here.
        return "", "unfittable"
    return category, "fresh"


def plan(
    rows: Iterable[Mapping[str, Any]],
    *,
    classify: Callable[[str], Any],
    memory: SeriesMemory,
) -> dict[str, Any]:
    """Decide what must be asked before asking anything.

    Returns the work split three ways, so a caller can see the cost before it is
    spent and so a run with nothing to ask makes no call at all:

    * ``settled``   — the keyword classifier decided; free, and it stays free
    * ``remembered``— decided before for this series, description unchanged
    * ``ask``       — new series, or one whose description has drifted
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        series, _ = series_of(row.get("Title", ""))
        grouped.setdefault(series, []).append(row)

    settled: dict[str, Any] = {}
    remembered: dict[str, Any] = {}
    ask: dict[str, list[Mapping[str, Any]]] = {}
    stale: list[str] = []
    restamped: list[str] = []

    for series, examples in grouped.items():
        description = str(examples[0].get("Description") or "")
        verdict = classify(series)
        confidence = float(getattr(verdict, "confidence", 0.0) or 0.0)
        category = str(getattr(verdict, "category", "") or "")
        if category and category != "Other" and confidence >= SETTLED_CONFIDENCE:
            settled[series] = {
                "category": category,
                "source": "keyword-classifier",
                "confidence": "high" if confidence >= 0.9 else "medium",
            }
            continue
        entry, state = memory.get(series, description)
        if state == "fresh":
            remembered[series] = {**entry, "source": "memory"}
            continue
        if state == "stale":
            stale.append(series)
        elif state == "restamped":
            restamped.append(series)
        ask[series] = examples

    return {
        "series_total": len(grouped),
        "settled": settled,
        "remembered": remembered,
        "ask": ask,
        "stale": stale,
        "restamped": restamped,
        "calls_needed": len(ask),
    }


def pending(
    rows: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    memory: Optional[SeriesMemory] = None,
    classifier: Optional[Any] = None,
) -> dict[str, Any]:
    """What the enrichment could resolve if it were run, and what it already has.

    :func:`plan` has always been able to answer this and nothing ever asked it,
    which is how 26 resolved series sat on disk unread. MEASURED on the pulled
    fortnight: the memory can match 38 of the 138 broadcasts the taxonomy and the
    synopsis together cannot place, but only 19 of those are FRESH -- the other
    19 carry a description the broadcaster has since rewritten, so by this
    module's own drift rule they must be asked again rather than answered from a
    stale reading.

    Reporting that is the point. An enrichment that silently answers 19 and says
    nothing about the 119 it could not looks finished when it is barely started.
    """
    if rows is None:
        from kairos.model import future_epg

        epg = future_epg.load_future_competitor_epg()
        epg = epg[0] if isinstance(epg, tuple) else epg
        if epg is None or not len(epg):
            return {"reason": "no forward schedule has been pulled", "series_total": 0}
        rows = epg.to_dict("records")
    if classifier is None:
        from kairos.data.classifier import ProgramClassifier

        classifier = ProgramClassifier.from_yaml()
    store = shipped_memory() if memory is None else memory

    def classify(series: str) -> Any:
        return classifier.classify(series)

    work = plan(rows, classify=classify, memory=store)
    return {
        "series_total": work["series_total"],
        "settled_by_taxonomy": len(work["settled"]),
        "answered_from_memory": len(work["remembered"]),
        "would_be_asked": work["calls_needed"],
        "stale": work["stale"],
        "reasked_after_rule_change": work["restamped"],
        "memory_size": len(store),
        "note": (
            f"{len(work['remembered'])} series answered from memory, "
            f"{work['calls_needed']} would need asking "
            f"({len(work['stale'])} of them because their synopsis changed)"
        ),
    }


def enrich(
    rows: list[dict[str, Any]],
    *,
    classify: Callable[[str], Any],
    categories: Iterable[str],
    memory: SeriesMemory,
    call: Optional[Callable[..., dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach Category, Season and Episode to every row.

    ``call`` is the provider seam (kairos.trade.extract_provider.StageCaller.call);
    without one, nothing is asked and every unresolved series is reported as
    such. That is the honest-absence shape the rest of this engine uses: no
    credentials must never mean an invented category.
    """
    work = plan(rows, classify=classify, memory=memory)
    schema = enrichment_schema(categories)
    answered: dict[str, Any] = {}
    failures: list[dict[str, str]] = []

    if call is not None:
        for series, examples in work["ask"].items():
            try:
                result = call(
                    stage="epg_category",
                    tier="mid",
                    system=SYSTEM,
                    content=build_prompt(series, examples),
                    tool_name="record_programme",
                    tool_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001 - one failure, one honest gap
                failures.append({"series": series, "error": type(exc).__name__})
                continue
            answered[series] = {**result, "source": "model"}
            memory.put(series, str(examples[0].get("Description") or ""), result)
        if answered:
            memory.save()

    decided = {**work["settled"], **work["remembered"], **answered}
    unfittable: list[str] = []
    out: list[dict[str, Any]] = []
    for row in rows:
        series, episode_name = series_of(row.get("Title", ""))
        entry = decided.get(series)
        category = (entry or {}).get("category") or ""
        if category == UNFITTABLE:
            unfittable.append(series)
        out.append({
            **row,
            "Series": series,
            "EpisodeName": episode_name,
            # An unresolved series carries an EMPTY category, never a guessed
            # one. Downstream already knows how to treat an unknown programme;
            # it does not know how to un-believe a wrong one.
            "Category": "" if category in ("", UNFITTABLE) else category,
            "CategorySource": (entry or {}).get("source", "unresolved"),
            "Season": (entry or {}).get("season"),
            "Episode": (entry or {}).get("episode"),
        })

    status = {
        "series_total": work["series_total"],
        "settled_free": len(work["settled"]),
        "from_memory": len(work["remembered"]),
        "asked": len(answered),
        "calls_needed": work["calls_needed"],
        "stale_reasked": work["stale"],
        "reasked_after_rule_change": work["restamped"],
        "failures": failures,
        # Named loudly: the taxonomy has no home for these, and that is a
        # decision for a person, not something to paper over with 'Other'.
        "unfittable": sorted(set(unfittable)),
        "unresolved": sorted({
            r["Series"] for r in out if r["CategorySource"] == "unresolved"
        }),
        "memory_size": len(memory),
    }
    return out, status
