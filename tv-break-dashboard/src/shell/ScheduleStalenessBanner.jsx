// ScheduleStalenessBanner: an honest "saved schedule is out of date" strip.
//
// The Schedule, Reports, and Overview pages all render off one saved CSV. When
// the operator changes settings, pricing, or constraints (or runs a live-only
// optimization preview), those pages keep showing the OLD saved schedule. The
// backend reports a freshness verdict on overview.schedule_freshness; this
// component consumes it and does not compute it.
//
// Honesty rules:
//   - status "fresh"   -> stay quiet (nothing to say).
//   - status "unknown" -> stay quiet (we do NOT claim staleness we cannot prove).
//   - status "stale"   -> show a calm amber strip naming what changed and when.
//
// Frozen input contract (overview.schedule_freshness):
//   { status: "fresh" | "stale" | "unknown",
//     computed_at: "<ISO-8601 UTC>" | null,
//     changed: [<operator-facing group label>, ...] }
// where the labels come from the backend's GROUP_LABELS: settings, constraints,
// overrides, coefficients, data, the impact model, inventory data, advertiser
// rules, campaign flights, program classifications. Unknown labels still render.

function ScheduleStalenessBanner({ freshness, locale, onRecompute, recomputeState }) {
  if (!freshness || typeof freshness !== 'object') return null;

  const status = String(freshness.status || '').toLowerCase();
  // Only speak when we KNOW the saved schedule is stale.
  if (status !== 'stale') return null;

  const t = (en, he) => (locale === 'he' ? he : en);

  // Friendly bilingual labels for each changed input group. The backend sends
  // the operator-facing GROUP_LABELS strings (for example "the impact model");
  // the raw internal keys are also accepted defensively. A group neither map
  // knows passes through verbatim below rather than being dropped.
  const changedLabels = {
    settings: t('settings', 'הגדרות'),
    constraints: t('constraints', 'אילוצים'),
    overrides: t('manual overrides', 'עקיפות ידניות'),
    coefficients: t("the model's learned values (coefficients)", 'הערכים שהמודל למד (מקדמים)'),
    data: t('source data', 'נתוני מקור'),
    impact_model: t('the impact model', 'מודל ההשפעה'),
    inventory: t('inventory data', 'נתוני המלאי'),
    advertiser: t('advertiser rules', 'כללי המפרסמים'),
    campaigns: t('campaign flights', 'נתוני הקמפיינים'),
    classifications: t('program classifications', 'סיווגי התוכניות'),
    events: t('special events', 'אירועים מיוחדים'),
    audience_model: t('the audience model', 'מודל קהל הצופים'),
    'the impact model': t('the impact model', 'מודל ההשפעה'),
    'inventory data': t('inventory data', 'נתוני המלאי'),
    'advertiser rules': t('advertiser rules', 'כללי המפרסמים'),
    'campaign flights': t('campaign flights', 'נתוני הקמפיינים'),
    'program classifications': t('program classifications', 'סיווגי התוכניות'),
    'special events': t('special events', 'אירועים מיוחדים'),
    'the audience model': t('the audience model', 'מודל קהל הצופים'),
  };

  const changed = Array.isArray(freshness.changed) ? freshness.changed : [];
  const friendly = changed
    .map((key) => changedLabels[key] || String(key ?? '').trim())
    .filter((label) => typeof label === 'string' && label.length > 0);

  // Join the changed groups naturally. Empty should not happen for a stale
  // verdict; the sentence below falls back to a generic subject when it is.
  const changedPhrase = friendly.length > 0 ? joinList(friendly, locale) : '';

  // Format the computation time in the browser locale, guarded against a null or
  // invalid date so the "on <date>" clause is simply omitted when unavailable.
  const computedLabel = formatComputedAt(freshness.computed_at, locale);

  const heading = t('Saved schedule is out of date', 'לוח השידור השמור אינו מעודכן');

  // The subject frame avoids a double verb when the changed list is empty and
  // sidesteps Hebrew number agreement for single-group lists.
  let detail;
  if (locale === 'he') {
    const subject = changedPhrase ? `חל שינוי ב${changedPhrase}` : 'חל שינוי בקלט הלוח';
    detail = computedLabel
      ? `${subject} מאז שהלוח חושב ב${computedLabel}. הריצו חישוב מחדש כדי לרענן את הלוח, הדוחות ותוכנית הברייקים.`
      : `${subject} מאז שהלוח חושב. הריצו חישוב מחדש כדי לרענן את הלוח, הדוחות ותוכנית הברייקים.`;
  } else {
    const subject = changedPhrase || 'schedule inputs';
    detail = computedLabel
      ? `${subject} changed since this schedule was computed on ${computedLabel}. Recompute to refresh the schedule, reports, and break plan.`
      : `${subject} changed since this schedule was computed. Recompute to refresh the schedule, reports, and break plan.`;
  }

  const recomputing = recomputeState === 'running';
  const buttonLabel = recomputing
    ? t('Recomputing', 'מחשב מחדש')
    : t('Recompute now', 'הריצו חישוב מחדש');

  return (
    <section
      className="schedule-staleness-banner"
      role="status"
      aria-label={heading}
    >
      <div className="schedule-staleness-text">
        <span className="schedule-staleness-heading">{heading}</span>
        <span className="schedule-staleness-detail">{detail}</span>
      </div>
      <button
        type="button"
        className="schedule-staleness-button"
        disabled={recomputing}
        onClick={() => onRecompute && onRecompute()}
      >
        {buttonLabel}
      </button>
    </section>
  );
}

// joinList joins friendly labels with locale-appropriate separators and a final
// conjunction, so "settings, constraints and source data" reads naturally.
function joinList(items, locale) {
  if (items.length <= 1) return items.join('');
  const conjunction = locale === 'he' ? ' ו' : ' and ';
  const head = items.slice(0, -1).join(locale === 'he' ? ', ' : ', ');
  const tail = items[items.length - 1];
  // Hebrew prefixes the conjunction directly onto the last word (no space after).
  return locale === 'he' ? `${head}${conjunction}${tail}` : `${head}${conjunction}${tail}`;
}

// formatComputedAt renders the ISO timestamp in the browser locale, returning
// null for a missing or unparseable value so the caller can omit the date clause.
function formatComputedAt(value, locale) {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
}

export default ScheduleStalenessBanner;
