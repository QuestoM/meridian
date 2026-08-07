import React, { useEffect, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Info } from 'lucide-react';
import { API_BASE, pageText, finiteNumber, formatNumber } from '../shell/surface-helpers';
import { AudienceModelBlock } from '../model/CalendarAudienceModel';
import { formatDay, formatDayOfMonth, formatStamp } from '../shell/dates';

// Companion module for the Calendar tab (CalendarEvents.jsx): the read-only
// "what the model relies on today" panel, the overlap panel, and the small
// display-only badge helpers the plan surfaces (Overview basis note, schedule
// canvas) import. Everything here renders values the backend measured or the
// operator asserted; nothing invents a number.

// Event type vocabulary shared by the management list, the overlap view and the
// plan-surface badges. Keys are the API's type enum.
export const EVENT_TYPES = {
  holiday: { en: 'Holiday', he: 'חג' },
  war: { en: 'War', he: 'מלחמה' },
  special: { en: 'Special event', he: 'אירוע מיוחד' },
  sport: { en: 'Sport', he: 'ספורט' },
  other: { en: 'Other', he: 'אחר' },
};

export function eventTypeLabel(type, locale) {
  const entry = EVENT_TYPES[String(type || 'other')] || EVENT_TYPES.other;
  return pageText(locale, entry.en, entry.he);
}

export function eventTypeChipClass(type) {
  return EVENT_TYPES[String(type || 'other')] ? `cal-type-chip ${String(type)}` : 'cal-type-chip other';
}

// ISO weekday key (pricing premiums are keyed 1=Monday .. 7=Sunday).
const ISO_DAY_NAMES = {
  1: ['Mon', 'שני'], 2: ['Tue', 'שלישי'], 3: ['Wed', 'רביעי'], 4: ['Thu', 'חמישי'],
  5: ['Fri', 'שישי'], 6: ['Sat', 'שבת'], 7: ['Sun', 'ראשון'],
};

// Presentation order for weekday displays: the Israeli week starts Sunday
// (ISO 7) and ends Saturday (ISO 6). Data stays keyed ISO; only the display
// order changes. Unknown keys sort after the known week, never dropped.
const ISRAELI_WEEKDAY_ORDER = [7, 1, 2, 3, 4, 5, 6];

export function israeliWeekdaySort(entries) {
  const rank = (entry) => {
    const index = ISRAELI_WEEKDAY_ORDER.indexOf(Number(entry?.iso_weekday));
    return index === -1 ? ISRAELI_WEEKDAY_ORDER.length : index;
  };
  return [...(entries || [])].sort((a, b) => rank(a) - rank(b));
}

// JS Date.getDay() order, mapped onto the planner's day keys.
const WEEKDAY_KEYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export function formatEventDate(value) {
  return formatDay(value);
}

function formatShortDate(value) {
  return formatDayOfMonth(value);
}

// Fetches the stored events once per refresh for the display-only plan-surface
// badges. Any failure (older backend without /api/events, offline) resolves to
// an empty list so the surfaces simply carry no badge, never a fabricated one.
export function usePlanEvents(refreshKey) {
  const [events, setEvents] = useState([]);
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/api/events`);
        if (!response.ok) throw new Error(String(response.status));
        const data = await response.json();
        if (active) setEvents(Array.isArray(data.events) ? data.events : []);
      } catch {
        if (active) setEvents([]);
      }
    })();
    return () => { active = false; };
  }, [refreshKey]);
  return events;
}

// Maps the server-computed plan overlap dates of every ACTIVE event onto the
// planner's weekday keys. Date to weekday is a calendar fact, not a model claim.
export function planEventWeekdayMap(events) {
  const map = {};
  for (const event of events || []) {
    if (!event || event.active === false) continue;
    for (const value of Array.isArray(event.plan_overlap_dates) ? event.plan_overlap_dates : []) {
      const date = new Date(`${String(value).slice(0, 10)}T00:00:00`);
      if (Number.isNaN(date.getTime())) continue;
      const key = WEEKDAY_KEYS[date.getDay()];
      if (!map[key]) map[key] = [];
      if (!map[key].includes(event.name)) map[key].push(event.name);
    }
  }
  return map;
}

// The Overview basis-note badge line: names of active events whose window
// overlaps the saved plan. Name only, display only, no invented numbers.
export function PlanEventBadges({ events, locale }) {
  const hits = (events || []).filter(
    (event) => event && event.active !== false && Array.isArray(event.plan_overlap_dates) && event.plan_overlap_dates.length > 0,
  );
  if (!hits.length) return null;
  const hint = pageText(
    locale,
    'Display only: the event is marked by name so you know these plan days sit inside it. No retention or revenue number changes, because event effects on retention are not measured yet.',
    'תצוגה בלבד: האירוע מסומן בשמו כדי שתדעו שימי התוכנית האלה נמצאים בתוכו. אף מספר שימור או הכנסה אינו משתנה, כי השפעת אירועים על השימור עדיין לא נמדדה.',
  );
  return (
    <p className="data-basis-note plan-event-badges">
      <Tooltip title={hint} arrow>
        <span className="plan-event-label">
          <Info size={12} aria-hidden="true" />
          {pageText(locale, 'Plan days inside an active event:', 'ימי תוכנית בתוך אירוע פעיל:')}
        </span>
      </Tooltip>
      {hits.map((event) => (
        <span className={eventTypeChipClass(event.type)} key={event.event_id || event.name}>{event.name}</span>
      ))}
    </p>
  );
}

// Reader for the /api/events model_context payload (kairos_api/events_api.py):
// weekday_premiums {available, basis, source, values: [{iso_weekday, multiplier}]},
// measurement {available, detrend_baseline_mode, seasonal_baseline, level_drift,
// computed_at}, training_window {start, end, days, total_breaks_measured} and
// wartime_disclosure {line, ceasefire_date, post_ceasefire_breaks,
// total_breaks_measured}. Anything absent resolves to null so each row can
// honestly skip instead of inventing a value.
function normalizeContext(context) {
  if (!context || typeof context !== 'object') return null;
  const premiumsRaw = context.weekday_premiums;
  const premiums = premiumsRaw && typeof premiumsRaw === 'object'
    ? (Array.isArray(premiumsRaw.values)
      ? premiumsRaw
      : { available: Object.keys(premiumsRaw).length > 0, values: Object.keys(premiumsRaw).map((key) => ({ iso_weekday: Number(key), multiplier: premiumsRaw[key] })) })
    : null;
  const measurement = context.measurement && typeof context.measurement === 'object' ? context.measurement : context;
  const seasonal = measurement.seasonal_baseline && typeof measurement.seasonal_baseline === 'object' ? measurement.seasonal_baseline : {};
  const window = context.training_window && typeof context.training_window === 'object' ? context.training_window : {};
  return {
    premiums,
    measurementAvailable: measurement.available !== false,
    measurementReason: typeof measurement.reason === 'string' ? measurement.reason : null,
    mode: measurement.detrend_baseline_mode || null,
    seasonalRecommended: typeof seasonal.recommended === 'boolean' ? seasonal.recommended : null,
    seasonalImprovement: finiteNumber(seasonal.holdout ? seasonal.holdout.relative_improvement : seasonal.relative_improvement),
    levelDrift: measurement.level_drift && typeof measurement.level_drift === 'object' ? measurement.level_drift : null,
    computedAt: typeof measurement.computed_at === 'string' ? measurement.computed_at : null,
    windowStart: window.start || null,
    windowEnd: window.end || null,
    windowDays: finiteNumber(window.days),
    windowBreaks: finiteNumber(window.total_breaks_measured),
    wartime: context.wartime_disclosure || null,
  };
}

function ContextRow({ label, hint, children }) {
  return (
    <div className="cal-context-row">
      <Tooltip title={hint} arrow>
        <span className="cal-context-label">
          <Info size={12} aria-hidden="true" />
          {label}
        </span>
      </Tooltip>
      <div className="cal-context-value">{children}</div>
    </div>
  );
}

function wartimeText(wartime, locale) {
  if (!wartime) return null;
  if (typeof wartime === 'string') return wartime;
  // English shows the backend's own disclosure line verbatim; Hebrew is
  // composed from the same backend numbers, never from invented ones.
  if (locale !== 'he' && typeof wartime.line === 'string' && wartime.line.trim()) return wartime.line;
  const cease = wartime.ceasefire_date || null;
  const tail = finiteNumber(wartime.post_ceasefire_breaks);
  const total = finiteNumber(wartime.total_breaks_measured ?? wartime.total_breaks);
  if (!cease || tail === null) return typeof wartime.line === 'string' ? wartime.line : null;
  const totalText = total === null ? '' : formatNumber(total, locale);
  return pageText(
    locale,
    `The whole training window was measured under wartime conditions. The ceasefire took effect only on ${String(cease).slice(0, 10)}, leaving ${formatNumber(tail, locale)}${totalText ? ` of ${totalText}` : ''} measured breaks after it. A holiday or war-intensity retention effect claimed from this window would be fabrication; it ships only once history with real contrast passes the held-out gate.`,
    `כל חלון האימון נמדד בתנאי מלחמה. הפסקת האש נכנסה לתוקף רק ב-${String(cease).slice(0, 10)}, כך שרק ${formatNumber(tail, locale)}${totalText ? ` מתוך ${totalText}` : ''} מהברייקים שנמדדו נופלים אחריה. מקדם שימור לחג או לעוצמת מלחמה שנטען מהחלון הזה יהיה בדיה; הוא ישוחרר רק כשהיסטוריה עם ניגוד אמיתי תעבור את מבחן ההחזקה.`,
  );
}

// Panel (b): what the model conditions on TODAY, rendered faithfully from the
// backend's model_context. Absent fields are skipped, never filled in.
export function ModelContextPanel({ context, locale }) {
  const ctx = normalizeContext(context);
  return (
    <section className="page-panel cal-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'What the model relies on today', 'על מה המודל מסתמך היום')}</h2>
      </div>
      <div className="cal-panel-body">
      {!ctx ? (
        <p className="cal-empty">{pageText(locale, 'The backend did not report the model context, so nothing is shown rather than an invented summary.', 'השרת לא דיווח על הקשר המודל, ולכן לא מוצג דבר במקום סיכום מומצא.')}</p>
      ) : (
        <div className="cal-context-list">
          {ctx.premiums && (
            <ContextRow
              label={pageText(locale, 'Weekday pricing premiums (rate-card assertions)', 'פרמיות יום בשבוע בתמחור (הצהרות מחירון)')}
              hint={pageText(locale, 'The live price multiplier per weekday. These are operator rate-card assertions, not values measured from audience history: the model applies them, it did not learn them.', 'מכפיל המחיר החי לכל יום בשבוע. אלה הצהרות מחירון של המפעיל, לא ערכים שנמדדו מהיסטוריית צפייה: המודל מחיל אותם, הוא לא למד אותם.')}
            >
              {ctx.premiums.available === false || !ctx.premiums.values?.length ? (
                <span>{pageText(locale, 'The rate-card weekday table could not be read, so no premiums are shown.', 'טבלת ימי השבוע של המחירון לא נקראה, ולכן לא מוצגות פרמיות.')}</span>
              ) : (
                <>
                  <div className="cal-weekday-table">
                    {israeliWeekdaySort(ctx.premiums.values).map((entry) => {
                      const day = ISO_DAY_NAMES[Number(entry.iso_weekday)];
                      const value = finiteNumber(entry.multiplier);
                      return (
                        <span className="cal-weekday-cell" key={entry.iso_weekday}>
                          <small>{day ? pageText(locale, day[0], day[1]) : String(entry.iso_weekday)}</small>
                          <span className="bidi-figure figure-nowrap">{value === null ? '-' : value.toFixed(2)}</span>
                        </span>
                      );
                    })}
                  </div>
                  <small className="cal-context-source">{pageText(locale, 'Rate-card assertion, not measured from audience history.', 'הצהרת מחירון, לא נמדדה מהיסטוריית צפייה.')}</small>
                </>
              )}
            </ContextRow>
          )}
          {!ctx.measurementAvailable && (
            <ContextRow
              label={pageText(locale, 'Retention measurement', 'מדידת השימור')}
              hint={pageText(locale, 'The coefficient file with the measurement metadata could not be read, so nothing is claimed about it.', 'קובץ המקדמים עם נתוני המדידה לא נקרא, ולכן לא נטען עליו דבר.')}
            >
              <span>{ctx.measurementReason ? <span className="bidi-figure figure-nowrap">{ctx.measurementReason}</span> : pageText(locale, 'Measurement metadata unavailable.', 'נתוני המדידה אינם זמינים.')}</span>
            </ContextRow>
          )}
          {ctx.mode && (
            <ContextRow
              label={pageText(locale, 'Retention measurement mode', 'אופן מדידת השימור')}
              hint={pageText(locale, 'The baseline the retention measurement detrends against. Global means one baseline for the whole window, with no weekday, date or holiday term inside the measurement.', 'קו הבסיס שממנו מדידת השימור מנקה מגמות. גלובלי פירושו קו בסיס אחד לכל החלון, ללא רכיב יום בשבוע, תאריך או חג בתוך המדידה.')}
            >
              <span>{ctx.mode === 'global' ? pageText(locale, 'Global baseline, calendar-blind', 'קו בסיס גלובלי, עיוור ללוח השנה') : <span className="bidi-figure figure-nowrap">{String(ctx.mode)}</span>}</span>
            </ContextRow>
          )}
          {ctx.seasonalImprovement !== null && (
            <ContextRow
              label={pageText(locale, 'Monthly seasonality verdict', 'הכרעת העונתיות החודשית')}
              hint={pageText(locale, 'A seasonal (month by minute) baseline exists in the code and was tested on held-out days. It is activated only when it beats the global baseline; the measured improvement decides.', 'קו בסיס עונתי (חודש לפי דקה) קיים בקוד ונבחן על ימים מוחזקים. הוא מופעל רק כשהוא מנצח את קו הבסיס הגלובלי; השיפור הנמדד מכריע.')}
            >
              <span>
                {ctx.seasonalRecommended
                  ? pageText(locale, `Measured held-out improvement ${(ctx.seasonalImprovement * 100).toFixed(1)}%, active`, `שיפור מדוד של ${(ctx.seasonalImprovement * 100).toFixed(1)}% במבחן מוחזק, פעילה`)
                  : pageText(locale, `Measured held-out improvement ${(ctx.seasonalImprovement * 100).toFixed(1)}%, stays off`, `שיפור מדוד של ${(ctx.seasonalImprovement * 100).toFixed(1)}% במבחן מוחזק, ולכן נשארת כבויה`)}
              </span>
            </ContextRow>
          )}
          {ctx.levelDrift && (
            <ContextRow
              label={pageText(locale, 'Week-to-week level drift', 'סחיפת רמה משבוע לשבוע')}
              hint={pageText(locale, 'The average measured break effect per training week. A consistent movement across weeks means the world was changing during training; the binding flag says whether that movement is large enough to matter next to the coefficient uncertainty.', 'אפקט הברייק הממוצע שנמדד בכל שבוע אימון. תנועה עקבית בין השבועות אומרת שהעולם השתנה במהלך האימון; דגל המחייבות אומר אם התנועה גדולה מספיק כדי להיות משמעותית לצד אי-הוודאות של המקדמים.')}
            >
              <div className="cal-drift-block">
                {finiteNumber(ctx.levelDrift.drift_per_week) !== null && (
                  <span>
                    {pageText(locale, 'Drift per week (log effect):', 'סחיפה לשבוע (אפקט לוג):')}
                    <span className="bidi-figure figure-nowrap">{Number(ctx.levelDrift.drift_per_week).toFixed(4)}</span>
                  </span>
                )}
                {typeof ctx.levelDrift.binding === 'boolean' && (
                  <span className={ctx.levelDrift.binding ? 'cal-chip warn' : 'cal-chip'}>
                    {ctx.levelDrift.binding ? pageText(locale, 'Binding', 'מחייבת') : pageText(locale, 'Not binding', 'לא מחייבת')}
                  </span>
                )}
                {Array.isArray(ctx.levelDrift.weekly_levels) && ctx.levelDrift.weekly_levels.length > 0 && (
                  <table className="cal-weekly-table">
                    <thead>
                      <tr>
                        <th>{pageText(locale, 'Week', 'שבוע')}</th>
                        <th>{pageText(locale, 'Breaks', 'ברייקים')}</th>
                        <th>{pageText(locale, 'Mean level (log)', 'רמה ממוצעת (לוג)')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ctx.levelDrift.weekly_levels.map((row, index) => (
                        <tr key={row.week ?? index}>
                          <td><span className="bidi-figure figure-nowrap">{formatNumber(row.week, locale)}</span></td>
                          <td><span className="bidi-figure figure-nowrap">{formatNumber(row.n, locale)}</span></td>
                          <td><span className="bidi-figure figure-nowrap">{finiteNumber(row.mean_log_effect) === null ? '-' : Number(row.mean_log_effect).toFixed(4)}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </ContextRow>
          )}
          {(ctx.windowStart || ctx.windowEnd) && (
            <ContextRow
              label={pageText(locale, 'Training window', 'חלון האימון')}
              hint={pageText(locale, 'The exact calendar span of the history the retention coefficients were measured on. Conditions outside this span, such as holidays, were never seen by the model.', 'טווח התאריכים המדויק של ההיסטוריה שעליה נמדדו מקדמי השימור. תנאים מחוץ לטווח הזה, למשל חגים, מעולם לא נראו על ידי המודל.')}
            >
              <span>
                <span className="bidi-figure figure-nowrap">{`${String(ctx.windowStart || '').slice(0, 10)} .. ${String(ctx.windowEnd || '').slice(0, 10)}`}</span>
                {ctx.windowDays !== null && ctx.windowBreaks !== null && (
                  <span>{pageText(locale, ` (${formatNumber(ctx.windowDays, locale)} days, ${formatNumber(ctx.windowBreaks, locale)} measured breaks)`, ` (${formatNumber(ctx.windowDays, locale)} ימים, ${formatNumber(ctx.windowBreaks, locale)} ברייקים שנמדדו)`)}</span>
                )}
              </span>
            </ContextRow>
          )}
          {ctx.computedAt && (
            <ContextRow
              label={pageText(locale, 'Coefficients computed at', 'המקדמים חושבו בתאריך')}
              hint={pageText(locale, 'When the current retention coefficients were last rebuilt from the source data.', 'מתי מקדמי השימור הנוכחיים נבנו מחדש בפעם האחרונה מנתוני המקור.')}
            >
              <span className="bidi-figure figure-nowrap">{formatStamp(ctx.computedAt)}</span>
            </ContextRow>
          )}
          {wartimeText(ctx.wartime, locale) && (
            <ContextRow
              label={pageText(locale, 'Wartime disclosure', 'גילוי נאות: מלחמה')}
              hint={pageText(locale, 'The training history was measured under a specific real-world condition. Coefficients carry that condition; there is no measured war or holiday effect to separate it out yet.', 'היסטוריית האימון נמדדה בתנאי מציאות ספציפיים. המקדמים נושאים את התנאי הזה; אין עדיין אפקט מלחמה או חג מדוד שמפריד אותו.')}
            >
              <span className="cal-wartime-line">{wartimeText(ctx.wartime, locale)}</span>
            </ContextRow>
          )}
        </div>
      )}
      <AudienceModelBlock locale={locale} />
      </div>
    </section>
  );
}

// Panel (c): per event, its overlap with the coefficient training window and
// with the saved plan dates. Both overlaps are computed server-side.
export function OverlapPanel({ events, locale }) {
  const rows = Array.isArray(events) ? events : [];
  return (
    <section className="page-panel cal-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'Event overlaps: training window and current plan', 'חפיפות אירועים: חלון האימון והתוכנית הנוכחית')}</h2>
        <span>{rows.length} {pageText(locale, 'events', 'אירועים')}</span>
      </div>
      <div className="cal-panel-body">
      <p className="cal-panel-note">{pageText(locale, 'For each event: whether the training data even saw that condition, and which saved-plan days sit inside it.', 'לכל אירוע: האם נתוני האימון בכלל ראו את התנאי הזה, ואילו ימים בתוכנית השמורה נמצאים בתוכו.')}</p>
      {rows.length === 0 ? (
        <p className="cal-empty">{pageText(locale, 'No events are stored yet, so there is nothing to intersect.', 'אין עדיין אירועים שמורים, ולכן אין מה להצליב.')}</p>
      ) : (
        <div className="cal-overlap-list">
          {rows.map((event) => {
            const windowDays = finiteNumber(event.window_overlap_days);
            const planDates = Array.isArray(event.plan_overlap_dates) ? event.plan_overlap_dates : [];
            return (
              <div className="cal-overlap-row" key={event.event_id || event.name}>
                <div className="cal-overlap-head">
                  <span className="cal-event-name">{event.name}</span>
                  <span className={eventTypeChipClass(event.type)}>{eventTypeLabel(event.type, locale)}</span>
                  {event.active === false && <span className="cal-chip off">{pageText(locale, 'Deactivated', 'מושבת')}</span>}
                </div>
                <div className="cal-overlap-facts">
                  <span>
                    {windowDays === null
                      ? pageText(locale, 'Training-window overlap not reported', 'חפיפה עם חלון האימון לא דווחה')
                      : windowDays === 0
                        ? pageText(locale, 'The training data did not see this condition', 'נתוני האימון לא ראו את התנאי הזה')
                        : pageText(locale, `${formatNumber(windowDays, locale)} days inside the training window`, `${formatNumber(windowDays, locale)} ימים בתוך חלון האימון`)}
                  </span>
                  <span className="cal-overlap-dates">
                    {planDates.length === 0
                      ? pageText(locale, 'No overlap with the saved plan', 'אין חפיפה עם התוכנית השמורה')
                      : planDates.map((date) => (
                        <span className="cal-date-chip bidi-figure figure-nowrap" key={date}>{formatShortDate(date)}</span>
                      ))}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
      </div>
    </section>
  );
}
