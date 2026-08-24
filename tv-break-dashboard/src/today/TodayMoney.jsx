import React, { useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { ChevronDown, ChevronUp, Download } from 'lucide-react';
import { Numeric, finiteNumber, formatCurrency, formatNumber, pageText } from '../shell/format';
import { useAddressParam } from '../shell/address-state';
import { formatDay, formatSpan } from '../shell/dates';
import { WALLS } from '../session.js';
import TargetForm from './TargetForm';
import TodayDayDetail from './TodayDayDetail';
import { isolate } from '../shell/bidi';
import { DAY_COLUMNS, download, scopeComment, toCsv } from './today-export';
import { Pressable } from '../studio/dom-controls';
import { Dialog } from '../studio/modal';

// Answer one: is this window on plan.
//
// One money figure, one scope, printed with the figure and never in a tooltip.
// The figure opens the rows behind it, and those rows sum back to it, which the
// payload states and this component prints. Where no target exists the verdict
// is an honest absence with the control that ends it, never a number.

const VERDICT_WORDS = {
  on_plan: ['On plan', 'עומד ביעד'],
  at_risk: ['At risk', 'בסיכון'],
  behind: ['Behind', 'בפיגור'],
};

// Unavailable is not one state, it is two, and they are different answers. The
// store was read and holds no target for this window, which is an absence a
// person ends with one control. Or the store was not read at all, which is not
// known and must never be printed as an absence.
const UNAVAILABLE_WORDS = {
  no_target: ['No target set', 'לא הוגדר יעד'],
  no_projection: ['No projection for this window', 'אין צפי לחלון הזה'],
};

const NOT_KNOWN_WORDS = ['Not known', 'לא ידוע'];

// The wall answers in the product's first language and its string is sent
// verbatim, so the words before the click and the words in the refusal cannot
// drift. This screen also has an English locale, and the reader this
// destination is built for is on it. The endpoint sends both halves; a payload
// that carries only the wall's own string is matched against the same constant
// the session module pins to it, and anything neither of those recognises is
// printed as it arrived rather than guessed at.
const REFUSALS = {
  [WALLS.readOnlyRole.detail]: 'A viewing account has no edit permission',
};

function refusalText(target, locale) {
  const hebrew = String(target.can_edit_reason_he || target.can_edit_reason || '').trim();
  const english = String(target.can_edit_reason_en || REFUSALS[hebrew] || hebrew).trim();
  if (!hebrew && !english) return '';
  return pageText(locale, english || hebrew, hebrew || english);
}

// One span, formatted one way, wherever a span is printed on this surface.
function spanLabel(start, end, locale) {
  if (!start || !end) return '';
  return formatSpan(start, end, locale);
}

export function windowLabel(money, locale) {
  const scope = money.scope || {};
  return spanLabel(scope.date_from, scope.date_to, locale);
}

// The basis, printed with the figure and never in a tooltip. Five facts: whose
// channel, over which dates, whether those edges are inside the range, in which
// zone the day boundaries fall, and out of which artifact the number came.
// The channel is the one part that comes from data rather than from this file,
// so it is bidi isolated. Without that, a Hebrew name in the English locale
// drags the window's first date across the separator that follows it.
function scopeLine(money, locale) {
  const scope = money.scope || {};
  const parts = [];
  if (scope.channel) parts.push(isolate(scope.channel));
  const span = windowLabel(money, locale);
  if (span) {
    const days = finiteNumber(scope.n_dates);
    parts.push(days ? pageText(locale, `${span} (${days} days)`, `${span} (${days} ימים)`) : span);
  }
  if (scope.inclusive) parts.push(pageText(locale, 'both dates included', 'שני התאריכים כלולים'));
  if (scope.timezone) parts.push(String(scope.timezone));
  parts.push(pageText(locale, 'from the saved plan', 'מהתוכנית השמורה'));
  return parts.join(' · ');
}

// Who set the target, and when. With login not set up there is no account to
// attribute it to, so the line says that rather than naming somebody unknown.
function authorLine(target, locale) {
  const when = formatDay(String(target.set_at || '').slice(0, 10));
  const who = String(target.set_by || '').trim();
  if (!who || who === 'unknown') {
    return pageText(locale, `Set on ${when}, before login was set up`, `נקבע ב${when}, לפני שהוגדרה כניסה למערכת`);
  }
  return pageText(locale, `Set by ${who} on ${when}`, `נקבע על ידי ${who} ב${when}`);
}

// A target a person supplied does not stop existing because the plan window
// moved under it. The store keys a target to the exact span it measures, and
// this screen deliberately refuses to read one span's number as another's, but
// refusing to read it is not the same as refusing to name it: a plan run that
// shifts the window would otherwise make somebody's number disappear from the
// one screen it was set on. The payload already carries them, so both branches
// print them, with the span and the amount, and the rule that keeps them apart.
const OTHER_WINDOWS_SHOWN = 3;

function OtherWindows({ target, locale }) {
  const rows = Array.isArray(target.other_windows) ? target.other_windows : [];
  const others = rows.filter((row) => row && row.period_start && row.period_end && finiteNumber(row.amount_ils) !== null);
  if (!others.length) return null;
  const shown = others.slice(0, OTHER_WINDOWS_SHOWN);
  const rest = others.length - shown.length;
  return (
    <div className="today-other-windows">
      <p className="today-note">
        {others.length === 1
          ? pageText(locale, 'A target is set on one other window for this channel.', 'נקבע יעד לחלון אחר אחד בערוץ הזה.')
          : pageText(locale, `A target is set on ${formatNumber(others.length, locale)} other windows for this channel.`, `נקבע יעד ל-${formatNumber(others.length, locale)} חלונות אחרים בערוץ הזה.`)}
        {' '}
        {pageText(locale, 'A target set for a different span is deliberately not read as this one’s, so it is named here rather than lost.', 'יעד שנקבע לטווח אחר אינו נקרא במכוון כיעד של החלון הזה, ולכן הוא נקוב כאן ולא נעלם.')}
      </p>
      {shown.map((row) => (
        <div className="today-other-row" key={`${row.period_start}-${row.period_end}`}>
          <span>{spanLabel(row.period_start, row.period_end, locale)}</span>
          <strong><Numeric>{formatCurrency(row.amount_ils, locale)}</Numeric></strong>
        </div>
      ))}
      {rest > 0 ? (
        <p className="today-note">
          {rest === 1
            ? pageText(locale, 'One more window has a target, on the targets this channel has stored.', 'לחלון נוסף אחד יש יעד, מבין היעדים השמורים בערוץ הזה.')
            : pageText(locale, `${formatNumber(rest, locale)} more windows have a target, on the targets this channel has stored.`, `ל-${formatNumber(rest, locale)} חלונות נוספים יש יעד, מבין היעדים השמורים בערוץ הזה.`)}
        </p>
      ) : null}
    </div>
  );
}

function VerdictChip({ verdict, locale }) {
  const state = String(verdict.state || 'unavailable');
  const words = VERDICT_WORDS[state] || UNAVAILABLE_WORDS[String(verdict.reason || '')] || NOT_KNOWN_WORDS;
  return <span className={`today-verdict ${state}`}>{pageText(locale, words[0], words[1])}</span>;
}

function TargetLine({ today, locale, onEdit, onClear }) {
  const target = today.target || {};
  const verdict = today.verdict || {};
  // The target store was not read at all on this path, so whether a target
  // exists is unknown. Printing the unset state here would report an absence
  // nobody measured, and offering the control that sets one would let a person
  // overwrite a target this screen never saw.
  if (target.state === 'unavailable') {
    return (
      <div className="today-target unset">
        <p>{pageText(locale, 'Whether this window has a target could not be read, so it is not known rather than unset.', 'לא ניתן היה לקרוא אם לחלון הזה נקבע יעד, ולכן המצב אינו ידוע ולא ריק.')}</p>
        <p className="today-note">{pageText(locale, 'Reload this screen to read the target again.', 'טענו מחדש את המסך כדי לקרוא שוב את היעד.')}</p>
      </div>
    );
  }
  if (target.state !== 'set') {
    return (
      <div className="today-target unset">
        <p>{pageText(locale, 'No target has been set for this window, so nothing here can say whether the week is on plan.', 'לא הוגדר יעד לחלון הזה, ולכן אי אפשר לומר כאן אם השבוע עומד ביעד.')}</p>
        <p className="today-note">{pageText(locale, 'A target is a number a person sets. The plan cannot supply it, because a plan measured against itself is always on plan.', 'יעד הוא מספר שאדם קובע. התוכנית לא יכולה לספק אותו, כי תוכנית שנמדדת מול עצמה תמיד עומדת ביעד.')}</p>
        <OtherWindows target={target} locale={locale} />
        {target.can_edit === false ? (
          <p className="today-note">{refusalText(target, locale)}</p>
        ) : (
          <Button className="today-primary" type="button" variant="contained" onClick={onEdit}>
            {pageText(locale, 'Set the target', 'הגדירו יעד')}
          </Button>
        )}
      </div>
    );
  }
  const variance = finiteNumber(verdict.variance_ils);
  const variancePercent = finiteNumber(verdict.variance_percent);
  const sign = variance !== null && variance > 0 ? '+' : '';
  return (
    <div className="today-target set">
      <div className="today-target-row">
        <span>{pageText(locale, 'Target', 'יעד')}</span>
        <strong><Numeric>{formatCurrency(target.amount_ils, locale)}</Numeric></strong>
      </div>
      <div className="today-target-row">
        <span>{pageText(locale, 'Against the target', 'מול היעד')}</span>
        <strong className={`today-variance ${verdict.state}`}>
          <Numeric>{variance === null ? '-' : `${sign}${formatCurrency(variance, locale)}`}</Numeric>
          {variancePercent === null ? null : <Numeric>{` (${sign}${formatNumber(variancePercent, locale)}%)`}</Numeric>}
        </strong>
      </div>
      <p className="today-note">{pageText(locale, verdict.threshold_en || '', verdict.threshold_he || '')}</p>
      <p className="today-note">{authorLine(target, locale)}</p>
      <OtherWindows target={target} locale={locale} />
      {target.can_edit === false ? (
        <p className="today-note">{refusalText(target, locale)}</p>
      ) : (
        <div className="today-target-actions">
          <Button className="today-secondary" type="button" onClick={onEdit}>
            {pageText(locale, 'Change the target', 'שינוי היעד')}
          </Button>
          <Button className="today-secondary" type="button" onClick={onClear}>
            {pageText(locale, 'Remove the target', 'הסרת היעד')}
          </Button>
        </div>
      )}
    </div>
  );
}

function DayRows({ money, locale, openDate, onToggleDate, onOpenPlan }) {
  const days = Array.isArray(money.days) ? money.days : [];
  const total = finiteNumber(money.amount_ils);
  if (!days.length) {
    return <p className="today-note">{pageText(locale, 'The per-day rows behind this figure are not available from this backend.', 'הפירוט היומי שמאחורי המספר הזה אינו זמין מהשרת הזה.')}</p>;
  }
  return (
    <div className="today-days">
      <div className="today-days-head">
        <span>{pageText(locale, 'Every day in the window, and what it is expected to earn', 'כל יום בחלון, וכמה הוא צפוי להכניס')}</span>
        <Button className="today-icon-button" type="button" onClick={() => exportDays(money)} aria-label={pageText(locale, 'Download these days', 'הורדת הימים האלה')}>
          <Download size={15} />
        </Button>
      </div>
      {days.map((day) => {
        const value = finiteNumber(day.projected_revenue);
        const share = total && value !== null ? (value / total) * 100 : null;
        const open = openDate === day.date;
        return (
          <React.Fragment key={day.date}>
            <Button className={`today-day-row${open ? ' open' : ''}`} type="button" aria-expanded={open} onClick={() => onToggleDate(open ? '' : day.date)}>
              <span className="today-day-name">
                {pageText(locale, day.weekday_en || '', day.weekday_he || '')}
                {day.is_weekend ? <span className="today-weekend">{pageText(locale, 'weekend', 'סוף שבוע')}</span> : null}
              </span>
              <span className="today-day-date"><Numeric>{formatDay(day.date)}</Numeric></span>
              <span className="today-day-breaks"><Numeric>{formatNumber(day.total_breaks, locale)}</Numeric></span>
              <span className="today-day-money"><Numeric>{formatCurrency(day.projected_revenue, locale)}</Numeric></span>
              <span className="today-day-share"><Numeric>{share === null ? '-' : `${formatNumber(Math.round(share * 10) / 10, locale)}%`}</Numeric></span>
            </Button>
            {open ? (
              <TodayDayDetail
                date={day.date}
                scope={money.scope || {}}
                locale={locale}
                onClose={() => onToggleDate('')}
                onWalk={onToggleDate}
                onOpenPlan={onOpenPlan}
              />
            ) : null}
          </React.Fragment>
        );
      })}
      <p className="today-note">
        {money.reconciled
          ? pageText(locale, 'These days sum to the figure above, to the shekel.', 'סכום הימים האלה שווה למספר שלמעלה, עד לשקל.')
          : pageText(locale, 'These days do not sum to the figure above. The difference is shown so it is not hidden.', 'סכום הימים האלה אינו שווה למספר שלמעלה. ההפרש מוצג כדי שלא יוסתר.')}
        {money.reconciled ? '' : ` ${formatCurrency(money.residual_ils, locale)}`}
      </p>
    </div>
  );
}

function exportDays(money) {
  const scope = money.scope || {};
  const text = [
    scopeComment(scope, [['grain', 'broadcast day'], ['rows', (money.days || []).length], ['window_total_ils', money.amount_ils]]),
    '',
    toCsv(DAY_COLUMNS, money.days || []),
  ].join('\n');
  download(`meridian-${scope.date_from || 'window'}-${scope.date_to || 'window'}-days.csv`, text);
}

export function TodayMoney({ today, locale, onOpenPlan, onOpenSettings, onSaveTarget, onClearTarget, saveState }) {
  // The open money day is an address (moneyDay in shell/nav.js). A day in the
  // address means the drill is open on it, so Back walks day -> drill -> panel
  // instead of leaping off the page.
  const [openDate, setOpenDateAddress] = useAddressParam('moneyDay', '');
  const [drillOpen, setDrillOpenState] = useState(() => Boolean(openDate));
  const setDrillOpen = (next) => {
    const resolved = typeof next === 'function' ? next(drillOpen) : next;
    setDrillOpenState(resolved);
    if (!resolved) setOpenDateAddress('');
  };
  const setOpenDate = (value) => {
    const resolved = typeof value === 'function' ? value(openDate) : value;
    setOpenDateAddress(resolved || '');
    if (resolved) setDrillOpenState(true);
  };
  const [formOpen, setFormOpen] = useState(false);
  const [clearReviewOpen, setClearReviewOpen] = useState(false);
  const cancelClearRef = useRef(null);
  const money = today.money || {};
  const verdict = today.verdict || {};
  const dayCount = Array.isArray(money.days) ? money.days.length : 0;
  const notCalendarWeek = today.window && today.window.available && !today.window.is_calendar_week;
  const withheld = money.unavailable;

  if (withheld) {
    return (
      <section className="page-panel today-answer today-answer-money" aria-label={pageText(locale, 'Is this week on plan', 'האם השבוע הזה עומד ביעד')}>
        <div className="today-answer-head">
          <h2>{pageText(locale, 'Is this week on plan', 'האם השבוע הזה עומד ביעד')}</h2>
          <span className="today-verdict unavailable">{pageText(locale, 'Cannot be answered yet', 'עוד אי אפשר לענות')}</span>
        </div>
        <p className="today-note">{pageText(locale, withheld.reason_en, withheld.reason_he)}</p>
        <div className="today-target-actions">
          <Button className="today-primary" type="button" variant="contained" onClick={() => onOpenSettings && onOpenSettings()}>
            {pageText(locale, withheld.needs_en, withheld.needs_he)}
          </Button>
        </div>
      </section>
    );
  }

  return (
    <section className="page-panel today-answer today-answer-money" aria-label={pageText(locale, 'Is this week on plan', 'האם השבוע הזה עומד ביעד')}>
      <div className="today-answer-head">
        <h2>{pageText(locale, 'Is this week on plan', 'האם השבוע הזה עומד ביעד')}</h2>
        <VerdictChip verdict={verdict} locale={locale} />
      </div>
      <div className="today-figure-block">
        <Pressable
          type="button"
          className="today-figure"
          onClick={() => setDrillOpen((open) => !open)}
          aria-expanded={drillOpen}
        >
          <Numeric>{formatCurrency(money.amount_ils, locale)}</Numeric>
          {drillOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </Pressable>
        <span className="today-figure-label">{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</span>
        <span className="today-scope">{scopeLine(money, locale)}</span>
        {notCalendarWeek ? (
          <span className="today-note">{pageText(locale, 'This window is the first seven days of the saved plan, not a calendar week. The plan holds no date inside the current week.', 'החלון הזה הוא שבעת הימים הראשונים של התוכנית השמורה, לא שבוע קלנדרי. בתוכנית אין תאריך שנופל בשבוע הנוכחי.')}</span>
        ) : null}
        <span className="today-drill-hint">
          {drillOpen
            ? pageText(locale, 'Open a day to see the plan rows behind it', 'פתחו יום כדי לראות את שורות התוכנית שמאחוריו')
            : pageText(locale, `Open the ${dayCount} days behind this figure`, `פתחו את ${dayCount} הימים שמאחורי המספר`)}
        </span>
      </div>
      {drillOpen ? (
        <DayRows money={money} locale={locale} openDate={openDate} onToggleDate={setOpenDate} onOpenPlan={onOpenPlan} />
      ) : null}
      {formOpen ? (
        <TargetForm
          today={today}
          locale={locale}
          saveState={saveState}
          onCancel={() => setFormOpen(false)}
          onSave={(values) => {
            setFormOpen(false);
            if (onSaveTarget) onSaveTarget(values);
          }}
        />
      ) : (
        <TargetLine today={today} locale={locale} onEdit={() => setFormOpen(true)} onClear={() => setClearReviewOpen(true)} />
      )}
      <Dialog
        open={clearReviewOpen}
        onClose={() => {
          if (saveState !== 'saving') setClearReviewOpen(false);
        }}
        title={pageText(locale, 'Remove this revenue target?', 'להסיר את יעד ההכנסה הזה?')}
        description={pageText(
          locale,
          'Review the exact scope before changing the commercial record.',
          'בדקו את ההיקף המדויק לפני שינוי הרשומה המסחרית.',
        )}
        closeLabel={pageText(locale, 'Close review', 'סגירת הסקירה')}
        dismissOnBackdrop={false}
        initialFocusRef={cancelClearRef}
        footer={(
          <>
            <Button ref={cancelClearRef} variant="outlined" onClick={() => setClearReviewOpen(false)} disabled={saveState === 'saving'}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
            <Button
              variant="contained"
              color="error"
              loading={saveState === 'saving'}
              onClick={async () => {
                await onClearTarget?.();
                setClearReviewOpen(false);
              }}
            >
              {pageText(locale, 'Remove target', 'הסרת היעד')}
            </Button>
          </>
        )}
      >
        <div className="today-clear-review">
          <dl>
            <div>
              <dt>{pageText(locale, 'Target', 'יעד')}</dt>
              <dd><Numeric>{formatCurrency(today.target?.amount_ils, locale)}</Numeric></dd>
            </div>
            <div>
              <dt>{pageText(locale, 'Window', 'חלון')}</dt>
              <dd>{windowLabel(today.money || {}, locale) || pageText(locale, 'Current saved window', 'החלון השמור הנוכחי')}</dd>
            </div>
          </dl>
          <p>{pageText(locale, 'This removes the target record. It does not change or rerun the saved broadcast plan.', 'הפעולה מסירה את רשומת היעד. היא אינה משנה או מריצה מחדש את תוכנית השידור השמורה.')}</p>
        </div>
      </Dialog>
    </section>
  );
}

export default TodayMoney;
