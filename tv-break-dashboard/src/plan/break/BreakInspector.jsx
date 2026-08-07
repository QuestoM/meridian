import React, { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, X } from 'lucide-react';
import { formatNumber, formatPercent, pageText } from '../../shell/format';
import { exactCurrency } from '../day/day-board-model';
import { violationLabel } from '../day/DayBoardReadout';
import ScheduleInspector, { confidenceLabel } from '../day/ScheduleInspector';
import PodBoard from './PodBoard';
import './break-inspector.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// One break, opened.
//
// Every figure carries the scope it was computed on, and the two money figures
// are never added together, because they are not the same currency yet:
// projected comes from the weekly plan and delivered comes from a daily spot
// ledger, and on this data they overlap on zero dates. So delivered is a state
// with a reason and a path forward, and it will never show a number nobody read.
//
// Opening a break keeps its place in the set it was opened from. The header
// prints the position in that set and carries the two arrows that walk it, so a
// person checking eight breaks in a row never returns to the board between them.
// The keyboard does the same with the up and down arrows.
//
// Nothing on this drawer is a dead end. The programme title opens the programme's
// own record, the hour opens the breaks the plan puts in it, and a saved
// placement opens the record naming who saved it and which restriction carries
// it. A figure a person cannot walk from is a figure they have to take on trust.
function BreakInspector({ breakId, locale, onClose, siblings, onNavigate, notify, onGlobalRefresh }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');
  // Which record is open on top of this one. The programme is a drawer of its
  // own; the hour and the saved placement are disclosures inside this drawer.
  const [programmeOpen, setProgrammeOpen] = useState(false);
  const [hourOpen, setHourOpen] = useState(false);
  const [pinOpen, setPinOpen] = useState(false);
  const set = Array.isArray(siblings) ? siblings : [];
  const index = set.indexOf(breakId);
  const walkable = index >= 0 && set.length > 1 && typeof onNavigate === 'function';
  // The objects behind the hour's own figure. An older payload without them
  // leaves the disclosure off rather than offering an empty list.
  const hourBreaks = (detail && detail.guardrails && detail.guardrails.hour_breaks) || [];

  useEffect(() => {
    let alive = true;
    setDetail(null);
    setError('');
    setProgrammeOpen(false);
    setHourOpen(false);
    setPinOpen(false);
    fetch(`${API_BASE}/api/breaks/${encodeURIComponent(breakId)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
      })
      .then((payload) => { if (alive) setDetail(payload); })
      .catch((fetchError) => { if (alive) setError(fetchError.message); });
    return () => { alive = false; };
  }, [breakId]);

  useEffect(() => {
    function step(direction) {
      if (!walkable) return;
      const next = set[index + direction];
      if (next) onNavigate(next);
    }
    function onKey(event) {
      // Escape closes the record on top, not the stack. With the programme open
      // over this drawer, one Escape puts the programme away and leaves the
      // break where it was, which is what a person who opened two things means.
      if (event.key === 'Escape') {
        if (programmeOpen) setProgrammeOpen(false);
        else onClose();
        return;
      }
      // The arrows walk this drawer's set, and the programme drawer over it has
      // fields of its own, so they stop at the top record while it is open.
      if (programmeOpen) return;
      if (event.key === 'ArrowDown' && walkable) {
        event.preventDefault();
        step(1);
      } else if (event.key === 'ArrowUp' && walkable) {
        event.preventDefault();
        step(-1);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, walkable, index, set, onNavigate, programmeOpen]);

  return (
    <>
    <aside className="break-inspector" dir={he ? 'rtl' : 'ltr'} role="dialog" aria-label={label('Break detail', 'פרטי הברייק')}>
      <header className="break-inspector-head">
        <div>
          <h2>{label('Break', 'ברייק')}</h2>
          <code dir="ltr">{breakId}</code>
        </div>
        {walkable && (
          <div className="break-inspector-walk">
            <span dir="ltr">{index + 1} / {set.length}</span>
            <button
              type="button"
              onClick={() => onNavigate(set[index - 1])}
              disabled={index === 0}
              aria-label={label('Previous break in this list', 'הברייק הקודם ברשימה')}
            >
              <ChevronUp size={14} />
            </button>
            <button
              type="button"
              onClick={() => onNavigate(set[index + 1])}
              disabled={index === set.length - 1}
              aria-label={label('Next break in this list', 'הברייק הבא ברשימה')}
            >
              <ChevronDown size={14} />
            </button>
          </div>
        )}
        <button type="button" onClick={onClose} aria-label={label('Close', 'סגירה')}>
          <X size={16} />
        </button>
      </header>

      {error && <p className="break-inspector-error" dir="auto">{error}</p>}
      {!detail && !error && <p>{label('Opening', 'פותח')}</p>}

      {detail && (
        <div className="break-inspector-body">
          <section>
            <h3>{label('Where it sits', 'היכן הוא יושב')}</h3>
            <dl>
              <dt>{pageText(locale, 'Programme', 'תוכנית')}</dt>
              <dd>
                <button
                  type="button"
                  className="break-open"
                  dir="auto"
                  onClick={() => setProgrammeOpen(true)}
                  aria-label={`${detail.programme.title}, ${label('open the programme record', 'פתיחת רשומת התוכנית')}`}
                >
                  {detail.programme.title}
                </button>
              </dd>
              <dt>{pageText(locale, 'Programme window', 'חלון התוכנית')}</dt>
              <dd dir="ltr">{detail.programme.start_clock} - {detail.programme.end_clock}</dd>
              <dt>{pageText(locale, 'Break window', 'חלון הברייק')}</dt>
              <dd dir="ltr">{detail.placement.start_clock} - {detail.placement.end_clock}</dd>
              <dt>{pageText(locale, 'Length', 'אורך')}</dt>
              <dd dir="ltr">{formatNumber(detail.placement.duration_seconds, locale)}s</dd>
              <dt>{pageText(locale, 'Order in the programme', 'סדר בתוך התוכנית')}</dt>
              <dd dir="ltr">{detail.identity.ordinal} / {detail.identity.breaks_in_programme}</dd>
              <dt>{pageText(locale, 'Placed by', 'נקבע על ידי')}</dt>
              <dd>{detail.placement.source === 'operator' ? label('the operator', 'המפעיל') : label('the plan', 'התוכנית')}</dd>
              {detail.placement.saved_placement && (
                <>
                  <dt>{label('Restriction holding it', 'המגבלה שנושאת אותה')}</dt>
                  <dd>
                    <button
                      type="button"
                      className="break-open"
                      dir="ltr"
                      aria-expanded={pinOpen}
                      onClick={() => setPinOpen((open) => !open)}
                    >
                      {detail.placement.saved_placement.constraint_id || label('none on record', 'לא רשומה')}
                    </button>
                  </dd>
                </>
              )}
            </dl>
            {detail.placement.saved_placement && pinOpen && (
              <dl className="break-price">
                <dt>{label('Saved by', 'נשמרה על ידי')}</dt>
                <dd dir="auto">{detail.placement.saved_placement.actor || label('no account recorded', 'לא נרשם חשבון')}</dd>
                <dt>{label('Saved at', 'נשמרה בתאריך')}</dt>
                <dd dir="ltr">{detail.placement.saved_placement.saved_at || label('not recorded', 'לא נרשם')}</dd>
                <dt>{label('Offset it holds', 'ההיסט שהיא מקבעת')}</dt>
                <dd dir="ltr">{formatNumber(Number(detail.placement.saved_placement.offset_seconds), locale)}s</dd>
                <dt>{label('Length it holds', 'האורך שהיא מקבעת')}</dt>
                <dd dir="ltr">{formatNumber(Number(detail.placement.saved_placement.duration_seconds), locale)}s</dd>
                <dt>{label('Note', 'הערה')}</dt>
                <dd dir="auto">{detail.placement.saved_placement.note || label('none', 'אין')}</dd>
              </dl>
            )}
            {detail.placement.saved_placement && (
              <p className="break-basis" dir="auto">
                {label(
                  'Select this break on the day board to remove the saved placement and let the plan place it again.',
                  'בחרו את הברייק הזה בלוח היום כדי להסיר את הנעיצה השמורה ולהחזיר את המיקום לתוכנית.',
                )}
              </p>
            )}
          </section>

          <section>
            <h3>{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</h3>
            <p className="break-money" dir="ltr">{exactCurrency(detail.money.projected.amount, locale)}</p>
            <p className="break-basis" dir="auto">{say(detail.money.projected, 'basis', he)}</p>
            <dl className="break-price">
              <dt>{label('Rate per point', 'מחיר לנקודת רייטינג')}</dt>
              <dd dir="ltr">{decimal(detail.programme.rate_per_point, 2)}</dd>
              <dt>{label('Baseline rating of the programme', 'רייטינג הבסיס של התוכנית')}</dt>
              <dd dir="ltr">{decimal(detail.programme.baseline_rating, 3)}</dd>
              <dt>{label('Retention once this break is present', 'השימור ברגע שהברייק הזה קיים')}</dt>
              <dd dir="ltr">{decimal(detail.money.projected.retention_at_this_break * 100, 4)}%</dd>
              <dt>{label('Rating this break is priced at', 'הרייטינג שלפיו מתומחר הברייק')}</dt>
              <dd dir="ltr">{decimal(detail.money.projected.rating_at_this_break, 6)}</dd>
              <dt>{label('Length over the rate unit', 'האורך חלקי יחידת המחירון')}</dt>
              <dd dir="ltr">{decimal(detail.placement.duration_seconds, 0)}s / {decimal(detail.programme.rate_unit_seconds, 0)}s</dd>
              <dt>{label('Premium', 'פרמיה')}</dt>
              <dd dir="ltr">{decimal(detail.programme.premium, 4)}</dd>
            </dl>
            <p className="break-basis" dir="auto">{say(detail.money.projected, 'formula', he)}</p>
            <p className="break-basis" dir="auto">{say(detail.money.projected, 'rating_formula', he)}</p>
            <h3>{label('Delivered', 'שסופק בפועל')}</h3>
            <p className="break-unavailable" dir="auto">{say(detail.money.delivered, 'reason', he)}</p>
            {detail.money.delivered.path_forward && (
              <p className="break-basis" dir="auto">{say(detail.money.delivered, 'path_forward', he)}</p>
            )}
          </section>

          <section>
            <h3>{pageText(locale, 'Retention cost', 'עלות שימור')}</h3>
            <dl>
              <dt>{label('Programme retention with this plan', 'שימור התוכנית בתוכנית הזו')}</dt>
              <dd dir="ltr">{formatPercent(detail.retention.programme_retention * 100, locale)}</dd>
              <dt>{label('Cost per break', 'עלות לכל ברייק')}</dt>
              <dd dir="ltr">{formatPercent(detail.retention.cost_per_break * 100, locale)}</dd>
              <dt>{label('Credible interval', 'רווח סמך')}</dt>
              <dd dir="ltr">
                {detail.retention.ci_low === null
                  ? label('point estimate only', 'אומדן נקודתי בלבד')
                  : `${formatPercent(detail.retention.ci_low * 100, locale)} .. ${formatPercent(detail.retention.ci_high * 100, locale)}`}
              </dd>
              <dt>{label('Measured on', 'נמדד על')}</dt>
              <dd dir="ltr">{formatNumber(detail.retention.sample_breaks, locale)}</dd>
              <dt>{label('Confidence', 'רמת ביטחון')}</dt>
              <dd dir="auto">{confidenceLabel(detail.retention.confidence, locale)}</dd>
            </dl>
          </section>

          <section>
            <h3>{pageText(locale, 'Regulatory guardrail', 'מגבלת רגולציה')}</h3>
            {detail.guardrails.hour && (
              <p dir="ltr">
                {String(detail.guardrails.hour.hour % 24).padStart(2, '0')}:00 {' '}
                {formatNumber(detail.guardrails.hour.ad_seconds, locale)}s / {formatNumber(detail.guardrails.hour.max_ad_seconds, locale)}s,
                {' '}{detail.guardrails.hour.breaks} / {detail.guardrails.hour.max_breaks}
              </p>
            )}
            {hourBreaks.length > 0 && (
              <button
                type="button"
                className="break-open break-open-row"
                aria-expanded={hourOpen}
                onClick={() => setHourOpen((open) => !open)}
              >
                <span dir="auto">{label('The breaks in this hour', 'הברייקים בשעה הזו')} ({hourBreaks.length})</span>
                {hourOpen ? <ChevronUp size={12} aria-hidden="true" /> : <ChevronDown size={12} aria-hidden="true" />}
              </button>
            )}
            {hourOpen && (
              <ul className="break-hour-list">
                {hourBreaks.map((row) => (
                  <li key={row.break_id}>
                    <button
                      type="button"
                      className="break-open"
                      disabled={row.break_id === breakId || typeof onNavigate !== 'function'}
                      onClick={() => onNavigate(row.break_id)}
                    >
                      <span dir="ltr">{row.start_clock}</span>
                      <span dir="auto">{row.programme}</span>
                      <span dir="ltr">{formatNumber(row.duration_seconds, locale)}s</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <p dir="auto">
              {label('Gap to the break before', 'מרווח מהברייק הקודם')}: <span dir="ltr">{gapText(detail.guardrails.spacing.gap_before_seconds, locale, label)}</span>
            </p>
            <p dir="auto">
              {label('Gap to the break after', 'מרווח לברייק הבא')}: <span dir="ltr">{gapText(detail.guardrails.spacing.gap_after_seconds, locale, label)}</span>
            </p>
            <ul className="break-violations">
              {(detail.compliance.violations || []).map((violation, index) => (
                <li key={`${violation.code}-${index}`}>
                  {violationLabel(violation.code, locale)} <span dir="ltr">{violation.observed} / {violation.limit}</span>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h3>{pageText(locale, 'Break contents', 'תוכן הברייק')}</h3>
            {detail.contents.state === 'real' && detail.contents.pod ? (
              // The same component the contents page draws, given the same pod.
              // The reorder acts are deliberately absent here: this drawer is
              // opened from a plan board to read a break, and a write with no
              // inverse on the surface that performed it is the defect P3 spent a
              // round closing. The pod's own page carries the acts and the inverse.
              <PodBoard pod={detail.contents.pod} locale={locale} busy onSaveOrder={noNotify} onRevertOrder={noNotify} />
            ) : (
              <>
                <p className="break-unavailable" dir="auto">{say(detail.contents, 'reason', he)}</p>
                <p className="break-basis" dir="auto">{say(detail.contents, 'path_forward', he)}</p>
                {(detail.contents.covered_days || []).length > 0 && (
                  <p className="break-basis" dir="auto">
                    {label('A traffic file covers', 'קובץ טראפיק מכסה')}: <span dir="ltr">{detail.contents.covered_days.join(', ')}</span>
                  </p>
                )}
              </>
            )}
          </section>
        </div>
      )}
    </aside>
    {programmeOpen && detail && (
      <ScheduleInspector
        segmentId={detail.programme.segment_id}
        channel={detail.identity.channel}
        day={detail.identity.day}
        locale={locale}
        notify={notify || noNotify}
        onClose={() => setProgrammeOpen(false)}
        onGlobalRefresh={onGlobalRefresh}
      />
    )}
    </>
  );
}

// The drawer is mounted from two boards and one of them may not carry the
// notifier. The programme record still opens; its own messages have nowhere to
// go, which is better than a click that throws.
function noNotify() {}

// A payload string in the reader's own language. Every honest empty state on
// this surface ships both, so a Hebrew operator never reads an English excuse.
function say(record, key, he) {
  if (!record) return '';
  return (he && record[`${key}_he`]) || record[key] || '';
}

// The price inputs are printed to the precision that reproduces the price, not to
// the shared formatter's one decimal. A rating of 1.617105 printed as 1.6 turns a
// basis a person can check into a basis they have to believe: the six digits here
// multiply back to the plan's own credit to within 0.01 ILS, measured across all
// 80 breaks of רשת 13 / 2024-11-01.
function decimal(value, digits) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return number.toFixed(digits);
}

function gapText(seconds, locale, label) {
  if (seconds === null || seconds === undefined) return label('no neighbour', 'אין שכן');
  return `${formatNumber(Math.round(seconds), locale)}s`;
}

export default BreakInspector;
