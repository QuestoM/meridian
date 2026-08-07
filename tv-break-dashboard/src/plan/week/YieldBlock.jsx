import React from 'react';
import { Button } from '@mui/material';
import { Coins } from 'lucide-react';
import { finiteNumber, formatCurrency, formatNumber, pageText } from '../../shell/format';
import { Code, Figure, Name } from '../../shell/bidi';
import { DataTable } from '../../shell/primitives';
import { daypartLabel as engineDaypartLabel } from '../../shell/surface-helpers';
import {
  bandWords,
  basisFormula,
  basisInputs,
  basisIsUnfamiliar,
  unfamiliarBasisWords,
  yieldFormulaWords,
} from './plan-week-basis';

// What a second of airtime is worth.
//
// This is the revenue and yield owner's own question and the rate card is the
// door that answers it, so the figure is computed once by the piece that owns
// the card and read here rather than computed a second time. It sits beside the supply
// because supply counts the seconds and this prices them, and the planner needs
// both in the same breath to choose an objective.
//
// The basis is printed with the figure and never in a tooltip: what the number
// was computed from, the scope it was summed on, and the measured uncertainty
// band on the retention cost, which is a real interval from the model's own
// calibrated coefficients rather than a rounding.
//
// It is printed in the operator's language. The server states its basis in
// engine field names and one English paragraph, and neither is a disclosure to
// somebody reading Hebrew, so plan-week-basis.js carries the words and this file
// keeps the engine's own text one disclosure away for whoever reconciles a
// figure against the code.

// A second is a small unit and the compact shell formatter rounds it to whole
// shekels, which turns 142.7044 into 143 and hides the difference between two
// rate cards. This is the same real number printed at the grain the question is
// asked at, never a different one.
function perSecond(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(number);
}

function BasisBlock({ basis, bandBasis, locale }) {
  const formula = basisFormula(basis);
  if (!formula) return null;
  const words = yieldFormulaWords(basis, locale);
  const inputs = basisInputs(basis);
  return (
    <div className="plan-basis">
      <p className="plan-basis-note">
        {pageText(locale, 'How it is computed: ', 'איך זה מחושב: ')}
        {words || (basisIsUnfamiliar(basis) ? unfamiliarBasisWords(locale) : null)}
      </p>
      {bandBasis && <p className="plan-basis-note">{bandWords(locale)}</p>}
      <details className="plan-basis-details">
        <summary>{pageText(locale, 'The wording the engine itself uses', 'הניסוח של המנוע עצמו')}</summary>
        <p className="numeric" lang="en">{formula}</p>
        {inputs.length > 0 && (
          <dl className="plan-basis-inputs" lang="en">
            {inputs.map((input) => (
              <div key={input.name}>
                <dt className="numeric"><Code>{input.name}</Code></dt>
                <dd>{input.note}</dd>
              </div>
            ))}
          </dl>
        )}
        {bandBasis && <p lang="en">{bandBasis}</p>}
      </details>
    </div>
  );
}

// The window these three money figures were summed over, printed with them.
//
// This destination reports the plan week everywhere else, and the worth of a
// second is computed across the whole saved plan, so the same label would carry
// two quantities on one screen unless each says which window it is. The span
// comes from the plan's own scope block on /api/plan-progress; when that read
// has not landed, the line still says which of the two windows this is and
// claims no dates it cannot show.
function planWindowLine(scope, locale) {
  const from = scope?.plan_date_from;
  const to = scope?.plan_date_to;
  const days = finiteNumber(scope?.plan_n_dates);
  if (!from || !to || days === null) {
    return pageText(locale, 'The whole saved plan, not the plan week', 'כל התוכנית השמורה, לא שבוע התוכנית');
  }
  return pageText(
    locale,
    `The whole saved plan, ${formatNumber(days, locale)} broadcast days, ${from} to ${to}, not the plan week above`,
    `כל התוכנית השמורה, ${formatNumber(days, locale)} ימי שידור, ${from} עד ${to}, ולא שבוע התוכנית שלמעלה`,
  );
}

export function YieldBlock({ data, locale, words, planScope }) {
  if (!data) return null;
  if (data.available === false) {
    return (
      <p className="plan-note plan-note-amber" role="status">
        {pageText(
          locale,
          'What a second of airtime is worth cannot be computed from the saved plan right now, so no figure is shown.',
          'לא ניתן לחשב כרגע מהתוכנית השמורה כמה שווה שנייה של זמן שידור, ולכן לא מוצג ערך.',
        )}
        {data.reason ? <small className="plan-note-detail"><Name>{data.reason}</Name></small> : null}
      </p>
    );
  }

  const totals = data.totals || {};
  const dayparts = Array.isArray(data.by_daypart) ? [...data.by_daypart] : [];
  dayparts.sort((a, b) => Number(b.yield_per_second || 0) - Number(a.yield_per_second || 0));
  const band = Number.isFinite(Number(data.retention_cost_low)) && Number.isFinite(Number(data.retention_cost_high));

  return (
    <div className="plan-yield" id="plan-yield">
      <div className="plan-section-subhead">
        <h3><Coins size={14} /> {pageText(locale, 'What a second of airtime is worth', 'כמה שווה שנייה של זמן שידור')}</h3>
        <Button
          className="secondary-button compact"
          type="button"
          variant="outlined"
          onClick={() => { window.location.hash = 'Pricing'; }}
        >
          {pageText(locale, 'Open the rate card', 'פתחו את כרטיס התעריפים')}
        </Button>
      </div>

      <p className="plan-money-window">{planWindowLine(planScope, locale)}</p>

      <div className="plan-figure-row">
        <div className="plan-figure is-headline">
          <span>{pageText(locale, 'Per second, across the saved plan', 'לשנייה, על פני התוכנית השמורה')}</span>
          <strong className="numeric"><Figure>{perSecond(totals.yield_per_second, locale)}</Figure></strong>
          <small>
            {pageText(
              locale,
              `${formatCurrency(totals.revenue, locale)} over ${formatNumber(totals.ad_seconds, locale)} ad seconds, ${formatNumber(totals.segment_count, locale)} programme segments`,
              `${formatCurrency(totals.revenue, locale)} על פני ${formatNumber(totals.ad_seconds, locale)} שניות פרסום, ${formatNumber(totals.segment_count, locale)} מקטעי תוכנית`,
            )}
          </small>
        </div>
        <div className="plan-figure">
          <span>{words.expectedRevenue}</span>
          <strong className="numeric"><Figure>{formatCurrency(data.revenue_ils, locale)}</Figure></strong>
        </div>
        <div className="plan-figure">
          <span>{words.retentionCost}</span>
          <strong className="numeric"><Figure>{formatCurrency(data.retention_cost_ils, locale)}</Figure></strong>
          {band ? (
            <small className="plan-band">
              {pageText(locale, 'band ', 'טווח ')}
              <span className="numeric">{formatCurrency(data.retention_cost_low, locale)}</span>
              {pageText(locale, ' to ', ' עד ')}
              <span className="numeric">{formatCurrency(data.retention_cost_high, locale)}</span>
            </small>
          ) : null}
        </div>
        <div className="plan-figure">
          <span>{pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור')}</span>
          <strong className="numeric"><Figure>{formatCurrency(data.revenue_net_ils, locale)}</Figure></strong>
        </div>
      </div>

      <div className="plan-section-subhead">
        <h3>{pageText(locale, 'Worth of a second by broadcast strip', 'שווי שנייה לפי רצועת שידור')}</h3>
        <span>{formatNumber(dayparts.length, locale)}</span>
      </div>
      <DataTable
        locale={locale}
        fit
        emptyLabel={pageText(locale, 'The saved plan carries no priced segments yet.', 'בתוכנית השמורה עדיין אין מקטעים מתומחרים.')}
        rows={dayparts}
        columns={[
          { key: 'group', label: pageText(locale, 'Broadcast strip', 'רצועת שידור'), render: (row) => engineDaypartLabel(row.group, locale) },
          { key: 'yield_per_second', label: pageText(locale, 'Per second', 'לשנייה'), render: (row) => perSecond(row.yield_per_second, locale) },
          { key: 'revenue', label: words.expectedRevenue, render: (row) => formatCurrency(row.revenue, locale) },
          { key: 'ad_seconds', label: pageText(locale, 'Ad seconds', 'שניות פרסום'), render: (row) => formatNumber(row.ad_seconds, locale) },
          { key: 'break_count', label: words.breaks, render: (row) => formatNumber(row.break_count, locale) },
        ]}
      />
      <BasisBlock basis={data.basis} bandBasis={band ? data.retention_cost_basis : null} locale={locale} />
    </div>
  );
}

export default YieldBlock;
