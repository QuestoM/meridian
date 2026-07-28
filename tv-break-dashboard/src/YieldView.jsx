import React, { useEffect, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Gauge } from 'lucide-react';
import {
  API_BASE,
  daypartLabel,
  finiteNumber,
  formatCurrency,
  formatRate,
  formatSeconds,
  normalizeRows,
  pageText,
  programTypeLabel,
} from './surface-helpers';

// YieldView: revenue and yield-per-second by daypart and programme, sourced from
// GET /api/yield-per-second. Surfaces where each ad-second earns the most and
// where inventory is under-monetized. The gross/cost/net money story is NOT
// repeated here: the money story card directly above this one renders exactly
// those three figures from the same payload, so this card keeps only its own
// content (the yield and ad-seconds stat chips plus the two breakdowns) and one
// quiet line pointing up at the money card.

// labelFor localizes the engine group key for display (daypart keys such as
// "prime", classifier program types such as "News"); the row key stays raw.
function YieldBars({ rows, locale, labelKey, labelFor }) {
  const maxYield = Math.max(...rows.map((row) => Number(row.yield_per_second || 0)), 1e-9);
  if (!rows.length) {
    return <div className="heatmap-empty">{pageText(locale, 'No rows available.', 'אין שורות זמינות.')}</div>;
  }
  return (
    <div className="yield-bar-list chart-ltr" dir="ltr">
      {rows.map((row, index) => {
        const yps = Number(row.yield_per_second || 0);
        const label = labelFor ? labelFor(row[labelKey]) : row[labelKey];
        return (
          <div className="yield-bar-row" key={`${row[labelKey] || index}`}>
            {/* Native title is a truncation echo of the ellipsised label, not an explanation. */}
            <span className="yield-bar-label" title={String(label || '')}>{label || pageText(locale, 'Unknown', 'לא ידוע')}</span>
            <i style={{ '--bar': yps / maxYield }} />
            <strong className="numeric" dir="ltr">{formatRate(yps, locale)}</strong>
            <small className="numeric" dir="ltr">{formatCurrency(row.revenue, locale)}</small>
          </div>
        );
      })}
    </div>
  );
}

export default function YieldView({ locale, refreshKey = 0 }) {
  const he = locale === 'he';
  const [state, setState] = useState({ status: 'loading', payload: null });

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/yield-per-second`)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (!active) return;
        setState({ status: 'ready', payload });
      })
      .catch(() => {
        if (!active) return;
        setState({ status: 'error', payload: null });
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const { status, payload } = state;
  const available = status === 'ready' && payload && payload.available !== false;
  const totals = payload?.totals || {};
  const byDaypart = normalizeRows(payload?.by_daypart);
  const byProgramme = normalizeRows(payload?.by_programme);
  const currency = payload?.currency || 'ILS';
  // Ground the per-second rate with a concrete read: what a standard 30 second
  // spot earns on average at this rate. Rendered only from the real total.
  const yieldValue = finiteNumber(totals.yield_per_second);
  const spot30 = yieldValue !== null ? formatCurrency(yieldValue * 30, locale) : null;
  const yieldTooltip = (
    <>
      {pageText(locale, 'Revenue divided by total ad seconds: what one second of commercial airtime earns on average across all of its rating, not per single rating point.', 'הכנסה חלקי סך שניות הפרסום: כמה מרוויחה בממוצע שנייה אחת של זמן פרסום על כל הרייטינג שלה, לא לנקודת רייטינג בודדת.')}
      {spot30 ? (
        <>
          {' '}
          {pageText(locale, 'For example, a 30 second spot earns on average about', 'לדוגמה, ספוט של 30 שניות מכניס בממוצע בערך')}
          {' '}
          <span className="ltr-run">{spot30}</span>
          {'.'}
        </>
      ) : null}
    </>
  );

  return (
    <section className="page-panel yield-view">
      <div className="panel-head">
        <h2>{pageText(locale, 'Yield per second', 'תשואה לשנייה')}</h2>
        <span>{pageText(locale, 'Where each ad-second earns most', 'היכן כל שניית פרסום מרוויחה הכי הרבה')}</span>
      </div>

      {status === 'loading' ? (
        <div className="frontier-skeleton" aria-hidden="true" />
      ) : !available ? (
        <div className="heatmap-empty">
          {status === 'error'
            ? pageText(locale, 'Yield data is unavailable right now.', 'נתוני התשואה אינם זמינים כרגע.')
            : pageText(locale, 'No yield data is available yet.', 'אין נתוני תשואה זמינים עדיין.')}
        </div>
      ) : (
        <>
          <p className="yield-net-pointer" dir={he ? 'rtl' : 'ltr'}>
            {pageText(locale, 'Full net in the from gross to net card.', 'נטו מלא בכרטיס מברוטו לנטו.')}
          </p>

          <div className="yield-totals" dir={he ? 'rtl' : 'ltr'}>
            <Tooltip title={yieldTooltip} arrow placement="bottom">
              <div className="yield-total-card">
                <span><Gauge size={13} /> {pageText(locale, 'Yield per second', 'תשואה לשנייה')}</span>
                <strong className="numeric" dir="ltr">{formatRate(totals.yield_per_second, locale)}</strong>
                <small className="yield-unit-line">{pageText(locale, `${currency} per ad second`, '₪ לשנ׳ פרסום')}</small>
              </div>
            </Tooltip>
            <div className="yield-total-card">
              <span>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</span>
              <strong className="numeric" dir="ltr">{formatSeconds(totals.ad_seconds, locale)}</strong>
            </div>
          </div>

          <div className="yield-split">
            <div className="yield-split-col yield-split-panel">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By daypart', 'לפי רצועת שידור')}</h3>
                <span>{byDaypart.length}</span>
              </div>
              <YieldBars rows={byDaypart} locale={locale} labelKey="group" labelFor={(value) => daypartLabel(value, locale)} />
            </div>
            <div className="yield-split-col yield-split-panel">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By programme', 'לפי תוכנית')}</h3>
                <span>{byProgramme.length}</span>
              </div>
              <YieldBars rows={byProgramme} locale={locale} labelKey="group" labelFor={(value) => programTypeLabel(value, locale)} />
            </div>
          </div>

          <p className="yield-foot-note">
            {pageText(locale, 'Yield per second = revenue / ad-seconds', 'תשואה לשנייה = הכנסה / שניות פרסום')}
            {' ('}
            <span className="ltr-run">{`${currency}/s`}</span>
            {').'}
          </p>
        </>
      )}
    </section>
  );
}
