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
} from '../shell/surface-helpers';
import { Figure } from '../shell/bidi';
import { Card } from '../studio';
import ChannelRefusal from './ChannelRefusal';
import { scopeState, unattributed } from './today-scope';

// YieldView: revenue and yield-per-second by daypart and programme, sourced from
// GET /api/yield-per-second. Surfaces where each ad-second earns the most and
// where inventory is under-monetized. The gross/cost/net money story is NOT
// repeated here: the money story card directly above this one renders exactly
// those three figures from the same payload, so this card keeps only its own
// content (the yield and ad-seconds stat chips plus the two breakdowns) and one
// quiet line pointing up at the money card.

// labelFor localizes the engine group key for display (daypart keys such as
// "prime", classifier program types such as "News"); the row key stays raw.
function YieldBars({ rows, locale, labelKey, labelFor, labelHeading, currency }) {
  const maxYield = Math.max(...rows.map((row) => Number(row.yield_per_second || 0)), 1e-9);
  if (!rows.length) {
    return <div className="heatmap-empty">{pageText(locale, 'No rows available.', 'אין שורות זמינות.')}</div>;
  }
  return (
    <div className="yield-table" role="table" aria-label={labelHeading}>
      <div className="yield-column-head" role="row">
        <span role="columnheader"><strong>{labelHeading}</strong></span>
        <span role="columnheader" className="yield-column-scale">
          <strong>{pageText(locale, 'Yield / ad second', 'תשואה / שניית פרסום')}</strong>
          <small>
            <Figure>{`${currency}/s`}</Figure>
            {' · '}
            {pageText(locale, 'relative scale: highest value in this group = 100%', 'סולם יחסי: הערך הגבוה בקבוצה = 100%')}
          </small>
        </span>
        <span role="columnheader">
          <strong>{pageText(locale, 'Projected revenue', 'הכנסה חזויה')}</strong>
          <small><Figure>{currency}</Figure></small>
        </span>
      </div>
      <div className="yield-bar-list" role="rowgroup">
        {rows.map((row, index) => {
          const yps = Number(row.yield_per_second || 0);
          const relativeYield = Math.max(0, Math.min(100, (yps / maxYield) * 100));
          const label = labelFor ? labelFor(row[labelKey]) : row[labelKey];
          const visibleLabel = label || pageText(locale, 'Unknown', 'לא ידוע');
          return (
            <div className="yield-bar-row" role="row" key={`${row[labelKey] || index}`}>
              <span className="yield-bar-label" role="rowheader">{visibleLabel}</span>
              <span className="yield-rate-cell" role="cell">
                <Figure className="numeric">{formatRate(yps, locale)}</Figure>
                <span
                  className="yield-relative-meter"
                  role="meter"
                  aria-label={pageText(locale, `Relative yield for ${visibleLabel}`, `תשואה יחסית עבור ${visibleLabel}`)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(relativeYield)}
                  aria-valuetext={pageText(locale, `${Math.round(relativeYield)} percent of the highest value in this group`, `${Math.round(relativeYield)} אחוז מהערך הגבוה בקבוצה`)}
                >
                  <i style={{ '--bar': relativeYield / 100 }} aria-hidden="true" />
                </span>
              </span>
              <span className="yield-revenue-cell" role="cell">
                <Figure className="numeric">{formatCurrency(row.revenue, locale)}</Figure>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function YieldView({ locale, refreshKey = 0, onOpenSettings = null }) {
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
  // The rate, the ad seconds and both breakdowns are the whole market's until a
  // channel is declared, and the payload says so. A per-daypart league table of
  // four broadcasters' revenue is the same breach as the total it sums to, so
  // the panel refuses as one.
  const withheld = unattributed(scopeState(payload, available));
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
          <span className="bidi-figure figure-nowrap">{spot30}</span>
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
      ) : withheld ? (
        <ChannelRefusal
          locale={locale}
          lead={pageText(
            locale,
            'Yield per second cannot be reported as yours yet.',
            'אי אפשר עדיין לדווח על התשואה לשנייה כשלכם.',
          )}
          onOpenSettings={onOpenSettings}
        />
      ) : (
        <>
          <p className="yield-net-pointer">
            {pageText(locale, 'Full net in the from gross to net card.', 'נטו מלא בכרטיס מברוטו לנטו.')}
          </p>

          <div className="yield-totals">
            <Tooltip title={yieldTooltip} arrow placement="bottom">
              <div className="yield-total-card">
                <span><Gauge size={13} /> {pageText(locale, 'Yield per second', 'תשואה לשנייה')}</span>
                <Figure className="numeric">{formatRate(totals.yield_per_second, locale)}</Figure>
                <small className="yield-unit-line">{pageText(locale, `${currency} per ad second`, '₪ לשנ׳ פרסום')}</small>
              </div>
            </Tooltip>
            <div className="yield-total-card">
              <span>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</span>
              <Figure className="numeric">{formatSeconds(totals.ad_seconds, locale)}</Figure>
            </div>
          </div>

          <div className="yield-split">
            <Card as="div" className="yield-split-col yield-split-panel">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By daypart', 'לפי רצועת שידור')}</h3>
                <span>{byDaypart.length}</span>
              </div>
              <YieldBars rows={byDaypart} locale={locale} labelKey="group" currency={currency}
                         labelHeading={pageText(locale, 'Daypart', 'רצועת שידור')}
                         labelFor={(value) => daypartLabel(value, locale)} />
            </Card>
            <Card as="div" className="yield-split-col yield-split-panel">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By programme', 'לפי תוכנית')}</h3>
                <span>{byProgramme.length}</span>
              </div>
              <YieldBars rows={byProgramme} locale={locale} labelKey="group" currency={currency}
                         labelHeading={pageText(locale, 'Programme', 'תוכנית')}
                         labelFor={(value) => programTypeLabel(value, locale)} />
            </Card>
          </div>

          <p className="yield-foot-note">
            {pageText(locale, 'Yield per second = revenue / ad-seconds', 'תשואה לשנייה = הכנסה / שניות פרסום')}
            {' ('}
            <span className="bidi-figure figure-nowrap">{`${currency}/s`}</span>
            {').'}
          </p>
        </>
      )}
    </section>
  );
}
