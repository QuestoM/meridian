import React, { useEffect, useState } from 'react';
import { CircleDollarSign, Gauge } from 'lucide-react';
import {
  API_BASE,
  formatCurrency,
  formatRate,
  formatSeconds,
  normalizeRows,
  pageText,
} from './surface-helpers';

// YieldView: revenue and yield-per-second by daypart and programme, sourced from
// GET /api/yield-per-second. Surfaces where each ad-second earns the most and
// where inventory is under-monetized. Revenue net of retention is shown in ILS
// when the endpoint computes it exactly (revenue_net_available:true), with the
// retention cost and a basis disclosure; when the endpoint cannot compute it
// honestly, the reason is shown and no figure is fabricated.

function YieldBars({ rows, locale, labelKey }) {
  const maxYield = Math.max(...rows.map((row) => Number(row.yield_per_second || 0)), 1e-9);
  if (!rows.length) {
    return <div className="heatmap-empty">{pageText(locale, 'No rows available.', 'אין שורות זמינות.')}</div>;
  }
  return (
    <div className="yield-bar-list chart-ltr" dir="ltr">
      {rows.map((row, index) => {
        const yps = Number(row.yield_per_second || 0);
        return (
          <div className="yield-bar-row" key={`${row[labelKey] || index}`}>
            <span className="yield-bar-label" title={String(row[labelKey] || '')}>{row[labelKey] || pageText(locale, 'Unknown', 'לא ידוע')}</span>
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
  const netAvailable = Boolean(payload?.revenue_net_available);
  const basisFormula = payload?.basis?.formula || '';

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
          <div className="yield-totals" dir={he ? 'rtl' : 'ltr'}>
            <div className="yield-total-card">
              <span><Gauge size={13} /> {pageText(locale, 'Yield per second', 'תשואה לשנייה')}</span>
              <strong className="numeric" dir="ltr">{formatRate(totals.yield_per_second, locale)} {currency}/s</strong>
            </div>
            <div className="yield-total-card">
              <span><CircleDollarSign size={13} /> {pageText(locale, 'Total revenue', 'הכנסה כוללת')}</span>
              <strong className="numeric" dir="ltr">{formatCurrency(totals.revenue, locale)}</strong>
            </div>
            <div className="yield-total-card">
              <span>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</span>
              <strong className="numeric" dir="ltr">{formatSeconds(totals.ad_seconds, locale)}</strong>
            </div>
            <div className={netAvailable ? 'yield-total-card' : 'yield-total-card muted'}>
              <span>{pageText(locale, 'Revenue net of retention', 'הכנסה בניכוי שימור')}</span>
              {netAvailable ? (
                <strong className="numeric" dir="ltr">{formatCurrency(payload.revenue_net_ils, locale)}</strong>
              ) : (
                <strong>{pageText(locale, 'Not available', 'לא זמין')}</strong>
              )}
            </div>
            {netAvailable && (
              <div className="yield-total-card muted">
                <span>{pageText(locale, 'Retention cost priced in', 'עלות שימור מתומחרת')}</span>
                <strong className="numeric" dir="ltr">{formatCurrency(payload.retention_cost_ils, locale)}</strong>
              </div>
            )}
          </div>

          <div className="yield-split">
            <div className="yield-split-col">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By daypart', 'לפי חלון שידור')}</h3>
                <span>{byDaypart.length}</span>
              </div>
              <YieldBars rows={byDaypart} locale={locale} labelKey="group" />
            </div>
            <div className="yield-split-col">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By programme', 'לפי תוכנית')}</h3>
                <span>{byProgramme.length}</span>
              </div>
              <YieldBars rows={byProgramme} locale={locale} labelKey="group" />
            </div>
          </div>

          <p className="yield-foot-note">
            {netAvailable
              ? pageText(
                  locale,
                  `Yield per second = revenue / ad-seconds (${currency}/s). Revenue net of retention prices the audience lost to breaks in ILS and subtracts it: ${basisFormula}`,
                  `תשואה לשנייה = הכנסה / שניות פרסום (${currency}/s). הכנסה בניכוי שימור מתמחרת בשקלים את הקהל שאבד לברייקים ומחסירה אותו: ${basisFormula}`,
                )
              : pageText(
                  locale,
                  `Yield per second = revenue / ad-seconds (${currency}/s). Revenue net of retention is not available: ${payload?.revenue_net_reason || 'the saved schedule lacks the per-segment audience needed to price it.'}`,
                  `תשואה לשנייה = הכנסה / שניות פרסום (${currency}/s). הכנסה בניכוי שימור אינה זמינה: ${payload?.revenue_net_reason || 'ללוח השמור חסר הקהל לכל מקטע הדרוש לתמחור.'}`,
                )}
          </p>
        </>
      )}
    </section>
  );
}
