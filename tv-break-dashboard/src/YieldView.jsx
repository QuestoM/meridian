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
// where inventory is under-monetized. revenue_net is unavailable per the
// endpoint (revenue_net_available:false), so it is labelled as such and never
// fabricated.

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

export default function YieldView({ locale }) {
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
  }, []);

  const { status, payload } = state;
  const available = status === 'ready' && payload && payload.available !== false;
  const totals = payload?.totals || {};
  const byDaypart = normalizeRows(payload?.by_daypart);
  const byProgramme = normalizeRows(payload?.by_programme);
  const currency = payload?.currency || 'ILS';

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
            <div className="yield-total-card muted">
              <span>{pageText(locale, 'Revenue net of retention', 'הכנסה בניכוי שימור')}</span>
              <strong>{pageText(locale, 'Not available', 'לא זמין')}</strong>
            </div>
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
            {pageText(
              locale,
              `Yield per second = revenue / ad-seconds (${currency}/s). Revenue net of retention is not exposed by this endpoint.`,
              `תשואה לשנייה = הכנסה / שניות פרסום (${currency}/s). הכנסה בניכוי שימור אינה נחשפת בנקודת קצה זו.`,
            )}
          </p>
        </>
      )}
    </section>
  );
}
