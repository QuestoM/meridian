import React, { useEffect, useState } from 'react';
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
import MoneyWaterfall from './MoneyWaterfall';

// YieldView: revenue and yield-per-second by daypart and programme, sourced from
// GET /api/yield-per-second. Surfaces where each ad-second earns the most and
// where inventory is under-monetized. The money stats render through the shared
// MoneyWaterfall: gross stays the dominant figure, the retention cost is an
// explicitly marked model estimate, and the net after retention cost is the
// bottom line, shown only when the endpoint computes it exactly
// (revenue_net_available:true); otherwise the reason is shown and no figure is
// fabricated.

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
            <div className="yield-total-card yield-money-card">
              <MoneyWaterfall
                variant="panel"
                locale={locale}
                gross={netAvailable ? (finiteNumber(payload.revenue_ils) ?? totals.revenue) : totals.revenue}
                retentionCost={netAvailable ? payload.retention_cost_ils : null}
                net={netAvailable ? payload.revenue_net_ils : null}
                retentionCostLow={payload.retention_cost_low}
                retentionCostHigh={payload.retention_cost_high}
                unavailableReason={netAvailable ? '' : String(payload.revenue_net_reason || '')}
              />
            </div>
            <div className="yield-total-card">
              <span><Gauge size={13} /> {pageText(locale, 'Yield per second', 'תשואה לשנייה')}</span>
              <strong className="numeric" dir="ltr">{formatRate(totals.yield_per_second, locale)} {currency}/s</strong>
            </div>
            <div className="yield-total-card">
              <span>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</span>
              <strong className="numeric" dir="ltr">{formatSeconds(totals.ad_seconds, locale)}</strong>
            </div>
          </div>

          <div className="yield-split">
            <div className="yield-split-col">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By daypart', 'לפי רצועת שידור')}</h3>
                <span>{byDaypart.length}</span>
              </div>
              <YieldBars rows={byDaypart} locale={locale} labelKey="group" labelFor={(value) => daypartLabel(value, locale)} />
            </div>
            <div className="yield-split-col">
              <div className="yield-subhead">
                <h3>{pageText(locale, 'By programme', 'לפי תוכנית')}</h3>
                <span>{byProgramme.length}</span>
              </div>
              <YieldBars rows={byProgramme} locale={locale} labelKey="group" labelFor={(value) => programTypeLabel(value, locale)} />
            </div>
          </div>

          <p className="yield-foot-note">
            {netAvailable
              ? pageText(
                  locale,
                  `Yield per second = revenue / ad-seconds (${currency}/s). The net after retention cost prices the audience lost to breaks in ILS and subtracts it: ${basisFormula}`,
                  `תשואה לשנייה = הכנסה / שניות פרסום (${currency}/s). הנטו אחרי עלות שימור מתמחר בשקלים את הקהל שאבד לברייקים ומחסיר אותו: ${basisFormula}`,
                )
              : pageText(
                  locale,
                  `Yield per second = revenue / ad-seconds (${currency}/s). The net after retention cost is not available: ${payload?.revenue_net_reason || 'the saved schedule lacks the per-segment audience needed to price it.'}`,
                  `תשואה לשנייה = הכנסה / שניות פרסום (${currency}/s). הנטו אחרי עלות שימור אינו זמין: ${payload?.revenue_net_reason || 'ללוח השמור חסר הקהל לכל מקטע הדרוש לתמחור.'}`,
                )}
          </p>
        </>
      )}
    </section>
  );
}
