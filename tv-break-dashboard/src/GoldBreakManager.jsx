import React, { useEffect, useState } from 'react';
import { Sparkles } from 'lucide-react';
import {
  API_BASE,
  formatCurrency,
  formatNumber,
  formatSeconds,
  normalizeRows,
  pageText,
} from './surface-helpers';

// GoldBreakManager: which breaks in the current plan are gold, and how many per
// day, from GET /api/gold-breaks (a live optimizer run on the saved settings).
// realized_premium / potential_premium come from the daily spot-pricing path,
// not the weekly optimizer, so the endpoint returns them as null with a
// source_pending marker; we render them as "pending" and never invent a premium.
// When gold breaks are disabled or none are configured, we show an honest empty
// state rather than a fabricated list.

function programTypeText(value, locale) {
  const text = String(value || '').trim();
  return text || pageText(locale, 'Mixed', 'מעורב');
}

export default function GoldBreakManager({ locale }) {
  const he = locale === 'he';
  const [state, setState] = useState({ status: 'loading', payload: null });

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/gold-breaks`)
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
  const breaks = normalizeRows(payload?.breaks);
  const byDay = normalizeRows(payload?.by_day);

  function body() {
    if (status === 'loading') {
      return <div className="frontier-skeleton" aria-hidden="true" />;
    }
    if (status === 'error') {
      return <div className="heatmap-empty">{pageText(locale, 'Gold-break data is unavailable right now.', 'נתוני ברייקים זהובים אינם זמינים כרגע.')}</div>;
    }
    if (payload?.available === false) {
      return <div className="heatmap-empty">{payload?.reason || pageText(locale, 'Gold-break data is unavailable.', 'נתוני ברייקים זהובים אינם זמינים.')}</div>;
    }
    if (payload?.enabled === false) {
      return (
        <div className="heatmap-empty">
          {pageText(locale, 'Gold breaks are turned off in settings. Enable them to flag premium placements.', 'ברייקים זהובים כבויים בהגדרות. הפעילו אותם כדי לסמן מיקומים פרימיום.')}
        </div>
      );
    }
    if (!breaks.length) {
      return (
        <div className="heatmap-empty">
          {payload?.reason || pageText(locale, 'No gold breaks in the current plan yet.', 'אין ברייקים זהובים בתוכנית הנוכחית עדיין.')}
        </div>
      );
    }
    return (
      <>
        <div className="gold-summary" dir={he ? 'rtl' : 'ltr'}>
          <div className="gold-summary-card">
            <span>{pageText(locale, 'Gold breaks', 'ברייקים זהובים')}</span>
            <strong className="numeric" dir="ltr">{formatNumber(payload?.count ?? breaks.length, locale)}</strong>
          </div>
          <div className="gold-summary-card">
            <span>{pageText(locale, 'Max per day', 'מקסימום ליום')}</span>
            <strong className="numeric" dir="ltr">{payload?.max_per_day != null ? formatNumber(payload.max_per_day, locale) : '-'}</strong>
          </div>
        </div>

        {byDay.length > 0 && (
          <div className="gold-by-day chart-ltr" dir="ltr">
            {byDay.map((row) => (
              <span className="gold-day-chip" key={row.day || 'unknown'}>
                {row.day || pageText(locale, 'Unknown', 'לא ידוע')}
                <strong className="numeric" dir="ltr">{formatNumber(row.count, locale)}</strong>
              </span>
            ))}
          </div>
        )}

        <div className="gold-table-wrap" dir={he ? 'rtl' : 'ltr'}>
          <table className="gold-table">
            <thead>
              <tr>
                <th>{pageText(locale, 'Day', 'יום')}</th>
                <th>{pageText(locale, 'Start', 'התחלה')}</th>
                <th>{pageText(locale, 'Programme', 'תוכנית')}</th>
                <th>{pageText(locale, 'Length', 'אורך')}</th>
                <th>{pageText(locale, 'Revenue', 'הכנסה')}</th>
                <th>{pageText(locale, 'Premium', 'פרמיה')}</th>
              </tr>
            </thead>
            <tbody>
              {breaks.map((row, index) => (
                <tr key={row.segment_id || index}>
                  <td>{row.day || '-'}</td>
                  <td className="numeric" dir="ltr">{row.start_time || '-'}</td>
                  <td>{programTypeText(row.program_type, locale)}</td>
                  <td className="numeric" dir="ltr">{formatSeconds(row.duration_seconds, locale)}</td>
                  <td className="numeric" dir="ltr">{formatCurrency(row.revenue, locale)}</td>
                  <td>
                    <span className="gold-premium-pending">{pageText(locale, 'Pending', 'ממתין')}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="gold-note">
          {pageText(
            locale,
            'Realized and potential premiums are computed on the daily spot-pricing path, not the weekly optimizer, so they are pending here.',
            'פרמיות ממומשות ופוטנציאליות מחושבות במסלול תמחור הספוטים היומי, לא באופטימייזר השבועי, ולכן הן ממתינות כאן.',
          )}
        </p>
      </>
    );
  }

  return (
    <section className="page-panel gold-break-manager">
      <div className="panel-head">
        <h2><Sparkles size={15} /> {pageText(locale, 'Gold breaks', 'ברייקים זהובים')}</h2>
        <span>{pageText(locale, 'Premium placements in the saved plan', 'מיקומי פרימיום בתוכנית השמורה')}</span>
      </div>
      {body()}
    </section>
  );
}
