import React, { useEffect, useState } from 'react';
import { AlertTriangle, BellRing } from 'lucide-react';
import {
  API_BASE,
  formatPercent,
  normalizeRows,
  pageText,
} from './surface-helpers';

// MakeGoodAlerts: under-delivery (make-good) risk per campaign, from
// GET /api/make-good-alerts. Today campaign_flights.csv is header-only, so the
// endpoint returns data_available:false with a reason and zero alerts. We render
// an honest empty state that names the path forward (upload campaign flights),
// and only show the alert list once real flight data exists.

function pct(value, locale) {
  const number = Number(value || 0) * 100;
  return formatPercent(number, locale);
}

export default function MakeGoodAlerts({ locale }) {
  const he = locale === 'he';
  const [state, setState] = useState({ status: 'loading', payload: null });

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/make-good-alerts`)
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
  const alerts = normalizeRows(payload?.alerts);
  const dataAvailable = status === 'ready' && payload?.data_available === true;
  const asOf = payload?.as_of;

  function body() {
    if (status === 'loading') {
      return <div className="frontier-skeleton" aria-hidden="true" />;
    }
    if (status === 'error') {
      return <div className="heatmap-empty">{pageText(locale, 'Make-good alerts are unavailable right now.', 'התראות פיצוי אינן זמינות כרגע.')}</div>;
    }
    if (!dataAvailable) {
      return (
        <div className="makegood-empty">
          <BellRing size={22} aria-hidden="true" />
          <strong>{pageText(locale, 'No campaign data yet', 'אין נתוני קמפיינים עדיין')}</strong>
          <p>
            {pageText(
              locale,
              'Under-delivery alerts need real campaign flights. Upload campaign_flights.csv with flight dates and goals to start tracking pacing and make-good risk.',
              'התראות תת-אספקה דורשות נתוני טיסות קמפיין אמיתיים. העלו את campaign_flights.csv עם תאריכי טיסה ויעדים כדי להתחיל לעקוב אחר קצב ואחר סיכון פיצוי.',
            )}
          </p>
          {payload?.reason && <small className="makegood-reason">{payload.reason}</small>}
        </div>
      );
    }
    if (!alerts.length) {
      return (
        <div className="makegood-empty ok">
          <strong>{pageText(locale, 'All campaigns are on pace', 'כל הקמפיינים בקצב תקין')}</strong>
          <p>{pageText(locale, 'No campaigns are projected to under-deliver.', 'לא צפויה תת-אספקה באף קמפיין.')}</p>
        </div>
      );
    }
    return (
      <div className="makegood-list" dir={he ? 'rtl' : 'ltr'}>
        {alerts.map((alert) => (
          <article className="makegood-row" key={alert.campaign_id}>
            <div className="makegood-row-head">
              <AlertTriangle size={15} aria-hidden="true" />
              <strong>{alert.campaign_id}</strong>
              <span className="makegood-shortfall numeric" dir="ltr">
                -{pct(alert.projected_shortfall, locale)}
              </span>
            </div>
            <div className="makegood-bars">
              <div className="makegood-bar">
                <span>{pageText(locale, 'Elapsed', 'חלף')}</span>
                <i style={{ '--bar': Math.min(1, Number(alert.elapsed_frac || 0)) }} />
                <small className="numeric" dir="ltr">{pct(alert.elapsed_frac, locale)}</small>
              </div>
              <div className="makegood-bar">
                <span>{pageText(locale, 'Delivered', 'סופק')}</span>
                <i style={{ '--bar': Math.min(1, Number(alert.delivered_frac || 0)) }} />
                <small className="numeric" dir="ltr">{pct(alert.delivered_frac, locale)}</small>
              </div>
              <div className="makegood-bar">
                <span>{pageText(locale, 'Projected', 'תחזית')}</span>
                <i style={{ '--bar': Math.min(1, Number(alert.projected_frac || 0)) }} />
                <small className="numeric" dir="ltr">{pct(alert.projected_frac, locale)}</small>
              </div>
            </div>
          </article>
        ))}
      </div>
    );
  }

  return (
    <section className="page-panel makegood-alerts">
      <div className="panel-head">
        <h2>{pageText(locale, 'Make-good alerts', 'התראות פיצוי')}</h2>
        <span>
          {dataAvailable && asOf
            ? `${pageText(locale, 'As of', 'נכון ל')} ${asOf}`
            : pageText(locale, 'Under-delivery risk', 'סיכון תת-אספקה')}
        </span>
      </div>
      {body()}
    </section>
  );
}
