import React, { useEffect, useState } from 'react';
import { Figure } from '../shell/bidi';
import { AlertTriangle, BellRing } from 'lucide-react';
import {
  API_BASE,
  formatPercent,
  normalizeRows,
  pageText,
} from '../shell/surface-helpers';

// MakeGoodAlerts: under-delivery (make-good) risk per campaign, from
// GET /api/make-good-alerts. Today campaign_flights.csv is header-only, so the
// endpoint returns data_available:false with a reason and zero alerts. We render
// an honest empty state that names the path forward (upload campaign flights),
// and only show the alert list once real flight data exists.
//
// This is a second ledger and not the one the campaign board reads. Its
// delivered fraction is projected from campaign_flights.csv against elapsed
// time, so every figure it prints names that source and the instant it was
// projected at, and neither is left for the reader to assume.

function pct(value, locale) {
  const number = Number(value || 0) * 100;
  return formatPercent(number, locale);
}

// Localized "as of" date. The endpoint sends an ISO date; the operator reads it
// in the page locale, never as a raw ISO string. An unparseable value falls
// back to the raw text rather than being dropped or invented.
function asOfLabel(value, locale) {
  const text = String(value || '').trim();
  if (!text) return '';
  const parsed = new Date(text);
  const formatted = Number.isNaN(parsed.getTime())
    ? text
    : parsed.toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  return locale === 'he' ? `נכון ל־${formatted}` : `As of ${formatted}`;
}

export default function MakeGoodAlerts({ locale, refreshKey = 0 }) {
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
  }, [refreshKey]);

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
              'Under-delivery alerts need real campaign flights. Upload campaign_flights.csv with start and end dates and delivery goals to start tracking pacing and make-good risk.',
              'התראות תת-אספקה דורשות נתוני קמפיינים אמיתיים. העלו את campaign_flights.csv עם תאריכי התחלה וסיום ויעדי אספקה כדי להתחיל לעקוב אחר קצב ואחר סיכון פיצוי.',
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
      <div className="makegood-list">
        <p className="data-basis-note">
          {pageText(
            locale,
            'Projected from campaign_flights.csv against the elapsed share of each flight. It is not the per-spot delivery ledger the campaign board counts.',
            'תחזית מתוך campaign_flights.csv מול החלק שחלף מכל טיסה. אין זה ספר האספקה ברמת התשדיר שלוח הקמפיינים סופר.',
          )}
          {asOf ? ` ${asOfLabel(asOf, locale)}.` : ''}
        </p>
        {alerts.map((alert) => (
          <article className="makegood-row" key={alert.campaign_id}>
            <div className="makegood-row-head">
              <AlertTriangle size={15} aria-hidden="true" />
              <strong>{alert.campaign_id}</strong>
              <span className="makegood-shortfall numeric">
                <Figure>-{pct(alert.projected_shortfall, locale)}</Figure>
              </span>
            </div>
            <div className="makegood-bars">
              <div className="makegood-bar">
                <span>{pageText(locale, 'Elapsed', 'חלף')}</span>
                <i style={{ '--bar': Math.min(1, Number(alert.elapsed_frac || 0)) }} />
                <small className="numeric"><Figure>{pct(alert.elapsed_frac, locale)}</Figure></small>
              </div>
              <div className="makegood-bar">
                <span>{pageText(locale, 'Delivered', 'סופק')}</span>
                <i style={{ '--bar': Math.min(1, Number(alert.delivered_frac || 0)) }} />
                <small className="numeric"><Figure>{pct(alert.delivered_frac, locale)}</Figure></small>
              </div>
              <div className="makegood-bar">
                <span>{pageText(locale, 'Projected', 'תחזית')}</span>
                <i style={{ '--bar': Math.min(1, Number(alert.projected_frac || 0)) }} />
                <small className="numeric"><Figure>{pct(alert.projected_frac, locale)}</Figure></small>
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
            ? asOfLabel(asOf, locale)
            : pageText(locale, 'Under-delivery risk', 'סיכון תת-אספקה')}
        </span>
      </div>
      {body()}
    </section>
  );
}
