import React, { useEffect, useState } from 'react';
import { Info } from 'lucide-react';
import {
  API_BASE,
  finiteNumber,
  formatCurrency,
  formatNumber,
  pageText,
} from './surface-helpers';

// The shared money story. Gross revenue is the real invoiced headline and stays
// visually dominant everywhere. The retention cost is an explicitly marked model
// estimate (with a quiet uncertainty range when the API provides one), and the
// net after retention cost is the bottom line of a small P&L-style waterfall,
// never a competing headline. Every figure renders from real API fields or an
// honest absent state; nothing is fabricated client-side.

function estimateExplainer(locale) {
  return pageText(
    locale,
    'The value of the audience worn away by the breaks, priced at the actual advertising rate. A model estimate, not an invoice.',
    'שווי הקהל שנשחק בעקבות ההפסקות, מתומחר לפי מחיר הפרסום בפועל. אומדן מהמודל, לא חשבונית.',
  );
}

// MoneyWaterfall: the presentational three-line money story.
// Props: gross / retentionCost / net are numbers (or absent), retentionCostLow
// and retentionCostHigh form the optional uncertainty band, variant picks the
// size ('headline' for the Overview card, 'panel' for embedded stat areas),
// The band provenance line: the backend sends one precise English sentence in
// retention_cost_basis; the Hebrew UI renders an equivalent localized line so a
// Hebrew page never carries a raw English paragraph. Shown only when the band
// itself is present.
function bandBasisNote(locale, payload) {
  if (!payload || !Number.isFinite(payload.retention_cost_low) || !Number.isFinite(payload.retention_cost_high)) return '';
  if (locale === 'he') {
    return 'הרצועה סביב עלות השימור נגזרת מרווחי הסמך המכוילים (95 אחוז) של מקדמי המודל.';
  }
  return typeof payload.retention_cost_basis === 'string' ? payload.retention_cost_basis : '';
}

// basisNote is an optional muted provenance sentence, and unavailableReason is
// shown when the cost and net cannot be computed honestly.
export default function MoneyWaterfall({
  gross,
  retentionCost,
  net,
  retentionCostLow = null,
  retentionCostHigh = null,
  locale = 'en',
  variant = 'panel',
  basisNote = '',
  unavailableReason = '',
}) {
  const grossValue = finiteNumber(gross);
  const costValue = finiteNumber(retentionCost);
  const netValue = finiteNumber(net);
  const bandLow = finiteNumber(retentionCostLow);
  const bandHigh = finiteNumber(retentionCostHigh);
  const hasBand = bandLow !== null && bandHigh !== null;
  const hasCostAndNet = costValue !== null && netValue !== null;
  const rootClass = variant === 'headline' ? 'money-waterfall headline' : 'money-waterfall';

  if (grossValue === null) {
    return (
      <div className={rootClass}>
        <p className="mw-unavailable">{pageText(locale, 'Money figures are not available yet.', 'נתוני הכסף אינם זמינים עדיין.')}</p>
      </div>
    );
  }

  return (
    <div className={rootClass}>
      <div className="mw-row mw-gross">
        <span className="mw-label">{pageText(locale, 'Gross revenue', 'הכנסות ברוטו')}</span>
        <strong className="mw-value numeric" dir="ltr">{formatCurrency(grossValue, locale)}</strong>
      </div>
      {hasCostAndNet ? (
        <>
          <div className="mw-row mw-cost">
            <span className="mw-label" title={estimateExplainer(locale)}>
              {pageText(locale, 'Retention cost', 'עלות שימור')}
              <span className="mw-chip">{pageText(locale, 'Model estimate', 'אומדן מודל')}</span>
              <Info size={12} className="mw-info" aria-hidden="true" />
            </span>
            <span className="mw-money">
              <strong className="mw-value numeric" dir="ltr">{`-${formatCurrency(costValue, locale)}`}</strong>
              {hasBand ? (
                <small className="mw-band numeric" dir="ltr" title={pageText(locale, 'The plausible range of the model estimate.', 'הטווח הסביר של אומדן המודל.')}>{`${formatCurrency(bandLow, locale)} - ${formatCurrency(bandHigh, locale)}`}</small>
              ) : null}
            </span>
          </div>
          <div className="mw-row mw-net">
            <span className="mw-label">{pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור')}</span>
            <strong className="mw-value numeric" dir="ltr">{formatCurrency(netValue, locale)}</strong>
          </div>
        </>
      ) : (
        <p className="mw-unavailable">
          {pageText(locale, 'Retention cost and net are not available for this plan.', 'עלות השימור והנטו אינם זמינים לתוכנית הזו.')}
          {unavailableReason ? <small>{unavailableReason}</small> : null}
        </p>
      )}
      {basisNote ? <small className="mw-basis">{basisNote}</small> : null}
    </div>
  );
}

// YieldMoneyPanel: the Overview money-story card. Reads the same
// GET /api/yield-per-second payload the yield panel uses and renders the
// headline-variant waterfall, with honest loading and absent states when the
// saved plan cannot price its money story.
export function YieldMoneyPanel({ locale, refreshKey = 0 }) {
  const [state, setState] = useState({ status: 'loading', payload: null });

  useEffect(() => {
    let active = true;
    setState({ status: 'loading', payload: null });
    fetch(`${API_BASE}/api/yield-per-second`)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (active) setState({ status: 'ready', payload });
      })
      .catch(() => {
        if (active) setState({ status: 'error', payload: null });
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const { status, payload } = state;
  const available = status === 'ready' && payload && payload.available !== false;
  const netAvailable = Boolean(payload?.revenue_net_available);
  const totals = payload?.totals || {};
  const gross = netAvailable ? (finiteNumber(payload.revenue_ils) ?? totals.revenue) : totals.revenue;

  return (
    <section className="page-panel money-story-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'From gross to net', 'מברוטו לנטו')}</h2>
        <span>{pageText(locale, 'From the saved weekly plan', 'לפי התוכנית השבועית השמורה')}</span>
      </div>
      <div className="money-story-body">
        {status === 'loading' ? (
          <p className="money-story-empty">{pageText(locale, 'Loading money figures.', 'טוען את נתוני הכסף.')}</p>
        ) : !available ? (
          <p className="money-story-empty">
            {pageText(locale, 'Money figures are not available right now.', 'נתוני הכסף אינם זמינים כרגע.')}
            {payload?.reason ? <small>{payload.reason}</small> : null}
          </p>
        ) : (
          <MoneyWaterfall
            variant="headline"
            locale={locale}
            gross={gross}
            retentionCost={netAvailable ? payload.retention_cost_ils : null}
            net={netAvailable ? payload.revenue_net_ils : null}
            retentionCostLow={payload.retention_cost_low}
            retentionCostHigh={payload.retention_cost_high}
            basisNote={bandBasisNote(locale, payload)}
            unavailableReason={netAvailable ? '' : String(payload.revenue_net_reason || '')}
          />
        )}
      </div>
    </section>
  );
}

// NetComparisonCard: the read-only comparison beside the engine focus control.
// Reads GET /api/optimizer/net-comparison and shows what the net-focused plan
// changes versus the current plan, as signed deltas. It adds no save path: the
// existing settings save and recompute flow stays the only way to switch focus.
export function NetComparisonCard({ locale, refreshSignal = '', currentFocus = 'blend' }) {
  const [state, setState] = useState({ status: 'loading', payload: null });

  useEffect(() => {
    if (refreshSignal === 'running') return undefined;
    let active = true;
    setState({ status: 'loading', payload: null });
    fetch(`${API_BASE}/api/optimizer/net-comparison`)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (active) setState({ status: 'ready', payload });
      })
      .catch(() => {
        if (active) setState({ status: 'error', payload: null });
      });
    return () => {
      active = false;
    };
  }, [refreshSignal]);

  const payload = state.payload;
  const remoteStatus = String(payload?.status || '');
  const loading = state.status === 'loading';
  const computing = state.status === 'ready' && remoteStatus === 'computing';
  const ready = state.status === 'ready' && remoteStatus === 'ready';
  const current = payload?.current || {};
  const focused = payload?.net_focused || {};
  const delta = payload?.delta || {};

  function deltaValue(key) {
    const provided = finiteNumber(delta[key]);
    if (provided !== null) return provided;
    const from = finiteNumber(current[key]);
    const to = finiteNumber(focused[key]);
    return from !== null && to !== null ? to - from : null;
  }

  function signed(value, formatter) {
    if (value === null) return '-';
    return `${value > 0 ? '+' : ''}${formatter(value, locale)}`;
  }

  const rows = ready
    ? [
        { key: 'gross', label: pageText(locale, 'Gross revenue', 'הכנסות ברוטו'), value: signed(deltaValue('gross'), formatCurrency), estimate: false },
        { key: 'retention_cost', label: pageText(locale, 'Retention cost', 'עלות שימור'), value: signed(deltaValue('retention_cost'), formatCurrency), estimate: true },
        { key: 'net', label: pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור'), value: signed(deltaValue('net'), formatCurrency), estimate: false },
        { key: 'breaks', label: pageText(locale, 'Breaks', 'ברייקים'), value: signed(deltaValue('breaks'), formatNumber), estimate: false },
      ]
    : [];

  return (
    <aside className="net-compare-card" aria-live="polite">
      <strong className="net-compare-head">{pageText(locale, 'What changes with net focus', 'מה משתנה במיקוד נטו')}</strong>
      {loading ? (
        <p className="net-compare-empty">{pageText(locale, 'Loading the comparison.', 'טוען את ההשוואה.')}</p>
      ) : computing ? (
        <p className="net-compare-empty">{pageText(locale, 'The comparison is computing in the background and will appear here once ready.', 'ההשוואה מחושבת ברקע ותופיע כאן כשתהיה מוכנה.')}</p>
      ) : !ready ? (
        <p className="net-compare-empty">{pageText(locale, 'The comparison is not available right now.', 'ההשוואה אינה זמינה כרגע.')}</p>
      ) : (
        <>
          <div className="net-compare-rows">
            {rows.map((row) => (
              <div className="net-compare-row" key={row.key}>
                <span className="net-compare-label">
                  {row.label}
                  {row.estimate ? <span className="mw-chip" title={estimateExplainer(locale)}>{pageText(locale, 'Model estimate', 'אומדן מודל')}</span> : null}
                </span>
                <strong className="numeric" dir="ltr">{row.value}</strong>
              </div>
            ))}
          </div>
          {locale === 'he' ? <p className="net-compare-basis">שני הצדדים הם אותה אופטימיזציה מלאה על יום מייצג של הערוץ שבבעלותכם, כך שההפרשים ניתנים להשוואה ישירה.</p> : typeof payload?.basis === 'string' && payload.basis ? <p className="net-compare-basis">{payload.basis}</p> : null}
          {currentFocus !== 'revenue_net' ? (
            <p className="net-compare-note">{pageText(locale, 'Switching to net focus will lower the displayed gross and raise the net, per the numbers here.', 'מעבר למיקוד נטו יוריד את הברוטו המוצג ויעלה את הנטו, בהתאם למספרים כאן.')}</p>
          ) : null}
        </>
      )}
    </aside>
  );
}
