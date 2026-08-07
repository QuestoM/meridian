import React, { useEffect, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Info } from 'lucide-react';
import {
  API_BASE,
  finiteNumber,
  formatCurrency,
  formatNumber,
  pageText,
} from '../shell/surface-helpers';
import { Figure, isolate } from '../shell/bidi';
import ChannelRefusal from './ChannelRefusal';
import { scopeState, unattributed } from './today-scope';
import { formatSpan } from '../shell/dates';

// The shared money story. Gross revenue is the real invoiced headline and stays
// visually dominant everywhere. The retention cost is an explicitly marked model
// estimate (with a quiet uncertainty range when the API provides one), and the
// net after retention cost is the bottom line of a small P&L-style waterfall,
// never a competing headline. Every figure renders from real API fields or an
// honest absent state; nothing is fabricated client-side.

function estimateExplainer(locale) {
  return pageText(
    locale,
    'The money cost of viewers stepping away during breaks: each break airs to a slightly smaller audience, so its rating and billing are lower. The gap is priced at the actual advertising rate. A model estimate, not an invoice.',
    'העלות הכספית של הצופים שמתרחקים בזמן הברייקים: כל ברייק משודר לקהל מעט קטן יותר, ולכן הרייטינג והחיוב שלו נמוכים יותר. ההפרש מתומחר לפי מחיר הפרסום בפועל. אומדן מהמודל, לא חשבונית.',
  );
}

// MoneyWaterfall: the presentational three-line money story.
// Props: gross / retentionCost / net are numbers (or absent), retentionCostLow
// and retentionCostHigh form the optional uncertainty band, variant picks the
// size ('headline' for the Overview card, 'panel' for embedded stat areas),
// The band provenance line, written here for the reader in both languages and
// shown only when the band itself is present.
//
// The payload also carries retention_cost_basis, one precise sentence that
// names the interval seam in the engine's own terms and prints five internal
// field names. That is the right answer for whoever reads the API and the wrong
// one for a general manager reading a money card, so it stays in the payload and
// reaches no screen. Both branches below say the same thing as that sentence in
// the words this reader already uses on this card.
function bandBasisNote(locale, payload) {
  if (!payload || !Number.isFinite(payload.retention_cost_low) || !Number.isFinite(payload.retention_cost_high)) return '';
  if (locale === 'he') {
    return 'טווח האומדן מראה כמה עלות השימור עשויה להיות נמוכה או גבוהה מהמספר המרכזי, לפי אי-הוודאות שנמדדה לכל ברייק (רמת ביטחון 95 אחוז).';
  }
  return 'The estimate range shows how much lower or higher the retention cost could be than the central figure, from the uncertainty measured for each break at the 95 percent confidence level.';
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

  // A proportional bar of where the gross goes: the net kept (green) plus the
  // retention cost given up (amber), summing to gross. Only shown on the headline
  // variant when the full split is real; every width comes from the real figures.
  const showBar = variant === 'headline' && hasCostAndNet && grossValue > 0;
  const netFrac = showBar ? Math.max(0, Math.min(1, netValue / grossValue)) : 0;
  const costFrac = showBar ? Math.max(0, Math.min(1, costValue / grossValue)) : 0;
  const bandLowFrac = showBar && hasBand ? Math.max(0, Math.min(1, bandLow / grossValue)) : null;
  const bandHighFrac = showBar && hasBand ? Math.max(0, Math.min(1, bandHigh / grossValue)) : null;

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
        <Figure className="mw-value numeric">{formatCurrency(grossValue, locale)}</Figure>
      </div>
      {hasCostAndNet ? (
        <>
          <div className="mw-row mw-cost">
            <Tooltip title={estimateExplainer(locale)} arrow placement="bottom">
              <span className="mw-label">
                {pageText(locale, 'Retention cost', 'עלות שימור')}
                <span className="mw-chip">{pageText(locale, 'Model estimate', 'אומדן מודל')}</span>
                <Info size={12} className="mw-info" aria-hidden="true" />
              </span>
            </Tooltip>
            <span className="mw-money">
              <Figure className="mw-value numeric">{`-${formatCurrency(costValue, locale)}`}</Figure>
            </span>
          </div>
          <div className="mw-row mw-net">
            <span className="mw-label">{pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור')}</span>
            <Figure className="mw-value numeric">{formatCurrency(netValue, locale)}</Figure>
          </div>
        </>
      ) : (
        <p className="mw-unavailable">
          {pageText(locale, 'Retention cost and net are not available for this plan.', 'עלות השימור והנטו אינם זמינים לתוכנית הזו.')}
          {unavailableReason ? <small>{unavailableReason}</small> : null}
        </p>
      )}
      {showBar ? (
        <div
          className="mw-bar"
          role="img"
          aria-label={pageText(
            locale,
            `Net ${formatCurrency(netValue, locale)} kept and retention cost ${formatCurrency(costValue, locale)} given up, out of ${formatCurrency(grossValue, locale)} gross.`,
            `נטו ${formatCurrency(netValue, locale)} נשמר ועלות שימור ${formatCurrency(costValue, locale)} נגרעה, מתוך ${formatCurrency(grossValue, locale)} ברוטו.`,
          )}
        >
          <div className="mw-bar-track">
            <span className="mw-bar-net" style={{ width: `${netFrac * 100}%` }} />
            <span className="mw-bar-cost" style={{ width: `${costFrac * 100}%` }} />
            {bandLowFrac !== null && bandHighFrac !== null ? (
              // No tooltip on the sliver itself: the legend key below carries the
              // themed MUI explanation for the band, and a hover-managed MUI
              // tooltip on an absolutely-positioned bar sliver is not worth the
              // wrapper.
              <i
                className="mw-bar-band"
                style={{ insetInlineEnd: `${bandLowFrac * 100}%`, width: `${Math.max(0, bandHighFrac - bandLowFrac) * 100}%` }}
              />
            ) : null}
          </div>
          <div className="mw-bar-legend">
            <span className="mw-bar-key"><i className="mw-dot net" />{pageText(locale, 'Net kept', 'נטו שנשמר')}</span>
            <span className="mw-bar-key"><i className="mw-dot cost" />{pageText(locale, 'Retention cost', 'עלות שימור')}</span>
            {hasBand ? (
              <Tooltip title={pageText(locale, 'The plausible range of the model estimate.', 'הטווח הסביר של אומדן המודל.')} arrow placement="bottom">
                <span className="mw-bar-key">
                  <i className="mw-dot band" />
                  {pageText(locale, 'Estimate range', 'טווח האומדן')}
                  <Figure className="mw-band-range">{`${formatCurrency(bandLow, locale)} - ${formatCurrency(bandHigh, locale)}`}</Figure>
                </span>
              </Tooltip>
            ) : null}
          </div>
        </div>
      ) : null}
      {basisNote ? <small className="mw-basis">{basisNote}</small> : null}
    </div>
  );
}

// YieldMoneyPanel: the Overview money-story card. Reads the same
// GET /api/yield-per-second payload the yield panel uses and renders the
// headline-variant waterfall, with honest loading and absent states when the
// saved plan cannot price its money story.
export function YieldMoneyPanel({ locale, refreshKey = 0, onOpenSettings = null }) {
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

  // Whose money story this is. The yield body serves the whole market's gross
  // and net whenever no channel is declared, and discloses that in the same
  // payload, so the panel refuses on the disclosure rather than printing the
  // market total under this operator's name.
  const withheld = unattributed(scopeState(payload, available));

  // Scope line from the API only (channel + plan calendar span). Never invent
  // "weekly" when the plan covers a different number of days.
  const scopeChannel = typeof payload?.scope_channel === 'string' && payload.scope_channel.trim()
    ? payload.scope_channel.trim()
    : null;
  const dateFrom = typeof payload?.date_from === 'string' && payload.date_from.trim()
    ? payload.date_from.trim().slice(0, 10)
    : null;
  const dateTo = typeof payload?.date_to === 'string' && payload.date_to.trim()
    ? payload.date_to.trim().slice(0, 10)
    : null;
  const nDates = finiteNumber(payload?.n_dates);
  const scopeParts = [];
  if (scopeChannel) {
    // Bidi isolated for the same reason the Today basis line isolates it: the
    // date that follows the separator would otherwise be dragged into the name.
    scopeParts.push(pageText(locale, `channel ${isolate(scopeChannel)}`, `ערוץ ${isolate(scopeChannel)}`));
  }
  if (dateFrom && dateTo) {
    const window = formatSpan(dateFrom, dateTo, locale);
    scopeParts.push(
      dateFrom === dateTo
        ? window
        : pageText(
          locale,
          `${window}${nDates !== null ? ` (${nDates} days)` : ''}`,
          `${window}${nDates !== null ? ` (${nDates} ימים)` : ''}`,
        ),
    );
  } else if (nDates !== null) {
    scopeParts.push(pageText(locale, `${nDates} plan days`, `${nDates} ימי תוכנית`));
  }
  const scopeLine = scopeParts.length
    ? pageText(
      locale,
      `From the saved plan · ${scopeParts.join(' · ')}`,
      `לפי התוכנית השמורה · ${scopeParts.join(' · ')}`,
    )
    : pageText(locale, 'From the saved plan', 'לפי התוכנית השמורה');

  return (
    <section className="page-panel money-story-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'From gross to net', 'מברוטו לנטו')}</h2>
        {/* The span is the scope of a figure. With no figure to scope, printing
            it would be the same silent drop the refusal exists to stop. */}
        {withheld ? null : <span>{scopeLine}</span>}
      </div>
      <div className="money-story-body">
        {status === 'loading' ? (
          <p className="money-story-empty">{pageText(locale, 'Loading money figures.', 'טוען את נתוני הכסף.')}</p>
        ) : !available ? (
          <p className="money-story-empty">
            {pageText(locale, 'Money figures are not available right now.', 'נתוני הכסף אינם זמינים כרגע.')}
            {payload?.reason ? <small>{payload.reason}</small> : null}
          </p>
        ) : withheld ? (
          <ChannelRefusal
            locale={locale}
            lead={pageText(
              locale,
              'The money story cannot be reported as yours yet.',
              'אי אפשר עדיין לדווח על סיפור הכסף כשלכם.',
            )}
            onOpenSettings={onOpenSettings}
          />
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
// existing settings save and plan run stays the only way to switch focus.
//
// Its basis line is written here in both languages for the same reason the money
// card's is: the payload's own basis sentence names the engine's measured
// coefficients, which is the right words for the API and the wrong ones for the
// person deciding. Both sentences below carry what that basis carries, including
// that this is a modeled forecast of one day rather than the saved plan's total.
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
                  {row.estimate ? <Tooltip title={estimateExplainer(locale)} arrow placement="bottom"><span className="mw-chip">{pageText(locale, 'Model estimate', 'אומדן מודל')}</span></Tooltip> : null}
                </span>
                <Figure className="numeric">{row.value}</Figure>
              </div>
            ))}
          </div>
          <p className="net-compare-basis">
            {pageText(
              locale,
              'Both sides are the same optimization of one representative broadcast day of the channel you own, under the saved rules, so the differences compare directly. A modeled forecast, not the saved plan total.',
              'שני הצדדים הם אותה אופטימיזציה על יום שידור מייצג של הערוץ שבבעלותכם, תחת אותם כללים שמורים, כך שההפרשים ניתנים להשוואה ישירה. תחזית מודל, לא סך התוכנית השמורה.',
            )}
          </p>
          {currentFocus !== 'revenue_net' ? (
            <p className="net-compare-note">{pageText(locale, 'Switching to net focus will lower the displayed gross and raise the net, per the numbers here.', 'מעבר למיקוד נטו יוריד את הברוטו המוצג ויעלה את הנטו, בהתאם למספרים כאן.')}</p>
          ) : null}
        </>
      )}
    </aside>
  );
}
