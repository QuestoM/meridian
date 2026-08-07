import React from 'react';
import { Numeric, finiteNumber, formatCurrency, formatNumber, formatPercent, pageText } from '../../shell/format';
import { Name } from '../../shell/bidi';
import { formatStamp } from '../../shell/dates';
import { dayLabel, impactSegmentLabel } from '../../shell/labels';
import { normalizeRows } from '../../shell/plan-model';

export function OptimizationRunSummary({ plan, locale }) {
  if (!plan?.summary) return null;
  const summary = plan.summary;
  // The preview optimizes one channel-day (that is what keeps it responsive).
  // Name that scope, so these figures are never read as weekly totals next to
  // the whole-week metrics above.
  const scopeParts = [plan.channel, plan.day ? dayLabel(plan.day, locale) : ''].filter(Boolean);
  const scopeLabel = scopeParts.length
    ? pageText(locale, `Preview scope: ${scopeParts.join(', ')} (one channel-day, not the weekly total)`, `היקף התצוגה המקדימה: ${scopeParts.join(', ')} (יום-ערוץ אחד, לא הסך השבועי)`)
    : pageText(locale, 'Preview scope: one channel-day, not the weekly total', 'היקף התצוגה המקדימה: יום-ערוץ אחד, לא הסך השבועי');
  return (
    <section className="optimizer-run-summary">
      <p className="data-basis-note optimizer-run-scope">{scopeLabel}</p>
      <div>
        <span>{pageText(locale, 'Optimized breaks', 'ברייקים באופטימום')}</span>
        <strong><Numeric>{formatNumber(summary.total_breaks, locale)}</Numeric></strong>
      </div>
      <div>
        <span>{pageText(locale, 'Plan revenue (this preview day)', 'הכנסת התוכנית (יום התצוגה המקדימה)')}</span>
        <strong><Numeric>{formatCurrency(summary.projected_revenue, locale)}</Numeric></strong>
      </div>
      <div>
        <span>{pageText(locale, 'Plan retention (this preview day)', 'שימור בתוכנית (יום התצוגה המקדימה)')}</span>
        <strong><Numeric>{formatPercent(summary.average_retention, locale)}</Numeric></strong>
      </div>
      <div>
        <span>{pageText(locale, 'Guardrail status', 'מצב בקרות')}</span>
        <strong>{summary.is_compliant ? pageText(locale, 'Compliant', 'תקין') : pageText(locale, 'Needs review', 'דורש בדיקה')}</strong>
      </div>
    </section>
  );
}

export function retentionCostConfidenceWord(confidence, copy) {
  const key = String(confidence || '').toLowerCase();
  return copy.retentionCostConfidence[key] || null;
}

export function RetentionCostSegment({ segment, copy, locale }) {
  const cost = segment?.retention_cost;
  if (!cost || typeof cost !== 'object') return null;

  const point = finiteNumber(cost.point);
  const used = finiteNumber(cost.used);
  const ciLow = finiteNumber(cost.ci_low);
  const ciHigh = finiteNumber(cost.ci_high);
  const count = finiteNumber(cost.n);
  const confidenceWord = retentionCostConfidenceWord(cost.confidence, copy);
  const isAssumption = count === 0 || String(cost.confidence || '').toLowerCase() === 'low';
  const hasInterval = ciLow !== null && ciHigh !== null;

  // Live-plan segments carry only segment_id; fall back to it so a confidence
  // row is never a nameless block of numbers.
  const name =
    impactSegmentLabel(segment.segment ?? segment.name ?? segment.program_type ?? '', locale) ||
    segment.label ||
    segment.segment_id ||
    '';

  return (
    <div className={isAssumption ? 'retention-cost-row assumption' : 'retention-cost-row'}>
      <div className="retention-cost-row-head">
        <strong><Name>{name}</Name></strong>
        <span className={`retention-cost-confidence ${String(cost.confidence || '').toLowerCase()}`}>
          {isAssumption ? copy.retentionCostAssumption : confidenceWord || copy.retentionCostAssumption}
        </span>
      </div>
      <div className="retention-cost-row-body">
        {used !== null && (
          <span>
            {copy.retentionCostUsed}
            <Numeric>{formatNumber(used, locale)}</Numeric>
          </span>
        )}
        {point !== null && (
          <span>
            {copy.retentionCostPoint}
            <Numeric>{formatNumber(point, locale)}</Numeric>
          </span>
        )}
        <span>
          {copy.retentionCostInterval}
          {hasInterval ? (
            <Numeric>{`[${formatNumber(ciLow, locale)}, ${formatNumber(ciHigh, locale)}]`}</Numeric>
          ) : (
            <small>{copy.retentionCostNoInterval}</small>
          )}
        </span>
        {count !== null && (
          <span>
            <Numeric>{formatNumber(count, locale)}</Numeric>
            <small>{copy.retentionCostBreaks}</small>
          </span>
        )}
      </div>
    </div>
  );
}

// CoefficientFreshnessChip: an honest status chip telling the operator whether
// the measured retention coefficients still match the underlying data, or have
// gone stale. The block is read from the live optimize plan first (most current
// to the run on screen), falling back to /api/parameters. When the API returns
// no coefficient_freshness block at all, nothing is rendered (no fabricated state).
export function freshnessDateLabel(value) {
  return formatStamp(value) || null;
}

export function CoefficientFreshnessChip({ plan, parameters, locale }) {
  const freshness = plan?.coefficient_freshness || parameters?.coefficient_freshness;
  if (!freshness || typeof freshness !== 'object') return null;

  const status = String(freshness.status || '').toLowerCase();
  if (status !== 'fresh' && status !== 'stale' && status !== 'unknown') return null;

  const computedLabel = freshnessDateLabel(freshness.computed_at);
  const changedFiles = normalizeRows(freshness.changed_files).filter(
    (name) => typeof name === 'string' && name.length > 0,
  );
  const reason = typeof freshness.reason === 'string' ? freshness.reason : '';

  const label =
    status === 'fresh'
      ? pageText(locale, 'Model measurements current', 'מדידות המודל עדכניות')
      : status === 'stale'
        ? pageText(locale, 'Model measurements out of date', 'מדידות המודל אינן עדכניות')
        : pageText(locale, 'Freshness unverifiable', 'לא ניתן לאמת עדכניות');

  return (
    <section className={`coefficient-freshness ${status}`} aria-label={label}>
      <div className="coefficient-freshness-head">
        <span className="coefficient-freshness-chip">{label}</span>
        {status === 'fresh' && computedLabel && (
          <span className="coefficient-freshness-date">
            {pageText(locale, 'Measured', 'נמדד')} <Numeric>{computedLabel}</Numeric>
          </span>
        )}
      </div>
      {status === 'stale' && (
        <div className="coefficient-freshness-detail">
          {changedFiles.length > 0 && (
            <p>
              {pageText(locale, 'Changed since measurement', 'השתנו מאז המדידה')}: {changedFiles.join(', ')}
            </p>
          )}
          {reason && <p>{reason}</p>}
        </div>
      )}
      {status === 'unknown' && reason && (
        <div className="coefficient-freshness-detail">
          <p>{reason}</p>
        </div>
      )}
    </section>
  );
}

// FirstBreakNote: when the measured first-break gate is active, the optimizer
// charges each programme's FIRST break extra retention cost. This renders a short
// bilingual note with the multiplier so the operator can see the adjustment is on.
// It reads first_break_active / first_break_multiplier from the live plan first,
// then /api/parameters. When the field is false or absent (the honest default;
// the lever is off by default), nothing is rendered.
export function readFirstBreak(source) {
  if (!source || typeof source !== 'object') return null;
  if (source.first_break_active === true) return source;
  const assumptions = source.assumptions;
  if (assumptions && typeof assumptions === 'object' && assumptions.first_break_active === true) {
    return assumptions;
  }
  return null;
}

export function FirstBreakNote({ plan, parameters, locale }) {
  const active = readFirstBreak(plan) || readFirstBreak(parameters);
  if (!active) return null;

  const multiplier = finiteNumber(active.first_break_multiplier);
  if (multiplier === null || multiplier <= 1) return null;
  const multiplierLabel = `x${multiplier.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

  return (
    <p className="first-break-note">
      {pageText(
        locale,
        "The first break of each programme is charged extra retention cost",
        'הברייק הראשון של כל תוכנית מתומחר בעלות שימור נוספת',
      )}{' '}
      (<Numeric>{multiplierLabel}</Numeric>).
    </p>
  );
}

export function RetentionCostPanel({ plan, parameters, copy, locale }) {
  const segments = normalizeRows(plan?.segments).filter(
    (segment) => segment?.retention_cost && typeof segment.retention_cost === 'object',
  );
  if (segments.length === 0) return null;

  return (
    <section className="retention-cost-panel" aria-label={copy.retentionCostTitle}>
      <div className="retention-cost-panel-head">
        <h2>{copy.retentionCostTitle}</h2>
        <p>{copy.retentionCostIntro}</p>
      </div>
      <FirstBreakNote plan={plan} parameters={parameters} locale={locale} />
      <div className="retention-cost-grid">
        {segments.map((segment, index) => (
          <RetentionCostSegment
            key={segment.id || segment.segment || segment.name || index}
            segment={segment}
            copy={copy}
            locale={locale}
          />
        ))}
      </div>
    </section>
  );
}
