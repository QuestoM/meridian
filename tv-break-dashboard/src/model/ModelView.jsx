import React from 'react';
import { Tooltip } from '@mui/material';
import {
  EMPTY_VALUE,
  Numeric,
  finiteNumber,
  formatCurrency,
  formatNumber,
  formatPercent,
  formatRetentionDelta,
  pageText,
} from '../shell/format';
import { fallbackSettings } from '../shell/fallbacks';
import { impactSegmentLabel, impactSourceLabel } from '../shell/labels';
import { normalizeRows } from '../shell/plan-model';

export function normalizeImpactRows(rows, segmentKey) {
  return normalizeRows(rows)
    .map((row) => {
      const coefficient =
        finiteNumber(row.average_coefficient) ??
        finiteNumber(row.average) ??
        finiteNumber(row.coefficient) ??
        finiteNumber(row.total_impact);
      return {
        segment: row.segment || row[segmentKey] || row.name || row.channel_name || '',
        coefficient,
        sampleCount: finiteNumber(row.sampleCount) ?? finiteNumber(row.sample_count) ?? finiteNumber(row.count),
        channelCount: finiteNumber(row.channelCount) ?? finiteNumber(row.channel_count),
        ciLow: finiteNumber(row.ciLow) ?? finiteNumber(row.ci_low),
        ciHigh: finiteNumber(row.ciHigh) ?? finiteNumber(row.ci_high),
      };
    })
    .filter((row) => row.segment);
}

// What did the model learn, and what rules govern the plan? Explainability plus
// the optimizer parameter ledger.
export function ModelView({ impact, parameters, locale }) {
  const measuredImpacts = impact.coefficient_impacts || {};
  const programImpacts = normalizeImpactRows(
    normalizeRows(measuredImpacts.program_type).length ? measuredImpacts.program_type : impact.program_type_impacts,
    'program_type',
  );
  const positionImpacts = normalizeImpactRows(
    normalizeRows(measuredImpacts.position).length ? measuredImpacts.position : impact.position_impacts,
    'position',
  );
  const lengthImpacts = normalizeImpactRows(
    normalizeRows(measuredImpacts.length).length ? measuredImpacts.length : impact.length_impacts,
    'length',
  );
  const impactSource = impactSourceLabel(measuredImpacts.source || 'legacy_csv', measuredImpacts.metadata, locale);
  return (
    <div className="data-tab-body">
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Model explainability', 'הסבריות מודל')}</h2>
          <span>{impactSource}</span>
        </div>
        <div className="impact-stack model-explain-grid">
          <ImpactPreview title={pageText(locale, 'Programme type impact', 'השפעת סוג תוכנית')} rows={programImpacts} locale={locale} />
          <ImpactPreview title={pageText(locale, 'Position impact', 'השפעת מיקום')} rows={positionImpacts} locale={locale} />
          <ImpactPreview title={pageText(locale, 'Length impact', 'השפעת אורך')} rows={lengthImpacts} locale={locale} />
          <DriftMonitorCard drift={impact.drift} locale={locale} />
        </div>
        {typeof measuredImpacts.pooling_note === 'string' && measuredImpacts.pooling_note.trim() && (
          <p className="data-basis-note">
            {pageText(locale, 'Model reliability note:', 'הערת מהימנות מהמודל:')}{' '}
            <span dir="ltr">{measuredImpacts.pooling_note}</span>
          </p>
        )}
      </section>
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Optimizer parameters', 'פרמטרי אופטימיזציה')}</h2>
          <span>{pageText(locale, 'Guardrails, assumptions, pricing', 'בקרות, הנחות ותמחור')}</span>
        </div>
        <ParameterLedger parameters={parameters} locale={locale} />
      </section>
    </div>
  );
}

export function ImpactPreview({ title, rows, locale }) {
  const first = normalizeImpactRows(rows, 'segment').slice(0, 4);
  const maxMagnitude = Math.max(...first.map((row) => Math.abs(row.coefficient || 0)), 0.01);
  return (
    <div className="impact-preview">
      <header>
        <strong>{title}</strong>
        <small>{pageText(locale, 'Retention delta per break', 'שינוי שימור לכל ברייק')}</small>
      </header>
      {first.length === 0 ? (
        <span>{pageText(locale, 'No measurements yet', 'אין עדיין מדידות')}</span>
      ) : (
        first.map((row, index) => {
          const magnitude = row.coefficient === null ? 0 : Math.abs(row.coefficient);
          // The range and the n= figure are Latin-and-digit runs inside RTL
          // text, so both carry the ltr-isolating numeric class; only the plain
          // Hebrew "sample pending" text renders without it.
          const range = row.ciLow !== null && row.ciHigh !== null
            ? `${formatRetentionDelta(row.ciLow, locale)} / ${formatRetentionDelta(row.ciHigh, locale)}`
            : row.sampleCount
              ? `n=${formatNumber(row.sampleCount, locale)}`
              : null;
          const coefficientLabel = formatRetentionDelta(row.coefficient, locale);
          return (
            <div className="impact-row" key={`${title}-${row.segment}-${index}`}>
              <span className="impact-label">{impactSegmentLabel(row.segment, locale)}</span>
              <span className="impact-meter" aria-hidden="true">
                <i style={{ '--impact-width': `${Math.max(8, (magnitude / maxMagnitude) * 100)}%` }} />
              </span>
              <strong>{row.coefficient === null ? coefficientLabel : <Numeric>{coefficientLabel}</Numeric>}</strong>
              <small className={range !== null ? 'numeric' : undefined}>{range !== null ? range : pageText(locale, 'sample pending', 'מדגם לא זמין')}</small>
            </div>
          );
        })
      )}
    </div>
  );
}

// Log-effect values are close enough to fractional level changes at the drift
// monitor's magnitudes, so value * 100 is shown as a signed percent-like figure.
export function formatDriftPercent(value, locale) {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const points = number * 100;
  const sign = points > 0 ? '+' : '';
  return `${sign}${points.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 })}%`;
}

// Audience level stability: surfaces the weekly level-drift measurement the
// coefficient rebuild stores in the artifact metadata and /api/impact echoes
// as `drift`. Renders the measured block or the honest absent reason; when the
// backend sends no verdict, none is invented here.
export function DriftMonitorCard({ drift, locale }) {
  const block = drift && typeof drift === 'object' ? drift : null;
  const title = pageText(locale, 'Audience level stability', 'יציבות רמת הצפייה');
  if (!block || block.status !== 'measured') {
    const reason = typeof block?.reason === 'string' && block.reason.trim() ? block.reason : null;
    return (
      <div className="impact-preview drift-card">
        <header>
          <strong>{title}</strong>
          <small>{pageText(locale, 'Weekly monitor', 'ניטור שבועי')}</small>
        </header>
        <p className="drift-note">{pageText(locale, 'No level-drift measurement is available for the current coefficients.', 'מדידת סחיפת הרמה אינה זמינה עבור המקדמים הנוכחיים.')}</p>
        {reason ? <p className="drift-reason" dir="ltr">{reason}</p> : null}
      </div>
    );
  }
  const driftLabel = formatDriftPercent(block.drift_per_week, locale);
  const seNumber = finiteNumber(block.drift_se);
  const seLabel = seNumber === null ? null : `± ${(seNumber * 100).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 })}%`;
  const bindingState = block.binding === true ? 'binding' : block.binding === false ? 'stable' : 'unknown';
  const chipLabel = bindingState === 'binding' ? pageText(locale, 'Needs attention', 'דורש תשומת לב') : bindingState === 'stable' ? pageText(locale, 'Stable', 'יציב') : pageText(locale, 'Not determined', 'לא נקבע');
  const weeks = normalizeRows(block.weekly_levels);
  const means = weeks.map((week) => finiteNumber(week.mean_log_effect)).filter((value) => value !== null);
  const minMean = means.length ? Math.min(...means) : 0;
  const meanSpan = means.length ? Math.max(...means) - minMean : 0;
  return (
    <div className="impact-preview drift-card">
      <header>
        <strong>{title}</strong>
        <small><Numeric>{formatNumber(block.n_weeks, locale)}</Numeric> {pageText(locale, 'weeks', 'שבועות')}, <Numeric>{formatNumber(block.n_breaks, locale)}</Numeric> {pageText(locale, 'breaks', 'ברייקים')}</small>
      </header>
      <div className="drift-headline">
        <div className="drift-stat">
          <strong><Numeric>{seLabel ? `${driftLabel} ${seLabel}` : driftLabel}</Numeric></strong>
          <small>{pageText(locale, 'Drift per week', 'סחיפה לשבוע')}</small>
        </div>
        <Tooltip
          title={typeof block.criterion === 'string' && block.criterion
            ? <span>{pageText(locale, 'The measured rule behind this verdict:', 'הכלל המדוד שמאחורי הקביעה הזו:')} <span dir="ltr">{block.criterion}</span></span>
            : ''}
          arrow
          placement="bottom"
        >
          <span className={`drift-chip ${bindingState}`}>{chipLabel}</span>
        </Tooltip>
      </div>
      {weeks.length > 0 ? (
        <div className="drift-week-block">
          <small className="drift-strip-caption">{pageText(locale, 'Weekly mean level', 'רמה שבועית ממוצעת')}</small>
          <div className="drift-week-strip">
            {weeks.map((week, index) => {
              const mean = finiteNumber(week.mean_log_effect);
              const ratio = mean === null || meanSpan <= 0 ? 1 : (mean - minMean) / meanSpan;
              return (
                <div className="drift-week" key={`drift-week-${week.week ?? index}`}>
                  <small>{pageText(locale, `Week ${week.week ?? index + 1}`, `שבוע ${week.week ?? index + 1}`)}</small>
                  <span className="drift-week-bar" aria-hidden="true"><i style={{ '--drift-week-width': `${Math.round(12 + ratio * 88)}%` }} /></span>
                  <strong><Numeric>{formatDriftPercent(mean, locale)}</Numeric></strong>
                  <small><Numeric>{`n=${formatNumber(week.n, locale)}`}</Numeric></small>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
      <p className="drift-note">{pageText(locale, "The plan's coefficients assume a steady audience level. A drift above the threshold means the weekly level moves more than the measurement's own precision, so recompute the coefficients when new data lands.", 'מקדמי התוכנית מניחים רמת צפייה יציבה. סחיפה מעל הסף פירושה שהרמה השבועית זזה יותר מדיוק המדידה עצמה, ולכן מומלץ לחשב את המקדמים מחדש כשנקלטים נתונים חדשים.')}</p>
    </div>
  );
}

export function ParameterLedger({ parameters, locale }) {
  const settings = parameters?.settings || fallbackSettings;
  const guardrails = parameters?.guardrails || {};
  const assumptions = parameters?.assumptions || {};
  const pricing = parameters?.pricing || {};
  const retentionAssumption = finiteNumber(assumptions.retention_impact_per_break);
  const basePrice = finiteNumber(pricing.base_price_per_second_per_tvr_point);
  const rows = [
    {
      label: pageText(locale, 'Ad minutes per hour', 'דקות פרסום לשעה'),
      value: `${formatNumber(settings.max_ad_minutes_per_hour, locale)} ${pageText(locale, 'min', 'דק׳')}`,
      detail: pageText(locale, 'Regulatory ceiling', 'תקרת רגולציה'),
    },
    {
      label: pageText(locale, 'Breaks per hour', 'ברייקים לשעה'),
      value: formatNumber(settings.max_breaks_per_hour, locale),
      detail: pageText(locale, 'Operational guardrail', 'בקרה תפעולית'),
    },
    {
      label: pageText(locale, 'Minimum spacing', 'מרווח מינימלי'),
      value: `${formatNumber(settings.min_break_spacing_minutes, locale)} ${pageText(locale, 'min', 'דק׳')}`,
      detail: pageText(locale, 'Between break starts', 'בין תחילות ברייקים'),
    },
    {
      label: pageText(locale, 'Retention floor', 'רף שימור'),
      value: formatPercent(Number(settings.min_retention_floor || 0) * 100, locale),
      detail: guardrails.min_retention_floor ? pageText(locale, 'Engine guardrail', 'בקרת מנוע') : pageText(locale, 'Saved setting', 'הגדרה שמורה'),
    },
    {
      label: pageText(locale, 'Retention assumption', 'הנחת שימור'),
      value: retentionAssumption === null ? '-' : formatRetentionDelta(retentionAssumption, locale),
      detail: pageText(locale, 'Default used when a segment has no measurements', 'ברירת מחדל כשסגמנט לא נמדד'),
    },
    {
      label: pageText(locale, 'Base price', 'מחיר בסיס'),
      value: basePrice === null ? '-' : formatCurrency(basePrice, locale),
      detail: pageText(locale, 'Per second per TVR point', 'לשנייה לכל נקודת TVR'),
    },
  ];
  const premiumRows = Object.entries(pricing.program_type_premiums || {})
    .slice(0, 6)
    .map(([name, value]) => ({ name, value: finiteNumber(value) }));
  return (
    <div className="parameter-ledger">
      <div className="parameter-grid">
        {rows.map((row) => (
          <div className="parameter-row" key={row.label}>
            <span>{row.label}</span>
            <strong><Numeric>{row.value}</Numeric></strong>
            <small>{row.detail}</small>
          </div>
        ))}
      </div>
      <div className="premium-list">
        <strong>{pageText(locale, 'Programme pricing premiums', 'פרמיות תמחור לפי סוג תוכנית')}</strong>
        {premiumRows.length === 0 ? (
          <span>{pageText(locale, 'No pricing model loaded', 'מודל תמחור לא נטען')}</span>
        ) : (
          premiumRows.map((row) => (
            <span key={row.name}>
              <b>{row.name}</b>
              <Numeric>{row.value === null ? '-' : `${formatNumber(row.value, locale)}x`}</Numeric>
            </span>
          ))
        )}
      </div>
    </div>
  );
}

export default ModelView;
