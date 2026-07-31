import React from 'react';
import { Tooltip } from '@mui/material';
import { finiteNumber, formatCurrency, formatNumber, formatPercent, pageText, summaryBasisLabel } from '../../shell/format';
import { dayLabel, scenarioNameLabel } from '../../shell/labels';
import { dayKeys, normalizeRows } from '../../shell/plan-model';
import { DataTable, PageHeader } from '../../shell/primitives';
import ScenarioCompare from './ScenarioCompare';
import FrontierPanel from './FrontierPanel';

export function ForecastsPage({ forecasts, overview, copy, locale, loading }) {
  // The API returns days in arbitrary (alphabetical) order; present them as a
  // week so the table reads Mon through Sun instead of a scrambled sequence.
  const days = normalizeRows(forecasts.by_day)
    .slice()
    .sort((a, b) => dayKeys.indexOf(a.day) - dayKeys.indexOf(b.day));
  const forecastBasis = summaryBasisLabel(overview.summary, locale);
  const scenarios = normalizeRows(forecasts.scenarios);
  const maxRevenue = Math.max(...scenarios.map((item) => Number(item.revenue || 0)), 1);
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Forecast scenarios"
        titleHe="תרחישי תחזית"
        bodyEn="Compare revenue-forward, balanced, and retention-protected plans before committing inventory."
        bodyHe="השוואה בין תוכניות שמעדיפות הכנסה, איזון או הגנת שימור לפני נעילת המלאי."
      />
      <div className="page-grid even">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Scenario curve', 'עקומת תרחישים')}</h2>
            <span>{copy.frontierMode}</span>
          </div>
          <div className="scenario-bars chart-ltr" dir="ltr">
            {scenarios.map((item) => {
              const weight = finiteNumber(item.revenue_weight);
              const weightTitle = weight === null
                ? ''
                : pageText(locale, `This scenario runs the optimizer at revenue weight ${weight} of 100 (higher chases revenue harder, lower protects viewers more)`, `התרחיש הזה מריץ את האופטימייזר עם משקל הכנסה ${weight} מתוך 100 (גבוה יותר רודף הכנסה, נמוך יותר מגן על הצופים)`);
              return (
                <Tooltip title={weightTitle} arrow placement="bottom" key={item.name}>
                  <div className="scenario-row">
                    <span>{scenarioNameLabel(item.name, locale)}</span>
                    <i style={{ '--bar': Number(item.revenue || 0) / maxRevenue }} />
                    <strong>{formatCurrency(item.revenue, locale)}</strong>
                    <small>{formatPercent(item.retention, locale)}</small>
                  </div>
                </Tooltip>
              );
            })}
          </div>
          <p className="data-basis-note">
            {pageText(
              locale,
              'Each scenario is a real optimizer run on one representative channel-day under the saved guardrails. These figures are not weekly totals; the daily forecast below is built from the saved weekly plan.',
              'כל תרחיש הוא ריצת אופטימיזציה אמיתית על יום-ערוץ מייצג אחד תחת הבקרות השמורות. אלה אינם סכומים שבועיים; התחזית היומית מטה נבנית מהתוכנית השבועית השמורה.',
            )}
            {forecastBasis ? ` ${pageText(locale, `Basis: ${forecastBasis}.`, `בסיס הנתונים: ${forecastBasis}.`)}` : ''}
          </p>
        </section>
        <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} loading={loading} operatorChannel={overview.settings?.operator_channel || ''} status={overview.frontier_status || ''} netPoint={overview.frontier_net_point || null} />
      </div>
      <ScenarioCompare locale={locale} savedRevenueWeight={finiteNumber(overview.settings?.revenue_weight)} />
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Daily forecast', 'תחזית יומית')}</h2>
          <span>{days.length} {pageText(locale, 'days', 'ימים')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No forecast rows were found.', 'לא נמצאו שורות תחזית.')}
          rows={days}
          columns={[
            { key: 'day', label: pageText(locale, 'Day', 'יום'), render: (row) => dayLabel(row.day, locale) },
            { key: 'breaks', label: pageText(locale, 'Breaks', 'ברייקים'), render: (row) => formatNumber(row.breaks, locale) },
            { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
            { key: 'retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.retention || 0) * 100, locale) },
          ]}
        />
      </section>
    </section>
  );
}

export default ForecastsPage;
