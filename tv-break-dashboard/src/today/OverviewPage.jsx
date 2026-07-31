import React from 'react';
import { Button } from '@mui/material';
import { Activity } from 'lucide-react';
import { Numeric, formatCurrency, formatNumber, formatPercent, pageText } from '../shell/format';
import { programTypeLabel, recommendationTitle } from '../shell/labels';
import { normalizeRows } from '../shell/plan-model';
import { PageHeader } from '../shell/primitives';
import SummaryMetrics from './SummaryMetrics';
import YieldView from './YieldView';
import { YieldMoneyPanel } from './MoneyWaterfall';
import ComplianceLedger from '../rules/ComplianceLedger';
import FrontierScopeChart from '../plan/week/FrontierScopeChart';

export function OverviewPage({ overview, compliance, files, copy, locale, setActiveView, onOpenRecommendation, loading, operatorChannel, savedRetentionFloor, onApplyFrontierFloor, applyWeightState, refreshKey, planEvents }) {
  const sourceCounts = overview.source_counts || {};
  const recommendations = normalizeRows(overview.recommendations);
  const fileRows = normalizeRows(files.files);
  const existingFiles = fileRows.filter((file) => file.exists).length;

  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Executive operating view"
        titleHe="תמונת ניהול תפעולית"
        bodyEn="A single read on revenue, retention, compliance, and the next decisions traffic teams need to make."
        bodyHe="מבט אחד על הכנסה, שימור צפייה, תאימות וההחלטות הבאות שצוותי הטראפיק צריכים לקבל."
        action={
          <Button className="run-button" type="button" variant="contained" onClick={() => setActiveView('Optimizer')}>
            <Activity size={15} />
            {copy.nav.Optimizer}
          </Button>
        }
      />
      <SummaryMetrics overview={overview} copy={copy} locale={locale} planEvents={planEvents} />
      <YieldMoneyPanel locale={locale} refreshKey={refreshKey} />
      <div className="page-grid two-one">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Priority decisions', 'החלטות בעדיפות גבוהה')}</h2>
            <span>{recommendations.length} {pageText(locale, 'actions', 'פעולות')}</span>
          </div>
          <div className="decision-list">
            {recommendations.slice(0, 5).map((item) => (
              <Button className="decision-row" type="button" key={item.id || item.title} onClick={() => (item.id && onOpenRecommendation ? onOpenRecommendation(item.id) : setActiveView('Optimizer'))}>
                <div>
                  <strong>{recommendationTitle(item, locale)}</strong>
                  <span>{programTypeLabel(item.program_type, locale) || pageText(locale, 'Mixed', 'מעורב')}</span>
                </div>
                <div>
                  <strong><Numeric>{formatCurrency(item.impact, locale)}</Numeric></strong>
                  <span><Numeric>{formatPercent(item.retention, locale)}</Numeric></span>
                </div>
              </Button>
            ))}
          </div>
        </section>
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Control room', 'חדר בקרה')}</h2>
            <span>{pageText(locale, 'Live model state', 'מצב מודל חי')}</span>
          </div>
          <div className="control-list">
            <div><span>{pageText(locale, 'Programmes', 'תוכניות')}</span><strong>{formatNumber(sourceCounts.programmes, locale)}</strong></div>
            <div><span>{pageText(locale, 'Spots', 'ספוטים')}</span><strong>{formatNumber(sourceCounts.spots, locale)}</strong></div>
            <div><span>{pageText(locale, 'Planned break rows', 'שורות תכנון ברייקים')}</span><strong>{formatNumber(sourceCounts.planned_break_rows, locale)}</strong></div>
            <div><span>{pageText(locale, 'Available source files', 'קבצי מקור זמינים')}</span><strong><Numeric>{existingFiles} / {fileRows.length}</Numeric></strong></div>
          </div>
        </section>
      </div>
      <div className="page-grid even">
        <ComplianceLedger compliance={compliance} copy={copy} locale={locale} />
        <FrontierScopeChart
          initialData={overview.frontier || []}
          copy={copy}
          locale={locale}
          loading={loading}
          operatorChannel={operatorChannel}
          savedRetentionFloor={savedRetentionFloor}
          onApplyFloor={onApplyFrontierFloor}
          applyState={applyWeightState}
          status={overview.frontier_status || ''}
        />
      </div>
      <YieldView locale={locale} refreshKey={refreshKey} />
    </section>
  );
}

export default OverviewPage;
