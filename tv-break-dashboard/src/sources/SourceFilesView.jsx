import React from 'react';
import { CalendarDays, ClipboardCheck, Database, TableProperties } from 'lucide-react';
import { formatNumber, pageText } from '../shell/format';
import { normalizeRows } from '../shell/plan-model';
import { DataTable, Metric, StatusBadge } from '../shell/primitives';

// Are my source files here and current? The counts plus the freshness table.
export function SourceFilesView({ files, overview, locale }) {
  const fileRows = normalizeRows(files.files);
  return (
    <div className="data-tab-body">
      <section className="metric-strip page-metrics">
        <Metric label={pageText(locale, 'Programmes', 'תוכניות')} value={formatNumber(overview.source_counts?.programmes, locale)} icon={CalendarDays} positive />
        <Metric label={pageText(locale, 'Spots', 'ספוטים')} value={formatNumber(overview.source_counts?.spots, locale)} icon={TableProperties} positive />
        <Metric label={pageText(locale, 'Plan rows', 'שורות תכנון')} value={formatNumber(overview.source_counts?.planned_break_rows, locale)} icon={ClipboardCheck} positive />
        <Metric label={pageText(locale, 'Sources online', 'מקורות זמינים')} value={`${fileRows.filter((file) => file.exists).length}/${fileRows.length}`} icon={Database} positive />
      </section>
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Source files', 'קבצי מקור')}</h2>
          <span>{pageText(locale, 'Presence, size and last update of every production input', 'קיום, גודל ומועד עדכון של כל קלט פרודקשן')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No source files were found.', 'לא נמצאו קבצי מקור.')}
          rows={fileRows}
          columns={[
            { key: 'path', label: pageText(locale, 'Path', 'נתיב') },
            { key: 'exists', label: pageText(locale, 'State', 'מצב'), status: true, minWidth: 104, flex: 0.45, render: (row) => <StatusBadge status={row.exists ? 'ready' : 'error'} locale={locale} mode="cell" /> },
            { key: 'size', label: pageText(locale, 'Size', 'גודל'), render: (row) => `${formatNumber(Number(row.size || 0) / 1024, locale)} KB` },
            { key: 'modified', label: pageText(locale, 'Modified', 'עודכן'), render: (row) => (row.modified ? new Date(row.modified).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US') : '-') },
          ]}
        />
      </section>
    </div>
  );
}

export default SourceFilesView;
