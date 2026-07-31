import React, { useState } from 'react';
import { Button, Tooltip } from '@mui/material';
import { Download } from 'lucide-react';
import { formatCurrency, formatMinutes, formatPercent, pageText } from '../../shell/format';
import { breakLengthLabel, breakPositionLabel, programTypeLabel } from '../../shell/labels';
import { normalizeRows } from '../../shell/plan-model';
import { buildCsv, downloadCsv } from '../../shell/downloads';
import { DataTable, PageHeader, StatusBadge } from '../../shell/primitives';
import ScheduleInspector from '../day/ScheduleInspector';

export function BreakLibraryPage({ breakLibrary, copy, locale, notify, onGlobalRefresh }) {
  const rows = normalizeRows(breakLibrary.breaks);
  // Click-to-open reuses the Schedule page's inspector drawer: every ranked row
  // carries the saved plan's segment_id, so the same /api/schedule/segment/{id}
  // detail and edit affordances open in place on this page. A row without a
  // segment id (older saved plan) gets an honest notice instead of a dead click.
  const [inspect, setInspect] = useState(null);
  const openBreak = (row) => {
    if (row && row.segment_id) {
      setInspect({ segmentId: row.segment_id, channel: row.channel, day: row.day });
    } else if (notify) {
      notify('This row carries no segment id in the saved plan, so there is no detail to open.', 'לשורה זו אין מזהה מקטע בתוכנית השמורה, ולכן אין פירוט לפתיחה.');
    }
  };
  // Exports every ranked row, not only the visible grid page. Vocabulary values
  // are localized; numbers stay raw so spreadsheet sorting works.
  const exportLibraryCsv = () => {
    const csv = buildCsv([
      { label: pageText(locale, 'Status', 'סטטוס'), value: (row) => row.status },
      { label: pageText(locale, 'Channel', 'ערוץ'), value: (row) => row.channel },
      { label: pageText(locale, 'Date', 'תאריך'), value: (row) => row.date },
      { label: pageText(locale, 'Start time', 'שעת התחלה'), value: (row) => row.start_time },
      { label: pageText(locale, 'Programme type', 'סוג תוכנית'), value: (row) => programTypeLabel(row.program_type, locale) },
      { label: pageText(locale, 'Position', 'מיקום'), value: (row) => breakPositionLabel(row.position, locale) },
      { label: pageText(locale, 'Break type', 'סוג ברייק'), value: (row) => breakLengthLabel(row.break_type, locale) },
      { label: pageText(locale, 'Length (seconds)', 'אורך (שניות)'), value: (row) => row.total_break_time },
      { label: pageText(locale, 'Expected revenue (ILS)', 'הכנסה צפויה (ש"ח)'), value: (row) => Math.round(Number(row.predicted_revenue) || 0) },
      { label: pageText(locale, 'Expected retention (%)', 'שימור צפוי (%)'), value: (row) => (Number(row.predicted_retention || 0) * 100).toFixed(2) },
      { label: pageText(locale, 'Segment id', 'מזהה מקטע'), value: (row) => row.segment_id },
    ], rows);
    downloadCsv('kairos-break-library.csv', csv);
    if (notify) notify('Break library exported as CSV.', 'ספריית הברייקים יוצאה כ־CSV.');
  };
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Break library"
        titleHe="ספריית ברייקים"
        bodyEn="The ranked shelf of the strongest breaks in the saved plan. Review the ranking, open a break for its full detail and edits, and export the list for the traffic meeting."
        bodyHe="מדף מדורג של הברייקים החזקים בתוכנית השמורה. עברו על הדירוג, פתחו ברייק לפרטים המלאים ולעריכה, וייצאו את הרשימה לישיבת הטראפיק."
      />
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Ranked break candidates', 'ברייקים מדורגים')}</h2>
          <div className="panel-head-tools">
            <span>{rows.length} {pageText(locale, 'breaks', 'ברייקים')}</span>
            <Tooltip title={pageText(locale, 'Exports every ranked break, not only the visible page.', 'הייצוא כולל את כל הברייקים המדורגים, לא רק את העמוד המוצג.')} arrow placement="bottom">
              <span>
                <Button className="secondary-button" type="button" variant="outlined" disabled={rows.length === 0} onClick={exportLibraryCsv}>
                  <Download size={14} />
                  {pageText(locale, 'CSV export', 'ייצוא CSV')}
                </Button>
              </span>
            </Tooltip>
          </div>
        </div>
        <p className="row-open-hint">{pageText(locale, 'Selecting a row opens the break detail.', 'לחיצה על שורה פותחת את פרטי הברייק.')}</p>
        <DataTable
          locale={locale}
          fit
          pageSize={25}
          onRowClick={openBreak}
          emptyLabel={pageText(locale, 'No break candidates were found.', 'לא נמצאו ברייקים מועמדים.')}
          rows={rows}
          columns={[
            { key: 'status', label: pageText(locale, 'Status', 'סטטוס'), status: true, minWidth: 104, flex: 0.55, render: (row) => <StatusBadge status={row.status} locale={locale} mode="cell" /> },
            { key: 'channel', label: pageText(locale, 'Channel', 'ערוץ') },
            { key: 'date', label: pageText(locale, 'Airing', 'שידור'), numeric: true, render: (row) => [row.date, row.start_time].filter(Boolean).join(' ') || '-' },
            { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית'), render: (row) => programTypeLabel(row.program_type, locale) },
            { key: 'position', label: pageText(locale, 'Position', 'מיקום'), render: (row) => breakPositionLabel(row.position, locale) },
            { key: 'break_type', label: pageText(locale, 'Type', 'סוג'), render: (row) => breakLengthLabel(row.break_type, locale) },
            { key: 'total_break_time', label: pageText(locale, 'Length', 'אורך'), render: (row) => formatMinutes(row.total_break_time, locale) },
            { key: 'predicted_revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.predicted_revenue, locale) },
            { key: 'predicted_retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.predicted_retention || 0) * 100, locale) },
          ]}
        />
      </section>
      {inspect && (
        <ScheduleInspector
          segmentId={inspect.segmentId}
          channel={inspect.channel}
          day={inspect.day}
          locale={locale}
          notify={notify}
          onClose={() => setInspect(null)}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
    </section>
  );
}

export default BreakLibraryPage;
