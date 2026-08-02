import { pageText } from '../shell/format';
import { fetchCompliance, fetchForecasts, fetchReportPreview } from './sources-api';

// The rows behind a report, read from the same place the download reads.
//
// Two of the five reports are streamed by the server, so the server serves
// their rows. The other three are built in the browser out of a live endpoint,
// so this module reads that same endpoint and maps the same columns the
// download maps. One source per report, never two, because two derivations of
// one number is how a screen and a file come to disagree.

const MAX_ROWS = 20;

const NO_ROWS = {
  code: 'no_rows',
  en: 'This report has no rows yet, so there is nothing to show.',
  he: 'לדוח הזה אין עדיין שורות, ולכן אין מה להציג.',
};

const UNREACHABLE = {
  code: 'unavailable',
  en: 'The rows behind this report could not be read.',
  he: 'לא ניתן היה לקרוא את השורות שמאחורי הדוח הזה.',
};

function empty(note) {
  return { available: false, columns: [], rows: [], total_rows: 0, scoped_rows: 0, shown_rows: 0, scope: null, notes: [note] };
}

function table(columns, records, source) {
  if (!records.length) return { ...empty(NO_ROWS), source };
  const shown = records.slice(0, MAX_ROWS);
  return {
    available: true,
    columns: columns.map((column) => column.label),
    rows: shown.map((record) => columns.map((column) => String(column.value(record) ?? ''))),
    total_rows: records.length,
    scoped_rows: records.length,
    shown_rows: shown.length,
    scope: null,
    notes: [],
    source,
  };
}

// The compliance download's own six columns, in its own order.
function complianceColumns(locale) {
  return [
    { label: pageText(locale, 'Check', 'בדיקה'), value: (row) => pageText(locale, row.label_en, row.label_he) || row.id },
    { label: pageText(locale, 'Observed', 'נמדד'), value: (row) => row.observed },
    { label: pageText(locale, 'Limit', 'מגבלה'), value: (row) => row.limit },
    { label: pageText(locale, 'Unit', 'יחידה'), value: (row) => row.unit },
    { label: pageText(locale, 'Status', 'סטטוס'), value: (row) => row.status },
    { label: pageText(locale, 'Violations', 'חריגות'), value: (row) => row.violations },
  ];
}

// The revenue download's own four columns, with the same rounding it writes.
function revenueColumns(locale) {
  return [
    { label: pageText(locale, 'Day', 'יום'), value: (row) => row.day },
    { label: pageText(locale, 'Revenue (ILS)', 'הכנסה (ש"ח)'), value: (row) => Math.round(Number(row.revenue) || 0) },
    { label: pageText(locale, 'Retention (%)', 'שימור (%)'), value: (row) => (Number(row.retention) * 100).toFixed(2) },
    { label: pageText(locale, 'Breaks', 'ברייקים'), value: (row) => row.breaks },
  ];
}

// The source audit download's own four columns, off the same file list.
function fileColumns(locale) {
  return [
    { label: pageText(locale, 'File', 'קובץ'), value: (row) => row.path },
    { label: pageText(locale, 'Present', 'קיים'), value: (row) => (row.exists ? pageText(locale, 'yes', 'כן') : pageText(locale, 'no', 'לא')) },
    { label: pageText(locale, 'Size (KB)', 'גודל (KB)'), value: (row) => (Number(row.size || 0) / 1024).toFixed(1) },
    { label: pageText(locale, 'Modified', 'עודכן'), value: (row) => (row.modified ? new Date(row.modified).toISOString() : '') },
  ];
}

export async function reportRows(reportId, files, locale) {
  if (reportId === 'weekly-plan' || reportId === 'daily-spots' || reportId === 'spots-ledger') {
    const result = await fetchReportPreview(reportId === 'spots-ledger' ? 'daily-spots' : reportId, MAX_ROWS);
    if (!result.online || !result.preview) return empty(UNREACHABLE);
    return { ...result.preview, source: reportId === 'weekly-plan' ? '/api/export/schedule.csv' : '/api/export/spots.csv' };
  }
  if (reportId === 'compliance') {
    const result = await fetchCompliance();
    if (!result.online || !result.body) return empty(UNREACHABLE);
    const checks = Array.isArray(result.body.checks) ? result.body.checks : [];
    return table(complianceColumns(locale), checks, '/api/compliance');
  }
  if (reportId === 'revenue') {
    const result = await fetchForecasts();
    if (!result.online || !result.body) return empty(UNREACHABLE);
    const byDay = Array.isArray(result.body.by_day) ? result.body.by_day : [];
    return table(revenueColumns(locale), byDay, '/api/forecasts');
  }
  if (reportId === 'data-quality') {
    const rows = Array.isArray(files && files.files) ? files.files : [];
    return table(fileColumns(locale), rows, '/api/files');
  }
  return empty(UNREACHABLE);
}
