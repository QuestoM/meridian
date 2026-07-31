import { API_BASE } from './api';
import { pageText } from './format';
import { normalizeRows } from './plan-model';

export function downloadJson(filename, payload) {
  if (typeof window === 'undefined') return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

// Bilingual friendly names for the schedule_freshness changed-group keys (the
// frozen contract shared with ScheduleStalenessBanner).
export function freshnessChangedLabels(freshness, locale) {
  const labels = {
    settings: pageText(locale, 'settings', 'הגדרות'),
    constraints: pageText(locale, 'constraints', 'אילוצים'),
    overrides: pageText(locale, 'manual overrides', 'עקיפות ידניות'),
    coefficients: pageText(locale, 'model measurements', 'מדידות המודל'),
    data: pageText(locale, 'source data', 'נתוני מקור'),
  };
  return normalizeRows(freshness?.changed)
    .map((key) => labels[key])
    .filter((label) => typeof label === 'string' && label.length > 0);
}

export async function downloadScheduleCsv(locale, notify, freshness) {
  if (typeof window === 'undefined') return;
  // The saved schedule CSV can lag the operator's latest edits. When the backend
  // reports it stale, name what changed and let the operator decide whether the
  // outdated export is still wanted, instead of silently shipping old numbers.
  if (freshness && String(freshness.status || '').toLowerCase() === 'stale') {
    const changed = freshnessChangedLabels(freshness, locale);
    const changedPhrase = changed.length > 0 ? changed.join(', ') : pageText(locale, 'inputs', 'קלטים');
    const question = pageText(locale, `The saved schedule is out of date (changed since it was computed: ${changedPhrase}). Download the outdated CSV anyway?`, `הלוח השמור אינו מעודכן (השתנו מאז חישובו: ${changedPhrase}). להוריד בכל זאת את ה־CSV הלא מעודכן?`);
    if (!window.confirm(question)) return;
  }
  try {
    const response = await fetch(`${API_BASE}/api/export/schedule.csv`);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const disposition = response.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/i);
    const filename = match ? match[1] : 'kairos-weekly-schedule.csv';
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    if (notify) {
      notify('Schedule exported as CSV.', 'הלוח יוצא כ־CSV.');
    }
  } catch (error) {
    if (notify) {
      const status = String(error.message || '');
      if (status.startsWith('404')) {
        notify('No schedule is available to export yet.', 'אין לוח זמין לייצוא עדיין.');
      } else {
        notify(`Schedule export failed (${error.message}).`, `ייצוא הלוח נכשל (${error.message}).`);
      }
    }
  }
}

// Streams the per-spot daily pricing ledger the server builds (every priced
// spot with its premium and revenue, plus every dropped spot with its reason).
export async function downloadSpotsLedgerCsv(locale, notify) {
  if (typeof window === 'undefined') return;
  try {
    const response = await fetch(`${API_BASE}/api/export/spots.csv`);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const disposition = response.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/i);
    const filename = match ? match[1] : 'kairos-daily-spots.csv';
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    if (notify) {
      notify('Daily spot ledger exported as CSV.', 'יומן הספוטים היומי יוצא כ־CSV.');
    }
  } catch (error) {
    if (notify) {
      const status = String(error.message || '');
      if (status.startsWith('404')) {
        notify('No spot ledger is available to export yet.', 'אין יומן ספוטים זמין לייצוא עדיין.');
      } else {
        notify(`Spot ledger export failed (${error.message}).`, `ייצוא יומן הספוטים נכשל (${error.message}).`);
      }
    }
  }
}

// Build a CSV string from real rows. Every field is quote-escaped so commas,
// quotes and newlines in the data never break a column. Header comes from the
// column labels; each cell reads its column key off the row.
export function buildCsv(columns, rows) {
  const escape = (value) => {
    const text = value === null || value === undefined ? '' : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const header = columns.map((column) => escape(column.label)).join(',');
  const body = rows.map((row) => columns.map((column) => escape(column.value(row))).join(',')).join('\n');
  // A BOM keeps Hebrew readable when the CSV is opened in Excel.
  return `﻿${header}\n${body}\n`;
}

export function downloadCsv(filename, text) {
  if (typeof window === 'undefined') return;
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

// Per-report downloaders. Each fetches the report's REAL content and saves it as a
// CSV; nothing is derived from the card metadata. The weekly plan streams from the
// server's own CSV export; the others build a CSV from the live endpoint.
export async function downloadComplianceReport(locale, notify) {
  try {
    const response = await fetch(`${API_BASE}/api/compliance`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    const checks = Array.isArray(data.checks) ? data.checks : [];
    const csv = buildCsv([
      { label: pageText(locale, 'Check', 'בדיקה'), value: (row) => pageText(locale, row.label_en, row.label_he) || row.id },
      { label: pageText(locale, 'Observed', 'נמדד'), value: (row) => row.observed },
      { label: pageText(locale, 'Limit', 'מגבלה'), value: (row) => row.limit },
      { label: pageText(locale, 'Unit', 'יחידה'), value: (row) => row.unit },
      { label: pageText(locale, 'Status', 'סטטוס'), value: (row) => row.status },
      { label: pageText(locale, 'Violations', 'חריגות'), value: (row) => row.violations },
    ], checks);
    downloadCsv('kairos-compliance.csv', csv);
    if (notify) notify('Compliance report exported as CSV.', 'דוח התאימות יוצא כ־CSV.');
  } catch (error) {
    if (notify) notify(`Compliance export failed (${error.message}).`, `ייצוא דוח התאימות נכשל (${error.message}).`);
  }
}

export async function downloadRevenueReport(locale, notify) {
  try {
    const response = await fetch(`${API_BASE}/api/forecasts`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    const byDay = Array.isArray(data.by_day) ? data.by_day : [];
    const csv = buildCsv([
      { label: pageText(locale, 'Day', 'יום'), value: (row) => row.day },
      { label: pageText(locale, 'Revenue (ILS)', 'הכנסה (ש"ח)'), value: (row) => Math.round(Number(row.revenue) || 0) },
      { label: pageText(locale, 'Retention (%)', 'שימור (%)'), value: (row) => (Number(row.retention) * 100).toFixed(2) },
      { label: pageText(locale, 'Breaks', 'ברייקים'), value: (row) => row.breaks },
    ], byDay);
    downloadCsv('kairos-revenue-forecast.csv', csv);
    if (notify) notify('Revenue forecast exported as CSV.', 'תחזית ההכנסה יוצאה כ־CSV.');
  } catch (error) {
    if (notify) notify(`Revenue export failed (${error.message}).`, `ייצוא תחזית ההכנסה נכשל (${error.message}).`);
  }
}

export function downloadDataQualityReport(fileRows, locale, notify) {
  const csv = buildCsv([
    { label: pageText(locale, 'File', 'קובץ'), value: (row) => row.path },
    { label: pageText(locale, 'Present', 'קיים'), value: (row) => (row.exists ? pageText(locale, 'yes', 'כן') : pageText(locale, 'no', 'לא')) },
    { label: pageText(locale, 'Size (KB)', 'גודל (KB)'), value: (row) => (Number(row.size || 0) / 1024).toFixed(1) },
    { label: pageText(locale, 'Modified', 'עודכן'), value: (row) => (row.modified ? new Date(row.modified).toISOString() : '') },
  ], fileRows);
  downloadCsv('kairos-source-audit.csv', csv);
  if (notify) notify('Source audit exported as CSV.', 'בקרת המקורות יוצאה כ־CSV.');
}
