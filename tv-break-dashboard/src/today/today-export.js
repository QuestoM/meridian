// The rows on the screen, as a file, and nothing else.
//
// The export is built from the payload the surface is already rendering rather
// than from a second request, so the file and the screen cannot disagree: they
// are the same numbers at the same grain, which is the property a downloaded
// figure has to have before anybody can reconcile it against the page it came
// from. Nothing is calculated again here and nothing is rounded here.

function cell(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(columns, rows) {
  const head = columns.map((column) => cell(column.header)).join(',');
  const body = rows.map((row) => columns.map((column) => cell(column.value(row))).join(','));
  return [head, ...body].join('\n');
}

// A stated scope travels with the file for the same reason it travels with the
// figure: a column of shekels with no channel and no window on it is not a
// number anybody can check.
export function scopeComment(scope, extra = []) {
  const parts = [
    `channel,${cell(scope.channel || '')}`,
    `from,${cell(scope.date_from || '')}`,
    `to,${cell(scope.date_to || '')}`,
    `inclusive,${scope.inclusive === false ? 'no' : 'yes'}`,
    `timezone,${cell(scope.timezone || '')}`,
    `currency,${cell(scope.currency || 'ILS')}`,
    `source,${cell(scope.source || '')}`,
    ...extra.map(([key, value]) => `${cell(key)},${cell(value)}`),
  ];
  return parts.join('\n');
}

export function download(name, text) {
  if (typeof document === 'undefined') return;
  const blob = new Blob([`﻿${text}`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export const DAY_COLUMNS = [
  { header: 'date', value: (row) => row.date },
  { header: 'weekday', value: (row) => row.weekday_en },
  { header: 'is_weekend', value: (row) => (row.is_weekend ? 'yes' : 'no') },
  { header: 'breaks', value: (row) => row.total_breaks },
  { header: 'ad_seconds', value: (row) => row.total_ad_seconds },
  { header: 'projected_revenue_ils', value: (row) => row.projected_revenue },
];

export const SEGMENT_COLUMNS = [
  { header: 'segment_id', value: (row) => row.segment_id },
  { header: 'start_clock', value: (row) => row.start_clock },
  { header: 'program_type', value: (row) => row.program_type },
  { header: 'daypart', value: (row) => row.daypart },
  { header: 'breaks', value: (row) => row.breaks },
  { header: 'ad_seconds', value: (row) => row.ad_seconds },
  { header: 'projected_revenue_ils', value: (row) => row.projected_revenue },
  { header: 'retention_percent', value: (row) => row.retention_percent },
  { header: 'share_of_day_percent', value: (row) => row.share_percent },
  { header: 'gold', value: (row) => (row.is_gold ? 'yes' : 'no') },
];
