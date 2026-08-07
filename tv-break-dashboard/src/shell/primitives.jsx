import React from 'react';
import { Tooltip } from '@mui/material';
import { ArrowDown, ArrowUp } from 'lucide-react';
import { Numeric, formatNumber, pageText } from './format';
import { Figure, Name } from './bidi';
import { normalizeRows } from './plan-model';
import { isolate } from './bidi';

export const LazyDataGrid = React.lazy(() => import('@mui/x-data-grid').then((module) => ({ default: module.DataGrid })));

export function Metric({ label, value, delta, sub, icon: Icon, positive = false, tone, title }) {
  const hasDelta = delta !== undefined && delta !== null && delta !== '';
  // MUI Tooltip rather than the native title attribute: the native bubble
  // ignores the document direction, so Hebrew explanations rendered
  // left-aligned; the themed tooltip follows the app's RTL setup.
  const body = (
    <div className="metric">
      <span className={`metric-icon ${tone || ''}`}>
        <Icon size={17} strokeWidth={1.8} />
      </span>
      <span className="metric-copy">
        <span>{label}</span>
        <strong><Numeric>{value}</Numeric></strong>
        {sub ? <small className="metric-sub">{sub}</small> : null}
      </span>
      {hasDelta ? (
        <span className={positive ? 'delta positive' : tone === 'risk' ? 'delta risk' : 'delta negative'}>
          {positive ? <ArrowUp size={12} /> : tone === 'risk' ? null : <ArrowDown size={12} />}
          <Numeric>{delta}</Numeric>
        </span>
      ) : null}
    </div>
  );
  return title ? <Tooltip title={title} arrow placement="bottom">{body}</Tooltip> : body;
}

export function PageHeader({ locale, titleEn, titleHe, bodyEn, bodyHe, action }) {
  return (
    <div className="page-header">
      <div>
        <h1>{pageText(locale, titleEn, titleHe)}</h1>
        <p>{pageText(locale, bodyEn, bodyHe)}</p>
      </div>
      {action}
    </div>
  );
}

export function StatusBadge({ status, locale, mode = 'inline' }) {
  const normalized = String(status || 'ready').toLowerCase();
  const labelMap = {
    ready: pageText(locale, 'Ready', 'מוכן'),
    compliant: pageText(locale, 'Compliant', 'תקין'),
    at_risk: pageText(locale, 'Needs review', 'דורש בדיקה'),
    attention: pageText(locale, 'Needs attention', 'דורש טיפול'),
    empty: pageText(locale, 'No rows yet', 'אין שורות עדיין'),
    error: pageText(locale, 'Error', 'שגיאה'),
  };
  return <span className={`status-badge ${mode} ${normalized}`}>{labelMap[normalized] || status}</span>;
}

export function DataTable({ columns, rows, emptyLabel, locale = 'en', onRowClick, pageSize = 10, fit = false }) {
  const safeRows = normalizeRows(rows);
  const gridRows = safeRows.map((row, index) => ({
    ...row,
    id: String(row.id || row.Campaign || row.path || row.break_id || `${index}-${columns[0]?.key || 'row'}`),
  }));
  const numericKeys = new Set([
    'spots',
    'seconds',
    'revenue',
    'target_spots',
    'num_breaks',
    'total_break_time',
    'predicted_revenue',
    'predicted_retention',
    'channels',
    'breaks',
    'retention',
    'size',
    'rows',
  ]);
  const startEdge = locale === 'he' ? 'right' : 'left';
  const gridColumns = columns.map((column) => ({
    field: column.key,
    headerName: column.label,
    flex: column.flex || 1,
    minWidth: column.minWidth || 120,
    sortable: column.sortable !== false,
    cellClassName: column.status ? 'status-data-grid-cell' : undefined,
    renderCell: (params) => {
      const isNumeric = column.numeric || numericKeys.has(column.key);
      const value = column.render
        ? column.render(params.row, params.api.getRowIndexRelativeToVisibleRows?.(params.id) || 0)
        : params.value ?? '';
      const className = [
        'grid-cell-content',
        isNumeric ? 'numeric-cell' : '',
        column.status ? 'status-grid-content' : '',
      ].filter(Boolean).join(' ');
      // A numeric column is a figure; anything else may be a name in either
      // script. Neither states a direction of its own, so the cell keeps the
      // grid's direction and the column stays aligned with its neighbours.
      return isNumeric
        ? <Figure className={className}>{value}</Figure>
        : <Name className={className}>{value}</Name>;
    },
    // The reading edge, for every column and its header alike. The grid takes
    // physical values only, so "start" is spelled out from the locale here, in
    // one place. A numeric column used to be pinned right in both locales, which
    // put it on the opposite edge from its neighbours in Hebrew and was the
    // misalignment the owner reported.
    align: column.align || startEdge,
    headerAlign: column.headerAlign || startEdge,
  }));

  const wrapClassName = [
    'data-table-wrap',
    'mui-grid-wrap',
    fit ? 'grid-fit' : '',
    onRowClick ? 'grid-row-clickable' : '',
  ].filter(Boolean).join(' ');
  return (
    <div className={wrapClassName}>
      <React.Suspense fallback={<div className="grid-loading">{emptyLabel}</div>}>
        <LazyDataGrid
          rows={gridRows}
          columns={gridColumns}
          density="compact"
          disableRowSelectionOnClick
          onRowClick={onRowClick ? (params) => onRowClick(params.row) : undefined}
          pageSizeOptions={[10, 25, 50]}
        initialState={{ pagination: { paginationModel: { pageSize, page: 0 } } }}
        localeText={{
          noRowsLabel: emptyLabel,
          paginationRowsPerPage: pageText(locale, 'Rows per page:', 'שורות בעמוד:'),
          paginationDisplayedRows: ({ from, to, count, estimated }) => {
            const total = count !== -1 ? formatNumber(count, locale) : pageText(locale, `more than ${to}`, `יותר מ-${to}`);
            const estimate = estimated && estimated > to ? formatNumber(estimated, locale) : total;
            // The from-to range is isolated as one run so the RTL line never
            // reorders it into to-from.
            return pageText(
              locale,
              `${formatNumber(from, locale)}-${formatNumber(to, locale)} of ${estimate}`,
              `${isolate(`${formatNumber(from, locale)}-${formatNumber(to, locale)}`)} מתוך ${estimate}`,
            );
          },
        }}
        autoHeight
      />
      </React.Suspense>
    </div>
  );
}
