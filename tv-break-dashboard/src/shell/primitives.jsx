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

// THE CARD, and the inset of everything inside it.
//
// One home, in two files: the shape and the rule are in shell/card.css, which
// carries the measurement that produced them and is worth reading before you
// change anything here. This is the component side of the same thing.
//
// A card is a bordered surface with a head and some content. Its inset is
// --card-inset and the card does NOT apply it to itself, because the head's
// rule and a table's row rules have to reach the card's border while their
// content sits at the inset. So the inset belongs to the pieces, not the box.
//
//   <Card>
//     <CardHead title="Ranked breaks" tools={<Export />} />
//     <CardBody>prose, controls, a form</CardBody>
//     <CardBleed><SomeTable /></CardBleed>
//   </Card>
//
// CardBody is the default and CardBleed is the exception, deliberately: on
// every surface the owner reported, the inset element read as correct and the
// flush element read as broken. Reach for CardBleed only when the row rules of
// a long table need to span the card, and know that it still aligns the
// table's first column to the card's inset. See card.css.
export function Card({ as: Tag = 'section', dense = false, className = '', children, ...rest }) {
  const names = ['card', dense ? 'card-dense' : '', className].filter(Boolean).join(' ');
  return <Tag className={names} {...rest}>{children}</Tag>;
}

// The head. `title` is the card's name; `sub` is a quiet fact about it, such as
// a row count; `tools` is the acts, which sit at the far end of the line.
export function CardHead({ title, sub, tools, className = '' }) {
  return (
    <div className={['card-head', className].filter(Boolean).join(' ')}>
      <h2>{title}</h2>
      {sub ? <span>{sub}</span> : null}
      {tools ? <div className="panel-head-tools">{tools}</div> : null}
    </div>
  );
}

export function CardBody({ children, className = '' }) {
  return <div className={['card-body', className].filter(Boolean).join(' ')}>{children}</div>;
}

// The named opt-in. A reader of the calling code can see that this content
// reaches the card's edge on purpose, which is the whole point of the name.
export function CardBleed({ children, className = '' }) {
  return <div className={['card-bleed', className].filter(Boolean).join(' ')}>{children}</div>;
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
