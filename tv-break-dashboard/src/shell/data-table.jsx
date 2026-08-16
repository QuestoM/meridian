import React from 'react';
import { Figure, Name, isolate } from './bidi';
import { formatNumber, pageText } from './format';
import { normalizeRows } from './plan-model';

function cx(...values) {
  return values.flat().filter(Boolean).join(' ');
}

export const LazyDataGrid = React.lazy(() => import('@mui/x-data-grid').then((module) => ({ default: module.DataGrid })));

export function DataTable({ columns, rows, emptyLabel, locale = 'en', onRowClick, pageSize = 10, fit = false }) {
  const safeRows = normalizeRows(rows);
  const gridRows = safeRows.map((row, index) => ({
    ...row,
    id: String(row.id || row.Campaign || row.path || row.break_id || `${index}-${columns[0]?.key || 'row'}`),
  }));
  const numericKeys = new Set([
    'spots', 'seconds', 'revenue', 'target_spots', 'num_breaks',
    'total_break_time', 'predicted_revenue', 'predicted_retention',
    'channels', 'breaks', 'retention', 'size', 'rows',
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
      const className = cx('grid-cell-content', isNumeric && 'numeric-cell', column.status && 'status-grid-content');
      return isNumeric
        ? <Figure className={className}>{value}</Figure>
        : <Name className={className}>{value}</Name>;
    },
    align: column.align || startEdge,
    headerAlign: column.headerAlign || startEdge,
  }));

  return (
    <div className={cx('data-table-wrap', 'mui-grid-wrap', fit && 'grid-fit', onRowClick && 'grid-row-clickable')}>
      <React.Suspense fallback={<div className="grid-loading" role="status">{emptyLabel}</div>}>
        <LazyDataGrid
          rows={gridRows}
          columns={gridColumns}
          rowHeight={48}
          columnHeaderHeight={48}
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
