import React, { useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { X } from 'lucide-react';
import { Numeric, formatNumber } from '../shell/format';
import { serverText, text } from './sources-copy';
import { reportRows } from './report-rows';

// The rows behind a report's row count. A card that says "8,704 rows in this
// download" and cannot show one of them is asking to be believed; this is the
// Stripe drill, where an amount opens the rows that make it up.
//
// Every preview comes from the same source as the file it describes, so the
// screen and the export are the same numbers at two grains rather than two
// derivations that agree until they do not.
export function ReportRowsDrawer({ report, files, locale, onClose }) {
  const [state, setState] = useState({ loading: true, preview: null });

  useEffect(() => {
    let active = true;
    setState({ loading: true, preview: null });
    reportRows(report.id, files, locale).then((preview) => {
      if (active) setState({ loading: false, preview });
    });
    return () => {
      active = false;
    };
  }, [report.id, files, locale]);

  const preview = state.preview;
  const scope = preview && preview.scope;
  const excluded = scope ? Number(scope.competitor_rows_excluded) || 0 : 0;

  return (
    <aside className="rows-drawer" role="dialog" aria-label={text('reportRowsTitle', locale)}>
      <header className="rows-drawer-head">
        <div>
          <strong>{report.title}</strong>
          <span>{serverText(report.unit, locale)}</span>
        </div>
        <div className="rows-drawer-walk">
          <Button className="icon-button" type="button" onClick={onClose} aria-label={text('close', locale)}>
            <X size={16} />
          </Button>
        </div>
      </header>

      {state.loading ? <p className="rows-drawer-note">{text('loading', locale)}</p> : null}

      {!state.loading && preview && !preview.available
        ? (preview.notes || []).map((entry) => (
            <p className="rows-drawer-note" key={entry.code}>{serverText(entry, locale)}</p>
          ))
        : null}

      {!state.loading && preview && preview.available ? (
        <>
          <dl className="rows-drawer-counts">
            <div>
              <dt>{text('rowsTotal', locale)}</dt>
              <dd><Numeric>{formatNumber(preview.total_rows, locale)}</Numeric></dd>
            </div>
            {excluded > 0 ? (
              <div>
                <dt>{text('rowsOwned', locale)}</dt>
                <dd><Numeric>{formatNumber(preview.scoped_rows, locale)}</Numeric></dd>
              </div>
            ) : null}
            <div>
              <dt>{text('rowsShown', locale)}</dt>
              <dd><Numeric>{formatNumber(preview.shown_rows, locale)}</Numeric></dd>
            </div>
          </dl>
          {preview.source ? (
            <p className="rows-drawer-note">
              <span>{text('reportRowsSource', locale)}</span> <span dir="ltr">{preview.source}</span>
            </p>
          ) : null}
          {(preview.notes || []).map((entry) => (
            <p className="rows-drawer-note" key={entry.code}>{serverText(entry, locale)}</p>
          ))}
          <div className="rows-drawer-table" dir="ltr">
            <table>
              <thead>
                <tr>
                  {preview.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, index) => (
                  <tr key={index}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </aside>
  );
}

export default ReportRowsDrawer;
