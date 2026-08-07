import React, { useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';
import { Numeric, formatNumber } from '../shell/format';
import { Code } from '../shell/bidi';
import { serverText, text } from './sources-copy';
import { fetchPreview } from './sources-api';

// The rows behind a row count. A count that cannot be opened is a dead end, so
// every count on a source card opens this and shows the file's own header and
// its first rows, with the counts that make the sample honest: how many rows
// are in the file, how many belong to this operator's channel, and how many are
// on screen.
//
// Opening one record keeps its place in the set it came from, which is Linear's
// `1 / 31` with its two arrows: a person who opens the third of seven inputs
// can walk the other six without going back for them.
export function RowsDrawer({ input, position, total, locale, onStep, onClose }) {
  const [state, setState] = useState({ loading: true, online: true, preview: null });

  useEffect(() => {
    let active = true;
    setState({ loading: true, online: true, preview: null });
    fetchPreview(input.kind, 20).then((result) => {
      if (!active) return;
      setState({ loading: false, online: result.online, preview: result.preview });
    });
    return () => {
      active = false;
    };
  }, [input.kind]);

  // Back and forward are logical directions, so the arrow that means "the one
  // before this" points right in Hebrew and left in English.
  const Back = locale === 'he' ? ChevronRight : ChevronLeft;
  const Forward = locale === 'he' ? ChevronLeft : ChevronRight;
  const preview = state.preview;
  const scope = preview && preview.scope;
  const excluded = scope ? Number(scope.competitor_rows_excluded) || 0 : 0;
  const hidden = preview ? Number(preview.columns_hidden) || 0 : 0;
  const name = locale === 'he' ? input.label_he || input.label_en : input.label_en;
  // The counter and its arrows appear only where there is a set to walk.
  const walkable = Number(total) > 1 && typeof onStep === 'function';

  return (
    <aside className="rows-drawer" role="dialog" aria-label={text('rowsTitle', locale)}>
      <header className="rows-drawer-head">
        <div>
          <strong>{name}</strong>
          <Code>{input.path}</Code>
        </div>
        <div className="rows-drawer-walk">
          {walkable ? (
            <>
              <Button
                className="icon-button"
                type="button"
                onClick={() => onStep(-1)}
                disabled={position <= 1}
                aria-label={text('previous', locale)}
              >
                <Back size={14} />
              </Button>
              <span className="rows-drawer-position">
                <Numeric>{formatNumber(position, locale)}</Numeric> {text('position', locale)} <Numeric>{formatNumber(total, locale)}</Numeric>
              </span>
              <Button
                className="icon-button"
                type="button"
                onClick={() => onStep(1)}
                disabled={position >= total}
                aria-label={text('next', locale)}
              >
                <Forward size={14} />
              </Button>
            </>
          ) : null}
          <Button className="icon-button" type="button" onClick={onClose} aria-label={text('close', locale)}>
            <X size={16} />
          </Button>
        </div>
      </header>

      {state.loading ? <p className="rows-drawer-note">{text('loading', locale)}</p> : null}
      {!state.loading && !state.online ? <p className="rows-drawer-note">{text('offline', locale)}</p> : null}
      {!state.loading && state.online && preview && !preview.available
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
            <div>
              <dt>{text('rowsOwned', locale)}</dt>
              <dd><Numeric>{formatNumber(preview.scoped_rows ?? preview.total_rows, locale)}</Numeric></dd>
            </div>
            <div>
              <dt>{text('rowsShown', locale)}</dt>
              <dd><Numeric>{formatNumber(preview.shown_rows, locale)}</Numeric></dd>
            </div>
          </dl>
          {excluded > 0 ? <p className="rows-drawer-note">{text('rowsExcluded', locale)}</p> : null}
          {hidden > 0 ? <p className="rows-drawer-note">{text('columnsHidden', locale)}</p> : null}
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

export default RowsDrawer;
