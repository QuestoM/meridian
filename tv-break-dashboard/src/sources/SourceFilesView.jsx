import React, { useState } from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { CheckCircle2, MinusCircle, Rows3 } from 'lucide-react';
import { Numeric, formatNumber, pageText } from '../shell/format';
import { normalizeRows } from '../shell/plan-model';
import { Code } from '../shell/bidi';
import { ROLE_LABELS, label, serverText, text } from './sources-copy';
import RowsDrawer from './RowsDrawer';
import { formatStamp } from '../shell/dates';

function formatSize(bytes, locale) {
  const number = Number(bytes) || 0;
  if (number <= 0) return '-';
  const kilobytes = number / 1024;
  // A 150-byte file is not a zero-byte file. Rounding it to 0 KB would say a
  // file with content is empty, which is the one thing a size column owes.
  if (kilobytes < 1) return `< 1 KB`;
  if (kilobytes < 1024) return `${formatNumber(Math.round(kilobytes), locale)} KB`;
  return `${formatNumber(Math.round(kilobytes / 1024), locale)} MB`;
}

function formatWhen(value) {
  if (!value) return '-';
  return formatStamp(value) || String(value);
}

// Every file the product reads or writes, with the two facts that are not the
// same fact: it is on disk, and something reads it. A file nothing reads
// carries the reason, and the reason names the file that is read instead.
export function SourceFilesView({ files, inputs, locale, highlight }) {
  const [rowsFor, setRowsFor] = useState(null);
  // Three lists, one table: the files the source audit counts, the files the
  // engine reads that the audit never named, and the files this product stored
  // for an input that nothing reads. Without the second one a card that says
  // "the engine reads this" would point at a list with no such row; without the
  // third one a file an operator just uploaded appears nowhere in the product.
  const rows = [
    ...normalizeRows(files && files.files),
    ...normalizeRows(files && files.also_read),
    ...normalizeRows(files && files.stored),
  ];
  const present = rows.filter((row) => row.exists).length;
  const read = rows.filter((row) => row.in_use).length;
  // A path whose rows this product can open is a path one of the inputs reads
  // its own rows from. The reference workbooks are named here because the
  // engine reads them, and this product has no reader for a workbook, so they
  // stay plain text: a link that opened a different file's rows under this
  // file's name would be worse than no link.
  const openable = new Map();
  normalizeRows(inputs).forEach((input) => {
    if (input.path && input.exists) openable.set(String(input.path), input);
  });
  return (
    <div className="sources-view">
      <section className="page-panel">
        <div className="panel-head">
          <div>
            <h2>{text('filesTitle', locale)}</h2>
            <small>{text('filesBody', locale)}</small>
          </div>
          <span>
            <Numeric>{formatNumber(read, locale)}</Numeric> {text('readOfPresent', locale)} <Numeric>{formatNumber(present, locale)}</Numeric> {text('present', locale)}
          </span>
        </div>
        <table className="source-files-table">
          <thead>
            <tr>
              <th>{text('filePath', locale)}</th>
              <th>{text('fileRole', locale)}</th>
              <th>{text('fileState', locale)}</th>
              <th className="numeric-head">{text('size', locale)}</th>
              <th className="numeric-head">{text('updated', locale)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.path} className={highlight && row.path === highlight ? 'highlighted' : undefined}>
                <td>
                  {openable.has(row.path) ? (
                    <Tooltip title={text('openRowsForFile', locale)} arrow placement="top">
                      <Button className="link-figure" type="button" onClick={() => setRowsFor(openable.get(row.path))}>
                        <Rows3 size={12} />
                        <Code className="source-file-path">{row.path}</Code>
                      </Button>
                    </Tooltip>
                  ) : (
                    <Code className="source-file-path">{row.path}</Code>
                  )}
                  {row.exists ? null : <span className="source-none">{text('filesMissing', locale)}</span>}
                  {serverText(row.note, locale) ? <span className="source-file-note">{serverText(row.note, locale)}</span> : null}
                </td>
                <td>{label(ROLE_LABELS, row.role || 'input', locale)}</td>
                <td>
                  <span className={row.in_use ? 'source-state ok' : 'source-state warn'}>
                    {row.in_use ? <CheckCircle2 size={12} /> : <MinusCircle size={12} />}
                    {row.in_use ? text('fileYes', locale) : text('fileNo', locale)}
                  </span>
                </td>
                <td className="numeric-cell"><Numeric>{formatSize(row.size, locale)}</Numeric></td>
                <td className="numeric-cell"><Numeric>{formatWhen(row.modified)}</Numeric></td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 ? (
          <p className="sources-note">{pageText(locale, 'No source files were found.', 'לא נמצאו קבצי מקור.')}</p>
        ) : null}
      </section>
      {rowsFor ? <RowsDrawer input={rowsFor} locale={locale} onClose={() => setRowsFor(null)} /> : null}
    </div>
  );
}

export default SourceFilesView;
