import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { RotateCcw, TriangleAlert } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { postJson, requestJson } from './assistant-stream';

// Undo, as an object you can open and read before you use it.
//
// Applying a change writes a restore point first. This is the control that
// makes that restore point addressable: it names it, opens it, and shows the
// exact fields and rows that going back would change, computed on the server by
// comparing the snapshot with the files as they stand right now. Nothing is
// estimated. A file whose snapshot is missing says so, a file that has not
// moved says so, and a restore point where nothing would change offers no
// button at all, because pressing it would be a change of unknown size for no
// reason.

function shortValue(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'object') {
    const text = JSON.stringify(value);
    return text.length > 60 ? `${text.slice(0, 57)}…` : text;
  }
  const text = String(value);
  return text.length > 60 ? `${text.slice(0, 57)}…` : text;
}

// Why a file cannot be read, or why nothing would change, said in the reader's
// language. The server sends the English sentence as the API record and a code
// beside it; this maps the code, because printing that sentence here put
// English prose under a Hebrew heading. An unknown code keeps the honest
// server text, isolated as technical detail rather than dressed as copy.
const REASON_TEXT = {
  snapshot_missing: ['The snapshot of this file is missing from the restore point.', 'השמירה של הקובץ הזה חסרה בנקודת השחזור.'],
  snapshot_unreadable: ['The snapshot of this file could not be read.', 'לא ניתן לקרוא את השמירה של הקובץ הזה.'],
  current_unreadable: ['The file as it stands now could not be read.', 'לא ניתן לקרוא את הקובץ במצבו הנוכחי.'],
  absent_at_snapshot: ['This file did not exist before the change, so undoing removes it again.', 'הקובץ הזה לא היה קיים לפני השינוי, ולכן הביטול מסיר אותו שוב.'],
  nothing_would_change: ['Nothing would change: every file already matches the restore point.', 'שום דבר לא ישתנה: כל הקבצים כבר תואמים לנקודת השחזור.'],
};

export function reasonText(code, locale) {
  const pair = REASON_TEXT[String(code || '')];
  return pair ? pageText(locale, pair[0], pair[1]) : '';
}

function ReasonLine({ code, detail, fallback, locale }) {
  const said = reasonText(code, locale);
  if (!said && !fallback) return null;
  return (
    <p className="asst-undo-note" dir="auto">
      <TriangleAlert size={12} />
      {said || pageText(locale, 'This file cannot be read.', 'לא ניתן לקרוא את הקובץ הזה.')}
      {detail ? <> <bdi dir="ltr">{String(detail)}</bdi></> : null}
      {!said && fallback ? <> <bdi dir="auto">{String(fallback)}</bdi></> : null}
    </p>
  );
}

function FieldRows({ rows, locale }) {
  return (
    <div className="asst-undo-grid">
      <div className="asst-undo-row head">
        <span dir="auto">{pageText(locale, 'Field', 'שדה')}</span>
        <span dir="auto">{pageText(locale, 'Now', 'עכשיו')}</span>
        <span dir="auto">{pageText(locale, 'After undo', 'אחרי הביטול')}</span>
      </div>
      {rows.map((row, index) => (
        <div className="asst-undo-row" key={`${row.field}-${index}`}>
          <span className="asst-undo-field" dir="ltr">{String(row.field ?? '')}</span>
          <span className="asst-undo-now" dir="ltr">{shortValue(row.current)}</span>
          <span className="asst-undo-after" dir="ltr">{shortValue(row.restored)}</span>
        </div>
      ))}
    </div>
  );
}

function FilePreview({ file, locale }) {
  const effect = String(file.effect || '');
  if (effect === 'unavailable') {
    return (
      <div className="asst-undo-file">
        <div className="asst-undo-file-head"><code dir="ltr">{String(file.file || '')}</code></div>
        <ReasonLine code={file.reason_code} detail={file.reason_detail} fallback={file.reason} locale={locale} />
      </div>
    );
  }
  if (effect === 'unchanged') {
    return (
      <div className="asst-undo-file">
        <div className="asst-undo-file-head"><code dir="ltr">{String(file.file || '')}</code></div>
        <p className="asst-undo-note" dir="auto">{pageText(locale, 'Unchanged since the restore point was taken.', 'לא השתנה מאז שנוצרה נקודת השחזור.')}</p>
      </div>
    );
  }
  const rows = Array.isArray(file.changes) ? file.changes : [];
  return (
    <div className="asst-undo-file">
      <div className="asst-undo-file-head">
        <code dir="ltr">{String(file.file || '')}</code>
        {Number.isFinite(file.change_count) ? (
          <span className="asst-undo-count">{file.change_count === 1 ? pageText(locale, 'one change', 'שינוי אחד') : pageText(locale, `${file.change_count} changes`, `${file.change_count} שינויים`)}</span>
        ) : null}
      </div>
      {file.kind === 'fields' && rows.length ? <FieldRows rows={rows} locale={locale} /> : null}
      {file.kind === 'rows' && rows.length ? (
        <div className="asst-undo-rows">
          {rows.map((row, index) => (
            <div className="asst-undo-rowline" key={`${row.row}-${index}`}>
              <span className="asst-undo-rowkey" dir="auto">{String(row.row ?? '')}</span>
              <span className={`asst-undo-state ${String(row.state || '')}`}>
                {row.state === 'added' ? pageText(locale, 'would be added back', 'יוחזר')
                  : row.state === 'removed' ? pageText(locale, 'would be removed', 'יוסר')
                  : pageText(locale, 'would change', 'ישתנה')}
              </span>
              {Array.isArray(row.fields) && row.fields.length ? <FieldRows rows={row.fields} locale={locale} /> : null}
            </div>
          ))}
        </div>
      ) : null}
      {file.note_code || file.kind === 'absent_at_snapshot' ? (
        <p className="asst-undo-note" dir="auto">{reasonText(file.note_code || 'absent_at_snapshot', locale)}</p>
      ) : null}
      {(file.kind === 'text' || file.kind === 'bytes') ? (
        <p className="asst-undo-note" dir="auto">{pageText(locale, `Size now ${file.bytes_now} bytes, after undo ${file.bytes_after_restore} bytes. A field-level view is not available for this file.`, `הגודל עכשיו ${file.bytes_now} בתים, ואחרי הביטול ${file.bytes_after_restore} בתים. אין תצוגה ברמת שדה לקובץ הזה.`)}</p>
      ) : null}
      {Number.isFinite(file.changes_omitted) && file.changes_omitted > 0 ? (
        <p className="asst-undo-note" dir="auto">{pageText(locale, `And ${file.changes_omitted} more changes not shown.`, `ועוד ${file.changes_omitted} שינויים שאינם מוצגים.`)}</p>
      ) : null}
    </div>
  );
}

export default function AssistantUndo({ locale, restoreId, notify, onDone }) {
  const [state, setState] = useState('idle');
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');
  const [confirming, setConfirming] = useState(false);
  const [done, setDone] = useState(null);

  const load = useCallback(() => {
    setState('loading');
    setError('');
    requestJson(`/api/assistant/restore/${encodeURIComponent(restoreId)}`)
      .then((body) => { setPreview(body); setState('ready'); })
      .catch((err) => { setError(err && err.message ? err.message : 'unknown'); setState('error'); });
  }, [restoreId]);

  useEffect(() => { if (state === 'open') load(); }, [state, load]);

  async function runUndo() {
    setConfirming(false);
    setState('restoring');
    try {
      const body = await postJson(`/api/assistant/restore/${encodeURIComponent(restoreId)}`, {});
      const restored = Array.isArray(body.restored) ? body.restored.length : 0;
      setDone({ restored, removed: Array.isArray(body.removed) ? body.removed.length : 0 });
      setState('done');
      if (notify) notify('The change was undone from its restore point.', 'השינוי בוטל מנקודת השחזור שלו.');
      if (onDone) onDone();
    } catch (err) {
      setError(err && err.message ? err.message : 'unknown');
      setState('error');
      if (notify) notify(`The undo failed (${err.message}).`, `הביטול נכשל (${err.message}).`);
    }
  }

  if (!restoreId) return null;

  if (state === 'idle') {
    return (
      <button type="button" className="asst-undo-open" onClick={() => setState('open')}>
        <RotateCcw size={12} />
        {pageText(locale, 'Undo this change', 'ביטול השינוי')}
        <code dir="ltr">{String(restoreId).slice(0, 8)}</code>
      </button>
    );
  }

  if (state === 'done') {
    return (
      <p className="asst-undo-done" dir="auto">
        {done && done.restored === 1
          ? pageText(locale, 'One file was put back. Run the plan so it reflects the restored data.', 'קובץ אחד הוחזר. הריצו את התוכנית כדי שתשקף את הנתונים המשוחזרים.')
          : pageText(locale, `${done ? done.restored : 0} files were put back. Run the plan so it reflects the restored data.`, `${done ? done.restored : 0} קבצים הוחזרו. הריצו את התוכנית כדי שתשקף את הנתונים המשוחזרים.`)}
      </p>
    );
  }

  const files = preview && Array.isArray(preview.files) ? preview.files : [];
  const restorable = Boolean(preview && preview.restorable);

  return (
    <div className="asst-undo">
      <div className="asst-undo-head">
        <span dir="auto">{pageText(locale, 'What undoing would change', 'מה הביטול ישנה')}</span>
        <code dir="ltr">{String(restoreId).slice(0, 8)}</code>
        <button type="button" className="asst-undo-close" onClick={() => setState('idle')}>
          {pageText(locale, 'Close', 'סגירה')}
        </button>
      </div>
      {state === 'loading' ? <div className="asst-loading">{pageText(locale, 'Reading the restore point', 'קורא את נקודת השחזור')}</div> : null}
      {state === 'restoring' ? <div className="asst-loading">{pageText(locale, 'Putting the previous state back', 'מחזיר את המצב הקודם')}</div> : null}
      {state === 'error' ? <div className="asst-error-note" dir="auto">{pageText(locale, `The restore point could not be read (${error}).`, `לא ניתן לקרוא את נקודת השחזור (${error}).`)}</div> : null}
      {preview && state !== 'loading' ? (
        <>
          {files.map((file, index) => <FilePreview key={`${file.file}-${index}`} file={file} locale={locale} />)}
          {!restorable && preview.reason ? (
            <p className="asst-undo-note" dir="auto">
              {reasonText(preview.reason_code, locale) || <bdi dir="auto">{String(preview.reason)}</bdi>}
            </p>
          ) : null}
          {preview.files_unavailable > 0 ? (
            <p className="asst-undo-note" dir="auto">{pageText(locale, 'Part of this restore point cannot be read, so undoing is not offered.', 'חלק מנקודת השחזור אינו קריא, ולכן הביטול אינו מוצע.')}</p>
          ) : null}
          {restorable && !confirming && state === 'ready' ? (
            <div className="asst-undo-actions">
              <Button variant="outlined" size="small" startIcon={<RotateCcw size={13} />} onClick={() => setConfirming(true)}>
                {pageText(locale, 'Undo now', 'בטלו עכשיו')}
              </Button>
            </div>
          ) : null}
          {confirming ? (
            <div className="asst-confirm" role="alertdialog">
              <p dir="auto">{pageText(locale, 'The files above go back to the state shown in the After undo column.', 'הקבצים שלמעלה יחזרו למצב שמוצג בעמודת אחרי הביטול.')}</p>
              <p dir="auto">{pageText(locale, 'A whole-file undo also reverts manual edits made to the same file since.', 'ביטול ברמת קובץ מבטל גם עריכות ידניות שנעשו באותו קובץ מאז.')}</p>
              <p dir="auto">{pageText(locale, 'A plan run cannot be un-run: the inputs go back, then run the plan again.', 'הרצת תוכנית אינה ניתנת לביטול: הנתונים חוזרים, ואז מריצים את התוכנית שוב.')}</p>
              <div className="asst-confirm-actions">
                <Button variant="contained" size="small" onClick={runUndo}>{pageText(locale, 'Undo now', 'בטלו עכשיו')}</Button>
                <Button variant="outlined" size="small" onClick={() => setConfirming(false)}>{pageText(locale, 'Cancel', 'ביטול')}</Button>
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
