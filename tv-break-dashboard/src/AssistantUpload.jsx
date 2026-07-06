import React, { useCallback, useRef, useState } from 'react';
import { FileSpreadsheet, Paperclip, Trash2, X } from 'lucide-react';
import { pageText } from './surface-helpers';
import { requestJson } from './assistant-stream';

// Agreement upload for the composer: a paperclip attaches an .xlsx/.xls/.csv
// agreement, which the server parses in memory and keeps only as a summary keyed
// to the account. On success a quiet chip names the file and a review prompt is
// suggested into the composer for the operator to send. A popover from the chip
// lists the account's own uploads with a delete. Every failure surfaces the
// backend detail (wrong type, too large) honestly; nothing is fabricated here.

const ACCEPT = '.xlsx,.xls,.csv';

const SUGGEST = ['Check the file I uploaded against the existing advertisers and propose an update per the agreement', 'בדוק את הקובץ שהעליתי מול המפרסמים הקיימים והצע עדכון לפי ההסכם'];

function uploadId(item) {
  return String((item && (item.upload_id || item.id)) || '');
}

function metaLabel(item, locale) {
  const rows = Number(item && item.total_rows);
  const sheets = Array.isArray(item && item.sheets)
    ? item.sheets.length
    : Array.isArray(item && item.sheet_names)
      ? item.sheet_names.length
      : Number(item && item.sheets);
  const parts = [];
  if (Number.isFinite(rows) && rows >= 0) parts.push(pageText(locale, `${rows} rows`, `${rows} שורות`));
  if (Number.isFinite(sheets) && sheets > 0) parts.push(pageText(locale, sheets === 1 ? '1 sheet' : `${sheets} sheets`, sheets === 1 ? 'גיליון אחד' : `${sheets} גיליונות`));
  return parts.join(' · ');
}

export default function AssistantUpload({ locale, notify, disabled, onSuggest }) {
  const inputRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [last, setLast] = useState(null);
  const [listOpen, setListOpen] = useState(false);
  const [list, setList] = useState({ state: 'idle', items: [], error: '' });

  const loadList = useCallback(async () => {
    setList((prev) => ({ ...prev, state: 'loading' }));
    try {
      const body = await requestJson('/api/assistant/uploads');
      const items = Array.isArray(body.uploads) ? body.uploads : Array.isArray(body.items) ? body.items : Array.isArray(body) ? body : [];
      setList({ state: 'ready', items, error: '' });
    } catch (err) {
      setList({ state: 'error', items: [], error: err.message });
    }
  }, []);

  const onFile = useCallback(async (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = '';
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const body = await requestJson('/api/assistant/upload', { method: 'POST', body: form });
      const filename = String(body.filename || file.name);
      setLast({ id: uploadId(body), filename });
      notify(`Uploaded ${filename}.`, `הקובץ ${filename} הועלה.`);
      onSuggest(pageText(locale, SUGGEST[0], SUGGEST[1]));
      if (listOpen) loadList();
    } catch (err) {
      notify(`The upload failed (${err.message}).`, `העלאת הקובץ נכשלה (${err.message}).`);
    } finally {
      setUploading(false);
    }
  }, [notify, onSuggest, locale, listOpen, loadList]);

  const toggleList = useCallback(() => {
    setListOpen((prev) => {
      const next = !prev;
      if (next) loadList();
      return next;
    });
  }, [loadList]);

  const removeUpload = useCallback(async (id) => {
    if (!id) return;
    try {
      await requestJson(`/api/assistant/uploads/${encodeURIComponent(id)}`, { method: 'DELETE' });
      setList((prev) => ({ ...prev, items: prev.items.filter((item) => uploadId(item) !== id) }));
      setLast((prev) => (prev && prev.id === id ? null : prev));
      notify('The file was removed.', 'הקובץ הוסר.');
    } catch (err) {
      notify(`Removing the file failed (${err.message}).`, `הסרת הקובץ נכשלה (${err.message}).`);
    }
  }, [notify]);

  return (
    <div className="asst-upload">
      <input ref={inputRef} type="file" accept={ACCEPT} className="asst-upload-input" onChange={onFile} tabIndex={-1} aria-hidden="true" />
      <button type="button" className="asst-upload-btn" onClick={() => { if (!disabled && !uploading && inputRef.current) inputRef.current.click(); }} disabled={disabled || uploading} aria-label={pageText(locale, 'Attach an agreement file', 'צירוף קובץ הסכם')} title={pageText(locale, 'Attach an agreement (Excel or CSV)', 'צירוף הסכם (אקסל או CSV)')}>
        <Paperclip size={16} />
      </button>
      <button type="button" className={`asst-upload-btn${listOpen ? ' active' : ''}`} onClick={toggleList} aria-expanded={listOpen} aria-label={pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')} title={pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')}>
        <FileSpreadsheet size={16} />
      </button>
      {uploading ? <span className="asst-upload-progress" role="status">{pageText(locale, 'Uploading', 'מעלה')}</span> : null}

      {last ? (
        <span className="asst-upload-chip">
          <button type="button" className="asst-upload-chip-open" onClick={toggleList} aria-expanded={listOpen}>
            <FileSpreadsheet size={13} />
            <span dir="auto">{last.filename}</span>
          </button>
          <button type="button" className="asst-upload-chip-x" onClick={() => setLast(null)} aria-label={pageText(locale, 'Dismiss', 'הסתרה')}>
            <X size={12} />
          </button>
        </span>
      ) : null}

      {listOpen ? (
        <div className="asst-upload-pop" role="dialog" aria-label={pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')}>
          <div className="asst-upload-pop-head">
            <span>{pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')}</span>
            <button type="button" onClick={() => setListOpen(false)} aria-label={pageText(locale, 'Close', 'סגירה')}><X size={13} /></button>
          </div>
          {list.state === 'loading' ? <div className="asst-loading">{pageText(locale, 'Loading files', 'טוען קבצים')}</div> : null}
          {list.state === 'error' ? <div className="asst-error-note">{pageText(locale, `Files could not be loaded (${list.error}).`, `לא ניתן לטעון את הקבצים (${list.error}).`)}</div> : null}
          {list.state === 'ready' && list.items.length === 0 ? <div className="asst-empty">{pageText(locale, 'No uploaded files. Attach an agreement to check it against the advertisers.', 'אין קבצים שהועלו. צרפו הסכם כדי לבדוק אותו מול המפרסמים.')}</div> : null}
          {list.state === 'ready' ? list.items.map((item, index) => {
            const id = uploadId(item);
            const meta = metaLabel(item, locale);
            return (
              <div className="asst-upload-item" key={id || `up-${index}`}>
                <div className="asst-upload-item-main">
                  <span className="asst-upload-name" dir="auto">{String((item && item.filename) || '')}</span>
                  {meta ? <span className="asst-upload-meta" dir="ltr">{meta}</span> : null}
                </div>
                <button type="button" className="asst-upload-del" onClick={() => removeUpload(id)} disabled={!id} aria-label={pageText(locale, 'Delete file', 'מחיקת קובץ')}><Trash2 size={13} /></button>
              </div>
            );
          }) : null}
        </div>
      ) : null}
    </div>
  );
}
