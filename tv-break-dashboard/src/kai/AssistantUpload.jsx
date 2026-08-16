import React, { useCallback, useRef, useState } from 'react';
import { Tooltip } from '@mui/material';
import { FileSpreadsheet, Paperclip, Trash2, X } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Name } from '../shell/bidi';
import { requestJson } from './assistant-stream';
import { isolate } from '../shell/bidi';
import ConsequenceDialog, { focusAfterDialogClose } from '../safety/ConsequenceDialog';
import { InputControl, Pressable } from '../studio/dom-controls';

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
  const listButtonRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [last, setLast] = useState(null);
  const [listOpen, setListOpen] = useState(false);
  const [list, setList] = useState({ state: 'idle', items: [], error: '' });
  const [deleteReview, setDeleteReview] = useState(null);
  const [deleting, setDeleting] = useState(false);

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
      notify(`Uploaded ${isolate(filename)}.`, `הקובץ ${isolate(filename)} הועלה.`);
      onSuggest(pageText(locale, SUGGEST[0], SUGGEST[1]));
      if (listOpen) loadList();
    } catch (err) {
      notify(`The upload failed (${isolate(err.message)}).`, `העלאת הקובץ נכשלה (${isolate(err.message)}).`);
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
    if (!id) return false;
    try {
      await requestJson(`/api/assistant/uploads/${encodeURIComponent(id)}`, { method: 'DELETE' });
      setList((prev) => ({ ...prev, items: prev.items.filter((item) => uploadId(item) !== id) }));
      setLast((prev) => (prev && prev.id === id ? null : prev));
      notify('The file was removed.', 'הקובץ הוסר.');
      return true;
    } catch (err) {
      notify(`Removing the file failed (${isolate(err.message)}).`, `הסרת הקובץ נכשלה (${isolate(err.message)}).`);
      return false;
    }
  }, [notify]);

  const confirmRemoveUpload = useCallback(async () => {
    const id = uploadId(deleteReview);
    if (!id) return;
    setDeleting(true);
    const removed = await removeUpload(id);
    setDeleting(false);
    if (removed) {
      setDeleteReview(null);
      focusAfterDialogClose(listButtonRef);
    }
  }, [deleteReview, removeUpload]);

  return (
    <div className="asst-upload">
      <InputControl ref={inputRef} type="file" accept={ACCEPT} className="asst-upload-input" onChange={onFile} tabIndex={-1} aria-hidden="true" />
      <Tooltip title={pageText(locale, 'Attach an agreement (Excel or CSV)', 'צירוף הסכם (אקסל או CSV)')} arrow placement="bottom">
        <span className="asst-upload-tipwrap">
          <Pressable type="button" className="asst-upload-btn" onClick={() => { if (!disabled && !uploading && inputRef.current) inputRef.current.click(); }} disabled={disabled || uploading} aria-label={pageText(locale, 'Attach an agreement file', 'צירוף קובץ הסכם')}>
            <Paperclip size={16} />
          </Pressable>
        </span>
      </Tooltip>
      <Tooltip title={pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')} arrow placement="bottom">
        <Pressable ref={listButtonRef} type="button" className={`asst-upload-btn${listOpen ? ' active' : ''}`} onClick={toggleList} aria-expanded={listOpen} aria-label={pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')}>
          <FileSpreadsheet size={16} />
        </Pressable>
      </Tooltip>
      {uploading ? <span className="asst-upload-progress" role="status">{pageText(locale, 'Uploading', 'מעלה')}</span> : null}

      {last ? (
        <span className="asst-upload-chip">
          <Pressable type="button" className="asst-upload-chip-open" onClick={toggleList} aria-expanded={listOpen}>
            <FileSpreadsheet size={13} />
            <Name>{last.filename}</Name>
          </Pressable>
          <Pressable type="button" className="asst-upload-chip-x" onClick={() => setLast(null)} aria-label={pageText(locale, 'Dismiss', 'הסתרה')}>
            <X size={12} />
          </Pressable>
        </span>
      ) : null}

      {listOpen ? (
        <div className="card asst-upload-pop" role="dialog" aria-label={pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')}>
          <div className="asst-upload-pop-head">
            <span>{pageText(locale, 'My uploaded files', 'הקבצים שהעליתי')}</span>
            <Pressable type="button" onClick={() => setListOpen(false)} aria-label={pageText(locale, 'Close', 'סגירה')}><X size={13} /></Pressable>
          </div>
          {list.state === 'loading' ? <div className="asst-loading">{pageText(locale, 'Loading files', 'טוען קבצים')}</div> : null}
          {list.state === 'error' ? <div className="asst-error-note">{pageText(locale, 'Files could not be loaded (', 'לא ניתן לטעון את הקבצים (')}<Name>{list.error}</Name>{').'}</div> : null}
          {list.state === 'ready' && list.items.length === 0 ? <div className="asst-empty">{pageText(locale, 'No uploaded files. Attach an agreement to check it against the advertisers.', 'אין קבצים שהועלו. צרפו הסכם כדי לבדוק אותו מול המפרסמים.')}</div> : null}
          {list.state === 'ready' ? list.items.map((item, index) => {
            const id = uploadId(item);
            const meta = metaLabel(item, locale);
            return (
              <div className="asst-upload-item" key={id || `up-${index}`}>
                <div className="asst-upload-item-main">
                  <span className="asst-upload-name"><Name>{String((item && item.filename) || '')}</Name></span>
                  {meta ? <span className="asst-upload-meta"><Figure>{meta}</Figure></span> : null}
                </div>
                <Pressable
                  type="button"
                  className="asst-upload-del"
                  onClick={() => setDeleteReview(item)}
                  disabled={!id}
                  aria-label={pageText(locale, `Review permanent deletion of ${isolate(String((item && item.filename) || 'file'))}`, `סקירת המחיקה הקבועה של ${isolate(String((item && item.filename) || 'הקובץ'))}`)}
                >
                  <Trash2 size={13} />
                </Pressable>
              </div>
            );
          }) : null}
        </div>
      ) : null}

      <ConsequenceDialog
        open={Boolean(deleteReview)}
        locale={locale}
        title={pageText(locale, 'Permanently delete this uploaded file summary?', 'למחוק לצמיתות את תקציר הקובץ שהועלה?')}
        description={pageText(locale, 'Mabat stores a bounded parsed summary, not the original spreadsheet. This deletion cannot be undone in the product.', 'מבט שומר תקציר מפוענח ומוגבל, ולא את גיליון המקור. לא ניתן לבטל את המחיקה הזו במוצר.')}
        object={deleteReview ? (
          <span className="consequence-review__object">
            <Name>{String(deleteReview.filename || '')}</Name>
            {metaLabel(deleteReview, locale) ? <> · <Figure>{metaLabel(deleteReview, locale)}</Figure></> : null}
            {' · ID '}<bdi>{uploadId(deleteReview)}</bdi>
          </span>
        ) : ''}
        scope={pageText(locale, 'Only the parsed upload summary stored for this account is deleted. Advertiser records, conversations and the original spreadsheet on your computer are not changed.', 'רק תקציר ההעלאה המפוענח ששמור לחשבון הזה יימחק. רשומות מפרסמים, שיחות וגיליון המקור במחשב שלכם אינם משתנים.')}
        consequence={pageText(locale, 'Mabat can no longer inspect or cite this upload. The stored summary is removed immediately and permanently.', 'מבט לא יוכל עוד לבדוק או לצטט את ההעלאה הזו. התקציר השמור יוסר מיד ולצמיתות.')}
        recovery={pageText(locale, 'There is no in-product restore. To use the agreement again, upload the original file again.', 'אין שחזור בתוך המוצר. כדי להשתמש שוב בהסכם, יש להעלות מחדש את קובץ המקור.')}
        confirmLabel={pageText(locale, 'Delete permanently', 'מחיקה לצמיתות')}
        workingLabel={pageText(locale, 'Deleting file summary', 'מוחק את תקציר הקובץ')}
        busy={deleting}
        onCancel={() => setDeleteReview(null)}
        onConfirm={confirmRemoveUpload}
      />
    </div>
  );
}
