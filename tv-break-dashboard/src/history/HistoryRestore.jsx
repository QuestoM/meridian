import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { Check, Lock, Pencil, RotateCcw, X } from 'lucide-react';
import { pageText } from '../shell/format';
import { Code, Figure } from '../shell/bidi';
import { InputControl, Pressable } from '../studio/dom-controls';
import { Dialog } from '../studio/modal';
import HistoryDiff from './HistoryDiff';
import { fetchVersionDiff, renameVersion, restoreVersion } from './history-api';
import { FILE_LABELS, RESTORE_BLOCKS, SOURCE_LABELS, pair, stampLabel } from './history-labels';

// A restore point, opened. The diff loads first and the restore control sits
// under it, so the preview is what a person acts on rather than a date. A
// point that cannot be put back into this deployment says why and offers no
// control, because a disabled button with no reason is a dead end.
//
// Reading and applying are separately permissioned. canEdit comes from the
// endpoint's own can_edit, so the refusal a person reads before the click is
// the string the server would send after it.
//
// That answer is per file as well as per surface, and the diff carries it. A
// restore writes whole files, and two of the nine are company-only to write by
// the front door: the settings document holds the audience model switch, the
// four regulatory limits and the channel declaration, and the calendar refuses
// a channel account on all three of its own write routes. So the control is
// held until the diff lands rather than offered over permissions it has not
// read, a file this account may not put back is shown with the exact refusal
// the server would send, and the rest of the point stays restorable.

export default function HistoryRestore({ entry, locale, canEdit, canEditReason, notify, onChanged }) {
  const facts = entry.facts || {};
  const versionId = String(facts.version_id || '');
  const files = Array.isArray(facts.files) ? facts.files.filter(Boolean) : [];
  const [diff, setDiff] = useState({ state: 'loading', data: null, files: {}, error: '' });
  const [selected, setSelected] = useState(() => new Set(files));
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(facts.label || '');
  const [reviewOpen, setReviewOpen] = useState(false);
  const cancelRestoreRef = useRef(null);

  useEffect(() => {
    let active = true;
    setDiff({ state: 'loading', data: null, files: {}, error: '' });
    setSelected(new Set(files));
    setLabel(facts.label || '');
    setEditing(false);
    setReviewOpen(false);
    fetchVersionDiff(versionId).then((result) => {
      if (!active) return;
      if (result.ok) setDiff({ state: 'ready', data: result.data.diff || {}, files: result.data.file_permissions || {}, error: '' });
      else setDiff({ state: 'error', data: null, files: {}, error: result.error });
    });
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versionId]);

  const permitted = (file) => ((diff.files || {})[file] || {}).can_edit !== false;
  const refusal = (file) => String((((diff.files || {})[file]) || {}).can_edit_reason || '');
  const chosen = files.filter((file) => selected.has(file) && permitted(file));
  const withheld = files.filter((file) => !permitted(file));

  const applyRestore = useCallback(async () => {
    setBusy(true);
    const result = await restoreVersion(versionId, chosen);
    setBusy(false);
    if (!result.ok) {
      notify(`The restore was refused. ${result.error}`, `השחזור נדחה. ${result.error}`);
      return;
    }
    const count = Array.isArray(result.data.restored) ? result.data.restored.length : 0;
    notify(
      `Put back ${count} file(s). A restore point was saved first, so this can be undone.`,
      `הוחזרו ${count} קבצים. נשמרה קודם נקודת שחזור, כך שאפשר לבטל.`,
    );
    if (onChanged) onChanged();
  }, [versionId, chosen, notify, onChanged]);

  const closeRestoreReview = useCallback(() => {
    if (!busy) setReviewOpen(false);
  }, [busy]);

  const confirmRestore = useCallback(() => {
    setReviewOpen(false);
    void applyRestore();
  }, [applyRestore]);

  const saveLabel = useCallback(async () => {
    setBusy(true);
    const result = await renameVersion(versionId, label.trim());
    setBusy(false);
    if (!result.ok) {
      notify(`The name could not be saved. ${result.error}`, `לא ניתן היה לשמור את השם. ${result.error}`);
      return;
    }
    setEditing(false);
    notify('The restore point name was saved.', 'שם נקודת השחזור נשמר.');
    if (onChanged) onChanged();
  }, [versionId, label, notify, onChanged]);

  const block = facts.restore_block ? pair(RESTORE_BLOCKS, facts.restore_block, locale) : '';

  return (
    <div className="hist-restore">
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Saved', 'נשמרה')}</span>
        <Figure>{stampLabel(entry.ts, locale)}</Figure>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'How', 'איך')}</span>
        <span>{pair(SOURCE_LABELS, facts.source, locale) || facts.source}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Name', 'שם')}</span>
        {editing ? (
          <span className="hist-rename">
            <InputControl
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              maxLength={120}
              aria-label={pageText(locale, 'Restore point name', 'שם נקודת השחזור')}
              disabled={busy}
            />
            <Pressable type="button" className="hist-icon-btn" onClick={saveLabel} disabled={busy} aria-label={pageText(locale, 'Save name', 'שמירת שם')}><Check size={13} /></Pressable>
            <Pressable type="button" className="hist-icon-btn" onClick={() => { setEditing(false); setLabel(facts.label || ''); }} disabled={busy} aria-label={pageText(locale, 'Cancel', 'ביטול')}><X size={13} /></Pressable>
          </span>
        ) : (
          <span className="hist-rename">
            <span>{facts.label || pageText(locale, 'Unnamed', 'ללא שם')}</span>
            {canEdit ? (
              <Pressable type="button" className="hist-icon-btn" onClick={() => setEditing(true)} aria-label={pageText(locale, 'Rename restore point', 'שינוי שם נקודת השחזור')}><Pencil size={12} /></Pressable>
            ) : null}
          </span>
        )}
      </div>

      <h4 className="hist-detail-h">{pageText(locale, 'What restoring this would change', 'מה ישתנה אם נחזור לכאן')}</h4>
      {diff.state === 'loading' ? <p className="hist-empty">{pageText(locale, 'Reading the difference', 'קורא את ההבדל')}</p> : null}
      {diff.state === 'error' ? <p className="hist-empty warn">{pageText(locale, `The difference could not be read. ${diff.error}`, `לא ניתן לקרוא את ההבדל. ${diff.error}`)}</p> : null}
      {diff.state === 'ready' ? <HistoryDiff diff={diff.data} locale={locale} /> : null}

      {facts.restorable === false ? (
        <p className="hist-block" role="note">{block}</p>
      ) : null}

      {facts.restorable !== false && !canEdit ? (
        <p className="hist-block" role="note">{canEditReason}</p>
      ) : null}

      {facts.restorable !== false && canEdit && diff.state === 'ready' ? (
        <div className="hist-restore-act">
          <span className="hist-detail-key">{pageText(locale, 'Put back', 'להחזיר')}</span>
          <div className="hist-restore-files">
            {files.map((file) => (
              <label key={file} className={permitted(file) ? 'hist-file-opt' : 'hist-file-opt blocked'}>
                <InputControl
                  type="checkbox"
                  checked={selected.has(file) && permitted(file)}
                  disabled={busy || !permitted(file)}
                  onChange={() => setSelected((current) => {
                    const next = new Set(current);
                    if (next.has(file)) next.delete(file);
                    else next.add(file);
                    return next;
                  })}
                />
                {pair(FILE_LABELS, file, locale) || file}
                {permitted(file) ? null : (
                  <span className="hist-file-why"><Lock size={11} />{refusal(file)}</span>
                )}
              </label>
            ))}
          </div>
          {withheld.length ? (
            <p className="hist-note">{pageText(locale, `${withheld.length} of these files are not this account's to put back, so they stay exactly as they are and the rest still come back.`, `${withheld.length} מהקבצים האלה אינם של החשבון הזה להחזרה, ולכן הם יישארו בדיוק כפי שהם והשאר יוחזרו.`)}</p>
          ) : null}
          <p className="hist-note">{pageText(locale, 'The current state is saved as a restore point first, so this restore can itself be undone.', 'המצב הנוכחי נשמר קודם כנקודת שחזור, כך שאפשר לבטל גם את השחזור הזה.')}</p>
          <Button variant="contained" size="small" startIcon={<RotateCcw size={13} />} disabled={busy || !chosen.length} onClick={() => setReviewOpen(true)}>
            {busy ? pageText(locale, 'Putting back', 'מחזיר') : pageText(locale, 'Put back', 'החזרה')}
          </Button>
        </div>
      ) : null}

      <Dialog
        open={reviewOpen}
        onClose={closeRestoreReview}
        title={pageText(locale, 'Review the restore', 'בדיקת השחזור')}
        description={pageText(locale, 'Confirm the exact restore point, selected domains and consequence before anything is put back.', 'אשרו את נקודת השחזור המדויקת, התחומים שנבחרו והתוצאה לפני שמשהו מוחזר.')}
        closeLabel={pageText(locale, 'Cancel and close the restore review', 'ביטול וסגירת בדיקת השחזור')}
        initialFocusRef={cancelRestoreRef}
        dismissOnBackdrop={false}
        footer={(
          <>
            <Button ref={cancelRestoreRef} type="button" variant="outlined" disabled={busy} onClick={closeRestoreReview}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
            <Button type="button" variant="contained" color="error" disabled={busy || !chosen.length} onClick={confirmRestore}>
              <RotateCcw size={13} aria-hidden="true" />
              {pageText(locale, 'Restore selected domains', 'שחזור התחומים שנבחרו')}
            </Button>
          </>
        )}
      >
        <div className="hist-kv">
          <div className="hist-detail-line">
            <span className="hist-detail-key">{pageText(locale, 'Restore point', 'נקודת שחזור')}</span>
            <span>{facts.label || pageText(locale, 'Unnamed', 'ללא שם')} {' '}<Code>{versionId}</Code></span>
          </div>
          <div className="hist-detail-line">
            <span className="hist-detail-key">{pageText(locale, 'Selected domains', 'תחומים שנבחרו')}</span>
            <span>{chosen.map((file) => pair(FILE_LABELS, file, locale) || file).join(', ')}</span>
          </div>
          <div className="hist-detail-line">
            <span className="hist-detail-key">{pageText(locale, 'Consequence', 'תוצאה')}</span>
            <span>{pageText(locale, 'Each selected domain is replaced with the copy saved at this point. The current state is saved as a new restore point first, so this restore can itself be undone.', 'כל תחום שנבחר מוחלף בעותק שנשמר בנקודה הזו. המצב הנוכחי נשמר קודם כנקודת שחזור חדשה, כך שאפשר לבטל גם את השחזור הזה.')}</span>
          </div>
        </div>
      </Dialog>
    </div>
  );
}
