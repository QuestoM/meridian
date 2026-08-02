import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { Check, Pencil, RotateCcw, X } from 'lucide-react';
import { pageText } from '../shell/format';
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

export default function HistoryRestore({ entry, locale, canEdit, canEditReason, notify, onChanged }) {
  const facts = entry.facts || {};
  const versionId = String(facts.version_id || '');
  const files = Array.isArray(facts.files) ? facts.files.filter(Boolean) : [];
  const [diff, setDiff] = useState({ state: 'loading', data: null, error: '' });
  const [selected, setSelected] = useState(() => new Set(files));
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(facts.label || '');

  useEffect(() => {
    let active = true;
    setDiff({ state: 'loading', data: null, error: '' });
    setSelected(new Set(files));
    setLabel(facts.label || '');
    setEditing(false);
    fetchVersionDiff(versionId).then((result) => {
      if (!active) return;
      if (result.ok) setDiff({ state: 'ready', data: result.data.diff || {}, error: '' });
      else setDiff({ state: 'error', data: null, error: result.error });
    });
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versionId]);

  const chosen = files.filter((file) => selected.has(file));

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
        <span dir="ltr">{stampLabel(entry.ts, locale)}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'How', 'איך')}</span>
        <span dir="auto">{pair(SOURCE_LABELS, facts.source, locale) || facts.source}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Name', 'שם')}</span>
        {editing ? (
          <span className="hist-rename">
            <input
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              dir="auto"
              maxLength={120}
              aria-label={pageText(locale, 'Restore point name', 'שם נקודת השחזור')}
              disabled={busy}
            />
            <button type="button" className="hist-icon-btn" onClick={saveLabel} disabled={busy} aria-label={pageText(locale, 'Save name', 'שמירת שם')}><Check size={13} /></button>
            <button type="button" className="hist-icon-btn" onClick={() => { setEditing(false); setLabel(facts.label || ''); }} disabled={busy} aria-label={pageText(locale, 'Cancel', 'ביטול')}><X size={13} /></button>
          </span>
        ) : (
          <span className="hist-rename">
            <span dir="auto">{facts.label || pageText(locale, 'Unnamed', 'ללא שם')}</span>
            {canEdit ? (
              <button type="button" className="hist-icon-btn" onClick={() => setEditing(true)} aria-label={pageText(locale, 'Rename restore point', 'שינוי שם נקודת השחזור')}><Pencil size={12} /></button>
            ) : null}
          </span>
        )}
      </div>

      <h4 className="hist-detail-h">{pageText(locale, 'What restoring this would change', 'מה ישתנה אם נחזור לכאן')}</h4>
      {diff.state === 'loading' ? <p className="hist-empty">{pageText(locale, 'Reading the difference', 'קורא את ההבדל')}</p> : null}
      {diff.state === 'error' ? <p className="hist-empty warn" dir="auto">{pageText(locale, `The difference could not be read. ${diff.error}`, `לא ניתן לקרוא את ההבדל. ${diff.error}`)}</p> : null}
      {diff.state === 'ready' ? <HistoryDiff diff={diff.data} locale={locale} /> : null}

      {facts.restorable === false ? (
        <p className="hist-block" role="note" dir="auto">{block}</p>
      ) : null}

      {facts.restorable !== false && !canEdit ? (
        <p className="hist-block" role="note" dir="auto">{canEditReason}</p>
      ) : null}

      {facts.restorable !== false && canEdit ? (
        <div className="hist-restore-act">
          <span className="hist-detail-key">{pageText(locale, 'Put back', 'להחזיר')}</span>
          <div className="hist-restore-files">
            {files.map((file) => (
              <label key={file} className="hist-file-opt">
                <input
                  type="checkbox"
                  checked={selected.has(file)}
                  disabled={busy}
                  onChange={() => setSelected((current) => {
                    const next = new Set(current);
                    if (next.has(file)) next.delete(file);
                    else next.add(file);
                    return next;
                  })}
                />
                {pair(FILE_LABELS, file, locale) || file}
              </label>
            ))}
          </div>
          <p className="hist-note" dir="auto">{pageText(locale, 'The current state is saved as a restore point first, so this restore can itself be undone.', 'המצב הנוכחי נשמר קודם כנקודת שחזור, כך שאפשר לבטל גם את השחזור הזה.')}</p>
          <Button variant="contained" size="small" startIcon={<RotateCcw size={13} />} disabled={busy || !chosen.length} onClick={applyRestore}>
            {busy ? pageText(locale, 'Putting back', 'מחזיר') : pageText(locale, 'Put back', 'החזרה')}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
