import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { Camera, Check, ChevronDown, ChevronUp, History, Pencil, RotateCcw, X } from 'lucide-react';
import { pageText } from './surface-helpers';
import { requestJson, postJson } from './assistant-stream';
import { fetchMe } from './Login';

// The versions timeline: a Google-Sheets-style append-only history of the four
// operation-state files (settings with pricing overrides, placement constraints,
// manual overrides, advertiser rules). Every change and restore records a version;
// a restore first saves a safety version of the current state, so it is always
// undoable. Rows expand to a diff of the current state versus that version (what
// restoring would change), with an inline restore that picks which files to put
// back. Writer roles can rename a version and create a named point; viewers see
// the timeline and diffs with the mutation affordances hidden. Every surface has
// honest loading, error and empty states, and nothing is fabricated on the client.

const FILE_ORDER = ['settings', 'constraints', 'overrides', 'advertisers'];

const FILE_LABELS = {
  settings: ['Settings', 'הגדרות'],
  constraints: ['Constraints', 'אילוצים'],
  overrides: ['Overrides', 'עקיפות'],
  advertisers: ['Advertisers', 'מפרסמים'],
};

const SOURCE_LABELS = {
  manual_edit: ['Manual edit', 'עריכה ידנית'],
  assistant_apply: ['Via the AI assistant', 'דרך עוזר ה-AI'],
  manual_snapshot: ['Manual point', 'נקודה ידנית'],
  pre_restore: ['Before restore', 'לפני שחזור'],
};

function fileLabel(file, locale) {
  return FILE_LABELS[file] ? pageText(locale, FILE_LABELS[file][0], FILE_LABELS[file][1]) : file;
}

function dateTimeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function shortValue(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'object') {
    const text = JSON.stringify(value);
    return text.length > 80 ? `${text.slice(0, 77)}…` : text;
  }
  return String(value);
}

function chipText(item) {
  if (item === null || item === undefined) return '-';
  if (typeof item === 'object') return shortValue(item.id ?? item.name ?? item);
  return String(item);
}

function TriRow({ head, field, cur, ver }) {
  return (
    <div className={`asst-diff-row${head ? ' head' : ''}`}>
      <span className="asst-diff-field" dir={head ? 'auto' : 'ltr'}>{field}</span>
      <span className="asst-diff-cur" dir={head ? 'auto' : 'ltr'}>{cur}</span>
      <span className="asst-diff-ver" dir={head ? 'auto' : 'ltr'}>{ver}</span>
    </div>
  );
}

function ChipList({ title, items }) {
  const list = Array.isArray(items) ? items : [];
  if (!list.length) return null;
  return (
    <div className="asst-diff-sub">
      <span className="asst-diff-sub-h">{title}</span>
      <div className="asst-diff-chips">{list.map((item, index) => <code dir="ltr" key={index}>{chipText(item)}</code>)}</div>
    </div>
  );
}

function fileHasChanges(file, data) {
  const detail = data && data[file];
  if (!detail || typeof detail !== 'object') return false;
  const changed = Array.isArray(detail.changed) ? detail.changed.length : 0;
  const added = Array.isArray(detail.added) ? detail.added.length : 0;
  const removed = Array.isArray(detail.removed) ? detail.removed.length : 0;
  return changed + added + removed > 0;
}

function FileDiff({ file, detail, locale }) {
  const headRow = <TriRow head field={pageText(locale, 'Field', 'שדה')} cur={pageText(locale, 'Current value', 'ערך נוכחי')} ver={pageText(locale, 'In version', 'בגרסה')} />;
  const changed = Array.isArray(detail.changed) ? detail.changed : [];
  let body;
  if (file === 'settings') {
    body = (
      <div className="asst-diff-grid">
        {headRow}
        {changed.map((row, index) => <TriRow key={index} field={String(row.field ?? '')} cur={shortValue(row.from)} ver={shortValue(row.to)} />)}
      </div>
    );
  } else if (file === 'advertisers') {
    const groups = [];
    const index = new Map();
    for (const row of changed) {
      const name = String((row && row.advertiser) || '');
      if (!index.has(name)) { index.set(name, groups.length); groups.push({ name, rows: [] }); }
      groups[index.get(name)].rows.push(row);
    }
    body = (
      <div className="asst-diff-store">
        <ChipList title={pageText(locale, 'Added', 'נוספו')} items={detail.added} />
        <ChipList title={pageText(locale, 'Removed', 'הוסרו')} items={detail.removed} />
        {groups.map((group) => (
          <div className="asst-diff-adv" key={group.name || 'row'}>
            <span className="asst-diff-adv-h" dir="auto">{group.name}</span>
            <div className="asst-diff-grid">
              {headRow}
              {group.rows.map((row, idx) => <TriRow key={idx} field={String(row.field ?? '')} cur={shortValue(row.from)} ver={shortValue(row.to)} />)}
            </div>
          </div>
        ))}
      </div>
    );
  } else {
    body = (
      <div className="asst-diff-store">
        <ChipList title={pageText(locale, 'Added', 'נוספו')} items={detail.added} />
        <ChipList title={pageText(locale, 'Removed', 'הוסרו')} items={detail.removed} />
        {changed.length ? (
          <div className="asst-diff-grid">
            {headRow}
            {changed.map((row, index) => <TriRow key={index} field={`${shortValue(row.id)} / ${String(row.field ?? '')}`} cur={shortValue(row.from)} ver={shortValue(row.to)} />)}
          </div>
        ) : null}
      </div>
    );
  }
  return (
    <section className="asst-diff-file">
      <h5 className="asst-diff-file-h">{fileLabel(file, locale)}</h5>
      {body}
    </section>
  );
}

function DiffView({ data, locale }) {
  const files = FILE_ORDER.filter((file) => data && data[file] && fileHasChanges(file, data));
  if (!files.length) {
    return <p className="asst-ver-empty" dir="auto">{pageText(locale, 'No differences from the current state.', 'אין הבדלים מול המצב הנוכחי.')}</p>;
  }
  return <div className="asst-ver-diff">{files.map((file) => <FileDiff key={file} file={file} detail={data[file]} locale={locale} />)}</div>;
}

function VersionRow({ entry, locale, canWrite, restoringId, onRename, onRestore, notify }) {
  const id = entry && entry.version_id != null ? String(entry.version_id) : '';
  const files = Array.isArray(entry.files) ? entry.files.filter(Boolean).map(String) : [];
  const sourcePair = SOURCE_LABELS[String(entry.source || '')] || null;
  const [diffOpen, setDiffOpen] = useState(false);
  const [diff, setDiff] = useState({ state: 'idle', data: null, error: '' });
  const [editing, setEditing] = useState(false);
  const [labelValue, setLabelValue] = useState(entry.label || '');
  const [renameBusy, setRenameBusy] = useState(false);
  const [restoreOpen, setRestoreOpen] = useState(false);
  const [selected, setSelected] = useState(() => new Set(files));
  const busy = restoringId === id;

  const toggleDiff = useCallback(async () => {
    const next = !diffOpen;
    setDiffOpen(next);
    if (next && diff.state === 'idle') {
      setDiff({ state: 'loading', data: null, error: '' });
      try {
        const body = await requestJson(`/api/versions/${encodeURIComponent(id)}/diff`);
        setDiff({ state: 'ready', data: body && typeof body.diff === 'object' ? body.diff : {}, error: '' });
      } catch (err) {
        setDiff({ state: 'error', data: null, error: err.message });
      }
    }
  }, [diffOpen, diff.state, id]);

  async function saveRename() {
    setRenameBusy(true);
    try {
      await onRename(id, labelValue.trim());
      setEditing(false);
    } catch (err) {
      notify(`Renaming the version failed (${err.message}).`, `שינוי שם הגרסה נכשל (${err.message}).`);
    } finally {
      setRenameBusy(false);
    }
  }

  function toggleFile(file) {
    setSelected((prev) => {
      const nextSet = new Set(prev);
      if (nextSet.has(file)) nextSet.delete(file);
      else nextSet.add(file);
      return nextSet;
    });
  }

  const chosen = FILE_ORDER.filter((file) => files.includes(file) && selected.has(file));

  return (
    <div className="asst-ver-row">
      <div className="asst-ver-head">
        <div className="asst-ver-main">
          <div className="asst-ver-line">
            <time dir="ltr">{dateTimeLabel(entry.created_at, locale)}</time>
            {sourcePair ? <span className="asst-ver-source">{pageText(locale, sourcePair[0], sourcePair[1])}</span> : <code dir="ltr">{String(entry.source || '?')}</code>}
            {entry.actor ? <span className="asst-ver-actor" dir="auto">{String(entry.actor)}</span> : null}
          </div>
          <div className="asst-ver-label-line">
            {editing ? (
              <span className="asst-ver-rename">
                <input value={labelValue} onChange={(event) => setLabelValue(event.target.value)} dir="auto" maxLength={120} placeholder={pageText(locale, 'Name this version', 'שם לגרסה')} aria-label={pageText(locale, 'Version name', 'שם הגרסה')} disabled={renameBusy} />
                <button type="button" className="asst-ver-rename-ok" onClick={saveRename} disabled={renameBusy} aria-label={pageText(locale, 'Save name', 'שמירת שם')}><Check size={13} /></button>
                <button type="button" className="asst-ver-rename-x" onClick={() => { setEditing(false); setLabelValue(entry.label || ''); }} disabled={renameBusy} aria-label={pageText(locale, 'Cancel', 'ביטול')}><X size={13} /></button>
              </span>
            ) : (
              <span className="asst-ver-label">
                {entry.label ? <span dir="auto">{entry.label}</span> : <span className="asst-ver-unlabeled">{pageText(locale, 'Unnamed', 'ללא שם')}</span>}
                {canWrite ? <button type="button" className="asst-ver-pencil" onClick={() => { setLabelValue(entry.label || ''); setEditing(true); }} aria-label={pageText(locale, 'Rename version', 'שינוי שם הגרסה')}><Pencil size={12} /></button> : null}
              </span>
            )}
          </div>
          {files.length ? <div className="asst-ver-files">{files.map((file) => <code dir="ltr" key={file}>{fileLabel(file, locale)}</code>)}</div> : null}
        </div>
        <div className="asst-ver-actions">
          <button type="button" className="asst-ver-toggle" onClick={toggleDiff} aria-expanded={diffOpen}>
            {diffOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            {pageText(locale, 'Show differences', 'הצג הבדלים')}
          </button>
          {canWrite ? (
            <Button variant="outlined" size="small" startIcon={<RotateCcw size={13} />} disabled={busy} onClick={() => setRestoreOpen((value) => !value)}>
              {busy ? pageText(locale, 'Restoring', 'משחזר') : pageText(locale, 'Restore', 'שחזר')}
            </Button>
          ) : null}
        </div>
      </div>

      {diffOpen ? (
        diff.state === 'loading' ? <div className="asst-loading">{pageText(locale, 'Loading the differences', 'טוען את ההבדלים')}</div>
        : diff.state === 'error' ? <div className="asst-error-note">{pageText(locale, `The differences could not be loaded (${diff.error}).`, `לא ניתן לטעון את ההבדלים (${diff.error}).`)}</div>
        : diff.state === 'ready' ? <DiffView data={diff.data} locale={locale} /> : null
      ) : null}

      {restoreOpen && canWrite ? (
        <div className="asst-ver-restore" role="group" aria-label={pageText(locale, 'What to restore', 'מה לשחזר')}>
          <span className="asst-ver-restore-h">{pageText(locale, 'What to restore', 'מה לשחזר')}</span>
          <div className="asst-ver-restore-files">
            {FILE_ORDER.map((file) => {
              const present = files.includes(file);
              return (
                <label key={file} className={`asst-ver-file-opt${present ? '' : ' off'}`}>
                  <input type="checkbox" disabled={!present || busy} checked={present && selected.has(file)} onChange={() => toggleFile(file)} />
                  {fileLabel(file, locale)}
                </label>
              );
            })}
          </div>
          <p className="asst-ver-restore-note" dir="auto">{pageText(locale, 'A safety version of the current state is saved automatically before restoring, so this restore can be undone.', 'לפני השחזור נשמרת אוטומטית גרסת בטיחות של המצב הנוכחי, כך שאפשר לבטל את השחזור.')}</p>
          <div className="asst-confirm-actions">
            <Button variant="contained" size="small" disabled={busy || !chosen.length} onClick={() => { setRestoreOpen(false); onRestore(id, chosen); }}>
              {pageText(locale, 'Restore now', 'שחזר עכשיו')}
            </Button>
            <Button variant="outlined" size="small" disabled={busy} onClick={() => setRestoreOpen(false)}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function AssistantVersions({ locale, notify, reloadKey, onChanged }) {
  const [state, setState] = useState('loading');
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [canWrite, setCanWrite] = useState(true);
  const [restoringId, setRestoringId] = useState(null);
  const [snapOpen, setSnapOpen] = useState(false);
  const [snapLabel, setSnapLabel] = useState('');
  const [snapBusy, setSnapBusy] = useState(false);

  useEffect(() => {
    let active = true;
    fetchMe().then((result) => {
      if (active && result && result.ok && result.data && result.data.role) setCanWrite(result.data.role !== 'viewer');
    }).catch(() => {});
    return () => { active = false; };
  }, []);

  const load = useCallback(async () => {
    setState('loading');
    try {
      const body = await requestJson('/api/versions?limit=50');
      setEntries(Array.isArray(body.entries) ? body.entries : []);
      setNote(typeof body.note === 'string' ? body.note : typeof body.scope_note === 'string' ? body.scope_note : '');
      setState('ready');
      setError('');
    } catch (err) {
      setState('error');
      setError(err.message);
    }
  }, []);

  useEffect(() => { load(); }, [load, reloadKey]);

  const renameVersion = useCallback(async (id, label) => {
    const body = await requestJson(`/api/versions/${encodeURIComponent(id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label }) });
    const nextLabel = body && typeof body.label === 'string' ? body.label : label;
    setEntries((prev) => prev.map((entry) => (String(entry.version_id) === String(id) ? { ...entry, label: nextLabel } : entry)));
    notify('The version name was updated.', 'שם הגרסה עודכן.');
  }, [notify]);

  const restoreVersion = useCallback(async (id, files) => {
    if (restoringId) return;
    setRestoringId(id);
    try {
      const body = await postJson(`/api/versions/${encodeURIComponent(id)}/restore`, { files });
      const restored = Array.isArray(body.restored) ? body.restored : [];
      const count = restored.length;
      if (count === 1) notify('Restored one file. A safety version was saved, so this restore can be undone.', 'שוחזר קובץ אחד. נשמרה גרסת בטיחות, כך שאפשר לבטל את השחזור.');
      else if (count) notify(`Restored ${count} files. A safety version was saved, so this restore can be undone.`, `שוחזרו ${count} קבצים. נשמרה גרסת בטיחות, כך שאפשר לבטל את השחזור.`);
      else notify('The restore completed. A safety version was saved, so it can be undone.', 'השחזור הושלם. נשמרה גרסת בטיחות, כך שאפשר לבטל אותו.');
      await load();
      if (onChanged) onChanged();
    } catch (err) {
      notify(`The restore failed (${err.message}).`, `השחזור נכשל (${err.message}).`);
    } finally {
      setRestoringId(null);
    }
  }, [restoringId, notify, load, onChanged]);

  const createSnapshot = useCallback(async () => {
    setSnapBusy(true);
    try {
      await postJson('/api/versions/snapshot', { label: snapLabel.trim() });
      setSnapLabel('');
      setSnapOpen(false);
      notify('A version point was saved.', 'נקודת גרסה נשמרה.');
      await load();
      if (onChanged) onChanged();
    } catch (err) {
      notify(`Saving the version point failed (${err.message}).`, `שמירת נקודת הגרסה נכשלה (${err.message}).`);
    } finally {
      setSnapBusy(false);
    }
  }, [snapLabel, notify, load, onChanged]);

  return (
    <div className="asst-versions">
      <p className="asst-ver-intro" dir="auto">{pageText(locale, 'Every change and restore saves a version you can review and roll back. A restore first saves a safety version of the current state, so it is always undoable.', 'כל שינוי וכל שחזור שומרים גרסה שאפשר לעיין בה ולחזור אליה. שחזור שומר תחילה גרסת בטיחות של המצב הנוכחי, כך שהוא תמיד הפיך.')}</p>

      {canWrite ? (
        <div className="asst-ver-snapshot">
          {snapOpen ? (
            <div className="asst-ver-snap-form">
              <input value={snapLabel} onChange={(event) => setSnapLabel(event.target.value)} dir="auto" maxLength={120} placeholder={pageText(locale, 'Name this point', 'שם לנקודה')} aria-label={pageText(locale, 'Version point name', 'שם נקודת הגרסה')} disabled={snapBusy} />
              <Button variant="contained" size="small" disabled={snapBusy} onClick={createSnapshot}>{snapBusy ? pageText(locale, 'Saving', 'שומר') : pageText(locale, 'Save point', 'שמור נקודה')}</Button>
              <Button variant="text" size="small" disabled={snapBusy} onClick={() => { setSnapOpen(false); setSnapLabel(''); }}>{pageText(locale, 'Cancel', 'ביטול')}</Button>
            </div>
          ) : (
            <button type="button" className="asst-ver-snap-btn" onClick={() => setSnapOpen(true)}>
              <Camera size={13} />
              {pageText(locale, 'Create a version point', 'צור נקודת גרסה')}
            </button>
          )}
        </div>
      ) : null}

      {state === 'loading' ? <div className="asst-loading">{pageText(locale, 'Loading the versions', 'טוען את הגרסאות')}</div> : null}
      {state === 'error' ? <div className="asst-error-note">{pageText(locale, `The versions could not be loaded (${error}).`, `לא ניתן לטעון את הגרסאות (${error}).`)}</div> : null}
      {state === 'ready' && entries.length === 0 ? (
        <div className="asst-empty">
          <History size={18} />
          {pageText(locale, 'No versions yet. A version is saved automatically before every change and every restore.', 'אין עדיין גרסאות. גרסה נשמרת אוטומטית לפני כל שינוי ולפני כל שחזור.')}
        </div>
      ) : null}

      {state === 'ready' && entries.length > 0 ? (
        <div className="asst-ver-list">
          {entries.map((entry, index) => (
            <VersionRow key={(entry && entry.version_id) || `ver-${index}`} entry={entry} locale={locale} canWrite={canWrite} restoringId={restoringId} onRename={renameVersion} onRestore={restoreVersion} notify={notify} />
          ))}
        </div>
      ) : null}

      {state === 'ready' && note ? <p className="asst-ver-scope" dir="auto">{pageText(locale, note, 'הגרסאות שומרות את קבצי מצב התפעול שהמפעילים עורכים: הגדרות (כולל עקיפות תמחור), אילוצי שיבוץ, עקיפות ידניות וכללי מפרסמים. ההיסטוריה נשמרת תמיד; שחזור מתעד קודם את המצב הנוכחי, ולכן ניתן לבטל אותו.')}</p> : null}
    </div>
  );
}
