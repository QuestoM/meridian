import React, { useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { RotateCcw } from 'lucide-react';
import { pageText } from './surface-helpers';

// The assistant rail's history and restore surfaces. The history tab shows the
// server audit log grouped by day; the restore tab lists restore points with
// an inline confirm before restoring. Both render honest loading, error and
// empty states: nothing here is ever fabricated on the client.

const EVENT_LABELS = {
  ask: ['Question asked', 'שאלה נשאלה'],
  answer: ['Answer returned', 'תשובה התקבלה'],
  propose: ['Actions proposed', 'הוצעו פעולות'],
  proposal: ['Actions proposed', 'הוצעו פעולות'],
  proposals: ['Actions proposed', 'הוצעו פעולות'],
  apply: ['Actions applied', 'פעולות הוחלו'],
  reject: ['Actions rejected', 'פעולות נדחו'],
  restore: ['Restore performed', 'בוצע שחזור'],
};

function timeLabel(ts, locale) {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US', { hour: '2-digit', minute: '2-digit' });
}

function dateTimeLabel(ts, locale) {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function groupByDay(entries, locale) {
  const groups = [];
  const index = new Map();
  for (const entry of entries) {
    const date = new Date(entry && entry.ts ? entry.ts : NaN);
    const valid = !Number.isNaN(date.getTime());
    const key = valid ? date.toISOString().slice(0, 10) : 'unknown';
    if (!index.has(key)) {
      const label = valid
        ? date.toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-US', { weekday: 'long', day: 'numeric', month: 'long' })
        : pageText(locale, 'Undated', 'ללא תאריך');
      index.set(key, groups.length);
      groups.push({ key, label, rows: [] });
    }
    groups[index.get(key)].rows.push(entry);
  }
  return groups;
}

function resultsSummary(results, locale) {
  if (!Array.isArray(results) || results.length === 0) return '';
  const applied = results.filter((row) => row && row.status === 'applied').length;
  const failed = results.filter((row) => row && row.status === 'failed').length;
  const parts = [];
  if (applied) parts.push(applied === 1 ? pageText(locale, 'one applied', 'אחת הוחלה') : pageText(locale, `${applied} applied`, `${applied} הוחלו`));
  if (failed) parts.push(failed === 1 ? pageText(locale, 'one failed', 'אחת נכשלה') : pageText(locale, `${failed} failed`, `${failed} נכשלו`));
  return parts.join(', ');
}

function filesCount(files) {
  if (Array.isArray(files)) return files.length;
  const number = Number(files);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function AuditLog({ audit, locale }) {
  const entries = Array.isArray(audit.entries) ? audit.entries : [];
  const groups = useMemo(() => groupByDay(entries, locale), [entries, locale]);
  if (audit.state === 'loading') {
    return <div className="asst-loading">{pageText(locale, 'Loading the activity log', 'טוען את יומן הפעילות')}</div>;
  }
  if (audit.state === 'error') {
    return <div className="asst-error-note">{pageText(locale, `The activity log could not be loaded (${audit.error}).`, `לא ניתן לטעון את יומן הפעילות (${audit.error}).`)}</div>;
  }
  if (entries.length === 0) {
    return <div className="asst-empty">{pageText(locale, 'No activity yet. Questions, approvals and restores are recorded here.', 'אין עדיין פעילות. שאלות, אישורים ושחזורים יירשמו כאן.')}</div>;
  }
  return (
    <div className="asst-audit">
      {groups.map((group) => (
        <section className="asst-day" key={group.key}>
          <h4 className="asst-day-label">{group.label}</h4>
          {group.rows.map((entry, index) => {
            const pair = EVENT_LABELS[String(entry && entry.event ? entry.event : '').toLowerCase()] || null;
            const summary = resultsSummary(entry && entry.results, locale);
            return (
              <div className="asst-audit-row" key={`${group.key}-${index}`}>
                <time dir="ltr">{timeLabel(entry && entry.ts, locale)}</time>
                <div className="asst-audit-main">
                  <span className="asst-audit-event">{pair ? pageText(locale, pair[0], pair[1]) : <code dir="ltr">{String((entry && entry.event) || '?')}</code>}</span>
                  {entry && entry.user ? <span className="asst-audit-user" dir="auto">{String(entry.user)}</span> : null}
                  {entry && entry.question ? <span className="asst-audit-detail" dir="auto">{String(entry.question)}</span> : null}
                  {summary ? <span className="asst-audit-detail">{summary}</span> : null}
                  {entry && entry.batch_id ? <code dir="ltr">{String(entry.batch_id).slice(0, 8)}</code> : null}
                </div>
              </div>
            );
          })}
        </section>
      ))}
    </div>
  );
}

function RestoreList({ restore, locale, restoringId, onRestorePoint }) {
  const [confirmId, setConfirmId] = useState(null);
  const points = Array.isArray(restore.points) ? restore.points : [];
  if (restore.state === 'loading') {
    return <div className="asst-loading">{pageText(locale, 'Loading restore points', 'טוען נקודות שחזור')}</div>;
  }
  if (restore.state === 'error') {
    return <div className="asst-error-note">{pageText(locale, `Restore points could not be loaded (${restore.error}).`, `לא ניתן לטעון את נקודות השחזור (${restore.error}).`)}</div>;
  }
  if (points.length === 0) {
    return <div className="asst-empty">{pageText(locale, 'No restore points yet. One is created automatically before every apply.', 'אין עדיין נקודות שחזור. נקודת שחזור נוצרת אוטומטית לפני כל החלת פעולות.')}</div>;
  }
  return (
    <div className="asst-restore-list">
      {points.map((point, index) => {
        const id = point && point.restore_id ? String(point.restore_id) : '';
        const count = filesCount(point && point.files);
        const busy = restoringId === id;
        return (
          <div className="asst-restore-row" key={id || `point-${index}`}>
            <div className="asst-restore-main">
              <time dir="ltr">{dateTimeLabel(point && point.created_at, locale)}</time>
              <span className="asst-restore-meta">
                {count !== null ? (count === 1 ? pageText(locale, 'One file', 'קובץ אחד') : pageText(locale, `${count} files`, `${count} קבצים`)) : pageText(locale, 'File count unknown', 'מספר הקבצים אינו ידוע')}
                {point && point.batch_id ? <code dir="ltr">{String(point.batch_id).slice(0, 8)}</code> : null}
              </span>
            </div>
            {confirmId === id ? (
              <div className="asst-confirm" role="alertdialog">
                <p>{pageText(locale, 'This restores the files to their previous state; a recompute may be needed afterwards.', 'ישחזר את הקבצים למצבם הקודם; ייתכן שיידרש חישוב מחדש.')}</p>
                <div className="asst-confirm-actions">
                  <Button variant="contained" size="small" disabled={busy || !id} onClick={async () => { setConfirmId(null); await onRestorePoint(id); }}>
                    {pageText(locale, 'Restore now', 'שחזר עכשיו')}
                  </Button>
                  <Button variant="outlined" size="small" disabled={busy} onClick={() => setConfirmId(null)}>
                    {pageText(locale, 'Cancel', 'ביטול')}
                  </Button>
                </div>
              </div>
            ) : (
              <Button variant="outlined" size="small" startIcon={<RotateCcw size={13} />} disabled={busy || !id} onClick={() => setConfirmId(id)}>
                {busy ? pageText(locale, 'Restoring', 'משחזר') : pageText(locale, 'Restore', 'שחזר')}
              </Button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function AssistantHistory({ tab, locale, audit, restore, restoringId, onRestorePoint }) {
  if (tab === 'restore') {
    return <RestoreList restore={restore} locale={locale} restoringId={restoringId} onRestorePoint={onRestorePoint} />;
  }
  return <AuditLog audit={audit} locale={locale} />;
}
