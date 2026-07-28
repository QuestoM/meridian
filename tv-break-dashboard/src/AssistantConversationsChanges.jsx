import React, { useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { ExternalLink, RotateCcw } from 'lucide-react';
import { pageText } from './surface-helpers';
import { fetchConversationChanges, restoreConversation } from './AssistantConversationsApi';

// The per-conversation applied-changes view: every proposal batch the active
// conversation produced, with kind, summary, status, who resolved it and when,
// a link into the restore page, and the conversation-level restore action. The
// restore confirm states the honest limits from the design spec word for word:
// recomputes cannot be un-run, a whole-file restore also reverts later manual
// edits to the same files, and the restore itself is undoable because a
// pre-restore snapshot is taken first.

const KIND_LABELS = {
  settings_change: ['Settings change', 'שינוי הגדרות'],
  constraint: ['New constraint', 'אילוץ חדש'],
  override: ['Override', 'עקיפה'],
  pricing_change: ['Pricing change', 'שינוי תמחור'],
  advertiser_change: ['Advertiser change', 'שינוי מפרסם'],
  recompute: ['Recompute', 'חישוב מחדש'],
};

const STATUS_LABELS = {
  pending: ['Pending', 'ממתין'],
  applied: ['Applied', 'הוחל'],
  failed: ['Failed', 'נכשל'],
  rejected: ['Rejected', 'נדחה'],
};

function timeLabel(iso, locale) {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

function StatusChip({ locale, status }) {
  const pair = STATUS_LABELS[status] || null;
  return (
    <span className={`asst-status-chip ${pair ? status : 'unknown'}`}>
      {pair ? pageText(locale, pair[0], pair[1]) : <code dir="ltr">{String(status || '?')}</code>}
    </span>
  );
}

export default function AssistantConversationsChanges({ locale, conversationId, notify, onShowRestore, reloadNonce }) {
  const [state, setState] = useState('loading');
  const [error, setError] = useState('');
  const [batches, setBatches] = useState([]);
  const [confirming, setConfirming] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [restoreResult, setRestoreResult] = useState(null);

  useEffect(() => {
    if (!conversationId) {
      setState('ready');
      setBatches([]);
      return undefined;
    }
    let active = true;
    setState('loading');
    setConfirming(false);
    setRestoreResult(null);
    fetchConversationChanges(conversationId)
      .then((body) => {
        if (!active) return;
        setBatches(Array.isArray(body.batches) ? body.batches.filter((batch) => batch && batch.batch_id) : []);
        setState('ready');
      })
      .catch((err) => {
        if (!active) return;
        setError(err && err.message ? err.message : 'unknown');
        setState('error');
      });
    return () => { active = false; };
  }, [conversationId, reloadNonce]);

  const hasApplied = batches.some((batch) => Array.isArray(batch.items) && batch.items.some((item) => item && item.status === 'applied'));

  async function runRestore() {
    setConfirming(false);
    setRestoring(true);
    try {
      const body = await restoreConversation(conversationId);
      setRestoreResult({
        files: Array.isArray(body.restored_files) ? body.restored_files.map(String) : [],
        versionsUsed: Array.isArray(body.version_ids_used) ? body.version_ids_used.length : 0,
        preId: body.pre_restore_version_id ? String(body.pre_restore_version_id) : null,
      });
      if (notify) notify('The conversation changes were restored and a pre-restore snapshot was saved.', 'השינויים של השיחה שוחזרו ונשמר צילום מצב שלפני השחזור.');
    } catch (err) {
      if (err && err.status === 409) setRestoreResult({ none: true });
      else if (notify) notify(`The conversation restore failed (${err.message}).`, `שחזור השיחה נכשל (${err.message}).`);
    } finally {
      setRestoring(false);
    }
  }

  if (!conversationId) return <div className="asst-empty">{pageText(locale, 'No active conversation yet.', 'אין עדיין שיחה פעילה.')}</div>;
  if (state === 'loading') return <div className="asst-loading">{pageText(locale, 'Loading the conversation changes', 'טוען את שינויי השיחה')}</div>;
  if (state === 'error') return <div className="asst-error-note">{pageText(locale, `The conversation changes could not be loaded (${error}).`, `לא ניתן לטעון את שינויי השיחה (${error}).`)}</div>;
  if (!batches.length) return <div className="asst-empty">{pageText(locale, 'The assistant has not proposed changes in this conversation yet.', 'העוזר עוד לא הציע שינויים בשיחה הזו.')}</div>;

  return (
    <>
      <p className="asst-chg-intro">{pageText(locale, 'Everything the assistant proposed and applied in this conversation.', 'כל מה שהעוזר הציע והחיל בשיחה הזו.')}</p>

      {hasApplied && !confirming && !restoring ? (
        <div className="asst-chg-restore-bar">
          <Button variant="outlined" size="small" startIcon={<RotateCcw size={13} />} onClick={() => { setRestoreResult(null); setConfirming(true); }}>
            {pageText(locale, 'Conversation restore', 'שחזור השיחה')}
          </Button>
        </div>
      ) : null}
      {!hasApplied ? (
        <p className="asst-chg-intro">{pageText(locale, 'No change from this conversation is currently applied, so there is nothing to restore.', 'שום שינוי מהשיחה הזו אינו מוחל, ולכן אין מה לשחזר.')}</p>
      ) : null}
      {restoring ? <div className="asst-loading">{pageText(locale, 'Restoring the conversation changes', 'משחזר את שינויי השיחה')}</div> : null}

      {confirming ? (
        <div className="asst-confirm" role="alertdialog">
          <p dir="auto">{pageText(locale, 'The files this conversation changed will be returned to their state before its first applied change.', 'הקבצים שהשיחה שינתה יוחזרו למצבם שלפני השינוי הראשון שהוחל בה.')}</p>
          <p dir="auto">{pageText(locale, 'Recomputes cannot be un-run: inputs are restored, then a recompute is offered.', 'חישובים מחדש אינם ניתנים לביטול: הנתונים משוחזרים, ואז מוצע חישוב מחדש.')}</p>
          <p dir="auto">{pageText(locale, 'A whole-file restore also reverts manual edits made to the same file after the conversation.', 'שחזור קובץ שלם מבטל גם עריכות ידניות שנעשו באותו קובץ אחרי השיחה.')}</p>
          <p dir="auto">{pageText(locale, 'The restore itself is undoable from the restore page: a pre-restore snapshot is saved first.', 'השחזור עצמו ניתן לביטול מעמוד השחזור: נשמר תחילה צילום מצב שלפני השחזור.')}</p>
          <div className="asst-confirm-actions">
            <Button variant="contained" size="small" onClick={runRestore}>{pageText(locale, 'Restore', 'שחזור')}</Button>
            <Button variant="outlined" size="small" onClick={() => setConfirming(false)}>{pageText(locale, 'Cancel', 'ביטול')}</Button>
          </div>
        </div>
      ) : null}

      {restoreResult && restoreResult.none ? (
        <div className="asst-error-note">{pageText(locale, 'No change from this conversation is currently applied, so there is nothing to restore.', 'שום שינוי מהשיחה הזו אינו מוחל, ולכן אין מה לשחזר.')}</div>
      ) : null}
      {restoreResult && !restoreResult.none ? (
        <div className="asst-chg-result">
          <p dir="auto">{restoreResult.files.length === 1 ? pageText(locale, 'One file was restored.', 'שוחזר קובץ אחד.') : pageText(locale, `${restoreResult.files.length} files were restored.`, `שוחזרו ${restoreResult.files.length} קבצים.`)}</p>
          {restoreResult.files.length ? (
            <div className="asst-chg-files">{restoreResult.files.map((file) => <code dir="ltr" key={file}>{file}</code>)}</div>
          ) : null}
          {restoreResult.preId ? (
            <p dir="auto">{pageText(locale, 'A pre-restore snapshot was saved, so this restore is undoable from the restore page.', 'נשמר צילום מצב שלפני השחזור, ולכן אפשר לבטל את השחזור הזה מעמוד השחזור.')}</p>
          ) : null}
          <p dir="auto">{pageText(locale, 'Run a recompute now so the plan reflects the restored data.', 'הריצו עכשיו חישוב מחדש כדי שהתוכנית תשקף את הנתונים המשוחזרים.')}</p>
          <button type="button" className="asst-ver-toggle" onClick={onShowRestore}>
            <ExternalLink size={12} />
            {pageText(locale, 'Restore page', 'עמוד השחזור')}
          </button>
        </div>
      ) : null}

      {batches.map((batch) => {
        const items = Array.isArray(batch.items) ? batch.items.filter((item) => item && typeof item === 'object') : [];
        return (
          <div className="asst-chg-batch" key={String(batch.batch_id)}>
            <div className="asst-chg-head">
              <span className="asst-chg-q" dir="auto">{batch.question ? String(batch.question) : pageText(locale, 'Actions batch', 'אצוות פעולות')}</span>
              <code dir="ltr">{String(batch.batch_id).slice(0, 8)}</code>
            </div>
            <div className="asst-chg-meta">
              {batch.status ? <StatusChip locale={locale} status={String(batch.status)} /> : null}
              {batch.created_by ? <span dir="auto">{pageText(locale, `Asked by ${batch.created_by}`, `נשאל על ידי ${batch.created_by}`)}</span> : null}
              {batch.created_at ? <time dir="ltr">{timeLabel(batch.created_at, locale)}</time> : null}
            </div>
            {items.map((item, index) => {
              const kindPair = KIND_LABELS[item.kind] || null;
              return (
                <div className="asst-chg-item" key={item.id != null ? String(item.id) : `row-${index}`}>
                  <div className="asst-chg-item-head">
                    <span className="asst-kind">{kindPair ? pageText(locale, kindPair[0], kindPair[1]) : <code dir="ltr">{String(item.kind || '?')}</code>}</span>
                    <StatusChip locale={locale} status={String(item.status || '')} />
                  </div>
                  {item.summary ? <p className="asst-chg-summary" dir="auto">{String(item.summary)}</p> : null}
                  {item.resolved_by ? (
                    <p className="asst-chg-resolved" dir="auto">{pageText(locale, `Resolved by ${item.resolved_by}`, `מבצע: ${item.resolved_by}`)}{item.resolved_at ? ` · ${timeLabel(item.resolved_at, locale)}` : ''}</p>
                  ) : null}
                </div>
              );
            })}
            {Array.isArray(batch.version_ids) && batch.version_ids.length ? (
              <button type="button" className="asst-ver-toggle" onClick={onShowRestore}>
                <ExternalLink size={12} />
                {pageText(locale, 'Version diff on the restore page', 'הצגת הגרסה בעמוד השחזור')}
              </button>
            ) : null}
          </div>
        );
      })}
    </>
  );
}
