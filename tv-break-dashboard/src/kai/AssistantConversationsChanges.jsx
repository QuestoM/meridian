import React, { useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { ExternalLink, RotateCcw } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { fetchConversationChanges, restoreConversation } from './AssistantConversationsApi';
import { KINDS } from './AssistantProposalCard';
import ProposalSummary from './AssistantProposalSummary';
import { isolate } from './kai-bidi';

// The per-conversation applied-changes view: every proposal batch the active
// conversation produced, with kind, summary, status, who resolved it and when,
// a link into the restore page, and the conversation-level restore action. The
// restore confirm states the honest limits from the design spec word for word:
// a plan run cannot be un-run, a whole-file restore also reverts later manual
// edits to the same files, and the restore itself is undoable because a
// pre-restore snapshot is taken first.

// One vocabulary for proposal kinds, owned by the card that renders them, so a
// kind can never be called two different things on two surfaces.
function kindPair(kind) {
  const entry = KINDS[kind];
  return entry ? [entry.en, entry.he] : null;
}

// The terms behind an item's summary, which this endpoint sends beside the
// items rather than inside them because the item key set is a frozen contract.
// An item with no entry keeps its own terms if it has any, and otherwise none,
// which is what the reading falls back on.
function withTerms(item, batch) {
  const table = batch && batch.item_terms && typeof batch.item_terms === 'object' ? batch.item_terms : null;
  const terms = table ? table[String(item.id)] : null;
  return terms ? { ...item, summary_terms: terms } : item;
}

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
      else if (notify) notify(`The conversation restore failed (${isolate(err.message)}).`, `שחזור השיחה נכשל (${isolate(err.message)}).`);
    } finally {
      setRestoring(false);
    }
  }

  if (!conversationId) return <div className="asst-empty">{pageText(locale, 'No active conversation yet.', 'אין עדיין שיחה פעילה.')}</div>;
  if (state === 'loading') return <div className="asst-loading">{pageText(locale, 'Loading the conversation changes', 'טוען את שינויי השיחה')}</div>;
  if (state === 'error') return <div className="asst-error-note">{pageText(locale, 'The conversation changes could not be loaded (', 'לא ניתן לטעון את שינויי השיחה (')}<bdi dir="auto">{error}</bdi>{').'}</div>;
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
          <p dir="auto">{pageText(locale, 'A plan run cannot be un-run: the inputs go back, then the plan is offered again.', 'הרצת תוכנית אינה ניתנת לביטול: הנתונים חוזרים, ואז מוצעת הרצה חדשה.')}</p>
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
          <p dir="auto">{pageText(locale, 'Run the plan now so it reflects the restored data.', 'הריצו עכשיו את התוכנית כדי שתשקף את הנתונים המשוחזרים.')}</p>
          <button type="button" className="asst-ver-toggle" onClick={() => onShowRestore(restoreResult.preId)}>
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
              {batch.created_by ? <span dir="auto">{pageText(locale, 'Asked by ', 'נשאל על ידי ')}<bdi dir="auto">{batch.created_by}</bdi></span> : null}
              {batch.created_at ? <time dir="ltr">{timeLabel(batch.created_at, locale)}</time> : null}
            </div>
            {items.map((item, index) => {
              const pair = kindPair(item.kind);
              return (
                <div className="asst-chg-item" key={item.id != null ? String(item.id) : `row-${index}`}>
                  <div className="asst-chg-item-head">
                    <span className="asst-kind">{pair ? pageText(locale, pair[0], pair[1]) : <code dir="ltr">{String(item.kind || '?')}</code>}</span>
                    <StatusChip locale={locale} status={String(item.status || '')} />
                  </div>
                  <ProposalSummary item={withTerms(item, batch)} locale={locale} className="asst-chg-summary" />

                  {item.resolved_by ? (
                    <p className="asst-chg-resolved" dir="auto">{pageText(locale, 'Resolved by ', 'מבצע: ')}<bdi dir="auto">{item.resolved_by}</bdi>{item.resolved_at ? <>{' · '}<bdi dir="ltr">{timeLabel(item.resolved_at, locale)}</bdi></> : null}</p>
                  ) : null}
                </div>
              );
            })}
            {Array.isArray(batch.version_ids) && batch.version_ids.length ? (
              // version_ids arrives newest first (the server's own manifest
              // order), so the first entry is the version this batch's most
              // recent apply produced.
              <button type="button" className="asst-ver-toggle" onClick={() => onShowRestore(batch.version_ids[0])}>
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
