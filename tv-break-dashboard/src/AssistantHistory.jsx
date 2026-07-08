import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { History, Trash2 } from 'lucide-react';
import { pageText } from './surface-helpers';
import { requestJson } from './assistant-stream';

// The assistant's conversation history: the operator's own saved questions and
// answers, read back from the server thread and shown as a readable transcript
// grouped by day. Loading, error and empty states are honest; nothing is
// fabricated on the client. A quiet clear action behind an inline confirm removes
// only the caller's own history. Pending actions and restores live in their own
// surfaces.

function dayKey(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return 'unknown';
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

function dayLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return pageText(locale, 'Earlier', 'מוקדם יותר');
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  const sameDay = (a, b) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  if (sameDay(date, today)) return pageText(locale, 'Today', 'היום');
  if (sameDay(date, yesterday)) return pageText(locale, 'Yesterday', 'אתמול');
  return date.toLocaleDateString(locale === 'he' ? 'he-IL' : undefined, { day: '2-digit', month: 'short', year: 'numeric' });
}

function timeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString(locale === 'he' ? 'he-IL' : undefined, { hour: '2-digit', minute: '2-digit' });
}

function groupByDay(entries, locale) {
  const groups = [];
  const index = new Map();
  for (const entry of entries) {
    const at = entry && entry.at;
    const key = dayKey(at);
    if (!index.has(key)) {
      const group = { key, label: dayLabel(at, locale), rows: [] };
      index.set(key, group);
      groups.push(group);
    }
    index.get(key).rows.push(entry);
  }
  return groups;
}

export default function AssistantHistory({ locale }) {
  const [state, setState] = useState('loading');
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState('');
  const [confirming, setConfirming] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearError, setClearError] = useState('');

  const load = useCallback(async () => {
    setState('loading');
    try {
      const body = await requestJson('/api/assistant/thread');
      setEntries(Array.isArray(body.entries) ? body.entries : []);
      setState('ready');
      setError('');
    } catch (err) {
      setState('error');
      setError(err.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const clearHistory = useCallback(async () => {
    setClearing(true);
    setClearError('');
    try {
      await requestJson('/api/assistant/thread', { method: 'DELETE' });
      setConfirming(false);
      await load();
    } catch (err) {
      setClearError(err.message);
    } finally {
      setClearing(false);
    }
  }, [load]);

  // Newest day first; within a day the exchanges read in the order they happened.
  const groups = useMemo(() => groupByDay([...entries].reverse(), locale).reverse(), [entries, locale]);

  if (state === 'loading') {
    return <div className="asst-loading">{pageText(locale, 'Loading the conversation history', 'טוען את היסטוריית השיחות')}</div>;
  }
  if (state === 'error') {
    return <div className="asst-error-note">{pageText(locale, `The conversation history could not be loaded (${error}).`, `לא ניתן לטעון את היסטוריית השיחות (${error}).`)}</div>;
  }
  if (entries.length === 0) {
    return (
      <div className="asst-empty">
        <History size={18} />
        {pageText(locale, 'No conversations yet. Your questions and the assistant answers are saved here for your account.', 'אין עדיין שיחות. השאלות שלכם ותשובות העוזר נשמרות כאן עבור החשבון שלכם.')}
      </div>
    );
  }

  return (
    <div className="asst-convo">
      {groups.map((group) => (
        <section className="asst-convo-day" key={group.key}>
          <h4 className="asst-convo-day-label">{group.label}</h4>
          {group.rows.map((entry, index) => (
            <div className="asst-convo-turn" key={`${group.key}-${index}`}>
              <div className="asst-convo-q">
                <p dir="auto">{String((entry && entry.question) || '')}</p>
                <time dir="ltr">{timeLabel(entry && entry.at, locale)}</time>
              </div>
              {entry && entry.answer ? <div className="asst-convo-a" dir="auto">{String(entry.answer)}</div> : null}
            </div>
          ))}
        </section>
      ))}

      {confirming ? (
        <div className="asst-confirm" role="alertdialog">
          <p>{pageText(locale, 'This permanently deletes your saved conversations on the server.', 'הפעולה מוחקת לצמיתות את השיחות השמורות שלכם בשרת.')}</p>
          {clearError ? <p className="asst-error-note">{clearError}</p> : null}
          <div className="asst-confirm-actions">
            <Button variant="contained" size="small" disabled={clearing} onClick={clearHistory}>
              {clearing ? pageText(locale, 'Deleting', 'מוחק') : pageText(locale, 'Delete now', 'מחק עכשיו')}
            </Button>
            <Button variant="outlined" size="small" disabled={clearing} onClick={() => setConfirming(false)}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </div>
        </div>
      ) : (
        <div className="asst-convo-actions">
          <button type="button" className="asst-convo-clear" onClick={() => setConfirming(true)}>
            <Trash2 size={13} />
            {pageText(locale, 'Clear my history', 'מחיקת ההיסטוריה שלי')}
          </button>
        </div>
      )}
    </div>
  );
}
