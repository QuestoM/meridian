import React, { useState } from 'react';
import { Button } from '../studio/actions';
import { Check, Pencil, Trash2, X } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Name } from '../shell/bidi';
import { formatStamp } from '../shell/dates';
import { InputControl, Pressable } from '../studio/dom-controls';

// The conversations section of the assistant rail: the saved conversations
// newest first with title and last-activity time, inline rename, and delete
// behind a confirm that names exactly what is
// removed. All mutations go through the useConversations hook so the panel,
// the chat column and this list stay on one source of truth.

function whenLabel(iso) {
  return iso ? formatStamp(iso) : '';
}

export default function AssistantConversationsRail({ locale, conv, disabled, onSelect = null }) {
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const busy = Boolean(disabled) || conv.busy;
  const rows = Array.isArray(conv.conversations) ? conv.conversations : [];

  function startRename(item) {
    setDeletingId(null);
    setRenamingId(String(item.id));
    setRenameValue(item.title ? String(item.title) : '');
  }

  async function saveRename(item) {
    const trimmed = renameValue.trim();
    setRenamingId(null);
    if (trimmed && trimmed !== String(item.title || '')) await conv.rename(String(item.id), trimmed);
  }

  return (
    <div className="asst-conv-section">
      <div className="asst-conv-head">
        <span className="asst-rail-title">{pageText(locale, 'Conversation history', 'היסטוריית שיחות')}</span>
      </div>

      {conv.listState === 'loading' ? (
        <div className="asst-loading">{pageText(locale, 'Loading the conversation list', 'טוען את רשימת השיחות')}</div>
      ) : null}
      {conv.listState === 'error' ? (
        <div className="asst-error-note">
          <span>{pageText(locale, 'The conversation list could not be loaded.', 'לא ניתן לטעון את רשימת השיחות.')}</span>
          <Button variant="text" size="small" onClick={conv.refreshList}>{pageText(locale, 'Retry', 'ניסיון חוזר')}</Button>
        </div>
      ) : null}
      {conv.listState === 'ready' && rows.length === 0 ? (
        <div className="asst-empty">{pageText(locale, 'No saved conversations yet. The first question starts one.', 'אין עדיין שיחות שמורות. השאלה הראשונה פותחת שיחה.')}</div>
      ) : null}

      {rows.length ? (
        <div className="asst-conv-list">
          {rows.map((item) => {
            const id = String(item.id);
            const title = item.title ? String(item.title) : pageText(locale, 'Untitled conversation', 'שיחה ללא כותרת');
            const count = Number(item.entry_count) || 0;
            return (
              <div className={`asst-conv-row${conv.activeId === id ? ' active' : ''}`} key={id}>
                {renamingId === id ? (
                  <span className="asst-conv-rename">
                    <InputControl
                      value={renameValue}
                      onChange={(event) => setRenameValue(event.target.value)}
                      onKeyDown={(event) => { if (event.key === 'Enter') saveRename(item); if (event.key === 'Escape') setRenamingId(null); }}
                      dir="auto"
                      autoFocus
                      maxLength={120}
                      aria-label={pageText(locale, 'Conversation title', 'כותרת השיחה')}
                    />
                    <Pressable type="button" className="asst-ver-rename-ok" onClick={() => saveRename(item)} aria-label={pageText(locale, 'Save the title', 'שמירת הכותרת')}><Check size={13} /></Pressable>
                    <Pressable type="button" className="asst-ver-rename-x" onClick={() => setRenamingId(null)} aria-label={pageText(locale, 'Cancel', 'ביטול')}><X size={13} /></Pressable>
                  </span>
                ) : (
                  <div className="asst-conv-line">
                    <Pressable type="button" className="asst-conv-open" onClick={() => { conv.select(id); if (onSelect) onSelect(id); }} disabled={busy} aria-current={conv.activeId === id ? 'true' : undefined}>
                      <span className="asst-conv-title"><Name>{title}</Name></span>
                      <span className="asst-conv-meta">
                        <time><Figure>{whenLabel(item.updated_at || item.created_at)}</Figure></time>
                        <span>{count === 1 ? pageText(locale, 'one question', 'שאלה אחת') : pageText(locale, `${count} questions`, `${count} שאלות`)}</span>
                      </span>
                    </Pressable>
                    <span className="asst-conv-actions">
                      <Pressable type="button" onClick={() => startRename(item)} disabled={busy} aria-label={pageText(locale, 'Rename', 'שינוי שם')}><Pencil size={12} /></Pressable>
                      <Pressable type="button" onClick={() => { setRenamingId(null); setDeletingId(id); }} disabled={busy} aria-label={pageText(locale, 'Delete', 'מחיקה')}><Trash2 size={12} /></Pressable>
                    </span>
                  </div>
                )}
                {deletingId === id ? (
                  <div className="asst-conv-confirm" role="alertdialog">
                    <p>
                      {pageText(locale, 'Delete the conversation ', 'למחוק את השיחה ')}
                      <Name>{`"${title}"`}</Name>
                      {count === 1 ? pageText(locale, ' and the one question saved in it?', ' ואת השאלה האחת השמורה בה?') : pageText(locale, ` and the ${count} questions saved in it?`, ` ואת ${count} השאלות השמורות בה?`)}
                    </p>
                    <p>{pageText(locale, 'Applied changes and restore points are kept.', 'שינויים שהוחלו ונקודות שחזור נשמרים.')}</p>
                    <div className="asst-confirm-actions">
                      <Button variant="contained" size="small" color="error" disabled={busy} onClick={async () => { setDeletingId(null); await conv.remove(id); }}>
                        {pageText(locale, 'Delete', 'מחיקה')}
                      </Button>
                      <Button variant="text" size="small" disabled={busy} onClick={() => setDeletingId(null)}>
                        {pageText(locale, 'Cancel', 'ביטול')}
                      </Button>
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
          {rows.length >= 25 ? (
            <p className="asst-conv-note">{pageText(locale, 'Up to 30 conversations are kept; the oldest are removed first.', 'נשמרות עד 30 שיחות; הישנות ביותר נמחקות תחילה.')}</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
