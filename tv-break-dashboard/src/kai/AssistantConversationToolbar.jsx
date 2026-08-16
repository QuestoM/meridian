import React from 'react';
import { History, MessageSquarePlus, Trash2 } from 'lucide-react';
import { Button } from '../studio/actions';
import { Pressable } from '../studio/dom-controls';
import { pageText } from '../shell/surface-helpers';

// Conversation-level navigation stays beside the conversation title. History
// is a view of the conversation surface; it is not an action-log category.
export default function AssistantConversationToolbar({
  locale, supported, listState, busy, asking, historyOpen, canClear, confirmClear,
  clearing, onNew, onToggleHistory, onRequestClear, onClear, onCancelClear,
}) {
  return (
    <div className="asst-conversation-actions">
      {supported !== false ? (
        <Pressable type="button" className="asst-conversation-action" onClick={onNew} disabled={asking || busy || supported !== true}>
          <MessageSquarePlus size={15} />
          {pageText(locale, 'New conversation', 'שיחה חדשה')}
        </Pressable>
      ) : null}
      {supported !== false ? (
        <Pressable
          type="button"
          className={`asst-conversation-action${historyOpen ? ' active' : ''}`}
          onClick={onToggleHistory}
          disabled={asking || (supported !== true && listState !== 'error')}
          aria-expanded={historyOpen}
          aria-controls="assistant-conversation-history"
        >
          <History size={15} />
          {historyOpen ? pageText(locale, 'Back to conversation', 'חזרה לשיחה') : pageText(locale, 'Conversation history', 'היסטוריית שיחות')}
        </Pressable>
      ) : null}
      {canClear ? (
        confirmClear ? (
          <span className="asst-clear-confirm">
            <span>{pageText(locale, 'Delete the whole conversation?', 'למחוק את כל השיחה?')}</span>
            <Button variant="contained" size="small" color="error" disabled={clearing} onClick={onClear}>
              {clearing ? pageText(locale, 'Deleting', 'מוחק') : pageText(locale, 'Delete', 'מחק')}
            </Button>
            <Button variant="text" size="small" disabled={clearing} onClick={onCancelClear}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </span>
        ) : (
          <Pressable type="button" className="asst-clear-btn" onClick={onRequestClear}>
            <Trash2 size={13} />
            {pageText(locale, 'Clear', 'מחיקה')}
          </Pressable>
        )
      ) : null}
    </div>
  );
}
