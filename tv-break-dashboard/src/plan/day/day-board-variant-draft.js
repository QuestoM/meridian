import { useEffect, useRef } from 'react';
import {
  MIN_DURATION_SECONDS,
  applyEdit,
  maxDurationFor,
  offsetBounds,
  snapTo,
} from './day-board-model';

export function normalizeVariantDraft(rawEdits, breaks, programmes) {
  const wanted = rawEdits && typeof rawEdits === 'object' ? rawEdits : {};
  const items = new Map((breaks || []).map((item) => [String(item.break_id), item]));
  let edits = {};
  for (const [breakId, raw] of Object.entries(wanted)) {
    const item = items.get(String(breakId));
    const programme = item ? programmes.get(item.segment_id) : null;
    const offset = Number(raw?.offset_seconds);
    const duration = Number(raw?.duration_seconds);
    if (!item || !programme || !Number.isFinite(offset) || !Number.isFinite(duration)) {
      return { ok: false, edits: {}, reason: `Draft break ${breakId} is not valid on this plan.` };
    }
    const safeDuration = snapTo(duration, 1, MIN_DURATION_SECONDS, maxDurationFor(programme, 0));
    const bounds = offsetBounds(programme, safeDuration);
    const safeOffset = snapTo(offset, 1, bounds.min, bounds.max);
    if (safeDuration !== duration || safeOffset !== offset) {
      return { ok: false, edits: {}, reason: `Draft break ${breakId} falls outside its current programme.` };
    }
    edits = applyEdit(edits, item, { offsetSeconds: safeOffset, durationSeconds: safeDuration });
  }
  return { ok: true, edits, reason: '' };
}

export function useDayBoardVariantDraft({
  board,
  breaks,
  programmes,
  command,
  setEdits,
  setSelected,
  setScore,
  setSettlement,
  resetHistory,
  notify,
}) {
  const appliedRef = useRef('');
  useEffect(() => {
    if (!board?.available || !command?.id || appliedRef.current === command.id) return;
    if (String(command.day) !== String(board.day)
      || String(command.channel) !== String(board.operator_channel)) return;
    const normalized = normalizeVariantDraft(command.edits, breaks, programmes);
    appliedRef.current = command.id;
    if (!normalized.ok) {
      notify?.(
        `This browser draft was not opened: ${normalized.reason}`,
        'טיוטת הדפדפן לא נפתחה מפני שהמזהים או הגבולות שלה אינם תואמים עוד לתוכנית המדויקת הזאת.',
      );
      return;
    }
    setEdits(normalized.edits);
    resetHistory();
    setScore(null);
    setSettlement(null);
    setSelected(Object.keys(normalized.edits)[0] || null);
  }, [board, breaks, command, notify, programmes, resetHistory, setEdits, setScore, setSelected, setSettlement]);
}
