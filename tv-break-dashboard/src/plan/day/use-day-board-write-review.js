import { useCallback, useEffect, useRef, useState } from 'react';

export default function useDayBoardWriteReview({ board, edits, liveOf, notify, onGold, onSave }) {
  const [review, setReview] = useState(null);
  const current = useRef({ board, edits, liveOf, notify, onGold, onSave });
  current.current = { board, edits, liveOf, notify, onGold, onSave };

  useEffect(() => { setReview(null); }, [board?.day]);

  const requestGold = useCallback((item) => {
    const state = current.current;
    if (Object.keys(state.edits).length > 0) {
      state.notify(
        'Save or discard the pending placement changes before changing gold. Gold re-plans the programme and can replace those break ids.',
        'יש לשמור או לבטל את שינויי המיקום הממתינים לפני שינוי הזהב. זהב מתכנן את התוכנית מחדש ועלול להחליף את מזהי הברייקים האלה.',
      );
      return;
    }
    setReview({
      kind: 'gold',
      item,
      live: state.liveOf(item),
      scope: [state.board?.operator_channel, state.board?.day, item?.segment_id].filter(Boolean).join(' / '),
    });
  }, []);

  const requestSave = useCallback(() => {
    const state = current.current;
    const breakIds = Object.keys(state.edits);
    if (breakIds.length === 0) return;
    setReview({
      kind: 'save',
      count: breakIds.length,
      breakIds,
      scope: [state.board?.operator_channel, state.board?.day].filter(Boolean).join(' / '),
    });
  }, []);

  const confirm = useCallback(() => {
    const action = review;
    setReview(null);
    if (action?.kind === 'gold') current.current.onGold(action.item);
    else if (action?.kind === 'save') current.current.onSave();
  }, [review]);

  return { review, requestGold, requestSave, confirm, cancel: () => setReview(null) };
}
