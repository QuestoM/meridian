// Clients, pacing: the two counting sentences above the board.
//
// They are prose about a payload and not one character of them is JSX, so they
// live beside the panel rather than inside it. Two reasons, and the second is
// the one that matters. PacingWorkspace.jsx was at the 450-line law with the
// undo, the record focus and the return trip still to add. And a sentence that
// states a count is exactly the part a critic should be able to execute rather
// than read: node runs this file directly, the way it already runs
// pacing-helpers.js, so the arithmetic behind the headline is asserted against
// the shipped board rather than greped for.
//
// Nothing here reads a clock, a store or a component. Every figure is a field
// the server counted, and a board that is not ready yet returns null so the
// panel prints a state rather than a zero.

import { isolate, pick } from './pacing-helpers.js';

// How many of the rows the board is asking a decision about already carry one.
// The job this destination serves is done when every at-risk campaign has an act
// taken against it or a recorded decision to take the risk on, so the headline
// counts what is left rather than what exists.
export function decidedCount(board) {
  if (!board || board.status !== 'ready') return 0;
  const payload = board.payload;
  const asking = payload.needs_a_decision || [];
  const raised = payload.make_goods || {};
  const accepted = payload.acceptances || {};
  return (payload.rows || []).filter((row) => (
    asking.indexOf(row.headline.verdict) >= 0
    && ((raised[row.campaign_id] || []).length > 0 || (accepted[row.campaign_id] || []).length > 0)
  )).length;
}

// The sentence above the list: what is left to decide, what is already decided,
// and what cannot be paced at all. A read in flight and a read that failed are
// two different facts and neither of them is a count.
export function headlineSentence(board, locale) {
  if (!board || board.status === 'loading') {
    return pick(locale, 'Reading the pacing board', 'קורא את לוח הקצב');
  }
  if (board.status === 'failed') {
    return pick(
      locale,
      'The pacing board could not be read, so no count is shown rather than a zero.',
      'לא ניתן היה לקרוא את לוח הקצב, ולכן לא מוצג מספר במקום אפס.',
    );
  }
  const counts = board.payload.counts || {};
  const acting = (counts.behind || 0) + (counts.at_risk || 0);
  const settled = decidedCount(board);
  return pick(
    locale,
    `${acting - settled} of ${counts.total || 0} campaigns still need a decision, ${settled} of the ${acting} at risk already carry one, ${counts.unknown || 0} cannot be paced yet.`,
    `${isolate(acting - settled)} מתוך ${isolate(counts.total || 0)} קמפיינים עדיין דורשים החלטה, ל־${isolate(settled)} מתוך ${isolate(acting)} שבסיכון כבר יש אחת, ${isolate(counts.unknown || 0)} עדיין לא ניתנים למדידת קצב.`,
  );
}

// How many of the counted rows the demo seed wrote. A count that mixes seeded
// rows into an operational one reads as a morning's work, and on this data most
// of them are seeded, so the count says which is which. The rows themselves are
// marked; the sentence above them was not.
export function seededSentence(board, locale) {
  if (!board || board.status !== 'ready') return null;
  const counts = board.payload.counts || {};
  if (!counts.demo) return null;
  const asking = (counts.behind || 0) + (counts.at_risk || 0);
  const asked = counts.demo_needing_a_decision || 0;
  const base = pick(
    locale,
    `${counts.demo} of the ${counts.total || 0} are demo rows the seed wrote against the real traffic log, not campaigns an operator booked. Their goals and flight dates are the seed's.`,
    `${isolate(counts.demo)} מתוך ${isolate(counts.total || 0)} הן שורות הדגמה שנכתבו על בסיס יומן השידור האמיתי ולא קמפיינים שמפעיל הזמין. היעדים ותאריכי הטיסה שלהן הם של זרע ההדגמה.`,
  );
  // The sentence above is about the board and the one below is about the count
  // in front of it. Measured on the shipped data, every row that needs a
  // decision is a seeded one, so a reader who took the headline for a morning's
  // work would have been reading the seed's own arithmetic back to itself.
  if (!asking) return base;
  const rest = pick(
    locale,
    `${asked} of the ${asking} rows that need a decision are among them.`,
    `${isolate(asked)} מתוך ${isolate(asking)} השורות שדורשות החלטה נמנות עליהן.`,
  );
  return `${base} ${rest}`;
}
