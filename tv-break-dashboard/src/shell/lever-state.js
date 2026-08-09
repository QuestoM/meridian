// One computed answer to "is this placement lever actually doing anything?".
//
// Three levers fold into the optimizer's one placement-preference map: advertiser
// demand, inventory awareness and delivery pacing. Every surface that shows or
// tunes one of them reads its state from here, so no screen carries a hardcoded
// verdict that will quietly go stale.
//
// Three states, the same three the pricing layers use: live, wired and off,
// absent.
//
// The mechanism reason, measured 2026-08-09 and pinned by
// tests/test_demand_weights_reach_only_greedy.py: the preference map is read
// only by the plan's first pass. The refinement step that follows re-optimises
// the schedule on revenue and retention and never reads the map, so a preference
// the first pass took on is taken straight back out. Over 30 real channel-days a
// full-range preference moved 754 segment break-counts before refinement and 2
// after it. Until that changes, no lever in this family can be live, whatever
// data it has -- which is why hasData alone must never drive the chip.

import { pageText } from './surface-helpers';

export const LEVER_LIVE = 'live';
export const LEVER_WIRED_OFF = 'wired_off';
export const LEVER_ABSENT = 'absent';

// Flip to true ONLY alongside a measurement showing the preference survives
// refinement. Doing so is a money change: shipping the preferred plan instead of
// the refined one measured -14.62% revenue over 30 operator channel-days.
const PREFERENCE_SURVIVES_REFINEMENT = false;

export function leverChipText(state, locale) {
  if (state === LEVER_LIVE) return pageText(locale, 'Live', 'פעיל');
  if (state === LEVER_ABSENT) return pageText(locale, 'Not built', 'לא קיים');
  return pageText(locale, 'Wired, off', 'מחווט-כבוי');
}

// The reason the whole family is off, regardless of data. Kept as one string per
// language so the UI wraps it, never the source.
function mechanismReason(locale) {
  return pageText(
    locale,
    'The plan is refined for revenue and retention after this preference is applied, and that step optimises the preference back out, so it does not change the schedule today.',
    'הלוח עובר ליטוש להכנסה ולשימור אחרי שההעדפה מוחלת, והשלב הזה ממטב את ההעדפה בחזרה החוצה, ולכן היא אינה משנה את הלוח היום.',
  );
}

function dataReason(lever, locale) {
  if (lever === 'pacing') {
    return pageText(locale, 'No campaign flights uploaded yet.', 'טרם הועלו יעדי דילוור לקמפיינים.');
  }
  if (lever === 'advertiser') {
    return pageText(locale, 'No scoped advertiser conditions defined yet.', 'טרם הוגדרו תנאים ממוקדים למפרסמים.');
  }
  return pageText(locale, 'No inventory rows load from the uploaded file.', 'לא נטענות שורות מלאי מהקובץ שהועלה.');
}

// lever: 'pacing' | 'advertiser' | 'inventory'
// hasData: true / false / null when genuinely unknown (never guess a green)
export function leverState(lever, hasData) {
  const state = PREFERENCE_SURVIVES_REFINEMENT && hasData === true ? LEVER_LIVE : LEVER_WIRED_OFF;
  return { lever, state, hasData };
}

// The lines a surface should show under its heading, most specific last.
export function leverReasons(lever, hasData, locale) {
  const reasons = [mechanismReason(locale)];
  if (hasData === false) reasons.push(dataReason(lever, locale));
  if (hasData === null) {
    reasons.push(pageText(
      locale,
      'Whether this lever has data could not be read, so its state is reported as unknown rather than assumed.',
      'לא ניתן היה לקרוא אם יש למנוף הזה נתונים, ולכן מצבו מדווח כלא ידוע ולא מונח.',
    ));
  }
  return reasons;
}
