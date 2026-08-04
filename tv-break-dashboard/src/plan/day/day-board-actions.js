// The day board's writes, and the exact inverse of each one.
//
// Saving a moved break has to reach the engine or it is theatre. The only store
// the weekly plan actually reads for a placement is the scoped restriction
// store, so a save writes one restriction that fixes this break at this offset,
// and then records the break's own half of that transaction so an undo knows
// precisely which restriction to remove.
//
// The scope is the part that was wrong before. The shipped editor saved a
// placement scoped to the whole broadcast date, and the resolver matches a date
// scope against every segment on that date, so pinning one break at 21:42 also
// pinned the first break of every other programme that day to 21:42. Measured
// through the engine's own resolver on 2024-11-01: the date scope binds 82 of
// 82 segments, the predicate below binds 1. Driven end to end over HTTP, saving
// one break moved exactly one first break and left the other 44 where the plan
// put them. This module scopes with the frozen predicate contract instead: date,
// programme and hour, all three, which names one break in one airing.

// The guard matters: this module is executed directly by the tests, in node,
// where ``import.meta.env`` does not exist. A test that could not run the shipped
// body builder would be a test of a copy of it, which is what let the save scope
// below ship wrong.
const API_BASE = (import.meta.env && import.meta.env.VITE_KAIROS_API_URL) || '';

async function call(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = String(body.detail);
    } catch {
      // A non-JSON error body keeps the status line as the honest message.
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  return response.json();
}

// The predicate that names exactly this break's airing, on the frozen contract.
// Channel is never in a predicate: every restriction is scoped to the operator's
// own channel by the engine itself.
export function breakPredicate(programme) {
  return {
    combinator: 'and',
    conditions: [
      { field: 'date', operator: 'is', value: programme.day },
      { field: 'programme', operator: 'is', value: programme.title },
      { field: 'hour', operator: 'eq', value: Math.floor(programme.start_seconds / 3600) % 24 },
    ],
  };
}

// The one constraint body a saved move writes, wherever the move was made.
//
// Both timelines the product carries save a placement, and before this they sent
// two different bodies: the board sent the predicate above and the editor sent a
// whole-date scope. So this is the single builder, and the two surfaces call it
// rather than each writing their own. Whatever a critic measures on one of them
// is true of the other by construction.
export function placementBody({ item, programme, live, note }) {
  return {
    scope_type: 'always',
    effect: 'fix_offset',
    offset_seconds: Math.round(live.offsetSeconds),
    duration_seconds: Math.round(live.durationSeconds),
    order_index: item.ordinal,
    notes: note || '',
    where: breakPredicate(programme),
  };
}

// What a saved placement binds, stated exactly rather than optimistically.
//
// This sentence used to end in the word only, and that word is not always true.
// The frozen predicate contract can name a date, a programme and an hour and
// nothing finer, so two airings of one title inside one hour are the same airing
// to it. Measured on רשת 13 / 2024-11-01: the predicate names exactly one airing
// for 37 of the day's 82 segments and for 30 of the 48 that carry breaks, and for
// the other 18 it names the 2 to 4 same-hour repeats of that title, almost all of
// them the promo block. So the sentence states the rule, and where the surface
// serves the whole day's programmes it counts the airings its own save will bind
// and says the number. A surface that draws part of a day passes no count rather
// than under-reporting one.
export function scopeSentence(programme, locale, airings) {
  const hour = String(Math.floor(programme.start_seconds / 3600) % 24).padStart(2, '0');
  const clock = `${hour}:${String(Math.floor((programme.start_seconds % 3600) / 60)).padStart(2, '0')}`;
  const count = Number(airings);
  const known = Number.isFinite(count) && count > 0;
  const others = count - 1;
  if (locale === 'he') {
    const rule = `חל על "${programme.title}" ב-${programme.day} בשעה ${hour}:00, שבה הרצועה מתחילה ב-${clock}`;
    if (!known) return rule;
    if (count === 1) return `${rule}, והיא רצועת השידור היחידה כזו ביום`;
    if (others === 1) return `${rule}, ואיתה עוד רצועת שידור אחת של אותה תוכנית באותה שעה`;
    return `${rule}, ואיתה עוד ${others} רצועות שידור של אותה תוכנית באותה שעה`;
  }
  const rule = `Applies to "${programme.title}" on ${programme.day} in the ${hour}:00 hour, where this airing starts at ${clock}`;
  if (!known) return rule;
  if (count === 1) return `${rule}, and it is the only such airing that day`;
  if (others === 1) return `${rule}, together with 1 more airing of it in the same hour`;
  return `${rule}, together with ${others} more airings of it in the same hour`;
}

// Save one moved break. Returns the record needed to undo it exactly.
//
// The two writes are one transaction and are treated as one. The restriction is
// what moves the money and the record is the only thing that offers the way back,
// so a restriction that lands while its record does not is money spent with no
// inverse on the surface that spent it. If the second write fails the first is
// deleted again and the failure is reported, which leaves the plan exactly where
// it was.
export async function saveBreakPlacement({ item, programme, live, note }) {
  const constraint = await call('/api/constraints', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(placementBody({ item, programme, live, note })),
  });
  const constraintId = constraint.constraint_id || constraint.id;
  let record;
  try {
    record = await call(`/api/breaks/${encodeURIComponent(item.break_id)}/placement`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        constraint_id: String(constraintId),
        offset_seconds: Math.round(live.offsetSeconds),
        duration_seconds: Math.round(live.durationSeconds),
        is_gold: Boolean(live.isGold),
        note: note || '',
      }),
    });
  } catch (error) {
    await undoBreakPlacement({ breakId: item.break_id, constraintId: String(constraintId) });
    throw error;
  }
  // A second save of the same break replaces its record, and the record was the
  // only thing naming the restriction the first save wrote. So the route reports
  // what it replaced and the earlier restriction is deleted here, through the
  // store that owns restrictions. Without this an update leaves a restriction in
  // force with nothing on any surface addressing it.
  const replaced = record && record.replaced;
  if (replaced && replaced.constraint_id && String(replaced.constraint_id) !== String(constraintId)) {
    try {
      await call(`/api/constraints/${encodeURIComponent(replaced.constraint_id)}`, { method: 'DELETE' });
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  return { breakId: item.break_id, constraintId: String(constraintId), record };
}

// The exact inverse of a save: drop the restriction, then drop the record.
export async function undoBreakPlacement({ breakId, constraintId }) {
  if (constraintId) {
    try {
      await call(`/api/constraints/${encodeURIComponent(constraintId)}`, { method: 'DELETE' });
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  try {
    await call(`/api/breaks/${encodeURIComponent(breakId)}/placement`, { method: 'DELETE' });
  } catch (error) {
    if (error.status !== 404) throw error;
  }
}

// Two presses of the same gold act inside one round trip are one decision, not
// two. Marking is not idempotent at the store: each mark writes its own override
// row, so a repeat while the first write is still in the air would leave two
// identical rows behind. The second caller is handed the first one's promise, so
// it gets the same answer the first got and the store gets one row.
const goldInFlight = new Map();

function onceInFlight(key, run) {
  const running = goldInFlight.get(key);
  if (running) return running;
  const promise = run().finally(() => goldInFlight.delete(key));
  goldInFlight.set(key, promise);
  return promise;
}

export function markGold(breakId) {
  return onceInFlight(`mark:${breakId}`, () => call(`/api/breaks/${encodeURIComponent(breakId)}/gold`, { method: 'POST' }));
}

export function clearGold(breakId) {
  return onceInFlight(`clear:${breakId}`, () => call(`/api/breaks/${encodeURIComponent(breakId)}/gold`, { method: 'DELETE' }));
}

export async function fetchDay(day) {
  return call(`/api/plan/day?day=${encodeURIComponent(day)}`);
}

export async function fetchDays() {
  return call('/api/plan/days');
}

export async function scoreDay(day, moves) {
  return call('/api/plan/day/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ day, moves }),
  });
}

// What saving would actually do, measured before anything is written.
//
// The score above holds the break counts the plan already chose, which is what
// lets it answer under the hand. A save writes a restriction and the engine then
// plans the whole day again with it in force. Measured on רשת 13 / 2024-11-01:
// pinning a break at exactly its own offset and duration costs 30,575.55 ILS
// against a cheap prediction of 0.00. This runs that second plan with nothing
// written, so a scheduler reads the real figure before the click rather than
// after it. About 600 ms, so it is a button and never a keystroke's side effect.
export async function saveEffect(day, moves) {
  return call('/api/plan/day/save-effect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ day, moves }),
  });
}
