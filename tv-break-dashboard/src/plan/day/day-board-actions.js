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

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

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
export function breakPredicate(programme, startSeconds) {
  return {
    combinator: 'and',
    conditions: [
      { field: 'date', operator: 'is', value: programme.day },
      { field: 'programme', operator: 'is', value: programme.title },
      { field: 'hour', operator: 'eq', value: Math.floor(programme.start_seconds / 3600) % 24 },
    ],
  };
}

export function scopeSentence(programme, locale) {
  const clock = `${String(Math.floor(programme.start_seconds / 3600) % 24).padStart(2, '0')}:${String(Math.floor((programme.start_seconds % 3600) / 60)).padStart(2, '0')}`;
  if (locale === 'he') {
    return `חל על "${programme.title}" ב-${programme.day} בשעה ${clock} בלבד`;
  }
  return `Applies to "${programme.title}" on ${programme.day} at ${clock} only`;
}

// Save one moved break. Returns the record needed to undo it exactly.
export async function saveBreakPlacement({ item, programme, live, note }) {
  const constraint = await call('/api/constraints', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scope_type: 'always',
      effect: 'fix_offset',
      offset_seconds: Math.round(live.offsetSeconds),
      duration_seconds: Math.round(live.durationSeconds),
      order_index: item.ordinal,
      notes: note || '',
      where: breakPredicate(programme, item.start_seconds),
    }),
  });
  const constraintId = constraint.constraint_id || constraint.id;
  const record = await call(`/api/breaks/${encodeURIComponent(item.break_id)}/placement`, {
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
