// Saving a dragged break from the schedule editor, and the words it reports back.
//
// Split out of ScheduleEditor.jsx, which stood at 466 lines against the 450-line
// law before this wave and is recorded as such in the baseline. Nothing here
// changed behaviour: the body, the endpoint and both messages are the ones the
// editor already sent, moved.
//
// One thing is worth reading before this is reused. The scope this save sends is
// the editor's own, which is the whole broadcast date or every airing of the
// programme. The restriction resolver matches a date scope against every segment
// on that date, so a pin saved with the date scope binds the chosen ordinal in
// every programme on the day, not only in the one that was dragged. Measured
// through the resolver on 2024-11-01, that is 82 of 82 segments against 1 for the
// predicate the day board sends. The day board scopes with the frozen predicate
// contract instead and names one airing; see day-board-actions.js.

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export async function postConstraint(body) {
  const response = await fetch(`${API_BASE}/api/constraints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (response.status === 404) {
    throw new Error('not-found');
  }
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

// The constraint body one dragged break produces, at the editor's chosen scope.
export function pinBody(item, scopeType, startSec, durationSec) {
  const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
  return {
    scope_type: scopeType,
    scope_value: scopeType === 'programme' ? (item.program_key || item.program_title) : (item.date || ''),
    channel: item.channel || '',
    effect: 'FIX_OFFSET',
    offset_seconds: offsetSeconds,
    duration_seconds: durationSec,
    order_index: Number(item.break_num_in_program || 0),
  };
}
