// What a save actually cost, measured after it landed.
//
// The readout's "change from the saved plan" is a PREDICTION. It prices the
// arrangement on screen against the plan's own basis while holding the break
// counts the plan already chose, which is what makes it answer in under a
// millisecond and what lets it be live under the hand during a drag.
//
// A save does something that prediction cannot see. It writes a restriction, the
// engine then runs the whole day again with that restriction in force, and on
// that second run it is free to place the rest of the day differently.
//
// Measured through the engine's own seam on רשת 13 / 2024-11-01, one process,
// same inputs, seconds apart: pinning a break at exactly the offset, duration and
// gold flag the plan had already given it moves the day from 1,067,845.55 to
// 1,037,270.00. That is 30,575.55 ILS against a prediction of 0.00, and the
// engine's objective falls from 0.541628827 to 0.537917737 even though the
// unpinned arrangement is still feasible under the pin. So the gap is the
// engine's search landing somewhere else, not a price the move carries.
//
// Before this module existed the board re-read the day after a save and the
// readout re-based itself against the fresh plan, so it printed a change of zero
// both before and after and the money a save cost was unobservable anywhere in
// the product. Now the board keeps the totals it had, compares them to the totals
// it got, and prints the difference beside the prediction it made.

// The same threshold the readout already uses to decide whether money moved. Half
// a cent, so display rounding can never register as a divergence.
export const AGREEMENT_TOLERANCE_ILS = 0.005;

export const DELTA_KEYS = ['revenue', 'retention', 'breaks', 'ad_seconds', 'gold_breaks'];

// An absent figure stays absent. Written out rather than leaning on Number,
// because Number(null) is 0 and Number('') is 0, and a missing prediction that
// silently became a prediction of zero is exactly the class of defect this
// module was built to close.
function num(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

// The difference between two totals payloads, key by key. A key missing on either
// side comes back null rather than zero, because an absent figure is not a zero.
export function totalsDelta(before, after) {
  const out = {};
  DELTA_KEYS.forEach((key) => {
    const a = num(before && before[key]);
    const b = num(after && after[key]);
    out[key] = a === null || b === null ? null : Number((b - a).toFixed(6));
  });
  return out;
}

// How much of the day the engine placed differently on that second run. Counted
// from the two break lists themselves: a break that changed its clock or its
// length, one the second run dropped, and one it added.
export function rearrangement(beforeBreaks, afterBreaks) {
  const before = new Map((beforeBreaks || []).map((row) => [row.break_id, row]));
  const after = new Map((afterBreaks || []).map((row) => [row.break_id, row]));
  const touched = new Set();
  let moved = 0;
  let added = 0;
  let removed = 0;
  before.forEach((row, id) => {
    const next = after.get(id);
    if (!next) {
      removed += 1;
      touched.add(row.segment_id);
      return;
    }
    const shifted = Math.abs(Number(next.start_seconds) - Number(row.start_seconds)) > 0.05;
    const resized = Math.abs(Number(next.duration_seconds) - Number(row.duration_seconds)) > 0.05;
    if (shifted || resized) {
      moved += 1;
      touched.add(row.segment_id);
    }
  });
  after.forEach((row, id) => {
    if (before.has(id)) return;
    added += 1;
    touched.add(row.segment_id);
  });
  return { moved, added, removed, programmes: touched.size, changed: moved + added + removed };
}

// One settled act: what the board predicted, what the plan became, and whether
// those two agree. Nothing here is rounded for display and nothing is invented:
// both sides are the engine's own day totals, read from the same route.
export function settlementOf({ act, basis, before, after, beforeBreaks, afterBreaks, predictedRevenue }) {
  if (!before || !after) return null;
  const realised = totalsDelta(before, after);
  const predicted = num(predictedRevenue);
  const difference = predicted === null || realised.revenue === null
    ? null
    : Number((realised.revenue - predicted).toFixed(6));
  let verdict = 'unknown';
  if (difference !== null) {
    verdict = Math.abs(difference) <= AGREEMENT_TOLERANCE_ILS ? 'agreed' : 'diverged';
  }
  return {
    act,
    basis: basis || null,
    before,
    after,
    realised,
    predicted,
    difference,
    verdict,
    rearranged: rearrangement(beforeBreaks, afterBreaks),
  };
}

// The prediction that belongs to an undo: the exact inverse of what the save it
// reverses turned out to cost. With no measured save behind it there is no
// prediction to make, and null is the honest answer rather than zero.
export function inverseOf(settlement) {
  if (!settlement || settlement.act !== 'save') return null;
  const realised = settlement.realised && settlement.realised.revenue;
  if (!Number.isFinite(realised)) return null;
  // Negated zero is negative zero, and the currency formatter prints it as
  // -0 ILS. Measured on screen: a save that cost nothing offered an undo whose
  // prediction read minus zero, which reads as a loss of an amount too small to
  // show rather than as the nothing it is.
  const inverse = -realised;
  return inverse === 0 ? 0 : inverse;
}
