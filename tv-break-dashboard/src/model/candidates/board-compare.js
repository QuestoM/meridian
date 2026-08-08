// The two pure functions the board's honesty rests on, plus the one small piece
// of browser work the panel does, all kept out of the component.
//
// `freshness` is the whole answer to "is this comparison about the files on the
// server now". It is tri-state on purpose: a route that did not answer is
// unknown, and unknown is not stale. Nothing here guesses, and nothing here
// falls back to showing the figures as though the question had been answered.
//
// `sortRows` never moves a row that has no figure to the top by treating its
// absence as a zero. A missing measurement sorts last in both directions,
// because it is not a small number, it is not a number.

export const SORTS = {
  rmse: (row) => row.rmse,
  rmse_delta: (row) => row.rmse_delta,
  paired_statistic: (row) => row.paired_statistic,
  cells: (row) => (row.cells || {}).cancelled_share,
  money: (row) => {
    const money = row.money || {};
    return money.state === 'measured' ? money.revenue_delta : null;
  },
};

// How large a candidate's movement is against the noise that candidate sits in.
//
// The denominator is the row's OWN fold dispersion, not a scale shared across
// the table, because that is the figure the verdict is decided against: a
// movement smaller than the difference between one temporal fold and the next is
// a property of which month was looked at and not of the model. So a bar that
// fills is a movement the size of its own noise, and every bar on this tree is a
// fraction of one.
//
// Null rather than zero when there is no dispersion to divide by. A row whose
// dispersion is zero has no noise to be compared against, and drawing an empty
// bar there would state that its movement is small rather than that there is
// nothing to state.
export function movementShare(row) {
  const moved = Math.abs(Number((row || {}).rmse_delta));
  const noise = Math.abs(Number((row || {}).fold_dispersion));
  if (!Number.isFinite(moved) || !Number.isFinite(noise) || noise <= 0) return null;
  return moved / noise;
}

// The live model's error as a share of the spread of the thing it predicts.
//
// One bar for the whole evaluation rather than one per row, because all six
// artifacts sit within a thousandth of each other on a scale of a quarter: six
// bars would be six identical full ones. Rendered once, under the figure it
// draws, it says the thing the table cannot: every model here, the live one
// included, is close to predicting nothing but the mean.
export function spreadShare(board) {
  const error = Number(((board || {}).shipped || {}).rmse);
  const spread = Number(((board || {}).evaluation || {}).target_sd);
  if (!Number.isFinite(error) || !Number.isFinite(spread) || spread <= 0) return null;
  return error / spread;
}

export function sortRows(rows, sort) {
  const read = SORTS[sort.key];
  if (!read) return rows.slice();
  const direction = sort.ascending ? 1 : -1;
  return rows.slice().sort((left, right) => {
    const a = read(left);
    const b = read(right);
    const aMissing = a === null || a === undefined || Number.isNaN(Number(a));
    const bMissing = b === null || b === undefined || Number.isNaN(Number(b));
    if (aMissing && bMissing) return String(left.id).localeCompare(String(right.id));
    if (aMissing) return 1;
    if (bMissing) return -1;
    return (Number(a) - Number(b)) * direction;
  });
}

// Bring the evidence panel into view when an artifact is picked.
//
// On a microtask, which is the first moment the panel is its full height: the
// page is shorter than the panel until then, so a synchronous scroll runs out
// of page and lands part of the way. A frame callback was tried first and is
// worse than either, because frames are suspended entirely in a window that is
// not in front, so the same click scrolled or did not depending on where the
// window was.
//
// Silent when there is no microtask queue and silent when the element cannot
// scroll itself, so no environment is required to have either.
export function reveal(ref, selector) {
  if (typeof queueMicrotask !== 'function') return;
  queueMicrotask(() => {
    const root = ref && ref.current;
    const node = root && root.querySelector ? root.querySelector(selector) : null;
    if (node && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ block: 'start' });
    }
  });
}

function short(digest) {
  return String(digest || '').slice(0, 12) || 'none';
}

export function freshness(board, live) {
  if (!live || live.status === 'loading') {
    return { state: 'unknown', moved: [], detail: '', checking: true };
  }
  if (live.status !== 'ok' || !live.payload) {
    return { state: 'unknown', moved: [], detail: live.detail || live.status, checking: false };
  }
  const payload = live.payload;
  const served = new Map((payload.candidates || []).map((row) => [row.id, row.sha256]));
  const shipped = ((payload.model_version || {}).artifacts || {}).retention || {};
  const moved = [];
  const reference = board.shipped || {};
  if (reference.sha256 && shipped.sha256 && reference.sha256 !== shipped.sha256) {
    moved.push({ id: 'shipped', measured_on: short(reference.sha256), served_now: short(shipped.sha256) });
  }
  (board.candidates || []).forEach((row) => {
    if (!served.has(row.id)) {
      moved.push({ id: row.id, measured_on: short(row.sha256), served_now: 'absent' });
      return;
    }
    if (served.get(row.id) !== row.sha256) {
      moved.push({ id: row.id, measured_on: short(row.sha256), served_now: short(served.get(row.id)) });
    }
  });
  const known = new Set((board.candidates || []).map((row) => row.id));
  (payload.candidates || []).forEach((row) => {
    if (!known.has(row.id)) {
      moved.push({ id: row.id, measured_on: 'absent', served_now: short(row.sha256) });
    }
  });
  return { state: moved.length ? 'stale' : 'current', moved, detail: '', checking: false };
}
