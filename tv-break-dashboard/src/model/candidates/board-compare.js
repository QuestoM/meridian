// Two pure functions the board's honesty rests on, kept out of the component.
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
