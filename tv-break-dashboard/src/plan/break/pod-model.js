// The pod: the ordered list of spots inside one break, and the arithmetic on it.
//
// Everything here is pure, so the surface above it draws and the rules live in
// one place a test can drive without a browser.
//
// Two rules run through all of it.
//
// A figure the payload marked unknown is never printed as a number. The server
// distinguishes a length it read from a length that is absent, and a surface that
// rendered the absent one as 0 would turn a missing declaration into a claim that
// the spot runs for no time. So a reader gets the word and the reason instead.
//
// Positions are 1 to 5 plus L, and L is not the fifth ordinal. One campaign can
// hold both the first and the last spot of a break, which is two positions in one
// pod, so Last carries its own code and never a number.

import { pageText } from '../../shell/format';

export const LAST_CODE = 'L';

// A duration or a gap, in the trade's own unit. There are no milliseconds in a
// traffic file, so seconds are printed whole.
export function secondsLabel(seconds, locale) {
  const number = Number(seconds);
  if (!Number.isFinite(number)) return pageText(locale, 'unknown', 'לא ידוע');
  return `${Math.round(number)}s`;
}

// Seconds as a clock, for the totals a person reads against a break length.
export function clockLabel(seconds, locale) {
  const number = Number(seconds);
  if (!Number.isFinite(number)) return pageText(locale, 'unknown', 'לא ידוע');
  const whole = Math.max(0, Math.round(number));
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, '0')}`;
}

// One spot's position, said the way the trade says it. An unrequested position
// is a real reading and says so; a missing one is unknown and says that instead.
export function positionLabel(position, locale) {
  if (!position || position.state !== 'real') return pageText(locale, 'unknown', 'לא ידוע');
  if (position.kind === 'last') return pageText(locale, 'Last', 'אחרון');
  if (position.kind === 'unpositioned') return pageText(locale, 'no position requested', 'לא התבקש מיקום');
  return String(position.code);
}

// The short badge the row carries. Last is L, an ordinal is its number, and an
// unpositioned spot carries a dash rather than a zero it never held.
export function positionCode(position) {
  if (!position || position.state !== 'real') return '?';
  if (position.kind === 'last') return LAST_CODE;
  if (position.kind === 'unpositioned') return '-';
  return String(position.code);
}

// A field the payload either read or named as missing.
export function fieldText(field, locale) {
  if (!field) return pageText(locale, 'unknown', 'לא ידוע');
  if (field.state === 'real' && field.value) return field.value;
  return pageText(locale, 'unknown', 'לא ידוע');
}

export function fieldReason(field, locale) {
  if (!field || field.state === 'real') return '';
  return (locale === 'he' && field.reason_he) || field.reason || '';
}

// A figure the server computed, or the reason there is none. Never a zero.
export function figureText(figure, locale) {
  if (!figure || figure.state !== 'real' || !Number.isFinite(Number(figure.seconds))) {
    return pageText(locale, 'unknown', 'לא ידוע');
  }
  return secondsLabel(figure.seconds, locale);
}

// The three figures a traffic operator reads before anything else: what the pod
// declares, how long it runs, and the difference. Each carries its own basis, so
// the surface prints the basis beside the figure rather than in a tooltip.
export function arithmeticRows(pod, locale) {
  const arithmetic = (pod && pod.arithmetic) || {};
  const rows = [
    {
      key: 'load',
      label: pageText(locale, 'Sum of the spots', 'סכום התשדירים'),
      figure: arithmetic.declared_load,
    },
    {
      key: 'span',
      label: pageText(locale, 'How long the pod runs', 'משך התוכן בפועל'),
      figure: arithmetic.span,
    },
    {
      key: 'unfilled',
      label: pageText(locale, 'Not covered by a spot', 'לא מכוסה בתשדיר'),
      figure: arithmetic.unfilled,
    },
  ];
  return rows.map((row) => ({
    ...row,
    text: figureText(row.figure, locale),
    basis: (row.figure && ((locale === 'he' && row.figure.basis_he) || row.figure.basis)) || '',
  }));
}

// The declared break length against the sum of its spots. This is the whole point
// of the surface, so when there is no declared length it says so in words and
// prints nothing that could be mistaken for a measurement.
export function declaredVerdict(pod, locale) {
  const against = (pod && pod.against_declared) || {};
  const declared = (pod && pod.declared_break_length) || {};
  if (against.state !== 'real') {
    return {
      state: 'unavailable',
      headline: pageText(locale, 'No declared break length', 'אין אורך ברייק מוצהר'),
      detail: (locale === 'he' && against.reason_he) || against.reason || '',
      covers: Array.isArray(declared.plan_covers) ? declared.plan_covers : [],
    };
  }
  const words = {
    exact: pageText(locale, 'The spots fill the break exactly', 'התשדירים ממלאים את הברייק במדויק'),
    gap: pageText(locale, 'Short of the declared length', 'חסר מול האורך המוצהר'),
    overflow: pageText(locale, 'Over the declared length', 'חריגה מהאורך המוצהר'),
  };
  return {
    state: 'real',
    verdict: against.verdict,
    headline: words[against.verdict] || '',
    seconds: against.seconds,
    declaredSeconds: against.declared_seconds,
    loadSeconds: against.load_seconds,
  };
}

// Move one spot to a new place in the order, and give back the keys in that
// order. Out of range indices leave the order alone rather than throwing, because
// a drag that ends outside the list is a person changing their mind.
export function moveSpot(keys, from, to) {
  const list = Array.isArray(keys) ? keys.slice() : [];
  if (!Number.isInteger(from) || !Number.isInteger(to)) return list;
  if (from < 0 || from >= list.length || to < 0 || to >= list.length || from === to) return list;
  const [moved] = list.splice(from, 1);
  list.splice(to, 0, moved);
  return list;
}

// The spots in the order the surface currently holds, which is the saved or file
// order until somebody drags, and the dragged order after that.
export function spotsInOrder(pod, keys) {
  const spots = (pod && Array.isArray(pod.spots) ? pod.spots : []);
  if (!Array.isArray(keys) || !keys.length) return spots;
  const byKey = new Map(spots.map((spot) => [spot.spot_key, spot]));
  const ordered = keys.map((key) => byKey.get(key)).filter(Boolean);
  return ordered.length === spots.length ? ordered : spots;
}

export function orderKeys(pod) {
  return (pod && Array.isArray(pod.spots) ? pod.spots : []).map((spot) => spot.spot_key);
}

// Whether the order on screen differs from the one the pod was served in. A save
// button that offers to write an unchanged order is a button that spends a write
// on nothing.
export function orderChanged(pod, keys) {
  const served = orderKeys(pod);
  if (!Array.isArray(keys) || keys.length !== served.length) return false;
  return keys.some((key, index) => key !== served[index]);
}

// Where the order on screen came from, in the reader's own language.
export function orderNote(pod, locale) {
  const order = (pod && pod.order) || {};
  return (locale === 'he' && order.reason_he) || order.reason || '';
}
