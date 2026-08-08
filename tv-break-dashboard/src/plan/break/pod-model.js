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

// A count with the word that agrees with it. Both languages read wrong when a
// plural noun stands beside the number one, and a pod of one spot is common.
export function countLabel(count, locale, enOne, enMany, heOne, heMany) {
  const total = Number(count) || 0;
  const he = locale === 'he';
  const word = total === 1 ? (he ? heOne : enOne) : (he ? heMany : enMany);
  return `${total} ${word}`;
}

// A duration or a gap, in the trade's own unit. There are no milliseconds in a
// traffic file, so seconds are printed whole.
//
// RULED 2026-08-09, after measuring rather than choosing. This product printed
// seconds three ways for one unit: `שניות` in twenty places, the standard Hebrew
// abbreviation `שנ'` in two, and a bare Latin `s` here. A Latin unit letter on a
// Hebrew screen is the one of the three that is simply wrong, and it was the one
// on the compact surfaces: the copy-check badge, the elapsed column and the
// length column all read `35s` to a Hebrew reader.
//
// The ruling is not "always spell it out". A column is not a sentence, and
// Hebrew has an accepted abbreviation, so this compact form uses it and prose
// keeps `שניות`. What is not allowed is a fourth form, or a Latin letter
// standing in for a Hebrew one.
export function secondsLabel(seconds, locale) {
  const number = Number(seconds);
  if (!Number.isFinite(number)) return pageText(locale, 'unknown', 'לא ידוע');
  return pageText(locale, `${Math.round(number)}s`, `${Math.round(number)} שנ'`);
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
      label: pageText(locale, "From the break's start to the last spot's end", 'מתחילת הברייק ועד סוף התשדיר האחרון'),
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

// The holes inside the uncovered figure, which a total cannot see: the dead air
// before the first spot, the gaps between spots, and any collisions. The server
// already measures all three; this only says them, because a pod reporting a
// large uncovered figure with no account of where it sits leaves the reader to
// find it by hand.
export function continuityRows(pod, locale) {
  const arithmetic = (pod && pod.arithmetic) || {};
  const head = arithmetic.gap_before_first_spot;
  const gaps = arithmetic.gaps_between_spots;
  const overlaps = arithmetic.overlaps_between_spots;
  const rows = [];
  if (head && head.state === 'real' && Math.round(Number(head.seconds)) > 0) {
    rows.push({
      key: 'head',
      count: pageText(locale, 'Before the first spot', 'לפני התשדיר הראשון'),
      seconds: secondsLabel(head.seconds, locale),
    });
  }
  if (gaps && gaps.count > 0) {
    rows.push({
      key: 'gaps',
      count: countLabel(gaps.count, locale, 'gap', 'gaps', 'רווח', 'רווחים'),
      seconds: secondsLabel(gaps.seconds, locale),
    });
  }
  if (overlaps && overlaps.count > 0) {
    rows.push({
      key: 'overlaps',
      count: countLabel(overlaps.count, locale, 'overlap', 'overlaps', 'חפיפה', 'חפיפות'),
      seconds: secondsLabel(overlaps.seconds, locale),
    });
  }
  return rows;
}

// The columns the spot list carries, headed. Two of them are bare numbers that
// mean different things, the running order and the position the traffic file declares,
// and side by side with no heading they are one number twice. Copy check sits
// between the copy version and the length, because that is the comparison it
// reports on: what the two figures either side of it say against each other.
export function spotColumns(locale) {
  const he = locale === 'he';
  const pick = (en, hebrew) => (he ? hebrew : en);
  return [
    { key: 'grip', label: '' },
    { key: 'seq', label: pick('Order', 'סדר') },
    { key: 'pos', label: pick('Pos', 'מיקום') },
    { key: 'advertiser', label: pick('Advertiser', 'מפרסם') },
    { key: 'creative', label: pick('Copy version', 'שם גרסה') },
    { key: 'copy', label: pick('Copy check', 'בדיקת גרסה') },
    { key: 'house', label: pick('House number', 'מספר בית') },
    { key: 'type', label: pick('Spot type', 'סוג תשדיר') },
    { key: 'clock', label: pick('Declared start', 'התחלה מוצהרת') },
    { key: 'elapsed', label: pick('Elapsed', 'זמן שחלף') },
    { key: 'len', label: pick('Length', 'אורך') },
  ];
}

// How far into the break each spot starts, counted by the current on-screen
// order rather than by the file's own absolute clock. The declared start
// column above answers what time the file names for a spot, which is fixed
// and does not move with a reorder; this answers how much of the break has
// already aired by the time this spot begins, which is exactly what moves
// when a spot moves. A traffic operator working a pod of thirty eight spots
// otherwise has to add durations by eye to answer it. Once one spot's own
// length is unknown, every figure after it would understate the true elapsed
// time by that unknown amount, so this reports unknown from that point on
// rather than a number that quietly assumes the missing length was zero.
export function elapsedSeconds(spots) {
  const list = Array.isArray(spots) ? spots : [];
  let running = 0;
  let known = true;
  return list.map((spot) => {
    const before = known ? running : null;
    const duration = spot.duration || {};
    if (duration.state === 'real') {
      running += Number(duration.seconds) || 0;
    } else {
      known = false;
    }
    return before;
  });
}

// How many of this pod's spots carry a copy version the copy check could run
// on at all, against the total. Most copy versions in the shipped file carry
// no parseable length, so a verification count of zero on a pod like that
// means nothing was checkable, not that everything passed, and the two read
// as the same quiet badge unless this is said.
export function copyCheckCoverage(spots) {
  const list = Array.isArray(spots) ? spots : [];
  const checked = list.filter((item) => {
    const state = (item.copy_length || {}).state;
    return state && state !== 'none';
  }).length;
  return { checked, total: list.length };
}

// The copy check badge: agrees is quiet, disagrees is the one thing a traffic
// operator must not miss, and no declared length is the honest answer for most
// rows, so it reads as absence rather than as a third kind of warning.
export function copyLengthBadge(copyLength, locale) {
  const state = (copyLength && copyLength.state) || 'none';
  if (state === 'agrees') return { state, text: pageText(locale, 'Agrees', 'תואם') };
  if (state === 'disagrees') {
    // The same seconds mark the length column uses. Three different marks for
    // one unit on one screen is three units as far as a reader is concerned.
    const copySeconds = secondsLabel(copyLength.copy_seconds, locale);
    return {
      state,
      text: pageText(locale, `Copy says ${copySeconds}`, `הגרסה נוקבת ב-${copySeconds}`),
      detail: (locale === 'he' && copyLength.reason_he) || copyLength.reason || '',
    };
  }
  return { state: 'none', text: pageText(locale, 'no length in copy', 'אין אורך בגרסה') };
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

// Where a drop should actually land, given the row the cursor is over and
// which half of it the cursor is on. `to` is expressed against the array after
// `from` has already been removed, which is what moveSpot expects, so the
// index shifts by one when `from` sits before the target and edge is 'before'
// or the target sits before `from` and edge is 'after'. This is what draws a
// single insertion line rather than a drop outcome that depends on which
// direction the drag happened to come from.
export function dropIndexFor(from, over, edge) {
  if (from === over) return from;
  if (edge === 'before') return from < over ? over - 1 : over;
  return from < over ? over : over + 1;
}

// A verdict on the pod's own current order against the positions the file
// declares, said in the reader's own language and named against the row it
// belongs to rather than left as a bare pair of numbers.
export function positionViolationMap(spots) {
  return new Map(positionViolations(spots).map((item) => [item.spot_key, item]));
}

// The same ranking check the server runs, ported so a drag or a keyboard move
// shows its own consequence before a save spends a write on it. JS-7's own
// sequence is reorder, then lock; a reader should learn a move broke the
// declared order at the moment they made it, not after they committed it.
// A numbered position names a spot's rank among the priced, positioned spots
// in this pod's own current order, not its absolute place in the whole pod,
// so unpositioned and unknown spots are skipped entirely here too.
export function positionViolations(spots) {
  const list = Array.isArray(spots) ? spots : [];
  const ranked = list.filter((item) => ['ordinal', 'last'].includes((item.position || {}).kind));
  const total = ranked.length;
  const violations = [];
  ranked.forEach((item, index) => {
    const rank = index + 1;
    const position = item.position || {};
    if (position.kind === 'ordinal') {
      const contracted = position.ordinal;
      if (contracted !== null && contracted !== undefined && Number(contracted) !== rank) {
        violations.push({ spot_key: item.spot_key, contracted_position: String(contracted), current_rank: rank });
      }
    } else if (position.kind === 'last' && total && rank !== total) {
      violations.push({ spot_key: item.spot_key, contracted_position: 'L', current_rank: rank });
    }
  });
  return violations;
}

// The traffic file declares a position; it does not say what a campaign bought.
// Measured on the shipped file: every pod's positions run 1 to N contiguously in
// the file's own order and reach 26, which is a settled rank inside the priced
// block, not a purchased slot. So the label says what the file says.
export function positionViolationLabel(violation, locale) {
  if (!violation) return '';
  return pageText(
    locale,
    `The traffic file places this spot at ${violation.contracted_position}, now ranked ${violation.current_rank}`,
    `קובץ הטראפיק מציב את התשדיר הזה במיקום ${violation.contracted_position}, וכעת הוא מדורג ${violation.current_rank}`,
  );
}

// Every real error the pod on screen carries, computed over whatever order
// the surface currently holds rather than only the order it was served in.
// The copy and length checks are already per-spot facts that a reorder cannot
// change; the position check is the one that can, so it is reranked here with
// the same function the server runs. This is what makes the verification tile
// live during a drag rather than only after a save.
export function liveVerificationErrors(spots) {
  const list = Array.isArray(spots) ? spots : [];
  const errors = [];
  list.forEach((item) => {
    const copyCheck = item.copy_length || {};
    if (copyCheck.state === 'disagrees') {
      errors.push({
        kind: 'copy_length',
        spot_key: item.spot_key,
        detail: `The copy version names ${copyCheck.copy_seconds}s, booked at ${copyCheck.booked_seconds}s`,
        detail_he: `שם הגרסה נוקב ב-${copyCheck.copy_seconds} שנ', בעוד ההזמנה היא ${copyCheck.booked_seconds} שנ'`,
      });
    }
    if ((item.duration || {}).state !== 'real') {
      errors.push({
        kind: 'missing_length',
        spot_key: item.spot_key,
        detail: 'This spot declares no length.',
        detail_he: 'לתשדיר הזה אין אורך מוצהר.',
      });
    }
  });
  positionViolations(list).forEach((violation) => {
    errors.push({
      kind: 'position_order',
      spot_key: violation.spot_key,
      detail: `The traffic file places this spot at ${violation.contracted_position}, currently ranked ${violation.current_rank} among the positioned spots`,
      detail_he: `קובץ הטראפיק מציב את התשדיר הזה במיקום ${violation.contracted_position}, וכעת הוא מדורג ${violation.current_rank} מבין התשדירים הממוקמים`,
    });
  });
  return errors;
}

// The error list the trade's own verification step calls for, named against
// the spot it belongs to rather than left as a count with nothing to open.
// Takes the spots in the order the surface currently shows, so the list a
// person reads is the list that would be saved if they saved right now.
export function verificationList(spots, locale) {
  const list = Array.isArray(spots) ? spots : [];
  const errors = liveVerificationErrors(list);
  const bySpot = new Map(list.map((spot) => [spot.spot_key, spot]));
  return errors.map((error, index) => {
    const spot = bySpot.get(error.spot_key);
    return {
      key: `${error.kind}-${error.spot_key}-${index}`,
      spotKey: error.spot_key,
      advertiser: spot ? fieldText(spot.advertiser, locale) : '',
      detail: (locale === 'he' && error.detail_he) || error.detail || '',
      kind: error.kind,
    };
  });
}

// How urgently a pod in the day's list needs a traffic operator's attention:
// a real error outranks a large uncovered figure, because an error is a
// mistake and an uncovered figure can be an honest, accepted gap. Used only to
// offer a sort; the default view stays in time order, which is how a traffic
// log is read.
export function podAttentionScore(pod) {
  const errors = Number((pod && pod.verification && pod.verification.count) || 0);
  const unfilled = (pod && pod.arithmetic && pod.arithmetic.unfilled) || {};
  const magnitude = unfilled.state === 'real' ? Math.abs(Number(unfilled.seconds) || 0) : 0;
  return errors * 100000 + magnitude;
}

// Whether this pod is locked, and by whom, read off the one place that fact
// lives so a caller never has to reach into `order` directly.
export function lockState(pod) {
  const order = (pod && pod.order) || {};
  return {
    locked: order.locked === true,
    lockedAt: order.locked_at || '',
    lockedBy: order.locked_by || '',
  };
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
