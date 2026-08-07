// Clients, pacing: the readings a row prints, kept out of the components.
//
// Two rules hold everywhere in this file. A figure is never formatted without the
// unit it is in, because rating points and shekels are two different currencies on
// one board. And a missing figure returns null rather than zero, so a component
// renders the reason it was handed instead of drawing a bar out of nothing.
//
// The names of the two things this destination is about come from the product
// vocabulary rather than from here. This surface had authored its own and drifted:
// it said קצב where the controlled word is קצב אספקה. The one module that holds
// those words is imported, and it imports nothing itself, so this file is still
// executable on its own the way its test runs it.
//
// The extension is on the specifier deliberately. Vite resolves an extensionless
// path and node does not, and this module is executed by node directly in
// tests/test_p11_surface_remedy.py so the assertions are about the file the
// browser loads. Measured: without it, node raised ERR_MODULE_NOT_FOUND and all
// six of those tests failed. Every other importer of this module in the product
// already writes the extension.

import { word } from '../../vocabulary.js';

export function term(key, locale) {
  return word(key, locale === 'he' ? 'he' : 'en');
}

export const ON_PACE = 'on_pace';
export const AT_RISK = 'at_risk';
export const BEHIND = 'behind';
export const UNKNOWN = 'unknown';

export const COVERED = 'covered';
export const SHORT_CERTAIN = 'short_certain';
export const NOT_BOOKED_YET = 'not_booked_yet';

export const RATING_POINTS = 'rating_points';
export const ILS = 'ils';

// The order the verdict strip reads in, worst first, which is the order the board
// itself is sorted in so the strip and the list never disagree.
export const VERDICT_ORDER = [BEHIND, AT_RISK, UNKNOWN, ON_PACE];

// The verdicts the board is asking a decision about. The server publishes the
// same list on the payload; this is the fallback for a payload written before it
// did, and the two are asserted equal by the test that reads both.
export const NEEDS_A_DECISION = [BEHIND, AT_RISK];

// The Israeli week reads Sunday first, and a broadcast day is named by its own
// weekday rather than left as an ISO string a reader has to derive one from.
const WEEKDAYS = [
  { en: 'Sunday', he: 'ראשון' },
  { en: 'Monday', he: 'שני' },
  { en: 'Tuesday', he: 'שלישי' },
  { en: 'Wednesday', he: 'רביעי' },
  { en: 'Thursday', he: 'חמישי' },
  { en: 'Friday', he: 'שישי' },
  { en: 'Saturday', he: 'שבת' },
];

export function weekday(isoDate, locale) {
  const parts = String(isoDate || '').slice(0, 10).split('-');
  if (parts.length !== 3) return '';
  const when = new Date(Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2])));
  if (Number.isNaN(when.getTime())) return '';
  const entry = WEEKDAYS[when.getUTCDay()];
  return entry ? pick(locale, entry.en, entry.he) : '';
}

// The two days the week ends on here, marked because a flight day that falls on
// one of them is not a day a media buyer reads the same way as a weekday.
export function isWeekend(isoDate) {
  const parts = String(isoDate || '').slice(0, 10).split('-');
  if (parts.length !== 3) return false;
  const when = new Date(Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2])));
  if (Number.isNaN(when.getTime())) return false;
  return when.getUTCDay() === 5 || when.getUTCDay() === 6;
}

export function pick(locale, en, he) {
  return locale === 'he' ? he : en;
}

// A value joined into a line of prose, isolated so its own order survives the
// direction around it.
//
// It takes a numeral, an ISO date, an identifier or a name, and never a phrase
// that already reads as a sentence in the surrounding language: a caller that
// isolates a figure which already carries its own unit puts the unit in front of
// its own number, which was measured on every one of 56 rows.
//
// The pair is the FIRST-STRONG isolate, U+2068 and U+2069, which is exactly what
// src/shell/bidi.jsx uses and what design-rules.md section 6 points a caller at.
// It infers the run's direction from the run's own first strong character, so
// one call is right for a numeral and right for the Hebrew channel name this
// surface also passes it. The left-to-right isolate this replaced would have
// laid that name out left to right.
//
// It is written here rather than imported because this module is executed
// directly by node in tests/test_p11_surface_remedy.py and bidi.jsx holds JSX
// that node cannot parse. The characters are escapes on purpose: they render as
// nothing, so a literal pair is invisible to review.
const FIRST_STRONG_ISOLATE = '⁨';
const POP_DIRECTIONAL_ISOLATE = '⁩';

export function isolate(text) {
  return `${FIRST_STRONG_ISOLATE}${text}${POP_DIRECTIONAL_ISOLATE}`;
}

export function decimals(value, places, locale) {
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: places,
    minimumFractionDigits: 0,
  }).format(Number(value));
}

// A figure with its unit. Rating points keep one decimal because a tenth of a
// point is a real trading quantity; money is whole shekels.
//
// The Hebrew form isolates the numeral and leaves the unit in the surrounding
// direction, so the phrase reads as Hebrew with a number in it. A caller never
// isolates the result again: doing that put the unit ahead of its own number on
// every row of the board.
export function amount(value, unit, locale) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  if (unit === ILS) {
    const shekels = decimals(Math.round(Number(value)), 0, locale);
    return pick(locale, `ILS ${shekels}`, `${isolate(shekels)} ש"ח`);
  }
  const points = decimals(Number(value), 1, locale);
  return pick(locale, `${points} rating points`, `${isolate(points)} נקודות רייטינג`);
}

// The unit on its own, for a field label that governs a number the reader types
// rather than one this surface prints. It is the same word amount() puts after a
// figure, so a form and the figures above it never name one quantity two ways.
// The ledger read carries no unit vocabulary, and a label that fell back to the
// store's key told a person to type an offer in "rating_points".
export function unitWord(unit, locale) {
  if (unit === ILS) return pick(locale, 'shekels', 'שקלים');
  return pick(locale, 'rating points', 'נקודות רייטינג');
}

// The same figure without its unit, for the left half of a pair that already
// names the unit once on its right. Two units in one phrase reads as two
// different quantities, and the phrase is a comparison of one.
export function bare(value, unit, locale) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  if (unit === ILS) return decimals(Math.round(Number(value)), 0, locale);
  return decimals(Number(value), 1, locale);
}

// A counted figure against the figure it is counted towards, with the unit said
// once at the end where it governs both. The left half is a bare numeral and is
// isolated here; the right half carries a unit and isolated its own numeral in
// amount, so wrapping it again would put that unit in front of its number.
export function pair(counted, goal, unit, locale) {
  const left = bare(counted, unit, locale);
  const right = amount(goal, unit, locale);
  if (left === null || right === null) return null;
  return pick(locale, `${left} of ${right}`, `${isolate(left)} מתוך ${right}`);
}

export function percent(ratio, locale) {
  if (ratio === null || ratio === undefined) return null;
  return `${decimals(Number(ratio) * 100, 0, locale)}%`;
}

// An instant as a person reads it, from the instant the store recorded. The
// offset is kept when the source carries one and never invented when it does
// not, because two instants on one row can be on two different clocks and a
// screen that hides that says they are on one.
export function instant(value) {
  const text = String(value || '').trim();
  if (!text) return '';
  const head = text.slice(0, 16).replace('T', ' ');
  if (head.length < 16) return text;
  const utc = /(\+00:00|Z)$/.test(text);
  return utc ? `${head} UTC` : head;
}

export function vocabularyLabel(entries, value, locale) {
  const found = (entries || []).find((entry) => entry.value === value);
  if (!found) return value || '';
  return pick(locale, found.label_en, found.label_he);
}

export function vocabularyMeaning(entries, value, locale) {
  const found = (entries || []).find((entry) => entry.value === value);
  if (!found) return '';
  return pick(locale, found.meaning_en, found.meaning_he);
}

export function localized(block, key, locale) {
  if (!block) return '';
  return String(block[locale === 'he' ? `${key}_he` : `${key}_en`] || '');
}

// The line the row's headline is about: the rating goal when the campaign carries
// one, the money goal otherwise. The board picks the same line, so the two never
// disagree about which unit a verdict is in.
export function headlineLine(row) {
  if (!row) return null;
  if (row.rating && row.rating.goal !== null && row.rating.goal !== undefined) return row.rating;
  if (row.money && row.money.goal !== null && row.money.goal !== undefined) return row.money;
  return null;
}

export function otherLine(row) {
  const headline = headlineLine(row);
  if (!headline) return null;
  return headline === row.rating ? row.money : row.rating;
}

// What to do about this row, resolved once so the button and the sentence can
// never describe two different acts. Four kinds and nothing else:
//
//   raise         the flight is fully booked and still short, which is a real owed
//                 amount, so the act is to raise a make-good for exactly it
//   open          a make-good is already open, so the act is to open it
//   book          the flight's remaining days carry no source, so the act is to
//                 book them; a gap to date is a watch signal here and not a debt
//   supply        something the board needed is missing, and the row says which
//
// The ordering is the product judgement this piece is most exposed on, so it is
// stated rather than left in the components. A campaign that is behind on day one
// of seven does not owe anybody anything: the flight can still make it up, and a
// make-good raised against a day-one gap would put a debt in the ledger that the
// week itself is about to settle. A make-good is offered when the shortfall is
// owed, which is when everything on the log is counted and it still falls short.
export function remedyFor(row, openIds) {
  const line = headlineLine(row);
  const open = (openIds || {})[row.campaign_id] || [];
  if (open.length) {
    return { kind: 'open', makeGoodId: open[0] };
  }
  if (!line) {
    return { kind: 'supply', block: row.headline };
  }
  if (line.forward && line.forward.state === SHORT_CERTAIN) {
    return { kind: 'raise', value: line.forward.remaining_to_goal, unit: line.unit, line };
  }
  if (line.pace && line.pace.verdict === UNKNOWN) {
    return { kind: 'supply', block: line.pace };
  }
  if (line.forward && line.forward.state === NOT_BOOKED_YET) {
    return {
      kind: 'book',
      days: line.forward.unsourced_remaining_days || [],
      remaining: line.forward.remaining_to_goal,
      unit: line.unit,
      line,
    };
  }
  return { kind: 'none', line };
}

// The other ending, resolved beside the remedy rather than inside it.
//
// A row on this board is finished with in one of two ways: something was done
// about it, or somebody read it and decided the risk stands. Only the second one
// applies to every row the board asks a decision about, and without it a campaign
// a person accepted looks exactly like one nobody opened.
//
// Three answers and nothing else:
//
//   accept    the board is asking a decision about this row and none is recorded
//   accepted  a decision is on the ledger, and the row shows it rather than the control
//   none      the board is not asking a decision about this row
export function acceptanceFor(row, acceptedIds, needsDecision) {
  const already = (acceptedIds || {})[row.campaign_id] || [];
  if (already.length) {
    return { kind: 'accepted', makeGoodId: already[0] };
  }
  const asking = needsDecision && needsDecision.length ? needsDecision : NEEDS_A_DECISION;
  if (asking.indexOf(row.headline.verdict) >= 0) {
    return { kind: 'accept', verdict: row.headline.verdict };
  }
  return { kind: 'none' };
}

// The two bars a row draws: what is counted, and where the reference sits on the
// same scale. Both are fractions of the goal, so they share one axis and a reader
// can see the gap rather than compute it.
export function barsFor(line) {
  if (!line || line.goal === null || line.goal === undefined || Number(line.goal) <= 0) return null;
  const goal = Number(line.goal);
  const counted = Number(line.counted.through_counted_day || 0);
  const booked = Number(line.counted.booked_total || 0);
  const reference = line.reference ? Number(line.reference.expected_through_counted_day) : null;
  return {
    counted: Math.max(0, Math.min(1, counted / goal)),
    booked: Math.max(0, Math.min(1, booked / goal)),
    reference: reference === null ? null : Math.max(0, Math.min(1, reference / goal)),
  };
}

export function dayCount(list) {
  return Array.isArray(list) ? list.length : 0;
}
