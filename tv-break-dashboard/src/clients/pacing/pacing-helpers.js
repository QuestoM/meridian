// Clients, pacing: the readings a row prints, kept out of the components.
//
// Two rules hold everywhere in this file. A figure is never formatted without the
// unit it is in, because rating points and shekels are two different currencies on
// one board. And a missing figure returns null rather than zero, so a component
// renders the reason it was handed instead of drawing a bar out of nothing.

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

export function pick(locale, en, he) {
  return locale === 'he' ? he : en;
}

// A number inside Hebrew prose, isolated so the digits never reorder around it.
export function isolate(text) {
  return `⁦${text}⁩`;
}

export function decimals(value, places, locale) {
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: places,
    minimumFractionDigits: 0,
  }).format(Number(value));
}

// A figure with its unit. Rating points keep one decimal because a tenth of a
// point is a real trading quantity; money is whole shekels.
export function amount(value, unit, locale) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  if (unit === ILS) {
    const shekels = decimals(Math.round(Number(value)), 0, locale);
    return pick(locale, `ILS ${shekels}`, `${shekels} ש"ח`);
  }
  const points = decimals(Number(value), 1, locale);
  return pick(locale, `${points} rating points`, `${points} נקודות רייטינג`);
}

export function percent(ratio, locale) {
  if (ratio === null || ratio === undefined) return null;
  return `${decimals(Number(ratio) * 100, 0, locale)}%`;
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
