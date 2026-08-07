import { isolate } from './bidi';

// Dates. This file is the only place in the dashboard that decides what a
// calendar day looks like to a reader, and verify-date-rules.mjs fails the build
// when a surface decides it anywhere else. A change to how a date reads is a
// change to this file and to nothing else.
//
// WHY IT EXISTS
//
// A pacing card printed the days with no delivery source as six ISO strings run
// together with commas, inside a Hebrew interface:
//
//   2025-04-28, 2025-04-29, 2025-04-30, 2025-05-01, 2025-05-02, 2025-05-03
//
// Three separate defects in one line. It is a machine format, not the format an
// Israeli operator reads. It spells out six days that are one unbroken run. And
// nothing collapsed, so a longer list would have run off the card. Every one of
// those was fixable at that call site, which is exactly why the fix belongs
// here: seventeen other surfaces were formatting dates for themselves too, each
// with its own answer, and fixing one of them fixes one of them.

// THE FORMAT IS dd/mm/yyyy, IN BOTH LOCALES.
//
// This is an Israeli media product and dd/mm/yyyy is the format its operators
// read. It is not a locale switch: an English-reading buyer sitting in Tel Aviv
// must not be shown 04/28/2025 while the person beside them reads 28/04/2025,
// because the two are indistinguishable on any day of the month up to twelve and
// a misread flight date books the wrong week. One format, always, and the
// English side happens to agree with it because en-GB writes dates this way too.
const DAY_SEPARATOR = '/';

// THE TWO SEPARATORS, AND WHY THEY LOOK NOTHING ALIKE.
//
// A rendered list holds two different joins: the one between a range's two ends,
// and the one between items. If a reader cannot tell them apart they cannot tell
// where one range stops and the next begins, and a list of three ranges reads as
// six loose dates. So the two are pulled apart on two independent cues at once.
//
// SHAPE: a hyphen has a horizontal stroke, a middle dot is a point. Neither can
// be mistaken for the other at any size, and neither is the slash already inside
// a date.
//
// SPACING: the range joiner carries no spaces, so its two ends bind tightly into
// one object. The item joiner carries a space on each side, so its neighbours
// read as separate objects. Tight means one thing, loose means two.
//
// The hyphen is also the bidi-safest choice available. It is a European
// separator, so between two runs of digits the algorithm folds the whole range
// into a single number run and cannot reorder the two ends. A dash or an arrow
// is a neutral, which in a Hebrew line resolves to right-to-left and prints the
// end date first. An arrow is worse still: it points somewhere, and in a
// right-to-left line the direction it points contradicts the reading order.
const RANGE_JOIN = '-';
const LIST_JOIN = ' · ';

// How many runs a list prints before it stops naming days and states a count. A
// reader scanning a card wants to know which days are affected; past a handful,
// what they actually want is how many. Six runs is the point where the line
// stops being scannable on a card at the widths this dashboard ships.
const MAX_RUNS = 6;

// A broadcast day is an Israeli day, so every INSTANT is read in one declared
// zone rather than in whatever zone the reader's machine is set to. The server
// groups on the same zone, so the day a row is filed under and the clock printed
// on it can never disagree.
//
// A calendar day is NOT an instant and never touches this. YYYY-MM-DD carries no
// time and no zone; feeding it through a clock is what makes a date render as
// the day before for a reader west of here. Everything below that takes a plain
// calendar day works on the string.
export const BROADCAST_ZONE = 'Asia/Jerusalem';

// The Israeli week reads Sunday first. The stored value stays the ISO date; only
// the reading is localized.
const WEEKDAYS = [
  { en: 'Sunday', he: 'ראשון' },
  { en: 'Monday', he: 'שני' },
  { en: 'Tuesday', he: 'שלישי' },
  { en: 'Wednesday', he: 'רביעי' },
  { en: 'Thursday', he: 'חמישי' },
  { en: 'Friday', he: 'שישי' },
  { en: 'Saturday', he: 'שבת' },
];

function say(locale, en, he) {
  return locale === 'he' ? he : en;
}

// A calendar day as three numbers, or null. Accepts a bare YYYY-MM-DD and any
// longer ISO string that starts with one, so a timestamp can be read for the day
// it falls on. Anything else is null, and every caller below prints such a value
// back verbatim rather than inventing a day for it.
export function parseDay(value) {
  const text = String(value ?? '').trim();
  if (!/^\d{4}-\d{2}-\d{2}/.test(text)) return null;
  const year = Number(text.slice(0, 4));
  const month = Number(text.slice(5, 7));
  const day = Number(text.slice(8, 10));
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  // Round-trip through UTC to reject the days a month does not have: 31 April
  // parses as three plausible numbers and is not a day.
  const stamp = Date.UTC(year, month - 1, day);
  const back = new Date(stamp);
  if (back.getUTCFullYear() !== year || back.getUTCMonth() !== month - 1 || back.getUTCDate() !== day) {
    return null;
  }
  return { year, month, day, stamp };
}

function pad(value, width) {
  return String(value).padStart(width, '0');
}

// One calendar day. Returns the value unchanged when it is not a calendar day,
// because a surface that cannot read its own payload should print what it was
// given rather than a date nobody sent.
export function formatDay(value) {
  const parts = parseDay(value);
  if (!parts) return String(value ?? '').trim();
  return `${pad(parts.day, 2)}${DAY_SEPARATOR}${pad(parts.month, 2)}${DAY_SEPARATOR}${pad(parts.year, 4)}`;
}

// One run of consecutive days, from its two ends.
//
// A RUN'S SHAPE NEVER CHANGES, and that is the whole answer to what a range
// looks like when it crosses a month.
//
// The tempting compression is to drop the parts the two ends share: 28-30/04/2025
// inside one month, 28/04-03/05/2025 across a month boundary, both ends in full
// only across a year. That produces three shapes for one idea, and which one a
// reader gets depends on where the run happens to fall in the calendar. It also
// puts a bare 28 next to a full date, which is a second number run for the bidi
// algorithm to reorder and a second thing for a reader to parse.
//
// So: both ends in full, always. Crossing a month is not a special case,
// crossing a year is not a special case, and there is nothing to get wrong at a
// boundary because no boundary is consulted. A reader learns the shape once.
//
// The result is isolated. Its content is digits and separators, which in a
// Hebrew line the algorithm would otherwise pull apart at the joiner.
export function formatDayRange(startValue, endValue) {
  const start = formatDay(startValue);
  const end = formatDay(endValue);
  if (!start) return isolate(end);
  if (!end || start === end) return isolate(start);
  return isolate(`${start}${RANGE_JOIN}${end}`);
}

// Sorted, de-duplicated calendar days grouped into maximal runs of consecutive
// days, followed by whatever did not parse as a day, each on its own.
//
// Exported so the guard's own cases and any caller that needs the grouping
// without the rendering read the same arithmetic the screen does.
export function groupConsecutiveDays(values) {
  const parsed = [];
  const unparsed = [];
  for (const value of Array.isArray(values) ? values : []) {
    const day = parseDay(value);
    if (day) parsed.push(day);
    else {
      const text = String(value ?? '').trim();
      if (text) unparsed.push(text);
    }
  }
  parsed.sort((a, b) => a.stamp - b.stamp);
  const runs = [];
  const DAY_MS = 86400000;
  for (const day of parsed) {
    const open = runs[runs.length - 1];
    if (open && day.stamp === open.endStamp) continue;
    if (open && day.stamp === open.endStamp + DAY_MS) {
      open.end = day;
      open.endStamp = day.stamp;
      open.length += 1;
      continue;
    }
    runs.push({ start: day, end: day, endStamp: day.stamp, length: 1 });
  }
  return { runs, unparsed };
}

// The tail a capped list ends on. It counts DAYS rather than runs, because the
// sentence above the list counts days too and two counts of different things on
// one card is how a reader stops trusting either.
function moreDaysText(count, locale) {
  if (count === 1) return say(locale, 'and 1 more day', 'ועוד יום אחד');
  return say(locale, `and ${isolate(count)} more days`, `ועוד ${isolate(count)} ימים`);
}

// A LIST OF DAYS, which is what this module was built for.
//
// Consecutive days collapse into runs, runs render in one shape, items are
// separated by something no reader can confuse with the joiner inside a run, and
// past MAX_RUNS the line states how many days it stopped naming instead of
// running off the card.
//
// Each run is isolated on its own and the items are NOT, on purpose. That leaves
// the ORDER of the list to the direction of the line it sits in, so a Hebrew
// reader gets the earliest run on the right where their eye starts and an
// English reader gets it on the left. Do not wrap the result in Figure: Figure
// forces left-to-right, which is right for one quantity and wrong for a list,
// because it would put the earliest run on the far side of a Hebrew line.
export function formatDayList(values, locale, { maxRuns = MAX_RUNS } = {}) {
  const { runs, unparsed } = groupConsecutiveDays(values);
  const items = [
    ...runs.map((run) => formatDayRange(
      `${pad(run.start.year, 4)}-${pad(run.start.month, 2)}-${pad(run.start.day, 2)}`,
      `${pad(run.end.year, 4)}-${pad(run.end.month, 2)}-${pad(run.end.day, 2)}`,
    )),
    ...unparsed.map((text) => isolate(text)),
  ];
  if (!items.length) return '';
  if (items.length <= maxRuns) return items.join(LIST_JOIN);
  const shownRuns = runs.slice(0, maxRuns);
  const shownDays = shownRuns.reduce((sum, run) => sum + run.length, 0);
  const allDays = runs.reduce((sum, run) => sum + run.length, 0) + unparsed.length;
  const head = items.slice(0, maxRuns).join(LIST_JOIN);
  return `${head}${LIST_JOIN}${moreDaysText(allDays - shownDays, locale)}`;
}

// A window with two named ends: a flight, a plan span, an event. It renders in
// the same shape a run does, because a reader should not have to learn that a
// booked window and a run of missing days are punctuated differently.
//
// An open end is stated rather than filled in. A window with no end date is a
// real state on this product's payloads and printing a question mark for it says
// the value is unknown when what is true is that there is not one yet.
export function formatSpan(startValue, endValue, locale) {
  const start = parseDay(startValue) ? formatDay(startValue) : '';
  const end = parseDay(endValue) ? formatDay(endValue) : '';
  if (start && end) return formatDayRange(startValue, endValue);
  if (start) return say(locale, `from ${isolate(start)}, no end date`, `מ-${isolate(start)}, ללא תאריך סיום`);
  if (end) return say(locale, `until ${isolate(end)}, no start date`, `עד ${isolate(end)}, ללא תאריך התחלה`);
  return say(locale, 'no dates set', 'לא נקבעו תאריכים');
}

// The weekday a calendar day falls on, 0 for Sunday, or -1 when the value is not
// a day. Read off the string through UTC, so no local zone can move it.
export function weekdayIndex(value) {
  const parts = parseDay(value);
  return parts ? new Date(parts.stamp).getUTCDay() : -1;
}

export function weekdayName(value, locale) {
  const entry = WEEKDAYS[weekdayIndex(value)];
  return entry ? say(locale, entry.en, entry.he) : '';
}

// Israeli week law: the weekend is Friday and Saturday, and nothing else. Three
// files each had their own copy of this and one of them read the weekday in the
// machine's local zone, which puts a Thursday evening in Europe on a Friday.
export function isWeekendDay(value) {
  const index = weekdayIndex(value);
  return index === 5 || index === 6;
}

// A day with the weekday it falls on, for a heading where the reader is choosing
// between days and the weekday is what they are choosing by. The comma is safe
// here in a way it is not in a list: it joins a word to a number rather than two
// dates to each other, so there is nothing for it to be confused with.
export function formatDayWithWeekday(value, locale) {
  const name = weekdayName(value, locale);
  const day = formatDay(value);
  if (!name) return isolate(day);
  return say(locale, `${name}, ${isolate(day)}`, `יום ${name}, ${isolate(day)}`);
}

// The zone-shifted parts of an INSTANT. Intl is the only thing that knows what
// Asia/Jerusalem was doing on a given date, so it is used here and nowhere else.
function zonedParts(value) {
  const when = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(when.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: BROADCAST_ZONE,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(when);
  const read = {};
  for (const part of parts) read[part.type] = part.value;
  if (!read.year) return null;
  // Midnight comes back as 24 from some engines when hour12 is false.
  return { ...read, hour: read.hour === '24' ? '00' : read.hour };
}

// The clock an instant reads at, in the broadcast zone.
export function formatClock(value) {
  const parts = zonedParts(value);
  if (!parts) return '';
  return isolate(`${parts.hour}:${parts.minute}`);
}

// An instant: the day it falls on, in this module's one date shape, and the
// clock it reads at. The date is composed here rather than handed to Intl so a
// timestamp and a calendar day print the same shape as each other.
export function formatStamp(value) {
  const parts = zonedParts(value);
  if (!parts) return '';
  const day = `${parts.day}${DAY_SEPARATOR}${parts.month}${DAY_SEPARATOR}${parts.year}`;
  return isolate(`${day}, ${parts.hour}:${parts.minute}`);
}

// The day of the month and the month, for a surface where the year is already
// established by its own heading and repeating it on every row is noise.
export function formatDayOfMonth(value) {
  const parts = parseDay(value);
  if (!parts) return String(value ?? '').trim();
  return isolate(`${pad(parts.day, 2)}${DAY_SEPARATOR}${pad(parts.month, 2)}`);
}

// The broadcast day an instant falls on, as YYYY-MM-DD. This is a key, not a
// reading: it is what a surface groups and looks up by, and it must never be
// shown to anybody.
export function isoDay(value) {
  const parts = zonedParts(value);
  if (!parts) return String(value ?? '').slice(0, 10);
  return `${parts.year}-${parts.month}-${parts.day}`;
}

export function todayIso() {
  return isoDay(new Date());
}

// A month heading, from a {year, month} cursor where month is 1..12. The month
// name is the one piece of a date that genuinely differs by language, so this is
// the one function here that reads differently in the two locales.
export function formatMonthTitle(year, month, locale) {
  const numeric = `${pad(month, 2)}${DAY_SEPARATOR}${pad(year, 4)}`;
  if (!Number.isFinite(year) || !Number.isFinite(month) || month < 1 || month > 12) return numeric;
  try {
    const name = new Intl.DateTimeFormat(locale === 'he' ? 'he-IL' : 'en-GB', {
      timeZone: 'UTC', month: 'long',
    }).format(new Date(Date.UTC(year, month - 1, 1)));
    return `${name} ${isolate(year)}`;
  } catch {
    return numeric;
  }
}
