import { isWeekendDay, weekdayIndex } from '../../shell/dates.js';

// The day board's pure model: the arrangement, the edits, and the undo history.
//
// Every act on the board is a named action with an inverse, held on one stack.
// That is what makes undo a keystroke rather than a page: the surface never asks
// the server what it used to look like, because the action that changed it
// carries the state it changed from. Redo is the same stack read the other way,
// exactly as a drawing tool does it, and a new act after an undo drops the
// redone tail rather than branching, which is the behaviour people already have
// in their fingers.
//
// The nudge table is published rather than hidden, the way a professional
// editing timeline publishes its own. Arrow moves one snap unit, Shift moves
// five, Alt moves one second regardless of the snap, and the same three apply to
// length with the up and down arrows.

export const SNAP_CHOICES = [30, 60];

export const DEFAULT_SNAP = 60;

// The smallest length a break may be dragged to, and the coarsest step a length
// edit takes. Thirty seconds is the standard spot unit the rate card quotes on.
export const MIN_DURATION_SECONDS = 30;
export const DURATION_STEP_SECONDS = 30;

export function snapTo(value, grid, min, max) {
  const step = Number(grid) > 0 ? Number(grid) : 1;
  const snapped = Math.round(Number(value) / step) * step;
  return Math.max(min, Math.min(max, snapped));
}

// How far one keypress moves a break, in seconds, given the current snap grid.
export function nudgeStep(event, grid) {
  if (event.altKey) return 1;
  if (event.shiftKey) return grid * 5;
  return grid;
}

export function durationStep(event) {
  if (event.altKey) return 1;
  if (event.shiftKey) return DURATION_STEP_SECONDS * 5;
  return DURATION_STEP_SECONDS;
}

// Where one arrow key would put a break, as a value rather than as an act.
//
// The left and right arrows move it inside its own programme and the up and down
// arrows change its length, both clamped to the programme's own window. Returned
// rather than applied, so the whole keyboard is one pure function the board calls
// and a test can drive directly, and so the board keeps only the decision to
// commit it. Any other key returns null and the board leaves the event alone.
export function nudgeTarget(event, programme, live, grid) {
  const bounds = offsetBounds(programme, live.durationSeconds);
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    const offsetSeconds = snapTo(live.offsetSeconds + direction * nudgeStep(event, grid), 1, bounds.min, bounds.max);
    return { next: { ...live, offsetSeconds }, name: 'move' };
  }
  if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
    const direction = event.key === 'ArrowUp' ? 1 : -1;
    const raw = live.durationSeconds + direction * durationStep(event);
    const durationSeconds = snapTo(raw, 1, MIN_DURATION_SECONDS, maxDurationFor(programme, live.offsetSeconds));
    return { next: { ...live, durationSeconds }, name: 'length' };
  }
  return null;
}

// Whether an edit is itself a gold act. An own-property test rather than a
// nullish one, because taking the mark off is the value false, and false is a
// decision the surface has to be able to hold.
function carriesGold(edit) {
  return Object.prototype.hasOwnProperty.call(edit, 'is_gold');
}

// A break's live position: its saved placement with any unsaved edit applied.
//
// Gold is not read off the edit unless the edit is itself a gold act, because
// gold is not a pending state on this board. Pressing G writes to the override
// store and reloads the day, so the plan is the only authority on it.
//
// Measured before this line changed, on רשת 13 / 2024-11-01, break 001~1: one
// press of ArrowRight left a pending edit that had snapshotted is_gold false, G
// then marked the programme, and the reload brought two sibling chips back gold
// while this one stayed grey, because false ?? true is false. The Gold button
// read off with it, so the act the operator had just performed was no longer
// reversible from the surface, and pressing G again wrote a second identical
// override row instead of clearing the mark.
export function liveBreak(item, edits) {
  const edit = edits[item.break_id];
  if (!edit) {
    return {
      offsetSeconds: item.offset_seconds,
      durationSeconds: item.duration_seconds,
      isGold: item.is_gold,
      edited: false,
    };
  }
  return {
    offsetSeconds: edit.offset_seconds ?? item.offset_seconds,
    durationSeconds: edit.duration_seconds ?? item.duration_seconds,
    isGold: carriesGold(edit) ? edit.is_gold : item.is_gold,
    edited: true,
  };
}

// The absolute clock second a break starts at, from its programme's start.
export function startSecondsOf(item, programme, live) {
  const base = programme ? programme.start_seconds : item.start_seconds - item.offset_seconds;
  return base + live.offsetSeconds;
}

// The window inside its own programme a break may be dragged within. A break
// never leaves the programme it belongs to, because the plan prices it on that
// programme's rating and rate.
export function offsetBounds(programme, durationSeconds) {
  if (!programme) return { min: 0, max: 0 };
  const span = Math.max(0, programme.duration_seconds - durationSeconds);
  return { min: 0, max: span };
}

export function maxDurationFor(programme, offsetSeconds) {
  if (!programme) return MIN_DURATION_SECONDS;
  return Math.max(MIN_DURATION_SECONDS, programme.duration_seconds - offsetSeconds);
}

// The edits, as the score endpoint's move list. Only genuinely changed breaks
// are sent, so an untouched day scores as itself.
export function movesFrom(edits) {
  return Object.entries(edits).map(([breakId, edit]) => ({
    break_id: breakId,
    offset_seconds: edit.offset_seconds ?? null,
    duration_seconds: edit.duration_seconds ?? null,
    is_gold: edit.is_gold ?? null,
  }));
}

// One history stack, shared by every act on the board.
export function emptyHistory() {
  return { past: [], future: [] };
}

export function pushAction(history, action) {
  return { past: [...history.past, action], future: [] };
}

export function undoAction(history) {
  if (!history.past.length) return { history, action: null };
  const action = history.past[history.past.length - 1];
  return {
    history: { past: history.past.slice(0, -1), future: [action, ...history.future] },
    action,
  };
}

export function redoAction(history) {
  if (!history.future.length) return { history, action: null };
  const action = history.future[0];
  return {
    history: { past: [...history.past, action], future: history.future.slice(1) },
    action,
  };
}

// Apply one edit to the edit map, dropping it when it returns the break to
// exactly where the plan put it, so an accidental round trip leaves no trace.
//
// A placement or a length edit writes no gold flag. It used to snapshot one,
// and the snapshot then shadowed the plan the moment the plan changed under it:
// see liveBreak above for the measurement. Only an act that declares itself a
// gold act, with goldEdit true, carries gold into the edit map.
export function applyEdit(edits, item, next) {
  const goldAct = next.goldEdit === true;
  const merged = {
    offset_seconds: next.offsetSeconds,
    duration_seconds: next.durationSeconds,
  };
  if (goldAct) merged.is_gold = next.isGold;
  const sameOffset = Math.abs(merged.offset_seconds - item.offset_seconds) < 0.001;
  const sameDuration = Math.abs(merged.duration_seconds - item.duration_seconds) < 0.001;
  // An edit that carries no gold changes no gold, so it is the same by definition.
  const sameGold = !goldAct || Boolean(merged.is_gold) === Boolean(item.is_gold);
  const nextEdits = { ...edits };
  if (sameOffset && sameDuration && sameGold) {
    delete nextEdits[item.break_id];
    return nextEdits;
  }
  nextEdits[item.break_id] = merged;
  return nextEdits;
}

export function clearEdit(edits, breakId) {
  const nextEdits = { ...edits };
  delete nextEdits[breakId];
  return nextEdits;
}

// What the readout should show right now.
//
// A score describes one arrangement. Between a reload and the score that follows
// it there is no score for what is on screen, and the honest stand-in is the
// board's own totals: the engine's figures for exactly this arrangement with
// nothing edited. Measured before this existed, an undo left the previous
// arrangement's 1,037,270 on screen for about 50 ms while the day had already
// gone back to 1,062,670.
export function boardView(score, board) {
  if (score) return score;
  if (!board) return null;
  return {
    basis: board.basis,
    saved: board.totals,
    current: board.totals,
    delta: { objective: 0, revenue: 0, retention: 0, breaks: 0, ad_seconds: 0, gold_breaks: 0 },
    changed_inputs: { placement: false, duration: false, gold: false },
    compliance: board.compliance,
    hours: board.hours,
    engine_ms: null,
  };
}

// Whether the board's own live figures agree with the weekly plan saved to disk.
//
// The board is a LIVE re-optimization of this channel-day against whatever
// settings, constraints, config and models are current, never a read of a file,
// and ``basis.committed`` is that other, genuinely different basis: the figures
// the weekly plan actually saved to output/weekly_break_schedule.csv, the
// artifact the week board, the overview and every export read. Config or a
// model can move between a save and the moment this board is opened, so the two
// are never assumed to agree; this is the one place that checks and says so.
export function committedGap(basis, saved) {
  const committed = basis && basis.committed;
  if (!committed || !saved) return { state: 'unavailable', committed: null };
  const revenueGap = Math.round((Number(saved.revenue) - Number(committed.revenue)) * 100) / 100;
  const breaksGap = Number(saved.breaks) - Number(committed.breaks);
  const percent = committed.revenue ? Math.round((revenueGap / committed.revenue) * 1000) / 10 : null;
  const matches = Math.abs(revenueGap) < 0.005 && breaksGap === 0;
  return { state: matches ? 'matches' : 'diverged', committed, revenueGap, breaksGap, percent };
}

// Whether this break can be put back where the plan had it, and what the inverse
// would have to delete.
//
// The answer is read from the break the server served, never from what this
// browser session happens to remember. That is the whole point of it: the undo
// stack below dies on a reload, while a saved placement is a restriction and a
// record that both outlive it. Measured before this existed, on רשת 13 /
// 2024-11-01: one ArrowRight and one Save moved the day by 25,400 ILS, and after
// a reload the surface that performed the act offered no way back at all, so the
// only routes were another destination or the API.
//
// A record without the restriction it names is still offered. The restriction may
// have been removed from the rules destination while the record stayed, and
// dropping the leftover record is exactly the repair that case needs.
export function inversePlacement(item) {
  return inverseOfRecord(item && item.saved_placement);
}

// The same inverse, read off a saved record on its own rather than off a break.
//
// A break chip can only carry the record when the plan still has a break with
// that id, and a save is free to take that id away. Measured on רשת 13 /
// 2024-11-01: pinning 001~2 one snap unit right re-plans that programme from four
// breaks to one, the day falls 1,067,845.55 to 1,020,401.35, and the id 001~2
// stops existing. The record and the restriction both survive on disk, so the
// inverse is still exact; it just has nothing on the board to hang from. The
// route serves those records as unbound_placements and this reads the inverse off
// one directly, which is why 47,444.20 ILS is no longer stranded.
export function inverseOfRecord(saved) {
  if (!saved || !saved.break_id) return null;
  return {
    breakId: String(saved.break_id),
    constraintId: saved.constraint_id ? String(saved.constraint_id) : '',
    savedAt: saved.saved_at || '',
    note: saved.note || '',
  };
}

// The break a clock hour resolves to, so an hour bar is an address and not a
// label. An hour holds breaks from several programmes, and the one a person
// means when they point at it is the first that starts inside it. Returns null
// for an hour the plan puts no break in, and the caller then changes nothing.
export function firstBreakInHour(breaks, programmes, liveOf, hour) {
  const wanted = Number(hour);
  const inside = (breaks || [])
    .map((item) => ({ item, start: startSecondsOf(item, programmes.get(item.segment_id), liveOf(item)) }))
    .filter((row) => Math.floor(row.start / 3600) === wanted)
    .sort((first, second) => first.start - second.start);
  return inside.length ? inside[0].item.break_id : null;
}

// How many airings a saved placement will actually bind, counted on the day.
//
// The frozen predicate contract names a date, a programme and an hour, so it
// binds every airing of that title starting in that hour, not only the one that
// was dragged. Measured on רשת 13 / 2024-11-01: 37 of the 82 segments are named
// uniquely and the rest share their (title, hour) with 1 to 3 others. The board
// is served the whole day, so it counts rather than assuming.
export function airingsBound(programmes, programme) {
  if (!programme) return 0;
  const hour = Math.floor(Number(programme.start_seconds) / 3600) % 24;
  const rows = Array.isArray(programmes) ? programmes : Array.from(programmes || []);
  return rows.filter((row) => row && row.title === programme.title
    && Math.floor(Number(row.start_seconds) / 3600) % 24 === hour).length;
}

// Programmes indexed by segment id, so a break resolves its own window once.
export function programmeIndex(programmes) {
  const index = new Map();
  (programmes || []).forEach((programme) => index.set(programme.segment_id, programme));
  return index;
}

// The Israeli week, presented Sunday first while the data stays ISO-keyed.
export const WEEKDAY_ORDER = [0, 1, 2, 3, 4, 5, 6];

// One set of day names for this tree, indexed by the ISO date's own weekday, so
// no surface here has to carry a second copy or fall back to an engine word. The
// plan payload names a day in English (Fri), which is the wire and not a label:
// a Hebrew surface that printed it read רשת 13 / Fri.
export const WEEKDAY_NAMES_HE = ['ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת'];
export const WEEKDAY_NAMES_EN = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export function weekdayName(isoDate, locale) {
  const index = weekdayIndex(isoDate);
  if (index < 0) return '';
  return (locale === 'he' ? WEEKDAY_NAMES_HE : WEEKDAY_NAMES_EN)[index];
}

export { weekdayIndex, isWeekendDay as isWeekend };

// Group a list of ISO days into Sunday-first weeks for the day picker.
export function weeksOf(days) {
  const weeks = [];
  let current = null;
  (days || []).forEach((day) => {
    const index = weekdayIndex(day);
    if (!current || index === 0) {
      current = { days: [] };
      weeks.push(current);
    }
    current.days.push(day);
  });
  return weeks;
}

// Money on this surface is printed exactly, never compacted.
//
// The shared formatter switches to compact notation at 100,000, so this day's
// revenue of 1,062,669.88 renders as 1.06M and hides 2,669.88 ILS. On a board
// whose whole purpose is watching money move, and whose break column has to be
// visibly the same total as its own footer, that rounding is the defect. Below
// 100,000 this is character for character what the shared formatter returns, so
// nothing that was already exact changes.
export function exactCurrency(value, locale = 'en') {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: 0,
    minimumFractionDigits: 0,
  }).format(number);
}

// What a chip can print inside itself at the width it is actually drawn at.
//
// Measured in Chrome on רשת 13 / 2024-11-01 before this existed: the board opens
// at 6 px per minute, every break of the day is 120 s, so all 80 chips are 12 px
// wide and every one of them clipped both of its own numbers, 160 in all. The
// clock 00:28:24 wants 47 px and had 4, the length 120s wants 25 px and had 4, so
// each rendered as one character, 0 over 1. A number clipped to its first digit
// is worse than no number at all, because it still reads as a figure.
//
// So a chip prints only what it can print in full, and what does not fit is drawn
// in the badge beside it, which is what a drawing tool does with an object too
// small to hold its own dimensions. Measured at the same font, 10 px Inter with
// tabular figures: no character of a clock or a length needs more than 6.25 px,
// and the body's own padding and border take 8 px before any text.
export const CHIP_CHAR_PX = 6.25;
export const CHIP_CHROME_PX = 8;

export function chipLabels(widthPx, clock, lengthText) {
  const inner = (Number(widthPx) || 0) - CHIP_CHROME_PX;
  const needs = (text) => Math.ceil(String(text || '').length * CHIP_CHAR_PX);
  return {
    clock: inner >= needs(clock),
    length: inner >= needs(lengthText),
  };
}

export function clockOf(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hour = Math.floor(total / 3600) % 24;
  const minute = Math.floor((total % 3600) / 60);
  const second = total % 60;
  return `${pad(hour)}:${pad(minute)}:${pad(second)}`;
}

export function shortClock(seconds) {
  return clockOf(seconds).slice(0, 5);
}

// A clock a person typed, back into seconds. Accepts HH:MM:SS and HH:MM, so the
// start of a break is a target you can state exactly rather than only a position
// you can drag to. Anything else returns null and the field puts itself back,
// because guessing at a half-typed time would move a break nobody asked to move.
export function parseClock(text) {
  const parts = String(text || '').trim().split(':');
  if (parts.length < 2 || parts.length > 3) return null;
  const numbers = parts.map((part) => Number(part));
  if (numbers.some((value) => !Number.isFinite(value) || value < 0)) return null;
  const [hour, minute, second = 0] = numbers;
  if (hour > 47 || minute > 59 || second > 59) return null;
  return hour * 3600 + minute * 60 + second;
}

function pad(value) {
  return String(value).padStart(2, '0');
}
