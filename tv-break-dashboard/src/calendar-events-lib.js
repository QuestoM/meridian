// Calendar-page helpers: pure date math over ISO date strings plus the persisted
// view choice. Everything here is a calendar fact (which days a stored event
// covers, how a month lays out), never a model claim; no number is invented.
// Israeli week law: the week starts Sunday and ends Saturday, weekend is Friday
// and Saturday only. Data stays ISO-keyed; only presentation is Sunday-first.

const VIEW_KEY = 'kairos.calendar.view';

// The persisted grid/list choice. Anything unreadable resolves to the grid,
// the page's primary view.
export function readStoredCalendarView() {
  try {
    return window.localStorage.getItem(VIEW_KEY) === 'list' ? 'list' : 'grid';
  } catch {
    return 'grid';
  }
}

export function storeCalendarView(view) {
  try {
    window.localStorage.setItem(VIEW_KEY, view === 'list' ? 'list' : 'grid');
  } catch {
    // localStorage may be unavailable (private mode); the session choice still works.
  }
}

// Local-timezone ISO day, so "today" on the grid is the operator's calendar day.
export function localIsoDate(date = new Date()) {
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${date.getFullYear()}-${month}-${day}`;
}

export function isoDay(value) {
  return String(value || '').slice(0, 10);
}

// Effective date range of an event. An open-ended event (empty end_date) is
// ongoing from its start onward, per the page's own stated convention.
const OPEN_END = '9999-12-31';

export function eventRange(event) {
  return { start: isoDay(event?.start_date), end: isoDay(event?.end_date) || OPEN_END };
}

export function eventOnDay(event, dayIso) {
  const { start, end } = eventRange(event);
  return Boolean(start) && start <= dayIso && dayIso <= end;
}

export function activeEventsOnDay(events, dayIso) {
  return (events || []).filter((event) => event && event.active !== false && eventOnDay(event, dayIso));
}

// Weekday column headers, Sunday-first.
export const WEEKDAY_HEADERS = {
  he: ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'שבת'],
  en: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
};

// Month arithmetic on a {year, month} cursor (month is 1..12).
export function addMonths(year, month, delta) {
  const index = year * 12 + (month - 1) + delta;
  return { year: Math.floor(index / 12), month: ((index % 12) + 12) % 12 + 1 };
}

export function monthTitle(year, month, locale) {
  try {
    return new Date(year, month - 1, 1).toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-GB', { month: 'long', year: 'numeric' });
  } catch {
    return `${year}-${String(month).padStart(2, '0')}`;
  }
}

// Sunday-first month matrix: full weeks of ISO dates covering the month,
// including the adjacent-month days that pad the first and last week.
export function monthMatrix(year, month) {
  const offset = new Date(year, month - 1, 1).getDay(); // 0 = Sunday
  const cursor = new Date(year, month - 1, 1 - offset);
  const weeks = [];
  do {
    const week = [];
    for (let i = 0; i < 7; i += 1) {
      week.push(localIsoDate(cursor));
      cursor.setDate(cursor.getDate() + 1);
    }
    weeks.push(week);
  } while (cursor.getMonth() === month - 1);
  return weeks;
}

// Lays the active events overlapping one week into horizontal lanes so a
// multi-day event renders as one continuous bar across its days. Greedy
// first-fit packing after sorting by start column then span; segments that do
// not fit into laneCap lanes are reported per column so the day cell can show
// an honest "+N" count instead of dropping them silently.
export function packWeekLanes(weekDates, events, laneCap = 4) {
  const weekStart = weekDates[0];
  const weekEnd = weekDates[weekDates.length - 1];
  const segments = [];
  for (const event of events || []) {
    if (!event || event.active === false) continue;
    const { start, end } = eventRange(event);
    if (!start || start > weekEnd || end < weekStart) continue;
    const from = start > weekStart ? start : weekStart;
    const to = end < weekEnd ? end : weekEnd;
    const startCol = weekDates.indexOf(from) + 1;
    const endCol = weekDates.indexOf(to) + 1;
    if (startCol < 1 || endCol < startCol) continue;
    segments.push({ event, startCol, endCol, continuesBefore: start < weekStart, continuesAfter: end > weekEnd });
  }
  segments.sort((a, b) => {
    if (a.startCol !== b.startCol) return a.startCol - b.startCol;
    const spanDiff = (b.endCol - b.startCol) - (a.endCol - a.startCol);
    if (spanDiff !== 0) return spanDiff;
    return String(a.event.name || '').localeCompare(String(b.event.name || ''), 'he');
  });
  const lanes = [];
  const hidden = [];
  for (const segment of segments) {
    const lane = lanes.find((rows) => rows.every((s) => s.endCol < segment.startCol || s.startCol > segment.endCol));
    if (lane) {
      lane.push(segment);
    } else if (lanes.length < laneCap) {
      lanes.push([segment]);
    } else {
      hidden.push(segment);
    }
  }
  const overflow = {};
  for (const segment of hidden) {
    for (let col = segment.startCol; col <= segment.endCol; col += 1) {
      overflow[col] = (overflow[col] || 0) + 1;
    }
  }
  return { lanes, overflow };
}

// The next active events from today: ongoing ones first (soonest to end),
// then future ones by start date. Deactivated and fully passed events never
// appear; this is orientation, not history.
export function upcomingEvents(events, todayIso, limit = 5) {
  const rows = (events || []).filter((event) => {
    if (!event || event.active === false) return false;
    const { start, end } = eventRange(event);
    return Boolean(start) && (end >= todayIso || start >= todayIso);
  });
  rows.sort((a, b) => {
    const rangeA = eventRange(a);
    const rangeB = eventRange(b);
    const ongoingA = rangeA.start <= todayIso ? 0 : 1;
    const ongoingB = rangeB.start <= todayIso ? 0 : 1;
    if (ongoingA !== ongoingB) return ongoingA - ongoingB;
    const keyA = ongoingA === 0 ? rangeA.end : rangeA.start;
    const keyB = ongoingB === 0 ? rangeB.end : rangeB.start;
    if (keyA !== keyB) return keyA.localeCompare(keyB);
    return String(a.name || '').localeCompare(String(b.name || ''), 'he');
  });
  return rows.slice(0, limit);
}
