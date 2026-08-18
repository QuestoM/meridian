// How this destination reads the delivery ledger. Framework free, so every
// reading below is testable without a renderer and none of them can reach the
// network.
//
// GET /api/clients/campaigns carries a per-campaign ledger that is tri-state by
// construction: a broadcast day either aired, is still scheduled, or has no
// per-spot source at all and is unknown. Figures are counted over the sourced
// days only, so a campaign holding one unknown day in its flight holds a floor
// and not a total.
//
// The rule these encode, and the reason this file exists: a reading never
// leaves here as a bare number. Every one of them carries the state it was
// counted in and whether it is a floor, so no surface can print the figure and
// drop the word that says what the figure is.

export const AIRED = 'aired';
export const SCHEDULED = 'scheduled';
export const UNKNOWN = 'unknown';
// A fourth display state, and not a fourth ledger state. It is what the ledger
// holds when days really were sourced and none of them carries a spot for this
// campaign: counted, and empty, which is a different claim from unknown.
export const COUNTED_EMPTY = 'counted_empty';

const NO_TOTALS = { spots: 0, seconds: 0, ratingPoints: 0, spendIls: 0, droppedByRule: 0 };

const NOTHING = {
  available: false,
  state: UNKNOWN,
  sourcedDays: 0,
  flightDays: 0,
  unknownDays: 0,
  unknownDates: [],
  aired: NO_TOTALS,
  scheduled: NO_TOTALS,
  isFloor: false,
};

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function round(value, places = 2) {
  const factor = 10 ** places;
  return Math.round(number(value) * factor) / factor;
}

// A broadcast day sits inside a flight when its date is inside the flight's own
// window. The store writes ISO dates, which compare correctly as strings, and an
// open end stays open rather than refusing to attribute the day at all.
export function inWindow(date, window) {
  if (!window) {
    return true;
  }
  const day = String(date || '');
  if (!day) {
    return false;
  }
  const from = String(window.starts_on || '');
  const to = String(window.ends_on || '');
  if (from && day < from) {
    return false;
  }
  if (to && day > to) {
    return false;
  }
  return true;
}

function totalsOf(days, state) {
  const rows = days.filter((day) => String(day.air_state || '') === state);
  return {
    spots: rows.reduce((sum, day) => sum + number(day.spots), 0),
    seconds: round(rows.reduce((sum, day) => sum + number(day.seconds), 0)),
    ratingPoints: round(rows.reduce((sum, day) => sum + number(day.rating_points_planned), 0), 4),
    spendIls: round(rows.reduce((sum, day) => sum + number(day.spend_ils), 0)),
    droppedByRule: rows.reduce((sum, day) => sum + number(day.spots_dropped_by_rule), 0),
  };
}

function payloadTotals(block) {
  const source = block || {};
  return {
    spots: number(source.spots),
    seconds: round(source.seconds),
    ratingPoints: round(source.rating_points_planned, 4),
    spendIls: round(source.spend_ils),
    droppedByRule: number(source.spots_dropped_by_rule),
  };
}

// Which of the four display states a counted slice is in. Order matters: what
// aired outranks what is still to come, because the column asks what has been
// delivered, and a day nobody has a source for is never allowed to read as a
// zero delivery.
function stateOf(sourcedDays, aired, scheduled) {
  if (sourcedDays <= 0) {
    return UNKNOWN;
  }
  if (aired.spots > 0) {
    return AIRED;
  }
  if (scheduled.spots > 0) {
    return SCHEDULED;
  }
  return COUNTED_EMPTY;
}

// One campaign's whole ledger, taken from the endpoint's own aggregates rather
// than re-summed here, so a cell and the payload behind it can never disagree
// about a figure they both hold.
function wholeCampaign(delivery) {
  const days = Array.isArray(delivery.days) ? delivery.days : [];
  const unknown = delivery.unknown || {};
  const aired = payloadTotals(delivery.aired);
  const scheduled = payloadTotals(delivery.scheduled);
  const sourcedDays = number(delivery.sourced_days);
  return {
    available: Boolean(delivery.available),
    state: stateOf(delivery.available ? sourcedDays : 0, aired, scheduled),
    sourcedDays,
    flightDays: number(delivery.flight_days),
    unknownDays: number(unknown.days),
    unknownDates: Array.isArray(unknown.dates) ? unknown.dates.map(String) : [],
    aired,
    scheduled,
    isFloor: number(unknown.days) > 0,
    days,
  };
}

// The same reading held to one flight's window. The ledger is written per
// campaign and per broadcast day, so a flight owns the days that fall inside its
// own dates, and a campaign whose ledger has no day inside this flight reads
// unknown for this flight rather than borrowing the campaign's figure.
function oneWindow(delivery, window) {
  const days = (Array.isArray(delivery.days) ? delivery.days : [])
    .filter((day) => inWindow(day.broadcast_date, window));
  const unknownRows = days.filter((day) => String(day.air_state || '') === UNKNOWN);
  const sourced = new Set(
    days.filter((day) => String(day.air_state || '') !== UNKNOWN).map((day) => String(day.broadcast_date || '')),
  );
  const aired = totalsOf(days, AIRED);
  const scheduled = totalsOf(days, SCHEDULED);
  return {
    available: sourced.size > 0,
    state: stateOf(sourced.size, aired, scheduled),
    sourcedDays: sourced.size,
    flightDays: sourced.size + unknownRows.length,
    unknownDays: unknownRows.length,
    unknownDates: unknownRows.map((day) => String(day.broadcast_date || '')),
    aired,
    scheduled,
    isFloor: unknownRows.length > 0,
    days,
  };
}

// The reading a surface renders. Pass a flight to scope it to that flight, or
// nothing to read the whole campaign. A missing ledger is the unknown state and
// never an empty total.
export function deliverySlice(delivery, window = null) {
  if (!delivery) {
    return { ...NOTHING, days: [] };
  }
  return window ? oneWindow(delivery, window) : wholeCampaign(delivery);
}

// The two word helpers both halves of the display need — the figures module and
// the basis module. They live here rather than in either of them so neither has
// to import the other, and so a spot is called a spot in one place.
//
// The locale switch is written out rather than imported, and that is not
// laziness. This file's first line promises it is framework free and reaches
// nothing; the shell's plain-JS helpers module reads `import.meta.env` at its
// top level, so importing one two-line function from it drags a Vite global
// into every bundle that touches delivery. Measured: doing exactly that turned
// eight passing renders in tests/test_p4_tree_renders_every_client.py into one
// failure and six errors, all of them "Cannot read properties of undefined
// (reading 'VITE_KAIROS_API_URL')".
const inLocale = (locale, en, he) => (locale === 'he' ? he : en);

export function decimals(value, places, locale) {
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: places,
    minimumFractionDigits: 0,
  }).format(Number(value));
}

export function spotWord(count, locale) {
  return inLocale(locale, count === 1 ? 'spot' : 'spots', count === 1 ? 'תשדיר' : 'תשדירים');
}

// Which files the counted days were read out of, named so the reader can go and
// look. Distinct and in the order the ledger holds them.
export function sourceFilesOf(slice) {
  const files = [];
  ((slice && slice.days) || []).forEach((day) => {
    const file = String(day.source_file || '').trim();
    if (file && !files.includes(file)) {
      files.push(file);
    }
  });
  return files;
}

// Which rules removed a spot from the counted days, in the words of what they
// capped. A dropped spot is a real spot the engine refused to place, so the
// count above it is short by exactly that many and the rule that did it is
// named rather than summarised.
//
// Named in a sentence, not by its id. The ledger records the cause as
// `dropped_rule_id` alone, an engine key, and this surface used to print it
// bare: a reader was told four spots were removed and handed
// DEFAULT_ONE_PER_BREAK to act on. The sentence is composed on the server from
// the rule's own row, by the same single translator the pacing drill uses, and
// arrives on the delivery block keyed by that id.
//
// A rule the rule file does not hold keeps its line and says so, with the path
// that would fix it. A cause this product cannot name is a third state, not an
// absence, and dropping the line would turn "unavailable" into "nothing here".
export function droppedRulesOf(slice, booking = null) {
  const seen = [];
  ((slice && slice.days) || []).forEach((day) => {
    const id = String(day.dropped_rule_id || '').trim();
    if (id && number(day.spots_dropped_by_rule) > 0 && !seen.includes(id)) {
      seen.push(id);
    }
  });
  return seen.map((id) => ({ id, block: (booking || {})[id] || null }));
}

// One goal's progress, read so that the percent can never travel without the
// state it was computed in. `floor` is the endpoint's own word for a percent
// whose denominator covers days nobody has a source for, and a reading that
// carries a percent with no state is treated as unknown rather than shown.
export function progressReading(progress) {
  if (!progress) {
    return { hasPercent: false, percent: null, state: UNKNOWN, isFloor: false };
  }
  const state = String(progress.state || UNKNOWN);
  const raw = progress.percent;
  const has = raw !== null && raw !== undefined && raw !== '' && Number.isFinite(Number(raw));
  return {
    hasPercent: has && state !== UNKNOWN,
    percent: has ? Number(raw) : null,
    state,
    isFloor: state === 'floor',
  };
}

// Every campaign's delivery on the board, keyed by campaign id. The client
// record shows the same campaigns the board does, and this is what lets it show
// the same ledger from the same read rather than a second one that could drift.
export function campaignDeliveryIndex(board) {
  const index = {};
  ((board && board.campaigns) || []).forEach((campaign) => {
    if (campaign && campaign.campaign_id) {
      index[campaign.campaign_id] = campaign.delivery || null;
    }
  });
  return index;
}
