// Pure helpers for the Clients destination. Framework free, so each one is
// trivially testable and none of them can reach the network.
//
// The rule these encode: a money figure never renders without the scope it was
// summed over. `basisLine` is that scope, built from the payload's own basis
// block, and every surface that prints a total prints it beside the total
// rather than in a tooltip.

import { pageText } from '../shell/format';
import { isolate } from '../shell/bidi';
import { formatDay, formatSpan } from '../shell/dates';

export const VIEWS = ['clients', 'money', 'campaigns', 'pacing', 'advertisers', 'agencies', 'agreements'];

// Exact shekels, standard notation, on every figure this destination prints.
// The shell's formatCurrency switches to compact above 100,000, which is right
// for a tile that conveys scale and wrong here: an analyst reconciling a client
// against an invoice reads the amount, and 699.45K is not an amount. Nothing is
// rounded away, so a one shekel difference between two rows stays visible.
export function exactMoney(value, locale = 'en') {
  if (value === null || value === undefined || value === '' || !Number.isFinite(Number(value))) {
    return '-';
  }
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(Number(value));
}

// The workspace keeps its view and its open record in the query string, never
// in the hash. The hash is the shell's router, and writing to it would navigate
// away from this destination; the query string is ignored by the shell except
// for `axis`, so these two keys are free.
export const VIEW_PARAM = 'clients';
export const RECORD_PARAM = 'client';

export function readParam(name, fallback = '') {
  if (typeof window === 'undefined') {
    return fallback;
  }
  const value = new URLSearchParams(window.location.search).get(name);
  return value === null ? fallback : value;
}

// The last view this destination wrote into the query string itself. It is the
// difference between an address the operator supplied and this destination's
// own leftover, and without it the two are indistinguishable on the next mount.
// A reload clears it, because after a reload the URL is the address again.
let writtenView = null;

export function writeParams(next) {
  if (typeof window === 'undefined' || !window.history) {
    return;
  }
  const params = new URLSearchParams(window.location.search);
  Object.entries(next).forEach(([key, value]) => {
    if (value) {
      params.set(key, value);
    } else {
      params.delete(key);
    }
  });
  const query = params.toString();
  const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
  window.history.replaceState(null, '', url);
  if (Object.prototype.hasOwnProperty.call(next, VIEW_PARAM)) {
    writtenView = next[VIEW_PARAM];
  }
}

// The view a navigation entry asked for, with an unknown one falling back to
// the destination's first view rather than to a blank screen.
export function requestedView(requested) {
  return VIEWS.includes(requested) ? requested : 'clients';
}

// Which view a mount opens on. Three navigation entries mount this destination,
// each naming the view it opens on, and every tab click stores the view in the
// query string. Those two facts collided: a mount re-read the param this
// destination had written itself, discarded the entry the operator had just
// pressed, and left all three entries dead after one tab click. So the stored
// view is honoured only while it is an address someone else supplied, which is
// the bookmark, and an entry wins over this destination's own leftover.
export function initialView(requested) {
  const fromUrl = readParam(VIEW_PARAM);
  if (VIEWS.includes(fromUrl) && fromUrl !== writtenView) {
    return fromUrl;
  }
  return requestedView(requested);
}

// A payload sentence in the reader's language. Every honest empty state and
// every stated limit crosses the wire as a pair, `<field>_en` and `<field>_he`,
// because a Hebrew-first product that explains itself in English has not
// explained itself. A missing pair falls back to the unsuffixed field rather
// than rendering nothing, so a reason is never lost.
export function localized(payload, field, locale) {
  if (!payload) {
    return '';
  }
  const wanted = locale === 'he' ? payload[`${field}_he`] : payload[`${field}_en`];
  return wanted || payload[field] || '';
}

// A refusal in the reader's language. The write layer carries both halves on the
// error it throws, so a Hebrew flow never renders the English sentence and an
// endpoint that sends one sentence still renders that one.
export function refusalText(error, locale) {
  if (!error) {
    return '';
  }
  if (locale === 'he') {
    return error.messageHe || error.message || '';
  }
  return error.messageEn || error.message || '';
}

// How this product came to know a client, as a word rather than as the token the
// store keys on. `observed` is a column value; a person reads "seen in the data".
// A value this table has never met is printed as it is, because hiding it would
// be worse than showing a token, and it is a state the payload really holds.
export function sourceLabel(source, locale) {
  const key = String(source || '');
  if (key === 'observed') {
    return pageText(locale, 'Seen in the data', 'נצפה בנתונים');
  }
  if (key === 'rules') {
    return pageText(locale, 'From a pricing rule', 'מתוך כלל תמחור');
  }
  if (key === 'manual') {
    return pageText(locale, 'Entered by hand', 'הוזן ידנית');
  }
  if (!key) {
    return pageText(locale, 'Source unknown', 'מקור לא ידוע');
  }
  return key;
}

// Which campaign names the ledger really holds a row for. A campaign observed on
// a client's record is a row on the money board's campaign grouping, and one that
// is not must not be offered as a control that opens nothing.
export function ledgerCampaignKeys(money) {
  if (!money || !money.available) {
    return [];
  }
  return (money.campaigns || []).map((row) => String(row.campaign));
}

// The same question for a break. A removed spot names the break it would have
// sat in, and that break is a row on the board's own break grouping wherever any
// priced spot landed in it. A break that holds nothing but removed spots has no
// row, so its id stays a label there and a control is never offered for a row
// that cannot be opened.
export function ledgerBreakKeys(money) {
  if (!money || !money.available) {
    return [];
  }
  return (money.breaks || []).map((row) => String(row.break_id));
}

// The one sentence that travels with every total on this destination: which
// file, which day, which channel, and how many of the file's rows were priced.
export function basisLine(basis, locale) {
  if (!basis || !basis.file) {
    return '';
  }
  const scope = basis.scope_channel || pageText(locale, 'no channel set', 'לא הוגדר ערוץ');
  const priced = Number(basis.priced_spots || 0);
  const rows = Number(basis.rows_in_file || 0);
  return pageText(
    locale,
    `${scope}, ${formatDay(basis.day)}, ${priced} priced spots of ${rows} in ${basis.file}`,
    `${scope}, ${formatDay(basis.day)}, ${isolate(priced)} תשדירים מתומחרים מתוך ${isolate(rows)} בקובץ ${basis.file}`,
  );
}

export function periodNote(basis, locale) {
  if (!basis) {
    return '';
  }
  return locale === 'he' ? basis.period_note_he || '' : basis.period_note_en || '';
}

export function widerPeriod(basis, locale) {
  if (!basis) {
    return '';
  }
  return locale === 'he' ? basis.wider_period_he || '' : basis.wider_period_en || '';
}

export function scopeNote(basis, locale) {
  if (!basis) {
    return '';
  }
  return locale === 'he' ? basis.scope_note_he || '' : basis.scope_note_en || '';
}

// The money board's drill is one value and the workspace owns it, because the
// record's money control has to open the row for the client that is on screen.
// While that key lived inside the board, the control could only switch the view
// and the reader landed on whatever row the board already held, which is another
// client's money presented as the answer to this client's question.
export const CLIENT_DRILL_GROUP = 'advertisers';
export const NO_DRILL = { group: CLIENT_DRILL_GROUP, key: '' };

// Whether the priced ledger holds a row for this client at all. The tree carries
// its money from the same board the money view renders, so a client whose
// figures are none has no row there and nothing to open behind them.
export function hasLedgerRow(client) {
  return Boolean(client) && client.gross !== null && client.gross !== undefined;
}

// What the record's money control opens. A client with a row opens that row,
// grouped by client so the key resolves against the rows on screen. A client
// without one opens nothing at all, so the reason printed on its record stays
// where the reader is looking instead of a board answering with the money of
// whoever happens to lead the ranking.
export function moneyTarget(client) {
  if (!hasLedgerRow(client) || !client.advertiser) {
    return null;
  }
  return { group: CLIENT_DRILL_GROUP, key: String(client.advertiser) };
}

// One predicate for what counts as finding a client, used at every level of the
// tree. A client is known by more than one string, so all of them are searched.
function clientMatches(client, needle) {
  return [client.advertiser, client.shown_name]
    .concat(client.aliases || [])
    .some((field) => String(field || '').toLowerCase().includes(needle));
}

// Search over the client tree. Matches an agency by name or id and a client by
// any of the strings it is known by, so one box finds either level.
export function filterAgencies(agencies, query) {
  const needle = String(query || '').trim().toLowerCase();
  if (!needle) {
    return agencies;
  }
  return agencies
    .map((agency) => {
      const agencyHit = [agency.name, agency.display_name, agency.agency_id]
        .some((field) => String(field || '').toLowerCase().includes(needle));
      if (agencyHit) {
        return agency;
      }
      const clients = agency.clients.filter((client) => clientMatches(client, needle));
      return clients.length ? { ...agency, clients } : null;
    })
    .filter(Boolean);
}

// The same search over a group of clients that hangs off no agency. Without it
// the box finds a client only where an agency claims it, and a client the tree
// renders but the search cannot reach is a client the reader cannot believe is
// there.
export function filterClients(clients, query) {
  const needle = String(query || '').trim().toLowerCase();
  if (!needle) {
    return clients || [];
  }
  return (clients || []).filter((client) => clientMatches(client, needle));
}

// Every client in the tree, flattened, so a record drawer can walk the whole
// set it was opened from. This is what makes the position counter honest: it
// counts the set the reader is actually in, not the page they can see.
//
// All three groups are in it, in the order the tree renders them. A client the
// payload computes and this function skips is a name that opens nothing when it
// is clicked, which is worse than a client that was never listed at all.
export function flattenClients(agencies, unlinked = [], bookedWithoutSpots = []) {
  const rows = [];
  agencies.forEach((agency) => {
    agency.clients.forEach((client) => {
      rows.push({ ...client, agency_id: agency.agency_id, agency_name: agency.name });
    });
  });
  [...unlinked, ...bookedWithoutSpots].forEach((client) => {
    rows.push({ ...client, agency_id: '', agency_name: '' });
  });
  return rows;
}

// Agency id to agency name, built from the client tree the workspace already
// loads. A board that prints AGY_10 is printing a database key at a person, so
// the name leads and the id stays as the secondary line that makes it findable.
// An id with no matching agency stays visible rather than resolving to a blank.
export function agencyIndex(tree) {
  const index = {};
  ((tree && tree.agencies) || []).forEach((agency) => {
    index[agency.agency_id] = agency.name || agency.display_name || '';
  });
  return index;
}

// The same index read the other way, agency name to agency id. The priced
// ledger groups by the name the daily file carries and every agency record is
// keyed by its id, so a figure headed by an agency name can only open the record
// it names once that name has been resolved. Measured on the shipped ledger: all
// nine agency names on it resolve. A name two agencies share resolves to
// neither, because a control that opens one of two records is a guess.
export function agencyIdsByName(tree) {
  const seen = {};
  const index = {};
  ((tree && tree.agencies) || []).forEach((agency) => {
    [agency.name, agency.display_name].forEach((value) => {
      const name = String(value || '').trim();
      if (!name || !agency.agency_id) {
        return;
      }
      seen[name] = seen[name] === undefined || seen[name] === agency.agency_id ? agency.agency_id : '';
      index[name] = seen[name];
    });
  });
  Object.keys(index).forEach((name) => {
    if (!index[name]) {
      delete index[name];
    }
  });
  return index;
}

// Campaign name to the booked campaign that carries it, from the campaign
// board's own rows. The ledger knows a campaign by the name in the daily file
// and the booked record is keyed by its id, which is the same gap the agency
// index closes. A name two bookings share resolves to neither, for the same
// reason.
export function campaignIdsByName(board) {
  const counts = {};
  const index = {};
  ((board && board.campaigns) || []).forEach((row) => {
    const name = String(row.name || '').trim();
    if (!name || !row.campaign_id) {
      return;
    }
    counts[name] = (counts[name] || 0) + 1;
    index[name] = row.campaign_id;
  });
  Object.keys(counts).forEach((name) => {
    if (counts[name] > 1) {
      delete index[name];
    }
  });
  return index;
}

// Which of a draft's fields actually differ from the record on file. An update
// that carries only what changed cannot disturb a field nobody touched, and it
// keeps the endpoint's duplicate refusal off a name that was not edited.
export function changedFields(stored, draft) {
  const changes = {};
  Object.entries(draft).forEach(([key, value]) => {
    const held = stored && stored[key] !== null && stored[key] !== undefined ? String(stored[key]) : '';
    const typed = String(value === null || value === undefined ? '' : value).trim();
    if (typed !== held) {
      changes[key] = typed;
    }
  });
  return changes;
}

export function positionOf(rows, advertiser) {
  const index = rows.findIndex((row) => row.advertiser === advertiser);
  return index < 0 ? null : { index, position: index + 1, total: rows.length };
}

export function step(rows, advertiser, delta) {
  const found = positionOf(rows, advertiser);
  if (!found || !rows.length) {
    return null;
  }
  const next = (found.index + delta + rows.length) % rows.length;
  return rows[next].advertiser;
}

// A goal is what was booked. It is deliberately not formatted as money even
// when its unit is shekels, because the money layer on this product is the
// break and a booked commitment is not a sum over breaks. The unit comes from
// the endpoint's own closed vocabulary when the caller has it.
export function goalLabel(flight, locale, vocabulary) {
  const value = Number(flight.goal_value || 0);
  const unit = vocabularyLabel(vocabulary, flight.goal_kind, locale);
  return `${value.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US')} ${unit}`.trim();
}

// A booked window, in the one shape a window has anywhere in this product. The
// arrow this used to draw was wrong twice over: it pointed left-to-right in a
// right-to-left line, and it left the open-ended case printing a question mark
// where the truth is that no end date has been set yet.
export function windowLabel(starts, ends, locale) {
  return formatSpan(starts, ends, locale);
}

// A closed value set arrives from the endpoint that owns it, as records
// carrying the word for each language and, for a state, what to do about it.
// The surface never holds a second copy of the table, so a value the API adds
// cannot render as a raw token here.
export function vocabularyEntry(vocabulary, value) {
  return (vocabulary || []).find((entry) => entry.value === value) || null;
}

export function vocabularyLabel(vocabulary, value, locale) {
  const entry = vocabularyEntry(vocabulary, value);
  if (!entry) {
    return String(value || '');
  }
  return locale === 'he' ? entry.label_he || entry.label_en : entry.label_en;
}

export function vocabularyRemedy(vocabulary, value, locale) {
  const entry = vocabularyEntry(vocabulary, value);
  if (!entry) {
    return '';
  }
  return locale === 'he' ? entry.what_to_do_he || '' : entry.what_to_do_en || '';
}

// Prose that names another destination has to reach it. The shell routes on the
// hash, so setting it is how a surface hands over, and it is the shell's own
// contract rather than a second router.
export function goToView(label) {
  if (typeof window === 'undefined') {
    return;
  }
  window.location.hash = label;
}
