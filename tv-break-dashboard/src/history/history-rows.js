import { genreLabel } from '../vocabulary.js';
import { WEEKDAYS } from './history-labels.js';
import {
  ROW_CHANNEL_SCOPE, ROW_DAYPARTS, ROW_EFFECTS, ROW_EVENT_TYPES, ROW_KINDS, ROW_MODES,
  ROW_MORE, ROW_PARTS, ROW_SOURCES, ROW_UNNAMED, ROW_UNREADABLE, ROW_WEEKDAY_INDEX,
  rowPhrase, rowWord,
} from './history-row-words.js';

// What one added or removed row is, to a person.
//
// The diff endpoint sends the record itself: a restriction arrives as its
// twenty-four columns, a pin as its fourteen. Printing that record is what this
// surface did before, and it failed the only question the preview exists to
// answer: measured on restore point e105c8d1da22, seven restrictions written
// from one rule each printed as their record cut at seventy-seven characters
// plus an ellipsis, which left a twelve-character id as the only difference and
// put the note, the occurrence, the effect and the author past the cut. So each
// store gets an identity of its own here: the one line that says which row this
// is, and the parts that tell it from its neighbours.
//
// Three rules hold across all eight stores.
//
// Nothing is computed. Every value below is a field the store already wrote,
// put into the operator's words and their own language; where a field is
// absent its part is absent, and where a shape cannot be read the chip says so
// rather than guessing.
//
// No engine key is printed as a name. The record id is printed last and marked
// as an id, because it is what a person quotes when they ask about a row, and
// never as the thing that identifies it to the eye.
//
// No channel name passes. A restriction predicate cannot carry one by its own
// frozen contract, the summary reads a closed set of six fields anyway, and the
// one legacy shape that can carry one, a flat channel scope, is named as a
// channel scope without its value. A pin is identified by its programme, its
// day and its clock rather than by target_id, which embeds a channel.

// The six fields a restriction predicate can test, per its frozen contract in
// docs/constraint-predicate-contract.md. The channel is deliberately not one of
// them, and reading a closed set rather than whatever a row carries is what
// keeps that true if a later grammar adds a seventh.
const SUMMARISED = ['programme', 'genre', 'daypart', 'weekday', 'date', 'hour'];

function text(row, key) {
  return String((row && row[key]) ?? '').trim();
}

function locWord(locale) {
  return locale === 'he' ? 'he' : 'en';
}

// A bare calendar day, read at local noon so a machine west of the broadcast
// zone cannot render it as the day before.
function plainDay(iso, locale) {
  const raw = String(iso ?? '').trim();
  if (!raw) return '';
  const date = new Date(`${raw.slice(0, 10)}T12:00:00`);
  if (Number.isNaN(date.getTime())) return raw;
  return date.toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-GB', {
    day: '2-digit', month: '2-digit', year: 'numeric',
  });
}

function weekdayName(index, locale) {
  const names = WEEKDAYS[index];
  if (!names) return '';
  return locale === 'he' ? names[1] : names[0];
}

function dayPart(value, locale) {
  const raw = String(value ?? '').trim();
  return rowWord(ROW_DAYPARTS, raw, locale) || raw;
}

function genreOrTitle(value, locale) {
  const raw = String(value ?? '').trim();
  if (!raw) return '';
  return genreLabel(raw, locWord(locale)) || raw;
}

function collector(locale) {
  const parts = [];
  const add = (key, values, ltr) => {
    const kept = (Array.isArray(values) ? values : [values])
      .map((value) => String(value ?? '').trim())
      .filter(Boolean);
    if (!kept.length) return;
    parts.push({ key, label: rowWord(ROW_PARTS, key, locale), values: kept, ltr: Boolean(ltr) });
  };
  return [parts, add];
}

// One condition of a restriction predicate, in words, or '' when this surface
// will not put it into words. An operator other than the plain one is left to
// the counted remainder rather than flattened into a claim it does not make.
function conditionValue(condition, locale) {
  const field = String((condition && condition.field) ?? '').trim();
  const operator = String((condition && condition.operator) ?? '').trim();
  if (!SUMMARISED.includes(field)) return '';
  if (field === 'hour' ? operator !== 'eq' : operator !== 'is') return '';
  const value = condition.value;
  if (value === null || value === undefined || typeof value === 'object') return '';
  if (field === 'hour') return `${String(value).padStart(2, '0')}:00`;
  if (field === 'date') return plainDay(value, locale);
  if (field === 'daypart') return dayPart(value, locale);
  if (field === 'weekday') return weekdayName(ROW_WEEKDAY_INDEX[String(value).slice(0, 3)], locale);
  if (field === 'genre') return genreOrTitle(value, locale);
  return String(value);
}

// The occurrence a restriction row covers. Real when the predicate parses and
// this surface can read its conditions, unknown when it will not parse, and
// absent when there is no predicate at all.
export function predicateValues(json, locale) {
  const raw = String(json ?? '').trim();
  if (!raw) return [];
  let parsed = null;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    return [rowPhrase(ROW_UNREADABLE, locale)];
  }
  const conditions = parsed && Array.isArray(parsed.conditions) ? parsed.conditions : [];
  if (!conditions.length) return [];
  const flat = String((parsed && parsed.combinator) || 'and').toLowerCase() === 'and';
  const named = [];
  let rest = 0;
  conditions.forEach((condition) => {
    const nested = condition && Array.isArray(condition.conditions);
    const value = flat && condition && !nested ? conditionValue(condition, locale) : '';
    if (value) named.push(value);
    else rest += 1;
  });
  if (rest) named.push(`${rest} ${rowPhrase(ROW_MORE, locale)}`);
  return named;
}

// A flat scope, which predates the predicate. A channel scope is named without
// its value: this surface cannot tell the operator's own channel from a rival's
// and may not print the second one.
function scopeValues(row, locale) {
  const type = text(row, 'scope_type').toLowerCase();
  const value = text(row, 'scope_value');
  if (type === 'channel') return [rowPhrase(ROW_CHANNEL_SCOPE, locale)];
  if (!value || type === 'always' || !type) return [];
  if (type === 'date') return [plainDay(value, locale)];
  if (type === 'weekday') return [weekdayName(Number(value) % 7, locale)];
  return [genreOrTitle(value, locale)];
}

function unnamed(file, locale) {
  const phrase = ROW_UNNAMED[file];
  return phrase ? rowPhrase(phrase, locale) : '';
}

// A restriction: what it forbids or fixes, on which occurrence, asked by whom.
function constraintRow(row, locale, file) {
  const [parts, add] = collector(locale);
  const effect = rowWord(ROW_EFFECTS, text(row, 'effect'), locale) || text(row, 'effect');
  const title = text(row, 'notes') || effect || unnamed(file, locale);
  add('applies_to', predicateValues(text(row, 'where_json') || text(row, 'rule_where_json'), locale));
  add('scope', scopeValues(row, locale));
  if (title !== effect) add('effect', effect);
  add('offset', text(row, 'offset_seconds'), true);
  add('length', text(row, 'duration_seconds'), true);
  add('breaks', text(row, 'count'), true);
  add('reason', text(row, 'reason'));
  add('asked_by', text(row, 'author'));
  add('starts', plainDay(text(row, 'starts_on'), locale), true);
  add('expires', plainDay(text(row, 'expires_on'), locale), true);
  add('id', text(row, 'constraint_id'), true);
  return { title, parts };
}

// A pin: what was decided, on which programme, on which day and at what time.
// target_id is not printed: it embeds a channel, and the anchor says the same
// thing in the words the day board itself uses.
function overrideRow(row, locale, file) {
  const [parts, add] = collector(locale);
  const kind = rowWord(ROW_KINDS, text(row, 'kind'), locale);
  const status = text(row, 'status');
  const source = text(row, 'source');
  add('programme', genreOrTitle(text(row, 'anchor_title'), locale));
  add('when', [plainDay(text(row, 'anchor_date'), locale), text(row, 'anchor_start')], true);
  add('worth', text(row, 'value'), true);
  add('note', text(row, 'notes'));
  if (status && status !== 'active') add('status', status);
  if (source && source !== 'manual') add('source', rowWord(ROW_SOURCES, source, locale) || source);
  add('id', text(row, 'override_id'), true);
  return { title: kind || text(row, 'notes') || unnamed(file, locale), parts };
}

// An advertiser or agency condition: whose it is, what it does to the price or
// to the placement, and where it applies.
function conditionRow(row, locale, file) {
  const agency = file === 'agency_conditions';
  const [parts, add] = collector(locale);
  const effect = text(row, 'effect');
  const mode = text(row, 'mode');
  const title = text(row, agency ? 'agency_id' : 'advertiser_id');
  add('effect', rowWord(ROW_EFFECTS, effect, locale) || effect);
  add('worth', [text(row, 'value'), rowWord(ROW_MODES, mode, locale) || mode]);
  add('positions', scoped(row, 'scope_positions'));
  add('scope', [
    ...scoped(row, 'scope_dayparts').map((value) => dayPart(value, locale)),
    ...scoped(row, 'scope_genres').map((value) => genreOrTitle(value, locale)),
    ...scoped(row, 'scope_programmes'),
    ...scoped(row, 'scope_weekdays'),
  ]);
  add('note', text(row, 'notes'));
  add('id', text(row, 'rule_id'), true);
  return {
    label: rowWord(ROW_PARTS, agency ? 'agency' : 'advertiser', locale),
    title: title || unnamed(file, locale),
    parts,
  };
}

// A scope column holds a comma-separated list, and ANY means it does not narrow
// anything, so it is left out rather than printed as a value.
function scoped(row, key) {
  return text(row, key)
    .split(',')
    .map((value) => value.trim())
    .filter((value) => value && value.toUpperCase() !== 'ANY');
}

function eventRow(row, locale, file) {
  const [parts, add] = collector(locale);
  const start = text(row, 'start_date');
  const end = text(row, 'end_date');
  const price = text(row, 'price_multiplier');
  add('type', rowWord(ROW_EVENT_TYPES, text(row, 'type'), locale) || text(row, 'type'));
  add('when', [plainDay(start, locale), end && end !== start ? plainDay(end, locale) : ''], true);
  if (Number(text(row, 'intensity')) > 1) add('intensity', text(row, 'intensity'), true);
  if (price && Number(price) !== 1) add('price', price, true);
  if (text(row, 'active').toLowerCase() === 'false') add('status', text(row, 'active'), true);
  add('note', text(row, 'notes'));
  add('id', text(row, 'event_id'), true);
  return { title: text(row, 'name') || unnamed(file, locale), parts };
}

function agencyRow(row, locale, file) {
  const [parts, add] = collector(locale);
  const status = text(row, 'status');
  add('type', text(row, 'agency_type'));
  if (status && status !== 'active') add('status', status);
  add('id', text(row, 'agency_id'), true);
  return { title: text(row, 'display_name') || text(row, 'name') || unnamed(file, locale), parts };
}

function linkRow(row, locale, file) {
  const [parts, add] = collector(locale);
  const source = text(row, 'source');
  add('agency', text(row, 'agency_id'), true);
  add('source', rowWord(ROW_SOURCES, source, locale) || source);
  add('seen_on', plainDay(text(row, 'observed_date'), locale), true);
  add('note', text(row, 'notes'));
  return {
    label: rowWord(ROW_PARTS, 'advertiser', locale),
    title: text(row, 'advertiser') || unnamed(file, locale),
    parts,
  };
}

const IDENTITY = {
  constraints: constraintRow,
  overrides: overrideRow,
  conditions: conditionRow,
  agency_conditions: conditionRow,
  events: eventRow,
  agencies: agencyRow,
  agency_links: linkRow,
};

// A store with no identity of its own cannot happen while the nine logical
// files are the nine this destination lists, so this reads the first field that
// names anything and prints the rest as nothing rather than inventing a shape.
function genericRow(row, locale) {
  const [parts, add] = collector(locale);
  const named = ['display_name', 'name', 'notes', 'advertiser'].map((key) => text(row, key)).find(Boolean);
  const id = Object.keys(row).filter((key) => key.endsWith('_id')).map((key) => text(row, key)).find(Boolean);
  add('id', id, true);
  return { title: named || id || '', parts };
}

// The identity of one row of one store: an optional label for a title that is a
// bare key, the title itself, and the parts that tell it from its neighbours.
export function rowIdentity(file, item, locale) {
  if (item === null || item === undefined) return { label: '', title: '', parts: [] };
  if (typeof item !== 'object') return { label: '', title: String(item), parts: [] };
  const build = IDENTITY[file] || genericRow;
  return { label: '', ...build(item, locale, file) };
}
