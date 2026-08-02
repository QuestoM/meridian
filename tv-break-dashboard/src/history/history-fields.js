import { copyByLocale } from '../shell/copy.js';
import { genreLabel } from '../vocabulary.js';
import { FORCE_LABELS, WEEKDAYS } from './history-labels.js';

// What one changed field of a stored file is, to a person.
//
// The diff endpoint sends a settings change as a field with the value on both
// sides exactly as the store holds it, and one of those values is a whole
// nested object: measured on this deployment, 61 of the 200 restore points
// carry a rate-card row, and printing it as a record cut at seventy-seven
// characters is the same defect the row chips had. A rate-card change is read
// here as the fields inside it that actually differ, each named in the reader's
// own language, each carrying the value on both sides.
//
// Three rules hold, and they are the row chips' rules.
//
// Nothing is computed. Every value printed is one of the two sides' own; which
// leaves to print is a choice about what to show, never about what is true.
//
// No engine key is printed as a name where the product already has a word for
// it. The words are read from the shell's own copy, from the run report's force
// table and from the Pricing page's layer names, so a rename lands here too.
//
// A value absent on one side says so rather than reading as an empty cell,
// because not set and set to nothing are different decisions to put back.

const MAX_LEAVES = 12;

const NOT_SET = ['Not set', 'לא הוגדר'];
const SWITCH_ON = ['On', 'פועל'];
const SWITCH_OFF = ['Off', 'כבוי'];

// The one settings field that can hold a channel name. The live side of the
// diff is the operator's own channel by definition, because the endpoint reads
// it from the settings store this product is scoped by. A stored side that
// differs from it is a channel this surface cannot vouch for, so it is named as
// one without its name, exactly as a restriction's legacy channel scope is.
const CHANNEL_FIELD = 'operator_channel';
const CHANNEL_HELD = ['Another channel, which this preview does not name', 'ערוץ אחר, והמסך הזה לא נוקב בשמו'];

// Settings keys the shell already names on the settings surface itself. The
// value is the key into the shell's own copy, so the two surfaces cannot drift.
const SHELL_WORDS = {
  effective_date: 'effectiveDate',
  gold_breaks_enabled: 'gold',
  locale: 'language',
  max_ad_minutes_per_hour: 'maxAdMinutes',
  max_breaks_per_hour: 'maxBreaks',
  max_daily_ad_minutes: 'dailyCap',
  min_break_spacing_minutes: 'spacing',
  min_retention_floor: 'retentionFloor',
  profile_name: 'profile',
  protected_program_max_ad_minutes_per_hour: 'protectedMax',
  protected_program_types: 'protectedTypes',
  require_manual_approval: 'approval',
  risk_lambda: 'riskCaution',
  sponsorships_enabled: 'sponsorships',
};

// The rest of the settings store, in the words its own destination uses: the
// pacing panel, the objective panel and the channel card each name their keys,
// and those names are the ones repeated here.
const SETTING_FIELDS = {
  audience_model_activation: ['Audience model switch', 'מתג מודל הקהל'],
  chart_direction: ['Chart direction', 'כיוון התרשימים'],
  currency: ['Currency', 'מטבע'],
  direction: ['Reading direction', 'כיוון קריאה'],
  notes: ['Note on the profile', 'הערה על הפרופיל'],
  objective_mode: ['Engine focus', 'מיקוד המנוע'],
  operator_channel: ['Your channel', 'הערוץ שלכם'],
  pacing_ahead_k: ['Over-delivery throttle', 'ריסון דילוור-יתר'],
  pacing_enabled: ['Campaign pacing', 'קצב קמפיינים'],
  pacing_epsilon: ['Pace denominator floor', 'רצפת מכנה הקצב'],
  pacing_reference_date: ['Pacing reference date', 'תאריך ייחוס לקצב'],
  pacing_urgency_k: ['Behind-pace strength', 'עוצמת פיגור בקצב'],
  pacing_urgency_max: ['Behind-pace cap', 'תקרת פיגור בקצב'],
  pacing_weight_floor: ['Over-delivery floor', 'רצפת דילוור-יתר'],
  pricing_overrides: ['Rate card', 'כרטיס תעריפים'],
  regulatory_source_url: ['Regulatory source', 'מקור רגולטורי'],
  timezone: ['Time zone', 'אזור זמן'],
};

// The two settings whose value is a closed set of tokens rather than a number.
const DIRECTION_VALUES = {
  ltr: ['Left to right', 'משמאל לימין'],
  rtl: ['Right to left', 'מימין לשמאל'],
};

const VALUE_WORDS = {
  locale: { en: ['English', 'אנגלית'], he: ['Hebrew', 'עברית'] },
  direction: DIRECTION_VALUES,
  chart_direction: DIRECTION_VALUES,
  objective_mode: {
    blend: ['Balanced, the default', 'מאוזן, ברירת המחדל'],
    revenue_net: ['Net focused', 'ממוקד נטו'],
  },
};

// Inside the rate card. The activation switch is keyed by the Pricing page's
// own layer name and the premium table by the engine's YAML name, so both
// spellings resolve to the one word the operator already reads on that page.
const PRICE_FIELDS = {
  base_price_per_second_per_tvr_point: ['Base price per rating point per second', 'מחיר בסיס לנקודת רייטינג לשנייה'],
};

const PRICE_LAYERS = {
  ad_type: ['Ad type', 'סוג פרסומת'],
  day: ['Day of week', 'יום בשבוע'],
  day_of_week: ['Day of week', 'יום בשבוע'],
  event: ['Calendar event', 'אירוע לוח שנה'],
  events: ['Calendar events', 'אירועי לוח שנה'],
  position: ['Position in break', 'מיקום בברייק'],
  position_in_break: ['Position in break', 'מיקום בברייק'],
  program: ['Programme type', 'סוג תוכנית'],
  program_type: ['Programme type', 'סוג תוכנית'],
  show: ['Specific show', 'תוכנית ספציפית'],
};

// The trade's positions inside a break are 1 to 5 and L, where L is LAST and is
// its own position. "last" is the rate card's legacy key for L and still reads.
const POSITION_KEYS = {
  1: ['First', 'ראשון'],
  2: ['Second', 'שני'],
  3: ['Third', 'שלישי'],
  4: ['Fourth', 'רביעי'],
  5: ['Fifth', 'חמישי'],
  L: ['Last (L)', 'אחרון (L)'],
  default_middle: ['Middle default', 'אמצע, ברירת מחדל'],
  last: ['Last (L)', 'אחרון (L)'],
};

function pairText(found, locale) {
  if (!found) return '';
  return locale === 'he' ? found[1] : found[0];
}

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function settingLabel(key, locale) {
  const raw = String(key ?? '');
  const shared = SHELL_WORDS[raw];
  if (shared) {
    const table = copyByLocale[locale === 'he' ? 'he' : 'en'] || {};
    if (table[shared]) return String(table[shared]);
  }
  return pairText(SETTING_FIELDS[raw], locale) || pairText(FORCE_LABELS[raw], locale);
}

function layerName(key, locale) {
  const raw = String(key ?? '');
  return pairText(PRICE_LAYERS[raw], locale) || raw;
}

// The key a premium is stored under. A programme type is a genre, a day is an
// ISO weekday read into the Sunday-first week, a position is a named engine
// slot, and a show title or an ad type is already a word.
function premiumKey(layer, key, locale) {
  const raw = String(key ?? '');
  if (!raw) return '';
  if (layer === 'program_type' || layer === 'program') return genreLabel(raw, locale === 'he' ? 'he' : 'en');
  if (layer === 'day_of_week' || layer === 'day') {
    const names = WEEKDAYS[Number(raw) % 7];
    return names ? pairText(names, locale) : raw;
  }
  if (layer === 'position_in_break' || layer === 'position') return pairText(POSITION_KEYS[raw], locale) || raw;
  return raw;
}

function priceLeaf(parts, locale) {
  const head = String(parts[0] ?? '');
  if (head === 'pricing_activation') {
    const layer = layerName(parts[1], locale);
    return locale === 'he' ? `שכבת ${layer}` : `${layer} layer`;
  }
  if (head === 'premiums') {
    const layer = layerName(parts[1], locale);
    const key = premiumKey(String(parts[1] ?? ''), parts[2], locale);
    if (!key) return locale === 'he' ? `מקדם ${layer}` : `${layer} premium`;
    return locale === 'he' ? `מקדם ${layer}, ${key}` : `${layer} premium, ${key}`;
  }
  return pairText(PRICE_FIELDS[head], locale) || parts.join(' · ');
}

// The name of one changed field. Only the settings store is named here: every
// other store sends a column name its own identity already reads, and this is
// the preview, not a rename.
export function fieldLabel(file, path, locale) {
  const parts = String(path ?? '').split('.');
  const head = parts[0] || '';
  const name = file === 'settings' ? settingLabel(head, locale) : '';
  const rest = parts.slice(1);
  if (!rest.length) return name || head;
  const inner = head === 'pricing_overrides' ? priceLeaf(rest, locale) : rest.join(' · ');
  return `${name || head} · ${inner}`;
}

function listText(value, locale) {
  if (value.some((item) => item && typeof item === 'object')) {
    return locale === 'he' ? `${value.length} רשומות` : `${value.length} entries`;
  }
  return value.map((item) => genreLabel(String(item), locale === 'he' ? 'he' : 'en')).join(', ');
}

// One value, as the reader sees it. Tri-state: a value, absent on this side, or
// a token whose closed set this surface has words for.
export function valueText(file, field, value, locale) {
  if (value === null || value === undefined || value === '') return pairText(NOT_SET, locale);
  if (typeof value === 'boolean') return pairText(value ? SWITCH_ON : SWITCH_OFF, locale);
  if (Array.isArray(value)) return listText(value, locale);
  if (typeof value === 'object') return pairText(NOT_SET, locale);
  const words = file === 'settings' ? VALUE_WORDS[String(field ?? '')] : null;
  const found = words ? words[String(value)] : null;
  return found ? pairText(found, locale) : String(value);
}

// Every scalar inside one side of a change, by the path it sits at. A list is a
// leaf of its own: its order is the value, so splitting it would print a field
// that is not in the store.
function sideLeaves(value, prefix, into) {
  const found = into || new Map();
  const at = prefix || '';
  if (isObject(value)) {
    Object.keys(value).forEach((key) => sideLeaves(value[key], at ? `${at}.${key}` : key, found));
    return found;
  }
  if (value !== undefined) found.set(at, value);
  return found;
}

function sameValue(one, other) {
  if (Array.isArray(one) || Array.isArray(other)) return JSON.stringify(one) === JSON.stringify(other);
  return one === other;
}

function moreText(rest, name, locale) {
  return locale === 'he' ? `עוד ${rest} שדות תחת ${name} השתנו גם הם` : `${rest} more fields under ${name} also changed`;
}

// One changed row, as the rows a person reads. A flat value is one row. A value
// that holds other values is the fields inside it that differ, so the decision
// is made on the difference rather than on a record cut mid-key.
export function changeRows(file, row, locale) {
  const field = String((row && row.field) ?? '');
  const now = row ? row.from : undefined;
  const atPoint = row ? row.to : undefined;
  if (!isObject(now) && !isObject(atPoint)) {
    const held = file === 'settings' && field === CHANNEL_FIELD && atPoint && atPoint !== now;
    return [{
      key: field,
      field: fieldLabel(file, field, locale),
      cur: valueText(file, field, now, locale),
      ver: held ? pairText(CHANNEL_HELD, locale) : valueText(file, field, atPoint, locale),
    }];
  }
  const nowLeaves = sideLeaves(now);
  const pointLeaves = sideLeaves(atPoint);
  const paths = [...new Set([...nowLeaves.keys(), ...pointLeaves.keys()])]
    .filter((path) => !sameValue(nowLeaves.get(path), pointLeaves.get(path)));
  const rows = paths.slice(0, MAX_LEAVES).map((path) => ({
    key: path ? `${field}.${path}` : field,
    field: fieldLabel(file, path ? `${field}.${path}` : field, locale),
    cur: valueText(file, field, nowLeaves.get(path), locale),
    ver: valueText(file, field, pointLeaves.get(path), locale),
  }));
  if (paths.length > MAX_LEAVES) {
    rows.push({
      key: `${field}.rest`,
      field: moreText(paths.length - MAX_LEAVES, fieldLabel(file, field, locale), locale),
      cur: '',
      ver: '',
    });
  }
  return rows;
}
