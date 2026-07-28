// Shared, dependency-free helpers for the insights surface components
// (YieldView, ScenarioCompare, GoldBreakManager, MakeGoodAlerts) and the
// upgraded frontier chart. These mirror the formatters in TVBreakDashboard.jsx
// so each surface stays self-contained without reaching into that 3.8k-line file.

// Empty default = same origin. In Vite dev the /api proxy reaches the Kairos
// API; in production the API serves the built SPA. An absolute override is only
// for split deployments.
export const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

export function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
}

export function finiteNumber(value) {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

// Honest empty-state sentinel: null/undefined/non-finite input renders as a
// plain hyphen, never a confident 0 that hides missing data. Callers that mean
// a real zero should pass 0 (or value || 0) to opt into the numeric path.
const EMPTY_VALUE = '-';

export function formatNumber(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  // Full grouped digits, never compact. Compact notation on a data count (e.g.
  // "2M" for 1,571,836) hides material differences and reads as dishonest, so do
  // not add notation:'compact' or drop precision here for large counts.
  return number.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: 1,
  });
}

export function formatPercent(value, locale = 'en') {
  if (finiteNumber(value) === null) return EMPTY_VALUE;
  return `${formatNumber(value, locale)}%`;
}

// Precise currency for DATA VALUES (tooltips, readout cards, deltas, stat
// figures). Compact notation with zero decimals hides material differences: it
// renders 1,571,836 and 1,100,000 both as "1M" and makes a 465,000 delta
// invisible, which is a legibility AND honesty failure. So the compact branch
// carries two fraction digits (1,571,836 -> "1.57M", 2,040,000 -> "2.04M") while
// minimumFractionDigits:0 keeps round values clean (2,000,000 -> "2M"). A 10,000
// ILS gap at the millions scale stays distinguishable (1.57M vs 1.58M). Axis
// ticks are the ONLY place compact-coarse is acceptable: use formatCurrencyAxis
// there, never this. Do not lower the fraction digits here or widen the compact
// threshold to swallow the 100K band; that reintroduces the trap.
export function formatCurrency(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const compact = Math.abs(number) >= 100000;
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 2 : 0,
    minimumFractionDigits: 0,
  });
  return formatter.format(number);
}

// Coarse currency for CHART AXIS TICKS ONLY, where space is tight and the label
// just conveys scale (1.6M, 465K, 12.5M). This is the deliberate compact-coarse
// exception to formatCurrency. Never use it for a data value the operator reads
// as a figure (tooltip point, readout card, delta, stat) - those must stay
// precise via formatCurrency so a 10,000 ILS difference is not rounded away.
export function formatCurrencyAxis(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    notation: 'compact',
    maximumFractionDigits: 1,
    minimumFractionDigits: 0,
  });
  return formatter.format(number);
}

export function formatMinutes(seconds, locale = 'en') {
  const number = finiteNumber(seconds);
  if (number === null) return EMPTY_VALUE;
  const minutes = Math.round(number / 60);
  return locale === 'he' ? `${minutes.toLocaleString('he-IL')} דק׳` : `${minutes.toLocaleString()} min`;
}

// Seconds shown as-is (yield-per-second works in seconds, not minutes).
export function formatSeconds(seconds, locale = 'en') {
  const number = Math.round(Number(seconds || 0));
  return locale === 'he' ? `${number.toLocaleString('he-IL')} שנ׳` : `${number.toLocaleString()} s`;
}

// Yield-per-second is a small currency rate; keep two fraction digits so a
// value like 12.4 ILS/s is not rounded into a misleading flat number.
export function formatRate(value, locale = 'en') {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

// Engine daypart keys (kairos/data/dayparts.py, plus the yield endpoint's
// "unclassified" fallback) mapped to operator-facing labels. The Hebrew names
// mirror the engine's own daypart table. An unknown key passes through
// verbatim so real data is never dropped or renamed.
const DAYPART_LABELS = {
  morning: ['Morning', 'בוקר'],
  noon: ['Noon', 'צהריים'],
  evening: ['Evening', 'ערב'],
  prime: ['Prime time', 'פריים טיים'],
  night: ['Night', 'לילה'],
  unclassified: ['Unclassified', 'ללא סיווג'],
};

export function daypartLabel(key, locale = 'en') {
  const text = String(key ?? '').trim();
  const entry = DAYPART_LABELS[text.toLowerCase()];
  if (!entry) return text;
  return locale === 'he' ? entry[1] : entry[0];
}

// Classifier program-type vocabulary (the saved schedule's program_type column)
// with Hebrew labels, mirroring the map TVBreakDashboard uses so genre names
// never leak as raw English into Hebrew surfaces. English shows the engine
// value itself; unknown values pass through verbatim.
const PROGRAM_TYPE_LABELS_HE = {
  News: 'חדשות',
  Reality: 'ריאליטי',
  Drama: 'דרמה',
  Sports: 'ספורט',
  Comedy: 'קומדיה',
  Promo: 'פרומו',
  Kids: 'ילדים',
  Children: 'ילדים',
  Digital: 'דיגיטל',
  Documentary: 'דוקומנטרי',
  Lifestyle: 'לייפסטייל',
  'Morning Program': 'תוכנית בוקר',
  Music: 'מוזיקה',
  Religious: 'תוכן דתי',
  'Special Event': 'אירוע מיוחד',
  'Talk Show': 'תוכנית אירוח',
  Other: 'אחר',
  Mixed: 'מעורב',
};

export function programTypeLabel(value, locale = 'en') {
  const text = String(value ?? '').trim();
  if (!text) return '';
  if (locale !== 'he') return text;
  return PROGRAM_TYPE_LABELS_HE[text] || text;
}
