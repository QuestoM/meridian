// Shared, dependency-free helpers for the Phase B surface components
// (YieldView, ScenarioCompare, GoldBreakManager, MakeGoodAlerts) and the
// upgraded frontier chart. These mirror the formatters in TVBreakDashboard.jsx
// so each surface stays self-contained without reaching into that 3.8k-line file.

export const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

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
  return number.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: 1,
  });
}

export function formatPercent(value, locale = 'en') {
  if (finiteNumber(value) === null) return EMPTY_VALUE;
  return `${formatNumber(value, locale)}%`;
}

export function formatCurrency(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const magnitude = Math.abs(number);
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: magnitude >= 100000 ? 0 : 1,
    notation: magnitude >= 100000 ? 'compact' : 'standard',
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

export async function fetchJsonOrError(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}
