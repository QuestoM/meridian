import React from 'react';

// Honest empty-state sentinel: null/undefined/non-finite input renders as a
// plain hyphen, never a confident 0 that hides missing data. Callers that mean
// a real zero should pass 0 (or value || 0) to opt into the numeric path.
export const EMPTY_VALUE = '-';

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

export function finiteNumber(value) {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatRetentionDelta(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) {
    return pageText(locale, 'Insufficient data', 'אין מספיק מדידות');
  }
  const points = number * 100;
  const sign = points > 0 ? '+' : '';
  return `${sign}${formatNumber(points, locale)}pp`;
}

export function Numeric({ children }) {
  return (
    <span className="numeric" dir="ltr">
      {children}
    </span>
  );
}

export function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

// stableSettingsKey produces an order-independent JSON signature for a settings
// object so the settings page can compare the in-progress draft against the
// saved settings (drives the "unsaved changes" affordance) without false
// positives from key order or fresh array identities.
export function stableSettingsKey(value) {
  if (Array.isArray(value)) {
    return `[${value.map(stableSettingsKey).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stableSettingsKey(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

// Format a plan calendar date for the basis line. ISO YYYY-MM-DD only; anything
// else is returned unchanged so we never invent a calendar day.
export function formatPlanDate(value, locale) {
  const text = String(value || '').trim();
  if (!/^\d{4}-\d{2}-\d{2}/.test(text)) return text;
  const iso = text.slice(0, 10);
  try {
    return new Date(`${iso}T00:00:00`).toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
  } catch {
    return iso;
  }
}

// Reads the backend's basis fields off the overview summary (which channel and
// which calendar span the headline numbers cover). Renders nothing when the
// backend does not provide them; no scope is ever invented here.
export function summaryBasisLabel(summary, locale) {
  const channel = typeof summary?.scope_channel === 'string' && summary.scope_channel.trim()
    ? summary.scope_channel.trim()
    : null;
  const nDates = finiteNumber(summary?.n_dates);
  const dateFrom = typeof summary?.date_from === 'string' && summary.date_from.trim()
    ? summary.date_from.trim()
    : null;
  const dateTo = typeof summary?.date_to === 'string' && summary.date_to.trim()
    ? summary.date_to.trim()
    : null;
  const parts = [];
  if (channel) {
    parts.push(pageText(locale, `your channel (${channel})`, `הערוץ שלכם (${channel})`));
  }
  if (dateFrom && dateTo) {
    const fromLabel = formatPlanDate(dateFrom, locale);
    const toLabel = formatPlanDate(dateTo, locale);
    if (dateFrom === dateTo) {
      parts.push(pageText(locale, `plan day ${fromLabel}`, `יום התוכנית ${fromLabel}`));
    } else {
      const span = nDates !== null
        ? pageText(
          locale,
          `${fromLabel} – ${toLabel} (${formatNumber(nDates, locale)} days on the saved plan)`,
          `${fromLabel} – ${toLabel} (${formatNumber(nDates, locale)} ימים בתוכנית השמורה)`,
        )
        : pageText(locale, `${fromLabel} – ${toLabel}`, `${fromLabel} – ${toLabel}`);
      parts.push(span);
    }
  } else if (nDates !== null) {
    parts.push(pageText(
      locale,
      `${formatNumber(nDates, locale)} days on the saved plan`,
      `${formatNumber(nDates, locale)} ימים בתוכנית השמורה`,
    ));
  }
  return parts.length ? parts.join(', ') : null;
}
