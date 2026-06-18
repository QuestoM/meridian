// Pure helpers for the Advertisers MANAGEMENT ZONE.
// Framework-free so they are trivially testable and reusable. These merge the
// /api/advertisers/stats aggregate with the editable advertiser rows, describe
// the four effect types, and drive the card grid's search / filter / sort.

import { pageText } from './advertisers-helpers';

// The four scoped-rule effect types, each with a stable colour tone and a
// bilingual label. Tones map to classes in advertiser-management.css. Order
// matches the backend effect_types list (premium, require, forbid, pressure).
export const EFFECT_META = [
  { key: 'premium', tone: 'teal', en: 'Coefficient', he: 'מקדם' },
  { key: 'require', tone: 'blue', en: 'Require', he: 'חובה' },
  { key: 'forbid', tone: 'red', en: 'Forbid', he: 'איסור' },
  { key: 'pressure', tone: 'muted', en: 'Placement', he: 'שיבוץ' },
];

// Look up the meta for one effect key, falling back to a neutral descriptor so
// an unknown backend effect is shown rather than silently dropped.
export function effectMeta(key) {
  return EFFECT_META.find((entry) => entry.key === key) || { key, tone: 'muted', en: key, he: key };
}

// Normalize the /stats response into a map keyed by advertiser_id so it can be
// merged against the editable rows without an O(n^2) scan.
export function indexStats(payload) {
  const list = payload && Array.isArray(payload.advertisers) ? payload.advertisers : [];
  const map = new Map();
  list.forEach((entry) => {
    if (entry && entry.advertiser_id != null) {
      map.set(String(entry.advertiser_id), entry);
    }
  });
  return map;
}

// Merge an editable advertiser row with its stats record. Stats are an
// enhancement: when the /stats call has not loaded (or the id is absent) the
// merged row carries null stat fields so the card can honestly show "-".
export function mergeRowWithStats(row, statsIndex) {
  const stats = statsIndex.get(String(row.advertiser_id)) || null;
  const conditionCount = Array.isArray(row.conditions) ? row.conditions.length : 0;
  return {
    ...row,
    // Prefer the engine-computed rule_count; fall back to the row's own
    // conditions array so a card is never blank while /stats is in flight.
    rule_count: stats ? stats.rule_count : conditionCount,
    effect_breakdown: stats ? stats.effect_breakdown : null,
    baseline_premium: stats ? stats.baseline_premium : null,
    avg_effective_premium: stats ? stats.avg_effective_premium : null,
    revenue: stats ? stats.revenue : null,
    profitability: stats ? stats.profitability : null,
    revenue_source: stats ? stats.revenue_source : null,
    stats_loaded: Boolean(stats),
  };
}

// Total scoped rules across one merged row's effect breakdown (or the
// rule_count fallback when the breakdown has not loaded).
export function totalRules(row) {
  if (row && row.effect_breakdown) {
    return Object.values(row.effect_breakdown).reduce((sum, count) => sum + Number(count || 0), 0);
  }
  return Number((row && row.rule_count) || 0);
}

// Count overlap findings flagged as hard conflicts on a merged row.
export function conflictCount(row) {
  const findings = row && Array.isArray(row.overlaps) ? row.overlaps : [];
  return findings.filter((finding) => finding && finding.kind === 'conflict').length;
}

// Format a premium multiplier for display: "1.20x" with two decimals, or "-"
// when the value is missing. NEVER fabricate: a null reads as a dash.
export function formatPremium(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return '-';
  }
  return `${Number(value).toFixed(2)}x`;
}

// Format the percent delta a premium multiplier implies (1.20 -> "+20%"),
// or null when there is no value or it is exactly rate card.
export function premiumDelta(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return null;
  }
  const pct = Math.round((Number(value) - 1) * 100);
  if (pct === 0) {
    return null;
  }
  return `${pct > 0 ? '+' : '−'}${Math.abs(pct)}%`;
}

// Apply the management-zone search + filter. Search matches id and notes;
// filters: all | with-rules | custom-premium | conflicts.
export function filterManaged(rows, { search, filter }) {
  const term = (search || '').trim().toLowerCase();
  return (rows || []).filter((row) => {
    if (term) {
      const haystack = `${row.advertiser_id || ''} ${row.notes || ''}`.toLowerCase();
      if (!haystack.includes(term)) {
        return false;
      }
    }
    if (filter === 'with-rules') {
      return totalRules(row) > 0;
    }
    if (filter === 'custom-premium') {
      const premium = row.avg_effective_premium ?? row.default_premium;
      return Number(premium ?? 1) !== 1;
    }
    if (filter === 'conflicts') {
      return conflictCount(row) > 0;
    }
    return true;
  });
}

// Sort a merged list. Keys: id | rules-desc | premium-desc | premium-asc.
// Premium sorts on avg_effective_premium when present, baseline otherwise.
export function sortManaged(rows, sortKey) {
  const list = [...(rows || [])];
  const premiumOf = (row) => Number(row.avg_effective_premium ?? row.default_premium ?? 1);
  if (sortKey === 'rules-desc') {
    return list.sort((a, b) => totalRules(b) - totalRules(a));
  }
  if (sortKey === 'premium-desc') {
    return list.sort((a, b) => premiumOf(b) - premiumOf(a));
  }
  if (sortKey === 'premium-asc') {
    return list.sort((a, b) => premiumOf(a) - premiumOf(b));
  }
  return list.sort((a, b) => String(a.advertiser_id || '').localeCompare(String(b.advertiser_id || '')));
}

// Roll up zone-level totals for the header strip. All figures are real counts
// derived from the merged rows; nothing is estimated.
export function managementSummary(rows) {
  const list = rows || [];
  return {
    total: list.length,
    withRules: list.filter((row) => totalRules(row) > 0).length,
    totalRules: list.reduce((sum, row) => sum + totalRules(row), 0),
    conflicts: list.reduce((sum, row) => sum + conflictCount(row), 0),
  };
}

// The honest provenance string for the pending revenue/profitability stat.
export function revenuePendingTooltip(locale) {
  return pageText(
    locale,
    'Source: the daily spot-pricing path (not yet available)',
    'מקור: מסלול תמחור הספוטים היומי (טרם זמין)',
  );
}
