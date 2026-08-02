// Pure helpers for the Advertisers MANAGEMENT ZONE.
// Framework-free so they are trivially testable and reusable. These merge the
// /api/advertisers/stats aggregate with the editable advertiser rows, describe
// the four effect types, and drive the card grid's search / filter / sort.

import { pageText } from './advertisers-helpers';
import { nameSearchHaystack, sortByDisplayName } from './advertiser-name-helpers';

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

// Index the identity read by the rules row each advertiser is bound to, so a
// card can carry the money of the advertiser it actually prices. A row bound to
// nobody appears in no key here, which is what makes its dash honest rather than
// a missing lookup.
export function indexIdentityByRow(payload) {
  const list = payload && Array.isArray(payload.advertisers) ? payload.advertisers : [];
  const map = new Map();
  list.forEach((record) => {
    const boundTo = record && record.rules ? record.rules.advertiser_id : null;
    if (boundTo) {
      map.set(String(boundTo), record);
    }
  });
  return map;
}

// Attach the bound advertiser's identity and money to one rules row. Nothing is
// invented: an unbound row gets nulls and the reason, and a bound row with no
// priced spot keeps its null and the ledger's own reason for it.
export function mergeRowWithIdentity(row, identityIndex) {
  const record = identityIndex.get(String(row.advertiser_id)) || null;
  const money = record && record.money ? record.money : null;
  return {
    ...row,
    bound_advertiser: record ? record.shown_name : '',
    bound_spots: money ? money.spots : null,
    revenue: money && money.spots ? money.gross : null,
    revenue_net: money && money.spots ? money.net : null,
    revenue_basis: money ? money.basis : '',
    revenue_reason: money ? money.reason : '',
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

// Apply the management-zone search + filter. Search matches the display name
// (both locales), the raw id and notes; filters: all | with-rules |
// custom-premium | conflicts.
export function filterManaged(rows, { search, filter }) {
  const term = (search || '').trim().toLowerCase();
  return (rows || []).filter((row) => {
    if (term) {
      const haystack = `${row.advertiser_id || ''} ${row.notes || ''} ${nameSearchHaystack(row)}`.toLowerCase();
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

// Sort a merged list. Keys: name | id | rules-desc | premium-desc | premium-asc.
// 'name' sorts by display name with unnamed raw-token records grouped last.
// Premium sorts on avg_effective_premium when present, baseline otherwise.
export function sortManaged(rows, sortKey, locale) {
  const list = [...(rows || [])];
  const premiumOf = (row) => Number(row.avg_effective_premium ?? row.default_premium ?? 1);
  if (sortKey === 'name') {
    return sortByDisplayName(list, locale);
  }
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
    // A row prices somebody only when its name cell carries an advertiser, so
    // this count is the real reach of the store rather than its row count.
    bound: list.filter((row) => String(row.name ?? '').trim()).length,
    withRules: list.filter((row) => totalRules(row) > 0).length,
    totalRules: list.reduce((sum, row) => sum + totalRules(row), 0),
    conflicts: list.reduce((sum, row) => sum + conflictCount(row), 0),
  };
}

// Why this row's revenue reads the way it does. Three states, never one blank:
// a bound row with priced spots names the daily file the figure came from, a
// bound row without one carries the ledger's own reason, and an unbound row says
// that it prices nobody and where it would be bound.
export function revenueProvenance(row, locale) {
  if (!String(row?.name ?? '').trim()) {
    return pageText(
      locale,
      'This pricing row carries no advertiser name, so it prices nobody and has no money. Name it on the client record.',
      'שורת התמחור הזו אינה נושאת שם מפרסם, ולכן היא אינה מתמחרת אף אחד ואין לה כסף. תנו לה שם בכרטיס הלקוח.',
    );
  }
  if (row?.revenue === null || row?.revenue === undefined) {
    return row?.revenue_reason || pageText(
      locale,
      'This advertiser has no priced spot in the daily file being read.',
      'למפרסם הזה אין תשדיר מתומחר בקובץ היומי הנקרא.',
    );
  }
  return pageText(
    locale,
    `Source: the priced daily ledger, ${row.revenue_basis}, gross before the agency rebate`,
    `מקור: הפנקס היומי המתומחר, ⁦${row.revenue_basis}⁩, ברוטו לפני רבייט הסוכנות`,
  );
}

// The honest provenance string for the pending revenue/profitability stat.
export function revenuePendingTooltip(locale) {
  return pageText(
    locale,
    'Source: the daily spot-pricing path (not yet available)',
    'מקור: מסלול תמחור הספוטים היומי (טרם זמין)',
  );
}
