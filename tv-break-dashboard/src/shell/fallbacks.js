// Offline fallbacks. Every field is null or empty on purpose: an unreachable
// API must drive the honest empty states, never invented numbers.

export const fallbackSettings = {
  profile_name: 'Israel commercial TV',
  locale: 'he',
  direction: 'rtl',
  chart_direction: 'ltr',
  timezone: 'Asia/Jerusalem',
  currency: 'ILS',
  effective_date: '2026-06-14',
  regulatory_source_url: 'https://www.rashut2.org.il/',
  max_ad_minutes_per_hour: 12,
  max_breaks_per_hour: 4,
  min_break_spacing_minutes: 7,
  min_retention_floor: 0.72,
  risk_lambda: 0.0,
  max_daily_ad_minutes: 160,
  protected_program_types: ['News', 'Kids', 'Children'],
  protected_program_max_ad_minutes_per_hour: 8,
  sponsorships_enabled: true,
  gold_breaks_enabled: true,
  gold_breaks_max_per_day: 3,
  require_manual_approval: true,
  notes: 'Configurable baseline. Validate with current counsel and broadcaster policy before production use.',
};

export const fallbackCompliance = {
  profile: fallbackSettings.profile_name,
  effective_date: fallbackSettings.effective_date,
  source_url: fallbackSettings.regulatory_source_url,
  // API offline: there is no schedule to evaluate, so report unknown rather than
  // asserting compliance against invented observed values.
  status: 'unknown',
  disclaimer: fallbackSettings.notes,
  checks: [],
};

export const fallbackOverview = {
  brand: 'Kairos',
  workspace: 'KAI Network',
  data_freshness: new Date().toISOString(),
  // API offline: do not fabricate metrics. Null fields drive the honest empty
  // states in the consuming components rather than confident invented numbers.
  summary: {
    total_breaks: null,
    total_ad_seconds: null,
    projected_revenue: null,
    average_retention: null,
    risk_score: null,
  },
  source_counts: null,
  recommendations: [],
  frontier: [],
  settings: fallbackSettings,
  compliance: fallbackCompliance,
};

// API offline: do not fabricate a schedule. Empty rows/programs/breaks drive the
// honest empty states in the consuming components, matching fallbackOverview
// (nulled metrics) and fallbackInventory (empty rows).
export const fallbackSchedule = {
  rows: [],
  break_operations: {
    programs: [],
    breaks: [],
    summary: { programs: 0, breaks: 0, ad_seconds: 0, revenue: 0 },
  },
  break_schedule: [],
};

export const fallbackInventory = {
  summary: { spots: 0, revenue: 0, seconds: 0 },
  scope_channel: null,
  by_daypart: [],
  by_hour: [],
};

export const fallbackBreakLibrary = { breaks: [] };
export const fallbackCampaigns = { campaigns: [] };
export const fallbackForecasts = { by_day: [], scenarios: [] };
export const fallbackReports = { reports: [] };
export const fallbackFiles = { files: [] };
export const fallbackImpact = {
  program_type_impacts: [],
  position_impacts: [],
  length_impacts: [],
  coefficient_impacts: {
    source: 'unavailable',
    metadata: {},
    program_type: [],
    position: [],
    length: [],
  },
};

export const fallbackParameters = {
  settings: fallbackSettings,
  guardrails: {},
  assumptions: {},
  pricing: {},
};
