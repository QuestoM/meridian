import {
  BroadcastIcon,
  CommercialIcon,
  GovernanceIcon,
  HistoryIcon,
  PlanIcon,
  SourcesIcon,
  TodayIcon,
} from './kairos-icons';

export const DOMAIN_DEFINITIONS = [
  { id: 'Today', icon: TodayIcon, en: 'Today', he: 'היום' },
  { id: 'Plan', icon: PlanIcon, en: 'Plan', he: 'תכנון' },
  { id: 'Broadcast', icon: BroadcastIcon, en: 'Broadcast', he: 'שידור' },
  { id: 'Commercial', icon: CommercialIcon, en: 'Commercial', he: 'מסחרי' },
  { id: 'Sources', icon: SourcesIcon, en: 'Sources', he: 'מקורות' },
  { id: 'Governance', icon: GovernanceIcon, en: 'Governance', he: 'ממשל וכללים' },
  { id: 'History', icon: HistoryIcon, en: 'History', he: 'היסטוריה' },
];

// Kept as pairs because Mabat's mention picker consumes this public shape.
export const navItems = DOMAIN_DEFINITIONS.map(({ id, icon }) => [id, icon]);

export const CONTEXT_ITEMS = {
  Today: [
    { id: 'overview', en: 'Overview', he: 'סקירה', target: { view: 'Today' } },
  ],
  Plan: [
    { id: 'objective', en: 'Objective', he: 'מטרה', target: { view: 'Plan', params: { plan: 'objective' } } },
    { id: 'run', en: 'Run', he: 'הרצה', target: { view: 'Plan', params: { plan: 'run' } } },
    { id: 'compare', en: 'Compare', he: 'השוואה', target: { view: 'Plan', params: { plan: 'compare' } } },
    { id: 'publish', en: 'Publish', he: 'הפצה', target: { view: 'Plan', params: { plan: 'publish' } } },
    { id: 'supply', en: 'Supply', he: 'היצע', target: { view: 'Plan', params: { plan: 'supply' } } },
    { id: 'board', en: 'Week board', he: 'לוח השבוע', target: { view: 'Plan', params: { plan: 'board' } } },
  ],
  Broadcast: [
    { id: 'day', en: 'Day timeline', he: 'ציר זמן יומי', target: { view: 'Broadcast', params: { broadcast: 'day' } } },
    { id: 'pods', en: 'Traffic pods', he: 'ברייקי טראפיק', target: { view: 'Broadcast', params: { broadcast: 'pods' } } },
    { id: 'library', en: 'Break library', he: 'ספריית ברייקים', target: { view: 'Broadcast', params: { broadcast: 'library' } } },
    { id: 'decisions', en: 'Manual decisions', he: 'החלטות ידניות', target: { view: 'Broadcast', params: { broadcast: 'decisions' } } },
  ],
  Commercial: [
    { id: 'clients', en: 'Clients', he: 'לקוחות', target: { view: 'Commercial', params: { clients: 'clients' } } },
    { id: 'money', en: 'Money', he: 'כסף', target: { view: 'Commercial', params: { clients: 'money' } } },
    { id: 'campaigns', en: 'Campaigns', he: 'קמפיינים', target: { view: 'Commercial', params: { clients: 'campaigns' } } },
    { id: 'pacing', en: 'Delivery', he: 'אספקה', target: { view: 'Commercial', params: { clients: 'pacing' } } },
    { id: 'advertisers', en: 'Advertisers', he: 'מפרסמים', target: { view: 'Commercial', params: { clients: 'advertisers' } } },
    { id: 'agencies', en: 'Agencies', he: 'סוכנויות', target: { view: 'Commercial', params: { clients: 'agencies' } } },
    { id: 'agreements', en: 'Trade agreements', he: 'הסכמי סחר', target: { view: 'Commercial', params: { clients: 'agreements' } } },
  ],
  Sources: [
    { id: 'inputs', en: 'Inputs', he: 'קלטים', target: { view: 'Sources', params: { sources: 'inputs' } } },
    { id: 'files', en: 'Files', he: 'קבצים', target: { view: 'Sources', params: { sources: 'files' } } },
    { id: 'downloads', en: 'Reports', he: 'דוחות', target: { view: 'Sources', params: { sources: 'downloads' } } },
  ],
  Governance: [
    { id: 'restrictions', en: 'Restrictions', he: 'הגבלות', target: { view: 'Governance', params: { rules: 'restrictions' } } },
    { id: 'licence', en: 'Licence', he: 'רישיון', target: { view: 'Governance', params: { rules: 'licence' } } },
    { id: 'rate_card', en: 'Rate card', he: 'מחירון', target: { view: 'Governance', params: { rules: 'rate_card' } } },
    { id: 'calendar', en: 'Calendar', he: 'לוח אירועים', target: { view: 'Governance', params: { rules: 'calendar' } } },
    { id: 'channel', en: 'Channel & model', he: 'ערוץ ומודל', target: { view: 'Governance', params: { rules: 'channel' } } },
    { id: 'levers', en: 'Planning levers', he: 'מנופי תכנון', target: { view: 'Governance', params: { rules: 'levers' } } },
    { id: 'model', en: 'Company model', he: 'מודל החברה', companyOnly: true, target: { view: 'Model' } },
  ],
  History: [
    { id: 'changes', en: 'Changes & restore', he: 'שינויים ושחזור', target: { view: 'History' } },
  ],
};

const CANONICAL_VIEWS = new Set(DOMAIN_DEFINITIONS.map((domain) => domain.id).concat('Model'));

// Every old address resolves to the same capability through one of the seven
// domains. These defaults are applied only when the old URL did not already
// carry a more specific scoped query value.
export const LEGACY_TARGETS = {
  Overview: { view: 'Today' },
  Optimizer: { view: 'Plan', params: { plan: 'objective' } },
  Schedule: { view: 'Plan', params: { plan: 'board' } },
  Inventory: { view: 'Plan', params: { plan: 'supply' } },
  Forecasts: { view: 'Plan', params: { plan: 'compare' } },
  'Break Library': { view: 'Broadcast', params: { broadcast: 'library' } },
  Overrides: { view: 'Broadcast', params: { broadcast: 'decisions' } },
  Campaigns: { view: 'Commercial', params: { clients: 'campaigns' } },
  Advertisers: { view: 'Commercial', params: { clients: 'advertisers' } },
  Agencies: { view: 'Commercial', params: { clients: 'agencies' } },
  Data: { view: 'Sources', params: { sources: 'inputs' } },
  Reports: { view: 'Sources', params: { sources: 'downloads' } },
  Settings: { view: 'Governance' },
  Calendar: { view: 'Governance', params: { rules: 'calendar' } },
  Pricing: { view: 'Governance', params: { rules: 'rate_card' } },
  Versions: { view: 'History' },
  Model: { view: 'Model' },
};

const PARAM_DOMAIN = {
  axis: 'Plan',
  plan: 'Plan',
  recommendation: 'Plan',
  broadcast: 'Broadcast',
  breakView: 'Broadcast',
  day: 'Broadcast',
  pod: 'Broadcast',
  clients: 'Commercial',
  client: 'Commercial',
  sources: 'Sources',
  source: 'Sources',
  sourceView: 'Sources',
  rules: 'Governance',
  modelSection: 'Governance',
  entry: 'History',
  historyKind: 'History',
  todaySection: 'Today',
};

const VALID_VALUES = {
  axis: new Set(['day', 'daypart', 'hour', 'type']),
  plan: new Set(['objective', 'run', 'compare', 'publish', 'supply', 'board']),
  broadcast: new Set(['day', 'pods', 'library', 'decisions']),
  breakView: new Set(['library', 'day', 'pod']),
  clients: new Set(['clients', 'money', 'campaigns', 'pacing', 'advertisers', 'agencies', 'agreements']),
  sources: new Set(['inputs', 'files', 'downloads']),
  source: new Set(['all', 'in_use', 'shadowed', 'not_read', 'empty', 'invalid', 'missing']),
  sourceView: new Set(['inputs', 'files', 'downloads']),
  rules: new Set(['restrictions', 'licence', 'rate_card', 'calendar', 'channel', 'levers']),
  modelSection: new Set(['gates', 'coverage', 'drift', 'candidates', 'training', 'versions', 'provenance']),
  todaySection: new Set(['economics', 'guardrails', 'yield']),
};

function decodedHash() {
  if (typeof window === 'undefined') return 'Today';
  try {
    return decodeURIComponent(window.location.hash.replace(/^#/, '')) || 'Today';
  } catch {
    return 'Today';
  }
}

function relativeUrl(url) {
  return `${url.pathname}${url.search}${url.hash}`;
}

function cleanScopedParams(url, domain, resetOwned = false) {
  Object.entries(PARAM_DOMAIN).forEach(([name, owner]) => {
    if (owner !== domain || resetOwned) url.searchParams.delete(name);
  });
}

function validateOwnedParams(url, domain) {
  Object.entries(PARAM_DOMAIN).forEach(([name, owner]) => {
    if (owner !== domain || !url.searchParams.has(name) || !VALID_VALUES[name]) return;
    if (!VALID_VALUES[name].has(url.searchParams.get(name))) url.searchParams.delete(name);
  });
}

export function domainForView(view) {
  if (view === 'Model') return 'Governance';
  if (CANONICAL_VIEWS.has(view)) return view;
  return LEGACY_TARGETS[view]?.view || 'Today';
}

export function domainLabel(domain, locale = 'he') {
  const record = DOMAIN_DEFINITIONS.find((item) => item.id === domain) || DOMAIN_DEFINITIONS[0];
  return locale === 'he' ? record.he : record.en;
}

export function contextItemsForDomain(domain, canAccessModel = false) {
  return (CONTEXT_ITEMS[domain] || []).filter((item) => !item.companyOnly || canAccessModel);
}

export function navigationUrl(label, params = {}) {
  if (typeof window === 'undefined') return '';
  const target = CANONICAL_VIEWS.has(label) ? { view: label } : (LEGACY_TARGETS[label] || { view: 'Today' });
  const domain = domainForView(target.view);
  const url = new URL(window.location.href);
  cleanScopedParams(url, domain, true);
  Object.entries({ ...(target.params || {}), ...params }).forEach(([name, value]) => {
    if (value === null || value === undefined || value === '') url.searchParams.delete(name);
    else url.searchParams.set(name, String(value));
  });
  url.hash = encodeURIComponent(target.view);
  return relativeUrl(url);
}

// Feature-owned drawers and error states can request shell navigation without
// importing the shell component or forcing a document reload. The mutable
// detail flag supplies a standalone fallback for component harnesses.
export function requestNavigation(view, params = {}) {
  if (typeof window === 'undefined') return false;
  const detail = { view, params, handled: false };
  window.dispatchEvent(new CustomEvent('kairos:navigate', { detail }));
  if (!detail.handled) {
    const next = navigationUrl(view, params);
    if (next) window.location.assign(next);
  }
  return detail.handled;
}

export function routeFromLocation({ fallbackView = 'Today', canAccessModel = true } = {}) {
  if (typeof window === 'undefined') {
    return { view: 'Today', domain: 'Today', assistant: false, normalizedUrl: '' };
  }

  const raw = decodedHash();
  const assistant = raw === 'Assistant';
  let target;
  if (assistant) {
    const safeFallback = CANONICAL_VIEWS.has(fallbackView) && fallbackView !== 'Model' ? fallbackView : 'Today';
    target = { view: safeFallback };
  } else if (CANONICAL_VIEWS.has(raw)) {
    target = { view: raw };
  } else {
    target = LEGACY_TARGETS[raw] || { view: 'Today' };
  }

  if (target.view === 'Model' && !canAccessModel) target = { view: 'Governance' };
  const domain = domainForView(target.view);
  const url = new URL(window.location.href);
  cleanScopedParams(url, domain);
  validateOwnedParams(url, domain);
  Object.entries(target.params || {}).forEach(([name, value]) => {
    if (!url.searchParams.has(name)) url.searchParams.set(name, value);
  });
  url.hash = encodeURIComponent(target.view);

  return {
    view: target.view,
    domain,
    assistant,
    normalizedUrl: relativeUrl(url),
  };
}

// Compatibility exports used by older shell code and tests.
export function viewFromLocation() {
  return routeFromLocation().view;
}

export function gridAxisFromLocation() {
  if (typeof window === 'undefined') return 'day';
  const axis = new URLSearchParams(window.location.search).get('axis');
  return VALID_VALUES.axis.has(axis) ? axis : 'day';
}

export function contextItemIsActive(item, activeView) {
  if (!item || !item.target) return false;
  if (item.target.view === 'Model') return activeView === 'Model';
  if (domainForView(activeView) !== domainForView(item.target.view)) return false;
  const params = typeof window === 'undefined' ? new URLSearchParams() : new URLSearchParams(window.location.search);
  const expected = item.target.params || {};
  const pairs = Object.entries(expected);
  if (pairs.length === 0) return true;
  return pairs.every(([name, value]) => params.get(name) === String(value));
}
