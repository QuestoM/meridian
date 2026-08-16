import { useEffect, useRef, useState } from 'react';
import { fetchJson } from './api';
import {
  fallbackBreakLibrary,
  fallbackCampaigns,
  fallbackFiles,
  fallbackForecasts,
  fallbackImpact,
  fallbackInventory,
  fallbackOverview,
  fallbackParameters,
  fallbackReports,
  fallbackSchedule,
} from './fallbacks';

const INITIAL_STATE = {
  overview: fallbackOverview,
  schedule: fallbackSchedule,
  inventory: fallbackInventory,
  breakLibrary: fallbackBreakLibrary,
  campaigns: fallbackCampaigns,
  forecasts: fallbackForecasts,
  reports: fallbackReports,
  files: fallbackFiles,
  impact: fallbackImpact,
  parameters: fallbackParameters,
  breakOperations: fallbackSchedule.break_operations,
  online: false,
  partial: false,
  loading: true,
  error: null,
};

// Transport fallbacks carry an explicit sentinel. Consumers that present or
// write saved settings must use this boundary instead of treating plausible
// display defaults as server-confirmed policy.
export function savedSettingsFromOverview(overview) {
  if (!overview || overview._unavailable === true) return null;
  const settings = overview.settings;
  if (!settings || settings._unavailable === true) return null;
  return settings;
}

function routeParams(search) {
  if (typeof URLSearchParams === 'undefined') return { get: () => null };
  if (search instanceof URLSearchParams) return search;
  if (typeof search === 'string') return new URLSearchParams(search);
  if (typeof window !== 'undefined') return new URLSearchParams(window.location.search);
  return new URLSearchParams();
}

// The shell owns orientation data; destinations own their operational reads.
// Keep this list deliberately small. Plan still receives schedule and inventory
// from the shell because its existing refresh contract relies on those props.
// Other destinations already fetch their own records and must not pay for a
// second global copy of them.
export function kairosDataResources(activeView = 'Today', search) {
  const params = routeParams(search);
  const resources = ['overview'];

  if (activeView === 'Today') {
    resources.push('schedule', 'files');
  } else if (activeView === 'Plan') {
    resources.push('schedule', 'inventory');
  } else if (activeView === 'Broadcast') {
    const view = params.get('broadcast') || 'day';
    const libraryView = params.get('breakView') || 'library';
    if (view === 'library' && libraryView === 'library') resources.push('breakLibrary');
  } else if (activeView === 'Sources') {
    const view = params.get('sourceView') || params.get('sources') || 'inputs';
    if (view === 'files') resources.push('files');
    if (view === 'downloads') resources.push('files', 'reports');
  } else if (activeView === 'Governance') {
    // A bare Governance address resolves from the account's job after mount,
    // and may land on Planning levers. Parameters therefore belong to the
    // destination profile rather than only to an explicit ?rules=levers URL.
    const section = params.get('rules');
    if (!section || section === 'levers') resources.push('parameters');
  }

  return resources;
}

async function loadSchedule() {
  const scheduleResult = await fetchJson('/api/schedule', fallbackSchedule);
  const embeddedOperations = scheduleResult.online && scheduleResult.data?.break_operations;
  let operationsResult = null;

  // Current servers embed the same board in /api/schedule. Older servers did
  // not, so retain the compatibility read only when it is actually necessary.
  if (!embeddedOperations) {
    operationsResult = await fetchJson('/api/break-operations', fallbackSchedule.break_operations);
  }

  const breakOperations = embeddedOperations || operationsResult?.data || fallbackSchedule.break_operations;
  return {
    data: { ...(scheduleResult.data || fallbackSchedule), break_operations: breakOperations },
    breakOperations,
    online: scheduleResult.online && (!operationsResult || operationsResult.online),
    error: scheduleResult.error || operationsResult?.error || null,
  };
}

const RESOURCE_LOADERS = {
  overview: () => fetchJson('/api/overview', fallbackOverview),
  schedule: loadSchedule,
  inventory: () => fetchJson('/api/inventory', fallbackInventory),
  breakLibrary: () => fetchJson('/api/break-library', fallbackBreakLibrary),
  reports: () => fetchJson('/api/reports', fallbackReports),
  files: () => fetchJson('/api/files', fallbackFiles),
  parameters: () => fetchJson('/api/parameters', fallbackParameters),
};

function mergeResource(state, name, record) {
  if (!record) return state;
  if (name === 'schedule') {
    return {
      ...state,
      schedule: record.data,
      breakOperations: record.breakOperations || record.data?.break_operations || fallbackSchedule.break_operations,
    };
  }
  return { ...state, [name]: record.data };
}

function routeStatus(state, resources, cache, version, loading) {
  const overview = cache.get('overview');
  const overviewCurrent = overview?.version === version;
  const currentRecords = resources
    .map((name) => cache.get(name))
    .filter((record) => record?.version === version);
  const online = overviewCurrent ? overview.online : state.online;
  return {
    ...state,
    online,
    partial: Boolean(overviewCurrent && overview.online && currentRecords.some((record) => !record.online)),
    loading,
    error: currentRecords.find((record) => record.error)?.error || null,
  };
}

export function useKairosData(refreshKey = 0, activeView = 'Today') {
  const [state, setState] = useState(INITIAL_STATE);
  const cache = useRef(new Map());
  const inFlight = useRef(new Map());
  const resources = kairosDataResources(activeView);
  const resourceKey = resources.join('|');

  useEffect(() => {
    let active = true;
    const selected = resourceKey.split('|').filter(Boolean);
    const missing = selected.filter((name) => cache.current.get(name)?.version !== refreshKey);

    function request(name) {
      const key = `${refreshKey}:${name}`;
      const existing = inFlight.current.get(key);
      if (existing) return existing;
      const loader = RESOURCE_LOADERS[name];
      const promise = loader()
        .then((result) => {
          const current = cache.current.get(name);
          if (!current || current.version <= refreshKey) {
            cache.current.set(name, { ...result, version: refreshKey });
          }
          return result;
        })
        .finally(() => {
          if (inFlight.current.get(key) === promise) inFlight.current.delete(key);
        });
      inFlight.current.set(key, promise);
      return promise;
    }

    function publish(loading) {
      setState((current) => {
        let next = current;
        for (const name of selected) {
          const record = cache.current.get(name);
          if (record?.version === refreshKey) next = mergeResource(next, name, record);
        }
        return routeStatus(next, selected, cache.current, refreshKey, loading);
      });
    }

    if (missing.length === 0) {
      publish(false);
      return () => {
        active = false;
      };
    }

    publish(true);
    Promise.all(missing.map(request)).then(() => {
      if (active) publish(false);
    });
    return () => {
      active = false;
    };
  }, [refreshKey, resourceKey]);

  // While the frontier is still computing on the server, poll the overview so
  // the chart self-heals on the default view instead of waiting for a manual
  // refresh. Bounded: one fetch about every 5 seconds, stopping when the status
  // flips away from computing or after roughly 2 minutes.
  const frontierStatus = state.overview?.frontier_status || '';
  useEffect(() => {
    if (state.loading || frontierStatus !== 'computing') return undefined;
    let active = true;
    let attempts = 0;
    const id = window.setInterval(async () => {
      attempts += 1;
      if (attempts > 24) {
        window.clearInterval(id);
        return;
      }
      const result = await fetchJson('/api/overview', null);
      if (!active || !result.online || !result.data) return;
      setState((current) => {
        const next = result.data;
        const statusChanged = (next.frontier_status || '') !== (current.overview?.frontier_status || '');
        const frontierChanged =
          JSON.stringify(next.frontier || []) !== JSON.stringify(current.overview?.frontier || []);
        // Re-render only on real movement, so the poll does not churn the app.
        if (!statusChanged && !frontierChanged) return current;
        const cached = cache.current.get('overview');
        if (cached) cache.current.set('overview', { ...cached, data: next });
        return { ...current, overview: next };
      });
    }, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [state.loading, frontierStatus]);

  return state;
}
