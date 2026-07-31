import { useEffect, useState } from 'react';
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

export function useKairosData(refreshKey = 0) {
  const [state, setState] = useState({
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
  });

  useEffect(() => {
    let active = true;
    async function load() {
      const [
        overviewResult,
        scheduleResult,
        inventoryResult,
        breakLibraryResult,
        campaignsResult,
        forecastsResult,
        reportsResult,
        filesResult,
        impactResult,
        parametersResult,
        breakOperationsResult,
      ] = await Promise.all([
        fetchJson('/api/overview', fallbackOverview),
        fetchJson('/api/schedule', fallbackSchedule),
        fetchJson('/api/inventory', fallbackInventory),
        fetchJson('/api/break-library', fallbackBreakLibrary),
        fetchJson('/api/campaigns', fallbackCampaigns),
        fetchJson('/api/forecasts', fallbackForecasts),
        fetchJson('/api/reports', fallbackReports),
        fetchJson('/api/files', fallbackFiles),
        fetchJson('/api/impact', fallbackImpact),
        fetchJson('/api/parameters', fallbackParameters),
        fetchJson('/api/break-operations', fallbackSchedule.break_operations),
      ]);
      if (!active) return;
      const results = [
        overviewResult,
        scheduleResult,
        inventoryResult,
        breakLibraryResult,
        campaignsResult,
        forecastsResult,
        reportsResult,
        filesResult,
        impactResult,
        parametersResult,
        breakOperationsResult,
      ];
      const schedulePayload = {
        ...scheduleResult.data,
        break_operations: scheduleResult.data?.break_operations || breakOperationsResult.data,
      };
      setState({
        overview: overviewResult.data,
        schedule: schedulePayload,
        inventory: inventoryResult.data,
        breakLibrary: breakLibraryResult.data,
        campaigns: campaignsResult.data,
        forecasts: forecastsResult.data,
        reports: reportsResult.data,
        files: filesResult.data,
        impact: impactResult.data,
        parameters: parametersResult.data,
        breakOperations: breakOperationsResult.data,
        // Online reflects the overview fetch (the app's backbone payload); a few
        // failing side endpoints degrade to an honest partial state instead of
        // flipping the whole app to offline.
        online: overviewResult.online,
        partial: overviewResult.online && !results.every((result) => result.online),
        loading: false,
        error: results.find((result) => result.error)?.error || null,
      });
    }
    load();
    return () => {
      active = false;
    };
  }, [refreshKey]);

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
