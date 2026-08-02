import React from 'react';
import { finiteNumber } from './format';
import { downloadJson } from './downloads';
import OverviewPage from '../today/OverviewPage';
import OptimizerWorkspace from '../plan/week/OptimizerWorkspace';
import SchedulePage from '../plan/week/SchedulePage';
import InventoryPage from '../plan/week/InventoryPage';
import ForecastsPage from '../plan/week/ForecastsPage';
import BreakLibraryPage from '../plan/break/BreakLibraryPage';
import CampaignsPage from '../clients/CampaignsPage';
import AdvertisersManager from '../clients/AdvertisersManager';
import AgencyManager from '../clients/AgencyManager';
import SettingsPanel from '../rules/SettingsPanel';
import ReportsPage from '../sources/ReportsPage';
import DataPage from '../sources/DataPage';
import OverrideConsole from '../plan/day/OverrideConsole';
import VersionsPage from '../history/VersionsPage';

// The workspace router, kept as a plain render function rather than a
// component so the React element tree is byte-for-byte what the single file
// produced: no extra layer, no new reconciliation boundary.
export function renderWorkspace({
  activeView,
  overview,
  schedule,
  inventory,
  breakLibrary,
  campaigns,
  forecasts,
  reports,
  files,
  impact,
  parameters,
  compliance,
  copy,
  locale,
  loading,
  notify,
  refreshKey,
  setRefreshKey,
  planEvents,
  settings,
  setActiveView,
  setActiveRecommendation,
  handleApplyFrontierFloor,
  applyWeightState,
  optimizerView,
  setOptimizerView,
  gridAxis,
  setGridAxis,
  showPrograms,
  setShowPrograms,
  showBreaks,
  setShowBreaks,
  showMetrics,
  setShowMetrics,
  selectedProgram,
  selectProgram,
  inspectorOpen,
  setInspectorOpen,
  activeRec,
  approved,
  rejected,
  optimizationPlan,
  scenario,
  approveRecommendation,
  rejectRecommendation,
  openRecommendationInOverrides,
  applySimilarRecommendations,
  handleRecomputeSchedule,
  recomputeState,
  overridePrefill,
  setOverridePrefill,
  saveState,
  persistSettings,
}) {
    const common = { overview, schedule, copy, locale, compliance, loading, notify, refreshKey };

    if (activeView === 'Overview') {
      return (
        <OverviewPage
          {...common}
          planEvents={planEvents}
          files={files}
          setActiveView={setActiveView}
          onOpenRecommendation={(id) => {
            // Land on the Optimizer with THIS decision active, not the default.
            setActiveRecommendation(id);
            setActiveView('Optimizer');
          }}
          operatorChannel={settings.operator_channel || ''}
          savedRetentionFloor={finiteNumber(settings.min_retention_floor)}
          onApplyFrontierFloor={handleApplyFrontierFloor}
          applyWeightState={applyWeightState}
          refreshKey={refreshKey}
        />
      );
    }

    if (activeView === 'Optimizer') {
      return (
        <OptimizerWorkspace
          {...common}
          activeViewMode={optimizerView}
          gridAxis={gridAxis}
          showPrograms={showPrograms}
          showBreaks={showBreaks}
          showMetrics={showMetrics}
          selectedProgramKey={selectedProgram?.key}
          inspectorOpen={inspectorOpen}
          selectedProgram={selectedProgram}
          activeRec={activeRec}
          approved={approved}
          rejected={rejected}
          optimizationPlan={optimizationPlan}
          parameters={parameters}
          onViewChange={(view) => setOptimizerView(view)}
          onGridAxisChange={(axis) => setGridAxis(axis)}
          onTogglePrograms={(checked) => setShowPrograms(checked)}
          onToggleBreaks={(checked) => setShowBreaks(checked)}
          onToggleMetrics={() => setShowMetrics((current) => !current)}
          onSelectProgram={selectProgram}
          onCloseInspector={() => {
            setInspectorOpen(false);
            notify('Break detail panel closed.', 'פאנל פרטי הברייק נסגר.', { transient: true });
          }}
          onApprove={() => activeRec && approveRecommendation(activeRec.id)}
          onReject={() => activeRec && rejectRecommendation(activeRec.id)}
          onOpenInOverrides={() => activeRec && openRecommendationInOverrides(activeRec)}
          onApplySimilar={applySimilarRecommendations}
          onExport={(exportScope) => {
            downloadJson('kairos-break-detail.json', { exportScope, selectedProgram, recommendation: activeRec, scenario });
            notify('Break detail exported as JSON.', 'פרטי הברייק יוצאו כ־JSON.');
          }}
        />
      );
    }

    if (activeView === 'Schedule') {
      return <SchedulePage {...common} planEvents={planEvents} onRecompute={handleRecomputeSchedule} recomputeState={recomputeState} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    if (activeView === 'Inventory') {
      return <InventoryPage inventory={inventory} overview={overview} copy={copy} locale={locale} />;
    }

    if (activeView === 'Break Library') {
      return <BreakLibraryPage breakLibrary={breakLibrary} copy={copy} locale={locale} notify={notify} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    if (activeView === 'Campaigns') {
      return <CampaignsPage campaigns={campaigns} copy={copy} locale={locale} refreshKey={refreshKey} />;
    }

    if (activeView === 'Forecasts') {
      return <ForecastsPage forecasts={forecasts} overview={overview} copy={copy} locale={locale} loading={loading} />;
    }

    // Calendar is now a tab inside the Rules (Settings) workspace. Old bookmarks
    // still resolve: the route activates Settings and sets the ?rules= param so
    // the workspace opens directly on the calendar section.
    if (activeView === 'Calendar') {
      if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search);
        params.set('rules', 'calendar');
        window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}#${encodeURIComponent('Settings')}`);
      }
      setActiveView('Settings');
      return null;
    }

    if (activeView === 'Reports') {
      return <ReportsPage reports={reports} files={files} overview={overview} copy={copy} locale={locale} notify={notify} />;
    }

    if (activeView === 'Data') {
      return (
        <DataPage
          files={files}
          impact={impact}
          parameters={parameters}
          overview={overview}
          copy={copy}
          locale={locale}
          notify={notify}
          onGlobalRefresh={() => setRefreshKey((k) => k + 1)}
        />
      );
    }

    if (activeView === 'Advertisers') {
      return <AdvertisersManager copy={copy} locale={locale} notify={notify} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    if (activeView === 'Agencies') {
      return <AgencyManager copy={copy} locale={locale} notify={notify} setActiveView={setActiveView} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    // Pricing is now a tab inside the Rules (Settings) workspace. Old bookmarks
    // still resolve: the route activates Settings and sets the ?rules= param so
    // the workspace opens directly on the rate card section.
    if (activeView === 'Pricing') {
      if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search);
        params.set('rules', 'rate_card');
        window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}#${encodeURIComponent('Settings')}`);
      }
      setActiveView('Settings');
      return null;
    }

    if (activeView === 'Overrides') {
      return (
        <OverrideConsole
          copy={copy}
          locale={locale}
          notify={notify}
          onGlobalRefresh={() => setRefreshKey((k) => k + 1)}
          prefill={overridePrefill}
          onPrefillConsumed={() => setOverridePrefill(null)}
        />
      );
    }

    if (activeView === 'Versions') {
      return <VersionsPage locale={locale} notify={notify} />;
    }

    return (
      <SettingsPanel
        settings={settings}
        parameters={parameters}
        copy={copy}
        locale={locale}
        saveState={saveState}
        onSave={persistSettings}
        onRecompute={handleRecomputeSchedule}
        recomputeState={recomputeState}
        notify={notify}
        onGlobalRefresh={() => setRefreshKey((k) => k + 1)}
      />
    );
}
