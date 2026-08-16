import React, { lazy, Suspense } from 'react';
import { finiteNumber, pageText } from './format';

const OverviewPage = lazy(() => import('../today/OverviewPage'));
const PlanWeek = lazy(() => import('../plan/week/PlanWeek'));
const DayPage = lazy(() => import('../plan/day/DayPage'));
const PodPage = lazy(() => import('../plan/break/PodPage'));
const BreakLibraryPage = lazy(() => import('../plan/break/BreakLibraryPage'));
const OverrideDecisions = lazy(() => import('../plan/day/OverrideDecisions'));
const ClientsWorkspace = lazy(() => import('../clients/ClientsWorkspace'));
const SourcesPage = lazy(() => import('../sources/SourcesPage'));
const RulesWorkspace = lazy(() => import('../rules/RulesWorkspace'));
const VersionsPage = lazy(() => import('../history/VersionsPage'));

function currentParam(name, fallback = '') {
  if (typeof window === 'undefined') return fallback;
  return new URLSearchParams(window.location.search).get(name) || fallback;
}

function WorkspaceFallback({ locale }) {
  return (
    <div className="workspace-route-loading" role="status" aria-live="polite">
      <span className="workspace-route-spinner" aria-hidden="true" />
      {pageText(locale, 'Opening workspace…', 'פותח את סביבת העבודה…')}
    </div>
  );
}

export function WorkspaceRouter({
  activeView,
  workspaceKey,
  overview,
  schedule,
  inventory,
  breakLibrary,
  campaigns,
  reports,
  files,
  compliance,
  copy,
  locale,
  loading,
  notify,
  refreshKey,
  setRefreshKey,
  planEvents,
  settings,
  settingsAvailable,
  setActiveView,
  approved,
  rejected,
  approveRecommendation,
  rejectRecommendation,
  applySimilarRecommendations,
  openRecommendationInOverrides,
  handleApplyFrontierFloor,
  applyWeightState,
  onReviewPlanRun,
  overridePrefill,
  setOverridePrefill,
  saveState,
  persistSettings,
  parameters,
}) {
  const common = { overview, schedule, copy, locale, compliance, loading, notify, refreshKey };
  const globalRefresh = () => setRefreshKey((key) => key + 1);

  let workspace;

  if (activeView === 'Today') {
    workspace = (
      <OverviewPage
        key={workspaceKey}
        {...common}
        planEvents={planEvents}
        files={files}
        setActiveView={setActiveView}
        onOpenRecommendation={(id) => {
          setActiveView('Plan', { plan: 'objective', recommendation: id });
        }}
        operatorChannel={settings.operator_channel || ''}
        savedRetentionFloor={finiteNumber(settings.min_retention_floor)}
        onApplyFrontierFloor={handleApplyFrontierFloor}
        applyWeightState={applyWeightState}
      />
    );
  } else if (activeView === 'Plan') {
    workspace = (
      <PlanWeek
        key={workspaceKey}
        entrance="Optimizer"
        overview={overview}
        loading={loading}
        schedule={schedule}
        inventory={inventory}
        copy={copy}
        locale={locale}
        notify={notify}
        planEvents={planEvents}
        onGlobalRefresh={globalRefresh}
        recommendations={overview.recommendations}
        approvedRecommendations={approved}
        rejectedRecommendations={rejected}
        onApproveRecommendation={approveRecommendation}
        onRejectRecommendation={rejectRecommendation}
        onApplySimilarRecommendations={applySimilarRecommendations}
        onOpenRecommendationInOverrides={openRecommendationInOverrides}
        onOpenSources={() => setActiveView('Sources', { sources: 'files' })}
      />
    );
  } else if (activeView === 'Broadcast') {
    const broadcastView = currentParam('broadcast', 'day');
    if (broadcastView === 'pods') {
      workspace = (
        <section className="page-workspace broadcast-pods-workspace" key={workspaceKey}>
          <PodPage locale={locale} notify={notify} />
        </section>
      );
    } else if (broadcastView === 'library') {
      workspace = (
        <BreakLibraryPage
          key={workspaceKey}
          breakLibrary={breakLibrary}
          copy={copy}
          locale={locale}
          notify={notify}
          onGlobalRefresh={globalRefresh}
        />
      );
    } else if (broadcastView === 'decisions') {
      workspace = (
        <OverrideDecisions
          key={workspaceKey}
          copy={copy}
          locale={locale}
          notify={notify}
          onGlobalRefresh={globalRefresh}
          prefill={overridePrefill}
          onPrefillConsumed={() => setOverridePrefill(null)}
        />
      );
    } else {
      workspace = (
        <DayPage
          key={workspaceKey}
          locale={locale}
          notify={notify}
          onGlobalRefresh={globalRefresh}
          refreshKey={refreshKey}
        />
      );
    }
  } else if (activeView === 'Commercial') {
    workspace = (
      <ClientsWorkspace
        key={workspaceKey}
        view={currentParam('clients', 'clients')}
        campaigns={campaigns}
        copy={copy}
        locale={locale}
        notify={notify}
        onGlobalRefresh={globalRefresh}
        setActiveView={setActiveView}
        refreshKey={refreshKey}
      />
    );
  } else if (activeView === 'Sources') {
    workspace = (
      <SourcesPage
        key={workspaceKey}
        view={currentParam('sources', 'inputs')}
        files={files}
        overview={overview}
        reports={reports}
        locale={locale}
        notify={notify}
        onGlobalRefresh={globalRefresh}
      />
    );
  } else if (activeView === 'History') {
    workspace = <VersionsPage key={workspaceKey} locale={locale} notify={notify} />;
  } else {
    workspace = (
      <RulesWorkspace
        key={workspaceKey}
        showInternalNavigation={false}
        settings={settings}
        settingsAvailable={settingsAvailable}
        settingsLoading={loading}
        parameters={parameters}
        copy={copy}
        locale={locale}
        saveState={saveState}
        onSave={persistSettings}
        onRecompute={onReviewPlanRun}
        recomputeState="idle"
        notify={notify}
        onGlobalRefresh={globalRefresh}
      />
    );
  }

  return <Suspense fallback={<WorkspaceFallback locale={locale} />}>{workspace}</Suspense>;
}

export function renderWorkspace(props) {
  return <WorkspaceRouter {...props} />;
}
