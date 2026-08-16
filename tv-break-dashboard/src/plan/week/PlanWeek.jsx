import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { pageText } from '../../shell/format';
import { queueWorkspaceContinuity } from '../../shell/workspace-continuity';
import { planEventWeekdayMap } from '../../rules/CalendarEventsModel';
// One destination, one stylesheet per component family, because the 450-line
// law applies to a stylesheet exactly as it applies to a module. No sheet
// overrides another: every selector belongs to the file it is declared in.
import './plan-week.css';
import './plan-week-panels.css';
import './plan-week-goal.css';
import './plan-week-palette.css';
import './plan-week-publish.css';
import './plan-week-basis.css';
import './plan-week-compare.css';
import { SECTION_IDS, SECTIONS, planWords, sectionForEntrance, sectionLabel } from './plan-week-model';
import { usePlanSurface } from './use-plan-surface';
import { useSectionData } from './use-section-data';
import { usePlanInventoryReadiness } from './use-plan-inventory-readiness';
import { usePlanOptimizationActions } from './use-plan-optimization-actions';
import { readInventory, readSchedule } from './plan-week-api';
import { usePlanKeyboard } from './use-plan-keyboard';
import { planCommands } from './plan-week-commands';
import CommandPalette from './CommandPalette';
import GoalStrip from './GoalStrip';
import PlanWeekHeader from './PlanWeekHeader';
import ObjectivePanel from './ObjectivePanel';
import RunPanel, { planScopeLine } from './RunPanel';
import ComparePanel from './ComparePanel';
import PublishPanel from './PublishPanel';
import SupplyPanel from './SupplyPanel';
import BoardPanel from './BoardPanel';
import RecommendationDecisionPanel from './RecommendationDecisionPanel';
import PlanConsequenceDialog from './PlanConsequenceDialog';
import PlanActionSafety from './PlanActionSafety';
import { PlanSectionDataGate, PlanSettingsGate } from './PlanStateGate';
import './plan-week-recommendations.css';
import './plan-week-instruments.css';
import './plan-week-board-v2.css';
import './plan-board-workbench.css';

// Plan, the week.
//
// One destination for the planner and the revenue owner, entered from whichever
// of the four surviving navigation entries a person clicked. The steps of JS-2
// are its own structure: choose the objective, run the plan, compare two ways of
// running it on revenue net of retention cost, freeze it. Supply and the week
// board sit beside those four because the same person reads them while doing it.
//
// The reference bars are Linear and Google Ads, and both are taken as mechanics
// rather than as decoration. From Linear: a command palette on Cmd K that prints
// each row's own shortcut, chorded moves, and view switching inside the content
// instead of in the navigation. From Google Ads: the goal and the progress
// against it read together on one strip, with the remedy on the same row as the
// diagnosis, and a state that is a published rule rather than a chart to
// interpret.

export function PlanWeek({
  entrance = 'Optimizer',
  overview,
  loading = false,
  schedule: providedSchedule,
  inventory: providedInventory,
  copy,
  locale,
  notify: providedNotify,
  planEvents,
  onGlobalRefresh,
  recommendations,
  approvedRecommendations,
  rejectedRecommendations,
  onApproveRecommendation,
  onRejectRecommendation,
  onApplySimilarRecommendations,
  onOpenRecommendationInOverrides,
  onOpenSources,
}) {
  const sectionFromAddress = useCallback(() => {
    if (typeof window === 'undefined') return sectionForEntrance(entrance);
    const addressed = new URLSearchParams(window.location.search).get('plan');
    return SECTION_IDS.includes(addressed) ? addressed : sectionForEntrance(entrance);
  }, [entrance]);
  const [section, setSection] = useState(sectionFromAddress);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [boardView, setBoardView] = useState('day');
  const [boardDate, setBoardDate] = useState(null);
  const [gridAxis, setGridAxis] = useState('day');
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  const [review, setReview] = useState(null);
  const recommendationFromAddress = useCallback(() => {
    if (typeof window === 'undefined') return '';
    return new URLSearchParams(window.location.search).get('recommendation') || '';
  }, []);
  const [selectedRecommendationId, setSelectedRecommendationId] = useState(recommendationFromAddress);
  const sectionRef = useRef(null);
  const words = useMemo(() => planWords(locale), [locale]);
  // Two of the four entrances are handed no notifier, so the destination
  // supplies one rather than letting a child call undefined.
  const notify = useMemo(() => providedNotify || (() => {}), [providedNotify]);
  const inventoryReadiness = usePlanInventoryReadiness();
  const surface = usePlanSurface({
    locale,
    notify: providedNotify,
    onGlobalRefresh,
    prepareCompare: section === 'compare',
    optimizationAllowed: inventoryReadiness.status === 'ready',
    inventoryReadiness,
    initialFreshness: overview?._unavailable === true ? null : overview?.schedule_freshness,
    initialFreshnessPending: loading,
  });
  const scheduleSection = useSectionData(providedSchedule, readSchedule, section === 'board');
  const inventorySection = useSectionData(providedInventory, readInventory, section === 'supply');
  const schedule = scheduleSection.data;
  const inventory = inventorySection.data;
  const dayEvents = useMemo(() => planEventWeekdayMap(planEvents), [planEvents]);

  useEffect(() => {
    setSection(sectionFromAddress());
  }, [sectionFromAddress]);

  useEffect(() => {
    function restoreAddressedSection() {
      setSection(sectionFromAddress());
      setSelectedRecommendationId(recommendationFromAddress());
    }
    window.addEventListener('popstate', restoreAddressedSection);
    return () => window.removeEventListener('popstate', restoreAddressedSection);
  }, [sectionFromAddress, recommendationFromAddress]);

  const selectRecommendation = useCallback((id) => {
    setSelectedRecommendationId(String(id || ''));
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    if (id) url.searchParams.set('recommendation', String(id));
    else url.searchParams.delete('recommendation');
    url.searchParams.set('plan', 'objective');
    window.history.pushState(
      { ...(window.history.state || {}), plan: 'objective', recommendation: id || null },
      '',
      `${url.pathname}${url.search}${url.hash}`,
    );
    setSection('objective');
  }, []);

  const go = useCallback((id, options = {}) => {
    if (!SECTION_IDS.includes(id)) return;
    setSection(id);
    if (typeof window !== 'undefined' && options.history !== false) {
      const url = new URL(window.location.href);
      url.searchParams.set('plan', id);
      window.history.pushState({ ...(window.history.state || {}), plan: id }, '', `${url.pathname}${url.search}${url.hash}`);
    }
    queueWorkspaceContinuity(() => {
      sectionRef.current?.focus({ preventScroll: true });
      const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      sectionRef.current?.scrollIntoView({ block: 'start', behavior: reduced ? 'auto' : 'smooth' });
    });
  }, []);

  // The two acts step 3 owes the planner, each with one implementation behind
  // both its doors: the card and the palette adopt through the same function,
  // and the day row and the sentence that names the decisive day open through
  // the same one.
  const adoptLeg = useCallback((leg) => {
    if (surface.adoptLeg(leg)) go('objective');
  }, [surface, go]);

  const optimization = usePlanOptimizationActions({
    surface, inventory: inventoryReadiness, locale, notify, go, setReview,
  });
  const { runNow, compareNow } = optimization;

  const requestPublish = useCallback((confirmCollapse) => {
    setReview({ kind: 'publish', confirmCollapse: Boolean(confirmCollapse) });
  }, []);

  const requestRestore = useCallback((version) => {
    if (!version) return;
    setReview({ kind: 'restore', version });
  }, []);

  const confirmReviewedAction = useCallback(async () => {
    const action = review;
    if (!action) return;
    setReview(null);
    if (action.kind === 'run') await optimization.confirmRun(action);
    else if (action.kind === 'publish') await surface.publish(action.confirmCollapse);
    else if (action.kind === 'restore') await surface.restore(action.version);
  }, [review, surface, optimization]);

  const openBoardDay = useCallback((date) => {
    if (!date) return;
    setBoardDate(String(date));
    setBoardView('day');
    go('board');
  }, [go]);

  const commands = useMemo(
    () => planCommands({
      go,
      words,
      surface,
      boardView,
      setBoardView,
      adoptLeg,
      runNow,
      compareNow,
      optimizationAllowed: optimization.optimizationAllowed,
      optimizationBlockedReason: optimization.blockedReason,
      requestPublish,
      openPalette: () => setPaletteOpen(true),
    }),
    [go, words, surface, boardView, adoptLeg, runNow, compareNow, optimization, requestPublish],
  );

  usePlanKeyboard({ commands, enabled: !paletteOpen });

  const scopeText = planScopeLine(schedule, locale);

  return (
    <section className="page-workspace plan-week" aria-label={words.place}>
      <PlanWeekHeader
        locale={locale}
        words={words}
        sections={SECTIONS}
        active={section}
        onGo={go}
        freshness={surface.freshness}
        freshnessState={surface.freshnessState}
        freshnessError={surface.freshnessError}
        runState={surface.runState}
        elapsed={surface.elapsed}
        onRun={runNow}
        runDisabled={!optimization.optimizationAllowed}
        runDisabledReason={optimization.blockedReason}
        versionCount={surface.versions.length}
        liveFrozenAs={surface.live?.frozen_as || null}
        scopeText={scopeText}
        onOpenPalette={() => setPaletteOpen(true)}
      />

      <GoalStrip progress={surface.progress} locale={locale} words={words} onGo={go} />
      <PlanActionSafety
        settingsState={surface.settingsState}
        settingsError={surface.settingsError}
        inventory={inventoryReadiness}
        locale={locale}
        onRetrySettings={surface.retrySettings}
        onRetryInventory={inventoryReadiness.retry}
        onOpenSources={onOpenSources}
      />

      <div className="plan-sections">
        <div
          ref={sectionRef}
          id={`plan-section-${section}`}
          className="plan-active-section"
          role="region"
          aria-labelledby={`plan-step-${section}`}
          tabIndex={-1}
        >
        {section === 'objective' && (
          <>
            <PlanSettingsGate
              state={surface.settingsState}
              error={surface.settingsError}
              locale={locale}
              onRetry={surface.retrySettings}
            >
              <ObjectivePanel
                draft={surface.draft}
                saved={surface.saved}
                dirty={surface.dirty}
                saveState={surface.saveState}
                adopted={surface.adopted}
                locale={locale}
                onChange={surface.changeDraft}
                onApplyTemplate={surface.applyTemplate}
                onSave={surface.saveObjective}
                onRevert={surface.revertDraft}
              />
            </PlanSettingsGate>
            <RecommendationDecisionPanel
              recommendations={recommendations}
              selectedId={selectedRecommendationId}
              approved={approvedRecommendations}
              rejected={rejectedRecommendations}
              locale={locale}
              notify={notify}
              onSelect={selectRecommendation}
              onApprove={onApproveRecommendation}
              onReject={onRejectRecommendation}
              onApplySimilar={onApplySimilarRecommendations}
              onOpenInOverrides={onOpenRecommendationInOverrides}
            />
          </>
        )}

        {section === 'run' && (
          <RunPanel
            locale={locale}
            words={words}
            freshness={surface.freshness}
            freshnessState={surface.freshnessState}
            runState={surface.runState}
            runProgress={surface.runProgress}
            runResult={surface.runResult}
            runError={surface.runError}
            elapsed={surface.elapsed}
            planScope={scopeText}
            onRun={runNow}
            actionDisabled={!optimization.optimizationAllowed}
            actionDisabledReason={optimization.blockedReason}
          />
        )}

        {section === 'compare' && (
          <PlanSettingsGate state={surface.settingsState} error={surface.settingsError} locale={locale} onRetry={surface.retrySettings}>
            <ComparePanel
              locale={locale}
              words={words}
              legA={surface.legA}
              legB={surface.legB}
              state={surface.compareState}
              payload={surface.comparePayload}
              error={surface.compareError}
              runWindow={surface.compareWindow}
              liveDays={surface.compareDays}
              prepared={surface.comparePrepared}
              actionDisabled={!optimization.optimizationAllowed}
              actionDisabledReason={optimization.blockedReason}
              onLegChange={surface.changeLeg}
              onCompare={compareNow}
              onCancel={surface.cancelCompare}
              onAdopt={adoptLeg}
              onOpenDay={openBoardDay}
            />
          </PlanSettingsGate>
        )}

        {section === 'publish' && (
          <PublishPanel
            locale={locale}
            words={words}
            versions={surface.versions}
            live={surface.live}
            canEdit={surface.canPublish}
            canEditReason={surface.canPublishReason}
            name={surface.versionName}
            note={surface.versionNote}
            publishState={surface.publishState}
            publishError={surface.publishError}
            selectedId={surface.selectedVersion}
            diff={surface.diff}
            onNameChange={surface.setVersionName}
            onNoteChange={surface.setVersionNote}
            onPublish={requestPublish}
            onSelect={surface.setSelectedVersion}
            onDiff={surface.loadDiff}
            onRestore={requestRestore}
          />
        )}

        {section === 'supply' && (
          <PlanSectionDataGate resource="inventory" state={inventorySection.state} error={inventorySection.error} locale={locale} onRetry={inventorySection.retry}>
            <SupplyPanel
              inventory={inventory}
              locale={locale}
              words={words}
              yieldPerSecond={surface.yieldPerSecond}
              planScope={surface.progress?.scope}
            />
          </PlanSectionDataGate>
        )}

        {section === 'board' && (
          <PlanSectionDataGate resource="schedule" state={scheduleSection.state} error={scheduleSection.error} locale={locale} onRetry={scheduleSection.retry}>
            <BoardPanel
              schedule={schedule}
              copy={copy}
              locale={locale}
              notify={notify}
              planEvents={planEvents}
              dayEvents={dayEvents}
              onGlobalRefresh={onGlobalRefresh}
              onRun={runNow}
              runState={surface.runState}
              runDisabled={!optimization.optimizationAllowed}
              runDisabledReason={optimization.blockedReason}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={(program) => setSelectedProgramKey(program?.key || null)}
              view={boardView}
              onViewChange={setBoardView}
              gridAxis={gridAxis}
              onGridAxisChange={setGridAxis}
              focusDate={boardDate}
              onFocusDateChange={setBoardDate}
              versions={surface.versions}
              live={surface.live}
              freshness={surface.freshness}
              canEdit={surface.canPublish}
              canEditReason={surface.canPublishReason}
              versionName={surface.versionName}
              versionNote={surface.versionNote}
              publishState={surface.publishState}
              publishError={surface.publishError}
              selectedVersion={surface.selectedVersion}
              diff={surface.diff}
              onVersionName={surface.setVersionName}
              onVersionNote={surface.setVersionNote}
              onPublish={requestPublish}
              onVersionDiff={surface.loadDiff}
              onRestore={requestRestore}
              onOpenHistory={() => go('publish')}
            />
          </PlanSectionDataGate>
        )}
        </div>
      </div>

      <p className="plan-foot-hint">
        {pageText(locale, 'Press Cmd K for every command, or G then a letter to move.', 'לחצו Cmd K לכל הפקודות, או G ואז אות כדי לנווט.')}
        <span className="plan-foot-sections">
          {SECTIONS.map((item) => `${sectionLabel(item.id, locale)} (G ${item.key.toUpperCase()})`).join(' · ')}
        </span>
      </p>

      <CommandPalette
        open={paletteOpen}
        commands={commands}
        locale={locale}
        onClose={() => setPaletteOpen(false)}
      />

      <PlanConsequenceDialog
        review={review}
        locale={locale}
        scopeText={scopeText}
        dirty={surface.dirty}
        versionName={surface.versionName}
        runAllowed={optimization.optimizationAllowed}
        onCancel={() => setReview(null)}
        onConfirm={confirmReviewedAction}
      />
    </section>
  );
}

export default PlanWeek;
