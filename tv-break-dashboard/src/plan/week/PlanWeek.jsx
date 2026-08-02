import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { pageText } from '../../shell/format';
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
import { SECTIONS, planWords, sectionForEntrance, sectionLabel } from './plan-week-model';
import { usePlanSurface } from './use-plan-surface';
import { useSectionData } from './use-section-data';
import { useBoardDay } from './use-board-day';
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
  schedule: providedSchedule,
  inventory: providedInventory,
  copy,
  locale,
  notify: providedNotify,
  planEvents,
  onGlobalRefresh,
}) {
  const [section, setSection] = useState(() => sectionForEntrance(entrance));
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [boardView, setBoardView] = useState('grid');
  const [boardDate, setBoardDate] = useState(null);
  const [gridAxis, setGridAxis] = useState('day');
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  const sectionRefs = useRef({});
  const words = useMemo(() => planWords(locale), [locale]);
  // Two of the four entrances are handed no notifier, so the destination
  // supplies one rather than letting a child call undefined.
  const notify = useMemo(() => providedNotify || (() => {}), [providedNotify]);
  // The comparison prepares itself only while step three is the open step, so
  // the machine is never spent on a week nobody is looking at.
  const surface = usePlanSurface({
    locale,
    notify: providedNotify,
    onGlobalRefresh,
    prepareCompare: section === 'compare',
  });
  const { data: schedule } = useSectionData(providedSchedule, readSchedule, section === 'board');
  const { data: inventory } = useSectionData(providedInventory, readInventory, section === 'supply');
  const dayEvents = useMemo(() => planEventWeekdayMap(planEvents), [planEvents]);

  useEffect(() => {
    setSection(sectionForEntrance(entrance));
  }, [entrance]);

  const go = useCallback((id) => {
    setSection(id);
    window.setTimeout(() => {
      sectionRefs.current[id]?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }, 0);
  }, []);

  // The two acts step 3 owes the planner, each with one implementation behind
  // both its doors: the card and the palette adopt through the same function,
  // and the day row and the sentence that names the decisive day open through
  // the same one.
  const adoptLeg = useCallback((leg) => {
    if (surface.adoptLeg(leg)) go('objective');
  }, [surface, go]);

  // Step two's act, with one implementation behind its three doors: the state
  // row's control, the palette's run row and the R key all call this, so no
  // control can carry a run's words while doing something else.
  const runNow = useCallback(() => {
    go('run');
    surface.runPlan();
  }, [go, surface]);

  const openBoardDay = useCallback((date) => {
    if (!date) return;
    setBoardDate(String(date));
    setBoardView('day');
    go('board');
  }, [go]);

  const clearBoardDay = useCallback(() => {
    setBoardDate(null);
    setBoardView('grid');
  }, []);

  const day = useBoardDay(boardDate);

  const commands = useMemo(
    () => planCommands({
      go,
      words,
      surface,
      boardView,
      setBoardView,
      adoptLeg,
      runNow,
      openPalette: () => setPaletteOpen(true),
    }),
    [go, words, surface, boardView, adoptLeg, runNow],
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
        versionCount={surface.versions.length}
        liveFrozenAs={surface.live?.frozen_as || null}
        scopeText={scopeText}
        onOpenPalette={() => setPaletteOpen(true)}
      />

      <GoalStrip progress={surface.progress} locale={locale} words={words} onGo={go} />

      <div className="plan-sections">
        <div ref={(node) => { sectionRefs.current.objective = node; }} hidden={section !== 'objective'}>
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
        </div>

        <div ref={(node) => { sectionRefs.current.run = node; }} hidden={section !== 'run'}>
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
            onRun={surface.runPlan}
          />
        </div>

        <div ref={(node) => { sectionRefs.current.compare = node; }} hidden={section !== 'compare'}>
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
            onLegChange={surface.changeLeg}
            onCompare={surface.compare}
            onCancel={surface.cancelCompare}
            onAdopt={adoptLeg}
            onOpenDay={openBoardDay}
          />
        </div>

        <div ref={(node) => { sectionRefs.current.publish = node; }} hidden={section !== 'publish'}>
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
            onPublish={surface.publish}
            onSelect={surface.setSelectedVersion}
            onDiff={surface.loadDiff}
            onRestore={surface.restore}
          />
        </div>

        <div ref={(node) => { sectionRefs.current.supply = node; }} hidden={section !== 'supply'}>
          <SupplyPanel
            inventory={inventory}
            locale={locale}
            words={words}
            yieldPerSecond={surface.yieldPerSecond}
            planScope={surface.progress?.scope}
          />
        </div>

        <div ref={(node) => { sectionRefs.current.board = node; }} hidden={section !== 'board'}>
          <BoardPanel
            schedule={schedule}
            copy={copy}
            locale={locale}
            notify={notify}
            planEvents={planEvents}
            dayEvents={dayEvents}
            onGlobalRefresh={onGlobalRefresh}
            onRun={surface.runPlan}
            runState={surface.runState}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={(program) => setSelectedProgramKey(program?.key || null)}
            view={boardView}
            onViewChange={setBoardView}
            gridAxis={gridAxis}
            onGridAxisChange={setGridAxis}
            focusDate={boardDate}
            dayPayload={day.payload}
            dayState={day.state}
            dayError={day.error}
            onClearFocus={clearBoardDay}
          />
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
    </section>
  );
}

export default PlanWeek;
