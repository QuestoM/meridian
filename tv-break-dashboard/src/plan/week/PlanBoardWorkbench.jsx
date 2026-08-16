import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GitCompareArrows, PencilRuler } from 'lucide-react';
import { Figure } from '../../shell/bidi';
import { flattenScheduleRows } from '../../shell/plan-model';
import { formatNumber, pageText } from '../../shell/format';
import { formatDay } from '../../shell/dates';
import BreakInspector from '../break/BreakInspector';
import DayBoard from '../day/DayBoard';
import DayPicker from '../day/DayPicker';
import ScheduleInspector from '../day/ScheduleInspector';
import { exactCurrency } from '../day/day-board-model';
import LocalPlanVariants from './LocalPlanVariants';
import PlanVersionRail from './PlanVersionRail';
import '../day/master-control-broadcast.css';

function datesInSchedule(schedule) {
  return Array.from(new Set(
    flattenScheduleRows(schedule?.rows || [])
      .map((row) => String(row.date || row.day || '').slice(0, 10))
      .filter((day) => /^\d{4}-\d{2}-\d{2}$/.test(day)),
  )).sort();
}

function Metric({ label, value, note, tone = '' }) {
  return (
    <div className={`plan-workbench-metric${tone ? ` is-${tone}` : ''}`}>
      <span>{label}</span>
      <strong><Figure>{value}</Figure></strong>
      <small>{note}</small>
    </div>
  );
}

function ComparisonStrip({ board, workState, locale }) {
  const score = workState?.score || null;
  const forecast = workState?.forecast || null;
  const basis = score?.basis || board?.basis || {};
  const record = basis.committed || null;
  const current = forecast?.after || score?.current || board?.totals || null;
  const projectedDelta = forecast?.delta?.revenue ?? score?.delta?.revenue ?? 0;
  const changeCount = forecast?.rearranged?.changed ?? workState?.editCount ?? 0;
  const rawGap = record && current ? Number(current.revenue) - Number(record.revenue) : null;
  const gap = rawGap !== null && Math.abs(rawGap) < 0.005 ? 0 : rawGap;
  const gapTone = gap === null || Math.abs(gap) < 0.005 ? 'flat' : gap > 0 ? 'gain' : 'loss';
  const deltaTone = Math.abs(projectedDelta) < 0.005 ? 'flat' : projectedDelta > 0 ? 'gain' : 'loss';

  return (
    <section className="card plan-comparison-strip" aria-label={pageText(locale, 'Plan of record and working day comparison', 'השוואת תוכנית הייחוס ויום העבודה')}>
      <div className="plan-comparison-title">
        <GitCompareArrows size={17} aria-hidden="true" />
        <div>
          <span>{pageText(locale, 'Same channel-day, two truths', 'אותו יום־ערוץ, שתי נקודות אמת')}</span>
          <strong>{formatDay(board?.day)}</strong>
        </div>
      </div>
      <Metric
        label={pageText(locale, 'Saved weekly plan', 'התוכנית השבועית השמורה')}
        value={record ? exactCurrency(record.revenue, locale) : '\u2014'}
        note={record
          ? `${formatNumber(record.breaks, locale)} ${pageText(locale, 'breaks', 'ברייקים')}`
          : pageText(locale, 'No committed row', 'אין שורה שמורה')}
      />
      <Metric
        label={forecast
          ? pageText(locale, 'Measured result if saved', 'תוצאה מדודה אם תישמר')
          : pageText(locale, 'Live working preview', 'תצוגת עבודה חיה')}
        value={current ? exactCurrency(current.revenue, locale) : '\u2014'}
        note={current ? `${formatNumber(current.breaks, locale)} ${pageText(locale, 'breaks', 'ברייקים')}` : pageText(locale, 'Calculating', 'מחשב')}
      />
      <Metric
        label={pageText(locale, 'Gap from plan of record', 'פער מתוכנית הייחוס')}
        value={gap === null ? '\u2014' : exactCurrency(gap, locale)}
        note={pageText(locale, 'Day scope', 'היקף יום')}
        tone={gapTone}
      />
      <Metric
        label={forecast ? pageText(locale, 'Breaks re-planned', 'ברייקים שתוכננו מחדש') : pageText(locale, 'Pending edits', 'עריכות ממתינות')}
        value={formatNumber(changeCount, locale)}
        note={forecast
          ? `${pageText(locale, 'save impact', 'השפעת שמירה')} · ${exactCurrency(projectedDelta, locale)}`
          : pageText(locale, 'Nothing written until reviewed save', 'דבר לא נכתב לפני שמירה שנבדקה')}
        tone={deltaTone}
      />
    </section>
  );
}

export default function PlanBoardWorkbench({
  schedule,
  locale,
  notify,
  onGlobalRefresh,
  focusDate,
  onFocusDateChange,
  versions,
  live,
  freshness,
  canEdit,
  canEditReason,
  versionName,
  versionNote,
  publishState,
  publishError,
  selectedVersion,
  diff,
  runState,
  runDisabled,
  runDisabledReason,
  onVersionName,
  onVersionNote,
  onPublish,
  onVersionDiff,
  onRestore,
  onRun,
  onOpenHistory,
}) {
  const availableDays = useMemo(() => datesInSchedule(schedule), [schedule]);
  const preferredDay = focusDate || schedule?.board?.date || availableDays[0] || '';
  const [day, setDay] = useState(preferredDay);
  const [board, setBoard] = useState(null);
  const [workState, setWorkState] = useState(null);
  const [draftCommand, setDraftCommand] = useState(null);
  const [openBreak, setOpenBreak] = useState(null);
  const [openProgramme, setOpenProgramme] = useState(null);
  const [breakIds, setBreakIds] = useState([]);
  const returnFocusRef = useRef(null);

  useEffect(() => {
    if (focusDate) setDay(String(focusDate));
  }, [focusDate]);

  useEffect(() => {
    if (!day && preferredDay) setDay(preferredDay);
  }, [day, preferredDay]);

  useEffect(() => {
    setBoard(null);
    setWorkState(null);
    setDraftCommand(null);
    setOpenBreak(null);
    setOpenProgramme(null);
  }, [day]);

  const selectDay = useCallback((nextDay) => {
    setDay(nextDay);
    onFocusDateChange?.(nextDay);
  }, [onFocusDateChange]);

  const onDayLoaded = useCallback((payload) => {
    setBoard(payload);
    setBreakIds((payload.breaks || []).map((row) => row.break_id));
  }, []);

  const onOpenBreak = useCallback((breakId) => {
    returnFocusRef.current = document.activeElement;
    setOpenProgramme(null);
    setOpenBreak(breakId);
  }, []);

  const onOpenProgramme = useCallback((programme) => {
    returnFocusRef.current = document.activeElement;
    setOpenBreak(null);
    setOpenProgramme({ segmentId: programme.segment_id, channel: programme.channel, day: programme.day });
  }, []);

  const closeInspector = useCallback(() => {
    setOpenBreak(null);
    setOpenProgramme(null);
    window.setTimeout(() => returnFocusRef.current?.focus?.(), 0);
  }, []);

  return (
    <div className="plan-workbench broadcast-day">
      <div className="plan-workbench-intro">
        <div>
          <span className="plan-workbench-kicker">{pageText(locale, 'Operator workbench', 'שולחן עבודת מפעיל')}</span>
          <h3>{pageText(locale, 'Work one broadcast day against the plan of record', 'עבודה על יום שידור אחד מול תוכנית הייחוס')}</h3>
          <p>{pageText(
            locale,
            'Select a break, change its placement or duration, measure the consequence, then use the existing reviewed save. The weekly record and frozen checkpoints remain distinct and named.',
            'בחרו ברייק, שנו מיקום או אורך, מדדו את ההשפעה ואז השתמשו בשמירה הקיימת שעוברת בדיקה. תוכנית הייחוס השבועית ונקודות הבקרה הקפואות נשארות נפרדות ומזוהות.',
          )}</p>
        </div>
        <span className="plan-workbench-mode">
          <PencilRuler size={15} aria-hidden="true" />
          {pageText(locale, 'Editing workspace', 'מרחב עריכה')}
        </span>
      </div>

      <DayPicker
        days={availableDays}
        value={day}
        onChange={selectDay}
        locale={locale}
        channel={board?.operator_channel || ''}
        windowed
      />

      <ComparisonStrip board={board} workState={workState} locale={locale} />

      <div className="plan-workbench-layout">
        <section className="plan-workbench-canvas" aria-label={pageText(locale, 'Editable daily timeline', 'ציר הזמן היומי לעריכה')}>
          <div className="plan-timeline-basis">
            <strong>{pageText(locale, 'Live editable timeline', 'ציר זמן חי לעריכה')}</strong>
            <span>{pageText(
              locale,
              'The saved weekly plan supplies the comparison totals. It does not retain a second placement timeline for this day, so no historical placements are fabricated here.',
              'התוכנית השבועית השמורה מספקת את נתוני ההשוואה. היא אינה שומרת ציר מיקומים שני ליום הזה, ולכן לא מוצגים כאן מיקומים היסטוריים מומצאים.',
            )}</span>
          </div>
          {day ? (
            <DayBoard
              day={day}
              locale={locale}
              notify={notify}
              onGlobalRefresh={onGlobalRefresh}
              onOpenBreak={onOpenBreak}
              onOpenProgramme={onOpenProgramme}
              onDayLoaded={onDayLoaded}
              onWorkState={setWorkState}
              draftCommand={draftCommand}
            />
          ) : (
            <div className="day-board-empty">
              <p>{pageText(locale, 'No broadcast day is available in the saved plan.', 'אין יום שידור זמין בתוכנית השמורה.')}</p>
            </div>
          )}
        </section>

        <div className="plan-workbench-rail-stack">
          <LocalPlanVariants
            board={board}
            live={live}
            freshness={freshness}
            workState={workState}
            locale={locale}
            notify={notify}
            onApplyDraft={setDraftCommand}
          />
          <PlanVersionRail
            locale={locale}
            versions={versions}
            live={live}
            freshness={freshness}
            canEdit={canEdit}
            canEditReason={canEditReason}
            name={versionName}
            note={versionNote}
            publishState={publishState}
            publishError={publishError}
            selectedId={selectedVersion}
            diff={diff}
            runState={runState}
            runDisabled={runDisabled}
            runDisabledReason={runDisabledReason}
            onNameChange={onVersionName}
            onNoteChange={onVersionNote}
            onPublish={onPublish}
            onCompare={(versionId) => onVersionDiff(versionId, 'live')}
            onRestore={onRestore}
            onRun={onRun}
            onOpenHistory={onOpenHistory}
          />
        </div>
      </div>

      {openBreak && (
        <BreakInspector
          breakId={openBreak}
          locale={locale}
          siblings={breakIds}
          onNavigate={setOpenBreak}
          onClose={closeInspector}
          notify={notify}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
      {openProgramme && (
        <ScheduleInspector
          segmentId={openProgramme.segmentId}
          channel={openProgramme.channel}
          day={openProgramme.day}
          locale={locale}
          notify={notify}
          onClose={closeInspector}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
    </div>
  );
}
