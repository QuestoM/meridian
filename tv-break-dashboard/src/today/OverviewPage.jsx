import React, { useEffect, useMemo, useState } from 'react';
import { formatPlanDate, pageText } from '../shell/format';
import { PageHeader } from '../shell/primitives';
import { fetchSession, needsJobPicker } from '../session.js';
import { word } from '../vocabulary.js';
import SummaryMetrics from './SummaryMetrics';
import YieldView from './YieldView';
import { YieldMoneyPanel } from './MoneyWaterfall';
import ComplianceLedger from '../rules/ComplianceLedger';
import FrontierScopeChart from '../plan/week/FrontierScopeChart';
import TodayMoney from './TodayMoney';
import TodayHealth from './TodayHealth';
import TodayDecisions from './TodayDecisions';
import JobPicker from './JobPicker';
import { isolate } from '../shell/bidi';
import { attributed, overviewScope } from './today-scope';
import { clearTarget, fetchToday, saveTarget, takePrimedToday, todayFromOverview } from './today-data';
import './today.css';
import './today-controls.css';

// Today: three answers, one screen, no clicks.
//
// The answers come from this surface's own read, which was started while the
// bundle was still booting, so they do not wait for the ten other endpoints the
// shell loads for the rest of the app. When that read is unavailable the same
// three answers are derived from the payload the shell already holds, so an
// older backend loses the target and the per-day rows and keeps everything else.

const HEALTH_VIEWS = { plan: 'Schedule', sources: 'Data', settings: 'Settings' };

// The run control lives in the staleness strip the shell renders above every
// workspace. A row that reports an out-of-date plan reaches that one control
// rather than growing a second one, because two buttons that start the same run
// is how an operator learns to distrust both. The selector is the shell's own
// and is stable; if the strip is not on screen the row falls back to the view.
const RUN_CONTROL = '.schedule-staleness-button';

function focusRunControl() {
  const control = typeof document === 'undefined' ? null : document.querySelector(RUN_CONTROL);
  if (!control) return false;
  if (control.scrollIntoView) control.scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (control.focus) control.focus({ preventScroll: true });
  return true;
}

export function useTodayData(refreshKey, overview) {
  const [state, setState] = useState({ status: 'loading', data: null, error: null });

  useEffect(() => {
    let active = true;
    const promise = (refreshKey === 0 && takePrimedToday()) || fetchToday();
    promise
      .then((data) => {
        if (active) setState({ status: 'ready', data, error: null });
      })
      .catch((error) => {
        if (active) setState({ status: 'error', data: null, error: String(error.message || error) });
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const fallback = useMemo(() => todayFromOverview(overview), [overview]);
  const today = state.data || fallback;
  // Answered means a real reading is in hand, from this surface's own endpoint
  // or from the shared payload. Until one of them is, the surface says it is
  // still reading, because the shared payload's own empty shape would otherwise
  // render as four zeroes and a compliant licence nobody checked.
  return { ...state, today, answered: Boolean(state.data) || today.answered === true };
}

export function OverviewPage({ overview, compliance, files, copy, locale, setActiveView, onOpenRecommendation, loading, operatorChannel, savedRetentionFloor, onApplyFrontierFloor, applyWeightState, refreshKey, planEvents, notify }) {
  const [localKey, setLocalKey] = useState(0);
  const { status, error, today, answered } = useTodayData(refreshKey + localKey, overview);
  const [saveState, setSaveState] = useState('idle');
  const [session, setSession] = useState(null);

  useEffect(() => {
    let active = true;
    fetchSession().then((result) => {
      if (active && result.ok) setSession(result.session);
    });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  async function persistTarget(values) {
    setSaveState('saving');
    try {
      await saveTarget(values);
      setSaveState('idle');
      setLocalKey((key) => key + 1);
      if (notify) notify('The target is saved.', 'היעד נשמר.');
    } catch (saveError) {
      setSaveState('idle');
      if (notify) notify(String(saveError.message || saveError), String(saveError.message || saveError));
    }
  }

  async function removeTarget() {
    setSaveState('saving');
    try {
      await clearTarget();
      setSaveState('idle');
      setLocalKey((key) => key + 1);
      if (notify) notify('The target is removed.', 'היעד הוסר.');
    } catch (clearError) {
      setSaveState('idle');
      if (notify) notify(String(clearError.message || clearError), String(clearError.message || clearError));
    }
  }

  function openHealth(check) {
    if (check.opens === 'licence') {
      const ledger = document.getElementById('today-licence');
      if (ledger && ledger.scrollIntoView) ledger.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (check.id === 'plan_out_of_date' || check.id === 'newer_model_version') {
      if (focusRunControl()) return;
    }
    const view = HEALTH_VIEWS[check.opens];
    if (view && setActiveView) setActiveView(view);
  }

  // The channel name is data. It is isolated where it is placed inside the
  // heading rather than here, so the raw name stays available to anything that
  // has to compare it, and only the printed copy carries the marks.
  const channel = today.channel || operatorChannel || '';
  const planRunAt = today.plan_run_at ? formatPlanDate(String(today.plan_run_at).slice(0, 10), locale) : '';
  // The source-file count the control-room panel used to carry, kept on the
  // input row it belongs to rather than lost with the panel.
  const fileRows = Array.isArray(files && files.files) ? files.files : [];
  const sourceFiles = fileRows.length ? { present: fileRows.filter((file) => file.exists).length, total: fileRows.length } : null;
  // The sentence below the three answers claims the four figures under it are
  // this operator's, over the window the money answer names. Both halves are
  // true only once the overview body has landed and could scope, so it is
  // printed on that one test. Withholding it on the refusal alone left it
  // asserting a channel and a window over four dashes on every cold boot, which
  // is the unknown state being printed as a fact.
  const openSettings = () => setActiveView && setActiveView('Settings');
  const summaryAttributed = attributed(overviewScope(overview));
  // The plan's week board. The day cannot travel with it: the shell renders the
  // plan surface from a frozen router and that surface takes no date, so a link
  // to one day needs a contract the plan piece has to publish first. Nothing
  // here names a day it cannot open.
  const openPlan = () => setActiveView && setActiveView('Schedule');

  return (
    <section className="page-workspace today-workspace">
      <PageHeader
        locale={locale}
        titleEn={`Today${channel ? `, ${isolate(channel)}` : ''}`}
        titleHe={`${word('place.today', 'he')}${channel ? `, ${isolate(channel)}` : ''}`}
        bodyEn="On plan, anything broken, what needs a decision. Three answers, no clicks."
        bodyHe="ביעד, משהו לא תקין, מה דורש החלטה. שלוש תשובות, בלי קליקים."
        action={
          planRunAt ? (
            <span className="today-plan-stamp">
              {pageText(locale, `${word('state.plan_current', 'en')} ${planRunAt}`, `${word('state.plan_current', 'he')}־${planRunAt}`)}
            </span>
          ) : null
        }
      />

      {status === 'error' && answered ? (
        <p className="today-degraded">
          {pageText(
            locale,
            `Today's own read did not answer, so these figures come from the shared payload and the target is not among them. ${error}`,
            `הקריאה של מסך היום לא נענתה, ולכן המספרים כאן מגיעים מהמטען המשותף והיעד אינו ביניהם. ${error}`,
          )}
        </p>
      ) : null}
      {status === 'error' && !answered ? (
        <p className="today-degraded">
          {pageText(
            locale,
            `Neither this surface's own read nor the shared payload answered, so there is nothing here to report. ${error}`,
            `לא הקריאה של המסך הזה ולא המטען המשותף נענו, ולכן אין כאן מה לדווח. ${error}`,
          )}
        </p>
      ) : null}

      {answered ? (
        <>
          <div className="today-answers">
            <TodayMoney
              today={today}
              locale={locale}
              saveState={saveState}
              onSaveTarget={persistTarget}
              onClearTarget={removeTarget}
              onOpenPlan={openPlan}
              onOpenSettings={openSettings}
            />
            <TodayHealth today={today} locale={locale} sourceFiles={sourceFiles} onOpen={openHealth} />
          </div>

          <TodayDecisions
            today={today}
            locale={locale}
            onOpenInOptimizer={(item) => (item.id && onOpenRecommendation ? onOpenRecommendation(item.id) : setActiveView && setActiveView('Optimizer'))}
            onOpenSettings={openSettings}
          />
        </>
      ) : status === 'error' ? null : (
        <p className="today-reading" role="status">
          {pageText(locale, 'Reading the three answers.', 'קורא את שלוש התשובות.')}
        </p>
      )}

      {/* Below the answers, never above them. An account with no job yet is the
          one visit where nobody has told this person what this screen is, so it
          is the visit that can least afford to open on a question: measured at
          913 px, the picker above pushed the third answer off the screen and
          charged the unaided reader a scroll for it. The three answers are what
          the screen is for, and the question is the first thing after them. */}
      {session && needsJobPicker(session) ? (
        <JobPicker
          session={session}
          locale={locale}
          copy={copy}
          notify={notify}
          setActiveView={setActiveView}
          onChosen={(next) => setSession(next)}
        />
      ) : null}

      {summaryAttributed ? (
        <p className="today-basis">
          {pageText(
            locale,
            'The same window, in four more figures. Every one is the saved plan for your channel over the window named above.',
            'אותו חלון, בארבעה מספרים נוספים. כולם מהתוכנית השמורה של הערוץ שלכם, על החלון שנקוב למעלה.',
          )}
        </p>
      ) : null}
      <SummaryMetrics overview={overview} copy={copy} locale={locale} planEvents={planEvents} onOpenSettings={openSettings} />
      <YieldMoneyPanel locale={locale} refreshKey={refreshKey} onOpenSettings={openSettings} />
      <div className="page-grid even">
        <div id="today-licence">
          <ComplianceLedger compliance={compliance} copy={copy} locale={locale} />
        </div>
        <FrontierScopeChart
          initialData={overview.frontier || []}
          copy={copy}
          locale={locale}
          loading={loading}
          operatorChannel={operatorChannel}
          savedRetentionFloor={savedRetentionFloor}
          onApplyFloor={onApplyFrontierFloor}
          applyState={applyWeightState}
          status={overview.frontier_status || ''}
        />
      </div>
      <YieldView locale={locale} refreshKey={refreshKey} onOpenSettings={openSettings} />
    </section>
  );
}

export default OverviewPage;
