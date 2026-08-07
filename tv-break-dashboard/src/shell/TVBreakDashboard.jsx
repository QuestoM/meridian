import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CacheProvider } from '@emotion/react';
import { CssBaseline, ThemeProvider } from '@mui/material';
import './coherence.css';
import { ltrCache, rtlCache, createKairosTheme } from './theme';
import { copyByLocale } from './copy';
import { fallbackCompliance, fallbackSettings } from './fallbacks';
import { viewFromLocation, gridAxisFromLocation } from './nav';
import { finiteNumber, pageText } from './format';
import { flattenScheduleRows, normalizeRows } from './plan-model';
import { useKairosData } from './use-kairos-data';
import { useSessionEffects } from './use-session';
import { createDecisionActions } from './decision-actions';
import { createPlanActions } from './plan-actions';
import { renderAuthScreen } from './auth-screens';
import { renderSideRail } from './side-rail';
import { renderTopBar } from './top-bar';
import { renderWorkspace } from './workspace-router';
import { AssistantPageProvider } from './assistant-page-context';
import ScheduleStalenessBanner from './ScheduleStalenessBanner';
import UserAdminDialog from './UserAdminDialog';
import { ChangePasswordDialog, requestLogout } from './Login';
import { RefreshCcw } from 'lucide-react';
import ActivityFeed from '../history/ActivityFeed';
import AssistantDock from '../kai/AssistantDock';
import { usePlanEvents } from '../rules/CalendarEventsModel';
import { DirectionRoot } from './bidi';

function TVBreakDashboard() {
  const [refreshKey, setRefreshKey] = useState(0);
  const { overview, schedule, inventory, breakLibrary, campaigns, forecasts, reports, files, impact, parameters, online, partial, loading, error } =
    useKairosData(refreshKey);
  const [activeRecommendation, setActiveRecommendation] = useState('rec-1');
  // Stored calendar events for the display-only plan-surface badges (Overview
  // basis note, schedule canvas). Resolves to [] on an older backend.
  const planEvents = usePlanEvents(refreshKey);
  const [approved, setApproved] = useState(new Set());
  const [rejected, setRejected] = useState(new Set());
  const [scenario, setScenario] = useState('Balanced');
  const [riskLambda, setRiskLambda] = useState(0);
  const riskLambdaTouched = useRef(false);
  // The assistant is a docked side column, not a page: an #Assistant hash on
  // load opens the dock and the workspace shows Overview instead.
  const [activeView, setActiveViewState] = useState(() => {
    const initial = viewFromLocation();
    return initial === 'Assistant' ? 'Overview' : initial;
  });
  // Dock open state survives reloads and view switches; the conversation
  // itself lives on the server, so nothing is lost while the dock is closed.
  const [assistantOpen, setAssistantOpen] = useState(() => {
    try {
      return window.localStorage.getItem('kairos.assistant.dockOpen') === '1' || viewFromLocation() === 'Assistant';
    } catch {
      return viewFromLocation() === 'Assistant';
    }
  });
  const [optimizerView, setOptimizerView] = useState('grid');
  const [gridAxis, setGridAxisState] = useState(gridAxisFromLocation);
  const [showPrograms, setShowPrograms] = useState(true);
  const [showBreaks, setShowBreaks] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [settings, setSettings] = useState(overview.settings || fallbackSettings);
  const [saveState, setSaveState] = useState('idle');
  const [recomputeState, setRecomputeState] = useState('idle');
  const [applyWeightState, setApplyWeightState] = useState('idle');
  const [optimizationState, setOptimizationState] = useState('idle');
  const [optimizationPlan, setOptimizationPlan] = useState(null);
  const [actionMessage, setActionMessage] = useState('');
  const [overridePrefill, setOverridePrefill] = useState(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [recomputeProgress, setRecomputeProgress] = useState(null);
  const toastTimer = useRef(null);
  // Persistent activity feed: every notify() lands here as a dated entry (not
  // only a transient toast), so nothing scrolls away unseen. Loaded from and
  // saved to localStorage so the record survives a reload. Entries are real
  // events, never fabricated.
  const [notifications, setNotifications] = useState(() => {
    try {
      const raw = window.localStorage.getItem('kairos.activity');
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.slice(-100) : [];
    } catch {
      return [];
    }
  });
  const [feedOpen, setFeedOpen] = useState(false);
  const notifyId = useRef(0);

  // Login / session state. The wall only renders when the backend says
  // authentication is set up; an uninitialized store keeps today's open
  // single-operator flow and shows an honest "open access" chip instead.
  const [auth, setAuth] = useState({ status: 'checking', user: null });
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [accountsDialogOpen, setAccountsDialogOpen] = useState(false);

  useSessionEffects(setAuth);

  function handleLoggedIn(user) {
    setAuth({ status: 'ready', user });
    // The pre-login data fetches were rejected by the wall; refetch with the
    // session cookie in place.
    setRefreshKey((key) => key + 1);
  }

  async function handleLogout() {
    setUserMenuAnchor(null);
    await requestLogout();
    notify('Signed out.', 'יצאת מהמערכת.');
    setAuth({ status: 'login', user: null });
  }

  useEffect(() => {
    try {
      window.localStorage.setItem('kairos.activity', JSON.stringify(notifications.slice(-100)));
    } catch {
      // localStorage may be unavailable (private mode); the in-memory feed still works.
    }
  }, [notifications]);

  // Honest progress affordance: a full-week rebuild is a synchronous call with no
  // percentage available, so we surface an elapsed-seconds timer (not a fake
  // progress bar) while an optimization or recompute is running.
  const isBusy = optimizationState === 'running' || recomputeState === 'running';
  useEffect(() => {
    if (!isBusy) {
      setElapsedSec(0);
      return undefined;
    }
    const started = Date.now();
    setElapsedSec(0);
    const id = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [isBusy]);

  function setActiveView(label) {
    // The old Assistant nav entry now opens and focuses the dock; the
    // workspace keeps showing whatever page the operator was on.
    if (label === 'Assistant') {
      setAssistantOpen(true);
      return;
    }
    setActiveViewState(label);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      url.hash = encodeURIComponent(label);
      window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    }
  }

  function setGridAxis(axis) {
    setGridAxisState(axis);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      if (axis === 'day') {
        url.searchParams.delete('axis');
      } else {
        url.searchParams.set('axis', axis);
      }
      if (!url.hash) {
        url.hash = encodeURIComponent(activeView);
      }
      window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    }
  }

  useEffect(() => {
    function handleHashChange() {
      const next = viewFromLocation();
      // #Assistant opens the dock beside the current page instead of
      // switching the workspace, so the old route keeps working.
      if (next === 'Assistant') {
        setAssistantOpen(true);
        return;
      }
      setActiveViewState(next);
    }
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem('kairos.assistant.dockOpen', assistantOpen ? '1' : '0');
    } catch {
      // localStorage may be unavailable (private mode); the session state still works.
    }
  }, [assistantOpen]);

  useEffect(() => {
    const nextSettings = overview.settings || fallbackSettings;
    setSettings((current) => ({ ...current, ...nextSettings }));
  }, [overview.settings]);

  useEffect(() => {
    if (riskLambdaTouched.current) return;
    const saved = finiteNumber(settings.risk_lambda);
    const fromParameters = finiteNumber(parameters?.settings?.risk_lambda);
    const base = saved !== null ? saved : fromParameters !== null ? fromParameters : 0;
    setRiskLambda(Math.round(Math.min(1, Math.max(0, base)) * 100));
  }, [settings.risk_lambda, parameters]);

  const locale = settings.locale === 'en' ? 'en' : 'he';
  const isHebrew = locale === 'he';
  const copy = copyByLocale[locale];
  // The optimization command group (scenario, risk, run, apply, planning-week)
  // is only meaningful on the planning surfaces; hide it on Data, Pricing,
  // Advertisers, Reports and the like where it does nothing.
  const showOptimizationControls = ['Overview', 'Optimizer', 'Schedule'].includes(activeView);
  const compliance = overview.compliance || fallbackCompliance;
  const theme = useMemo(() => createKairosTheme(isHebrew ? 'rtl' : 'ltr'), [isHebrew]);
  const muiCache = isHebrew ? rtlCache : ltrCache;
  const activeNotificationCount = notifications.filter((n) => !n.dismissed).length;

  function notify(en, he, options = {}) {
    setActionMessage(pageText(locale, en, he));
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setActionMessage(''), 2600);
    // Transient notices (pure navigation feedback) show the toast only and stay
    // out of the persistent activity feed, which records real events.
    if (options.transient) return;
    // Also record the event in the persistent activity feed. A stable id avoids
    // Math.random; the bilingual pair is stored so the feed renders in whatever
    // language the operator later views it in.
    notifyId.current += 1;
    const entry = { id: `n${Date.now()}-${notifyId.current}`, en, he, ts: Date.now(), dismissed: false };
    setNotifications((current) => [...current, entry].slice(-100));
  }

  useEffect(() => () => {
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
  }, []);

  const schedulePrograms = useMemo(() => flattenScheduleRows(schedule.rows || []), [schedule]);

  const selectedProgram = useMemo(() => {
    if (selectedProgramKey) {
      const selected = schedulePrograms.find((program) => program.key === selectedProgramKey);
      if (selected) return selected;
    }
    const marked = schedulePrograms.find((program) => program.selected);
    return marked || schedulePrograms[0] || null;
  }, [schedulePrograms, selectedProgramKey]);

  const activeRec =
    overview.recommendations?.find((rec) => rec.id === activeRecommendation) ||
    overview.recommendations?.[0];

  const { openRecommendationInOverrides, approveRecommendation, rejectRecommendation, applySimilarRecommendations } =
    createDecisionActions({
      overview, activeRec, selectedProgram, scenario, notify,
      setApproved, setRejected, setRefreshKey, setOverridePrefill, setActiveView,
    });

  function selectProgram(program) {
    if (!program) return;
    setSelectedProgramKey(program.key);
    setInspectorOpen(true);
    const related =
      normalizeRows(overview.recommendations).find((rec) => rec.program_type === program.program_type) ||
      normalizeRows(overview.recommendations)[0];
    if (related?.id) setActiveRecommendation(related.id);
  }

  function handleRefresh() {
    setRefreshKey((current) => current + 1);
    notify('Data refreshed from the Kairos API.', 'הנתונים רועננו מה־API של Kairos.', { transient: true });
  }

  function dismissNotification(id) {
    setNotifications((current) => current.map((n) => (n.id === id ? { ...n, dismissed: true } : n)));
  }
  function restoreNotification(id) {
    setNotifications((current) => current.map((n) => (n.id === id ? { ...n, dismissed: false } : n)));
  }
  function dismissAllNotifications() {
    setNotifications((current) => current.map((n) => ({ ...n, dismissed: true })));
  }
  function restoreAllNotifications() {
    setNotifications((current) => current.map((n) => ({ ...n, dismissed: false })));
  }

  const { persistSettings, handleRunOptimization, handleApplyFrontierFloor, handleRecomputeSchedule, handleApplyOptimization } =
    createPlanActions({
      settings, setSettings, scenario, riskLambda, locale, notify, setRefreshKey,
      setSaveState, setRecomputeState, setRecomputeProgress, setApplyWeightState,
      setOptimizationState, setOptimizationPlan, setActiveView, setOptimizerView, setInspectorOpen,
    });

  // Auth gate: nothing from the workspace renders before the session check
  // resolves, so the app never flashes behind the login wall.
  const authScreen = renderAuthScreen({ auth, setAuth, muiCache, theme, handleLoggedIn });
  if (authScreen) {
    return authScreen;
  }

  return (
    <CacheProvider value={muiCache}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
    <AssistantPageProvider page={{ view: activeView, label: copy.nav[activeView] || activeView }}>
    <DirectionRoot locale={locale} className={`kairos-shell ${isHebrew ? 'rtl' : 'ltr'}`} lang={locale}>
      {renderSideRail({
        copy, locale, activeView, setActiveView, assistantOpen, auth,
        userMenuAnchor, setUserMenuAnchor, setPasswordDialogOpen, setAccountsDialogOpen, handleLogout,
      })}

      {passwordDialogOpen && (
        <ChangePasswordDialog
          locale={locale}
          onClose={() => setPasswordDialogOpen(false)}
          onDone={() => {
            setPasswordDialogOpen(false);
            notify('Password updated.', 'הסיסמה עודכנה.');
          }}
        />
      )}
      {accountsDialogOpen && auth.user && auth.user.role === 'admin' && (
        <UserAdminDialog
          locale={locale}
          selfUsername={auth.user.username}
          notify={notify}
          onClose={() => setAccountsDialogOpen(false)}
        />
      )}

      <main className="workspace">
        {renderTopBar({
          copy, locale, activeView, setActiveView, showOptimizationControls,
          schedule, overview, settings, scenario, setScenario,
          riskLambda, setRiskLambda, riskLambdaTouched, online, partial,
          notify, handleRefresh, assistantOpen, setAssistantOpen,
          activeNotificationCount, setFeedOpen, persistSettings, optimizationState,
          handleRunOptimization, recomputeState, handleApplyOptimization, elapsedSec,
        })}

        {isBusy && (
          <div className="rebuild-note" role="status">
            <RefreshCcw size={14} className="upload-spinner" />
            <span>{recomputeProgress ? pageText(locale, `Rebuilding the schedule: day ${recomputeProgress.done} of ${recomputeProgress.total}. Elapsed ${elapsedSec}s.`, `בונה מחדש את הלוח: יום ${recomputeProgress.done} מתוך ${recomputeProgress.total}. חלפו ${elapsedSec} שניות.`) : pageText(locale, `Rebuilding the whole weekly schedule. This can take up to a couple of minutes. Elapsed ${elapsedSec}s.`, `בונה מחדש את כל הלוח השבועי. זה יכול להימשך עד כמה דקות. חלפו ${elapsedSec} שניות.`)}</span>
          </div>
        )}

        <ScheduleStalenessBanner
          freshness={overview?.schedule_freshness}
          locale={locale}
          onRecompute={handleRecomputeSchedule}
          recomputeState={recomputeState}
        />

        {renderWorkspace({
          activeView, overview, schedule, inventory, breakLibrary, campaigns, forecasts,
          reports, files, impact, parameters, compliance, copy, locale, loading,
          notify, refreshKey, setRefreshKey, planEvents, settings, setActiveView,
          setActiveRecommendation, handleApplyFrontierFloor, applyWeightState,
          optimizerView, setOptimizerView, gridAxis, setGridAxis,
          showPrograms, setShowPrograms, showBreaks, setShowBreaks, showMetrics, setShowMetrics,
          selectedProgram, selectProgram, inspectorOpen, setInspectorOpen,
          activeRec, approved, rejected, optimizationPlan, scenario,
          approveRecommendation, rejectRecommendation, openRecommendationInOverrides,
          applySimilarRecommendations, handleRecomputeSchedule, recomputeState,
          overridePrefill, setOverridePrefill, saveState, persistSettings,
        })}

        {actionMessage && <div className="toast">{actionMessage}</div>}
        {loading && <div className="toast">{copy.loading}</div>}
        {feedOpen && (
          <ActivityFeed
            notifications={notifications}
            locale={locale}
            onDismiss={dismissNotification}
            onRestore={restoreNotification}
            onClearAll={dismissAllNotifications}
            onRestoreAll={restoreAllNotifications}
            onClose={() => setFeedOpen(false)}
          />
        )}
        {!loading && !online && error && <div className="toast muted">{copy.apiUnavailable}</div>}
      </main>

      {assistantOpen && (
        <AssistantDock locale={locale} notify={notify} onClose={() => setAssistantOpen(false)} />
      )}
    </DirectionRoot>
    </AssistantPageProvider>
      </ThemeProvider>
    </CacheProvider>
  );
}

export default TVBreakDashboard;
