import React, { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { CacheProvider } from '@emotion/react';
import { CssBaseline, ThemeProvider } from '@mui/material';
import './coherence.css';
import { ltrCache, rtlCache, createKairosTheme } from './theme';
import { copyByLocale } from './copy';
import { fallbackCompliance, fallbackSettings } from './fallbacks';
import {
  contextItemIsActive,
  contextItemsForDomain,
  domainForView,
  domainLabel,
  navigationUrl,
  routeFromLocation,
} from './nav';
import { pageText } from './format';
import { savedSettingsFromOverview, useKairosData } from './use-kairos-data';
import { createDecisionActions } from './decision-actions';
import { createPlanActions } from './plan-actions';
import { renderSideRail } from './side-rail';
import { renderTopBar } from './top-bar';
import { renderWorkspace } from './workspace-router';
import { AssistantPageProvider } from './assistant-page-context';
import ScheduleStalenessBanner from './ScheduleStalenessBanner';
import { ChangePasswordDialog, requestLogout } from './Login';
import { usePlanEvents } from '../rules/CalendarEventsModel';
import { DirectionRoot, useDocumentDirection } from './bidi';
import { queueWorkspaceContinuity, transitionWorkspaceUpdate } from './workspace-continuity';
import { Toast } from './primitives';
const ActivityFeed = lazy(() => import('../history/ActivityFeed'));
const AssistantDock = lazy(() => import('../kai/AssistantDock'));
const UserAdminDialog = lazy(() => import('./UserAdminDialog'));
const ModelConsole = lazy(() => import('../model/console/ModelConsole'));

function currentRelativeUrl() {
  if (typeof window === 'undefined') return '';
  return `${window.location.pathname}${window.location.search}${window.location.hash}`;
}

function TVBreakDashboard({ auth, setAuth }) {
  const canAccessModel = auth.status === 'open' || auth.user?.affiliation === 'company';
  const [refreshKey, setRefreshKey] = useState(0);
  const [activeView, setActiveViewState] = useState(
    () => routeFromLocation({ canAccessModel }).view,
  );
  const { overview, schedule, inventory, breakLibrary, campaigns, forecasts, reports, files, impact, parameters, online, partial, loading, error } =
    useKairosData(refreshKey, activeView);
  const confirmedOverviewSettings = savedSettingsFromOverview(overview);
  const planEvents = usePlanEvents(refreshKey, activeView === 'Today' || activeView === 'Plan');
  const [approved, setApproved] = useState(new Set());
  const [rejected, setRejected] = useState(new Set());
  const [workspaceKey, setWorkspaceKey] = useState(0);
  const [addressRevision, setAddressRevision] = useState(0);
  const [assistantOpen, setAssistantOpen] = useState(() => {
    const initialRoute = routeFromLocation({ canAccessModel });
    try {
      return window.localStorage.getItem('kairos.assistant.dockOpen') === '1' || initialRoute.assistant;
    } catch {
      return initialRoute.assistant;
    }
  });
  const [settings, setSettings] = useState(overview.settings || fallbackSettings);
  const settingsAvailable = Boolean(confirmedOverviewSettings) && settings._unavailable !== true;
  const [saveState, setSaveState] = useState('idle');
  const [applyWeightState, setApplyWeightState] = useState('idle');
  const [actionMessage, setActionMessage] = useState('');
  const [overridePrefill, setOverridePrefill] = useState(null);
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

  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [accountsDialogOpen, setAccountsDialogOpen] = useState(false);

  async function handleLogout() {
    setUserMenuAnchor(null);
    const result = await requestLogout();
    if (!result.ok) { notify('Sign out failed. Your session is still active; try again.', 'היציאה נכשלה. ההפעלה הנוכחית עדיין פעילה; יש לנסות שוב.'); return; }
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

  const activeViewRef = useRef(activeView);
  activeViewRef.current = activeView;
  const lastSyncedAddress = useRef('');

  function readAndNormalizeAddress({ remount = false, force = false } = {}) {
    const route = routeFromLocation({
      fallbackView: activeViewRef.current,
      canAccessModel,
    });
    if (route.assistant) setAssistantOpen(true);
    if (route.normalizedUrl && route.normalizedUrl !== currentRelativeUrl()) {
      window.history.replaceState(
        { ...(window.history.state || {}), kairos: true, normalized: true },
        '',
        route.normalizedUrl,
      );
    }
    const address = currentRelativeUrl();
    if (!force && address === lastSyncedAddress.current) return;
    lastSyncedAddress.current = address;
    setActiveViewState(route.view);
    setAddressRevision((value) => value + 1);
    if (remount) setWorkspaceKey((value) => value + 1);
  }

  function setActiveView(label, params = {}) {
    // Mabat is a contextual dock action, not an eighth destination.
    if (label === 'Assistant') {
      setAssistantOpen(true);
      return;
    }
    const next = navigationUrl(label, params);
    if (!next) return;
    window.history.pushState(
      { ...(window.history.state || {}), kairos: true, view: label },
      '',
      next,
    );
    transitionWorkspaceUpdate(() => readAndNormalizeAddress({ remount: true, force: true }), { focusMain: true });
  }

  useEffect(() => {
    // Normalization repairs unknown and legacy addresses in place. It never
    // creates a Back entry; only an operator navigation calls pushState.
    readAndNormalizeAddress({ force: true });

    const historyObject = window.history;
    const originalPushState = historyObject.pushState;
    const originalReplaceState = historyObject.replaceState;
    const announceAddress = () => window.dispatchEvent(new Event('kairos:addresschange'));
    const wrap = (original) => function wrappedHistoryState(...args) {
      const before = window.location.href;
      const result = original.apply(this, args);
      if (window.location.href !== before) announceAddress();
      return result;
    };
    const wrappedPushState = wrap(originalPushState);
    const wrappedReplaceState = wrap(originalReplaceState);
    historyObject.pushState = wrappedPushState;
    historyObject.replaceState = wrappedReplaceState;

    const onTraversal = () => transitionWorkspaceUpdate(() => readAndNormalizeAddress({ remount: true }), { focusMain: true });
    const onNavigationRequest = (event) => {
      if (!event.detail?.view) return;
      event.detail.handled = true;
      setActiveView(event.detail.view, event.detail.params || {});
    };
    const onAddressChange = () => {
      // Local controls use history state for shareability. They already own
      // their component state, so update the shell chrome without remounting.
      setAddressRevision((value) => value + 1);
      lastSyncedAddress.current = currentRelativeUrl();
      queueWorkspaceContinuity();
    };
    window.addEventListener('popstate', onTraversal);
    window.addEventListener('hashchange', onTraversal);
    window.addEventListener('kairos:addresschange', onAddressChange);
    window.addEventListener('kairos:navigate', onNavigationRequest);
    return () => {
      window.removeEventListener('popstate', onTraversal);
      window.removeEventListener('hashchange', onTraversal);
      window.removeEventListener('kairos:addresschange', onAddressChange);
      window.removeEventListener('kairos:navigate', onNavigationRequest);
      if (historyObject.pushState === wrappedPushState) historyObject.pushState = originalPushState;
      if (historyObject.replaceState === wrappedReplaceState) historyObject.replaceState = originalReplaceState;
    };
  }, [canAccessModel]);

  useEffect(() => {
    try {
      window.localStorage.setItem('kairos.assistant.dockOpen', assistantOpen ? '1' : '0');
    } catch {
      // localStorage may be unavailable (private mode); the session state still works.
    }
  }, [assistantOpen]);

  useEffect(() => {
    if (!confirmedOverviewSettings) return;
    setSettings({ ...confirmedOverviewSettings });
  }, [confirmedOverviewSettings]);

  const locale = settings.locale === 'en' ? 'en' : 'he';
  const isHebrew = locale === 'he';
  useDocumentDirection(locale);
  useEffect(() => {
    try {
      window.localStorage.setItem('kairos.locale', locale);
    } catch {
      // The document still follows the saved backend setting for this visit.
    }
  }, [locale]);
  const copy = copyByLocale[locale];
  // Today names the active saved week in the chrome. The write controls live in
  // Plan beside their scope, review and result; the shell is orientation only.
  const showPlanContext = activeView === 'Today';
  const activeDomain = domainForView(activeView);
  const activeDomainLabel = domainLabel(activeDomain, locale);
  // Workspace-owned tabs must not be repeated in the shell. Broadcast and
  // Governance are the two domains whose local navigation lives here.
  const shellOwnsLocalNavigation = activeDomain === 'Broadcast' || activeDomain === 'Governance';
  const localItems = shellOwnsLocalNavigation
    ? contextItemsForDomain(activeDomain, canAccessModel).map((item) => ({
      ...item,
      active: contextItemIsActive(item, activeView, addressRevision),
    }))
    : [];
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

  const activeRec = overview.recommendations?.[0];
  const { openRecommendationInOverrides, approveRecommendation, rejectRecommendation, applySimilarRecommendations } =
    createDecisionActions({
      overview, activeRec, selectedProgram: null, scenario: 'Balanced', notify,
      setApproved, setRejected, setRefreshKey, setOverridePrefill, setActiveView,
    });

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

  const { persistSettings, handleApplyFrontierFloor } =
    createPlanActions({
      settings, setSettings, locale, notify, setRefreshKey, setSaveState, setApplyWeightState,
      settingsAvailable,
    });

  return (
    <CacheProvider value={muiCache}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
    <AssistantPageProvider page={{ view: activeView, label: copy.nav[activeView] || activeView }}>
    <DirectionRoot locale={locale} className={`kairos-shell ${isHebrew ? 'rtl' : 'ltr'}${activeView === 'Model' ? ' model-route-host' : ''}${assistantOpen ? ' assistant-open' : ''}`} lang={locale}>
      {activeView === 'Model' ? (
        <Suspense fallback={<div className="workspace-route-loading" role="status">{pageText(locale, 'Opening the company model…', 'פותח את מודל החברה…')}</div>}>
          <ModelConsole
            locale={locale}
            onBack={() => {
              if (window.history.state?.kairos && window.history.length > 1) window.history.back();
              else setActiveView('Governance');
            }}
            onOpenRules={() => setActiveView('Governance', { rules: 'restrictions' })}
            onOpenEvents={() => setActiveView('Governance', { rules: 'calendar' })}
          />
        </Suspense>
      ) : (
      <>
      <a className="skip-link" href="#kairos-main">{pageText(locale, 'Skip to main workspace', 'דילוג לאזור העבודה הראשי')}</a>
      {renderSideRail({
        copy, locale, activeDomain, setActiveView, assistantOpen, auth, canAccessModel,
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
        <Suspense fallback={null}>
          <UserAdminDialog
            locale={locale}
            selfUsername={auth.user.username}
            notify={notify}
            onClose={() => setAccountsDialogOpen(false)}
          />
        </Suspense>
      )}

      <main id="kairos-main" className="workspace" tabIndex={-1} aria-label={pageText(locale, 'Kairos work area', 'אזור העבודה של Kairos')}>
        {renderTopBar({
          copy, locale, activeView, activeDomain, activeDomainLabel, localItems,
          onNavigateLocal: (item) => setActiveView(item.target.view, item.target.params || {}),
          setActiveView, showPlanContext,
          schedule, overview, settings, settingsAvailable, online, partial,
          notify, handleRefresh, assistantOpen, setAssistantOpen,
          activeNotificationCount, setFeedOpen, persistSettings,
        })}

        {activeDomain === 'Today' && <ScheduleStalenessBanner
          freshness={overview?.schedule_freshness}
          locale={locale}
          onReviewRun={() => setActiveView('Plan', { plan: 'run' })}
        />}

        {renderWorkspace({
          activeView, workspaceKey, overview, schedule, inventory, breakLibrary, campaigns, forecasts,
          reports, files, impact, parameters, compliance, copy, locale, loading,
          notify, refreshKey, setRefreshKey, planEvents, settings, settingsAvailable, setActiveView,
          handleApplyFrontierFloor, applyWeightState,
          approved, rejected,
          approveRecommendation, rejectRecommendation, openRecommendationInOverrides,
          applySimilarRecommendations, onReviewPlanRun: () => setActiveView('Plan', { plan: 'run' }),
          overridePrefill, setOverridePrefill, saveState, persistSettings,
        })}

        {actionMessage && <Toast className="toast">{actionMessage}</Toast>}
        {loading && <Toast className="toast">{copy.loading}</Toast>}
        {feedOpen && (
          <Suspense fallback={null}>
            <ActivityFeed
              notifications={notifications}
              locale={locale}
              onDismiss={dismissNotification}
              onRestore={restoreNotification}
              onClearAll={dismissAllNotifications}
              onRestoreAll={restoreAllNotifications}
              onClose={() => setFeedOpen(false)}
            />
          </Suspense>
        )}
        {!loading && !online && error && <div className="toast muted" role="alert" aria-live="assertive">{copy.apiUnavailable}</div>}
      </main>

      {assistantOpen && (
        <Suspense fallback={null}>
          <AssistantDock locale={locale} notify={notify} onClose={() => setAssistantOpen(false)} />
        </Suspense>
      )}
      </>
      )}
    </DirectionRoot>
    </AssistantPageProvider>
      </ThemeProvider>
    </CacheProvider>
  );
}

export default TVBreakDashboard;
