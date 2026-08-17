import React from 'react';
import { Tooltip } from '@mui/material';
import { Button, IconButton } from '../studio/actions';
import { Bell, ChevronDown, Languages, RefreshCcw } from 'lucide-react';
import { pageText } from './format';
import { planningWeekLabel } from './plan-model';
import { Figure } from './bidi';
import { formatClock } from './dates';
import { MabatIcon } from './kairos-icons';

/*
 * The application header is orientation, not a second workspace. Planning
 * levers and write actions live in Plan, beside the scope and result they
 * change. Keeping them here made every destination inherit a control panel it
 * did not own and left a full-week rewrite one click away from Today.
 */
export function renderTopBar({
  copy,
  locale,
  activeView,
  activeDomain,
  activeDomainLabel,
  localItems,
  onNavigateLocal,
  setActiveView,
  showPlanContext,
  schedule,
  overview,
  online,
  partial,
  loading,
  handleRefresh,
  assistantOpen,
  setAssistantOpen,
  activeNotificationCount,
  setFeedOpen,
  settings,
  settingsAvailable,
  persistSettings,
}) {
  // Before the first response resolves nothing is known, and unknown is not
  // negative: the chip says "checking", never "no connection", until a fetch
  // has actually failed. The offline styling is kept so the state still
  // reads as not-yet-good.
  const connectionLabel = online
    ? (partial ? copy.partialData : copy.liveApi)
    : (loading ? copy.checkingApi : copy.snapshot);

  return (
    <header className="top-bar">
      <div className="top-bar-primary">
        <div className="title-group">
          <span className="section-title">{activeDomainLabel || copy.nav[activeView] || copy.optimizer}</span>
          {showPlanContext ? (
            <Button
              className="date-control"
              variant="text"
              onClick={() => setActiveView('Plan', { plan: 'board' })}
              endIcon={<ChevronDown size={14} />}
            >
              {planningWeekLabel(schedule, locale)}
            </Button>
          ) : null}
        </div>

        <div className="status-group">
          <div className="connection-state" role="status" aria-live="polite" aria-atomic="true">
            <span className={online ? (partial ? 'api-state offline partial' : 'api-state online') : 'api-state offline'}>
              {connectionLabel}
            </span>
            <Tooltip
              title={pageText(locale, 'Time the data was last updated from the API', 'מועד עדכון הנתונים האחרון מה־API')}
              arrow
              placement="bottom"
            >
              <span className="freshness">
                {online && overview.data_freshness ? `${copy.dataUpdated} ${formatClock(overview.data_freshness)}` : `${copy.dataUpdated} -`}
              </span>
            </Tooltip>
          </div>

          <Tooltip title={copy.refresh} arrow placement="bottom">
            <IconButton className="icon-button" aria-label={copy.refresh} onClick={handleRefresh}>
              <RefreshCcw size={17} />
            </IconButton>
          </Tooltip>
          <Tooltip title={pageText(locale, 'Mabat, the Kairos operations assistant', 'מבט, העוזר התפעולי של קיירוס')} arrow placement="bottom">
            <IconButton
              className={assistantOpen ? 'icon-button assistant-toggle open' : 'icon-button assistant-toggle'}
              aria-label={pageText(locale, 'Open or close Mabat, the assistant', 'פתיחה או סגירה של מבט, העוזר')}
              aria-pressed={assistantOpen}
              onClick={() => setAssistantOpen((current) => !current)}
            >
              <MabatIcon size={18} />
            </IconButton>
          </Tooltip>
          <IconButton
            className="icon-button"
            aria-label={copy.notifications}
            onClick={() => setFeedOpen((value) => !value)}
          >
            <span className="bell-wrap">
              <Bell size={17} />
              {activeNotificationCount > 0 ? (
                <Figure className="bell-badge">{activeNotificationCount > 9 ? '9+' : activeNotificationCount}</Figure>
              ) : null}
            </span>
          </IconButton>
          <Button
            className="locale-toggle"
            variant="text"
            startIcon={<Languages size={16} />}
            disabled={!settingsAvailable}
            title={!settingsAvailable
              ? pageText(locale, 'Saved settings are unavailable; refresh before changing language.', 'ההגדרות השמורות אינן זמינות; יש לרענן לפני שינוי השפה.')
              : undefined}
            onClick={() => persistSettings({ ...settings, locale: locale === 'he' ? 'en' : 'he', direction: locale === 'he' ? 'ltr' : 'rtl' })}
          >
            <span className="locale-toggle-label">{locale === 'he' ? copy.english : copy.hebrew}</span>
          </Button>
        </div>
      </div>

      {localItems.length ? (
        <nav className="context-local-nav" aria-label={pageText(locale, `${activeDomainLabel || activeDomain} sections`, `מדורי ${activeDomainLabel || activeDomain}`)}>
          {localItems.map((item) => (
            <Button
              key={item.id}
              variant="text"
              className={item.active ? 'context-nav-item active' : 'context-nav-item'}
              aria-current={item.active ? 'page' : undefined}
              onClick={() => onNavigateLocal(item)}
            >
              {locale === 'he' ? item.he : item.en}
            </Button>
          ))}
        </nav>
      ) : null}
    </header>
  );
}
