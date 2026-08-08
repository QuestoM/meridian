import React from 'react';
import { Button, FormControl, IconButton, InputLabel, MenuItem, Select, Slider, Tooltip } from '@mui/material';
import { Bell, Bot, CalendarDays, ChevronDown, GitCompare, Info, Languages, Play, RefreshCcw } from 'lucide-react';
import { Numeric, pageText } from './format';
import { planningWeekLabel } from './plan-model';
import { Figure } from './bidi';
import { formatClock } from './dates';

export function renderTopBar({
  copy,
  locale,
  activeView,
  setActiveView,
  showOptimizationControls,
  schedule,
  overview,
  settings,
  scenario,
  setScenario,
  riskLambda,
  setRiskLambda,
  riskLambdaTouched,
  online,
  partial,
  notify,
  handleRefresh,
  assistantOpen,
  setAssistantOpen,
  activeNotificationCount,
  setFeedOpen,
  persistSettings,
  optimizationState,
  handleRunOptimization,
  recomputeState,
  handleApplyOptimization,
  elapsedSec,
}) {
  return (
        <header className="top-bar">
          <div className="title-group">
            <span className="section-title">{copy.nav[activeView] || copy.optimizer}</span>
            {showOptimizationControls && (
              <Button
                className="date-control"
                type="button"
                variant="outlined"
                onClick={() => {
                  setActiveView('Schedule');
                  notify('Opened the schedule for the active planning week.', 'נפתח לוח השידורים לשבוע התכנון הפעיל.', { transient: true });
                }}
              >
                {planningWeekLabel(schedule, locale)}
                <ChevronDown size={14} />
              </Button>
            )}
          </div>

          {showOptimizationControls && (
          <div className="command-group">
            <FormControl className="scenario-select" size="small">
              <InputLabel id="scenario-label">{copy.scenario}</InputLabel>
              <Select
                labelId="scenario-label"
                value={scenario}
                label={copy.scenario}
                onChange={(event) => {
                  setScenario(event.target.value);
                  notify('Scenario selected. Run optimization to preview this planning mode.', 'התרחיש נבחר. הריצו אופטימיזציה כדי לצפות במצב תכנון זה.', { transient: true });
                }}
              >
                <MenuItem value="Balanced">{copy.scenarios[0]}</MenuItem>
                <MenuItem value="Revenue priority">{copy.scenarios[1]}</MenuItem>
                <MenuItem value="Retention guardrail">{copy.scenarios[2]}</MenuItem>
              </Select>
            </FormControl>
            <div className="risk-lambda-control">
              <div className="risk-lambda-head">
                <span className="risk-lambda-label">{copy.riskCaution}</span>
                <Tooltip title={copy.riskCautionHelp} arrow placement="bottom">
                  <Info size={13} className="risk-lambda-info" aria-label={copy.riskCautionHelp} />
                </Tooltip>
                <Numeric>{`${Math.round(Math.min(100, Math.max(0, riskLambda)))}/100`}</Numeric>
              </div>
              <Slider
                size="small"
                value={riskLambda}
                min={0}
                max={100}
                step={5}
                aria-label={copy.riskCaution}
                valueLabelDisplay="off"
                onChange={(event, value) => {
                  riskLambdaTouched.current = true;
                  setRiskLambda(Array.isArray(value) ? value[0] : value);
                }}
              />
            </div>
            <Button
              className="secondary-button"
              type="button"
              variant="outlined"
              onClick={() => {
                setActiveView('Forecasts');
                notify('Opened scenario comparison.', 'נפתחה השוואת תרחישים.', { transient: true });
              }}
            >
              <GitCompare size={15} />
              {copy.compare}
            </Button>
          </div>
          )}

          <div className="status-group">
            <span className={online ? (partial ? 'api-state offline partial' : 'api-state online') : 'api-state offline'}>
              {online ? (partial ? copy.partialData : copy.liveApi) : copy.snapshot}
            </span>
            <Tooltip title={locale === 'he' ? 'מועד עדכון הנתונים האחרון מה־API' : 'Time the data was last updated from the API'} arrow placement="bottom">
              <span className="freshness">{online && overview.data_freshness ? `${copy.dataUpdated} ${formatClock(overview.data_freshness)}` : `${copy.dataUpdated} -`}</span>
            </Tooltip>
            <IconButton className="icon-button" type="button" aria-label={copy.refresh} size="small" onClick={handleRefresh}>
              <RefreshCcw size={15} />
            </IconButton>
            <Tooltip title={pageText(locale, 'Kai, the Kairos operations assistant', 'קאי, העוזר התפעולי של קיירוס')} arrow placement="bottom">
              <IconButton
                className={assistantOpen ? 'icon-button assistant-toggle open' : 'icon-button assistant-toggle'}
                type="button"
                aria-label={pageText(locale, 'Open or close Kai, the assistant', 'פתיחה או סגירה של קאי, העוזר')}
                aria-pressed={assistantOpen}
                size="small"
                onClick={() => setAssistantOpen((current) => !current)}
              >
                <Bot size={15} />
              </IconButton>
            </Tooltip>
            <IconButton
              className="icon-button"
              type="button"
              aria-label={copy.notifications}
              size="small"
              onClick={() => setFeedOpen((v) => !v)}
            >
              <span className="bell-wrap">
                <Bell size={15} />
                {activeNotificationCount > 0 && (
                  <Figure className="bell-badge">{activeNotificationCount > 9 ? '9+' : activeNotificationCount}</Figure>
                )}
              </span>
            </IconButton>
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => persistSettings({ ...settings, locale: locale === 'he' ? 'en' : 'he', direction: locale === 'he' ? 'ltr' : 'rtl' })}
            >
              <Languages size={14} />
              {locale === 'he' ? copy.english : copy.hebrew}
            </Button>
            {showOptimizationControls && (
              <>
                <Button className="run-button" type="button" variant="contained" disabled={optimizationState === 'running'} onClick={handleRunOptimization}>
                  {optimizationState === 'running' ? <RefreshCcw size={15} className="upload-spinner" /> : <Play size={15} fill="currentColor" />}
                  {optimizationState === 'running' ? `${pageText(locale, `Running ${elapsedSec}s`, `מריץ ${elapsedSec} שנ'`)}` : copy.runOptimization}
                </Button>
                <Tooltip title={pageText(locale, 'Saves these levers and rebuilds the whole weekly schedule, not just the preview', 'שומר את ההגדרות האלה ובונה מחדש את כל הלוח השבועי, לא רק את התצוגה המקדימה')} arrow placement="bottom">
                  <span>
                    <Button
                      className="apply-button"
                      type="button"
                      variant="outlined"
                      disabled={optimizationState === 'running' || recomputeState === 'running'}
                      onClick={handleApplyOptimization}
                    >
                      {recomputeState === 'running' ? <RefreshCcw size={15} className="upload-spinner" /> : <CalendarDays size={15} />}
                      {recomputeState === 'running' ? `${pageText(locale, `Applying ${elapsedSec}s`, `מחיל ${elapsedSec} שנ'`)}` : pageText(locale, 'Apply to weekly schedule', 'החל על לוח השבוע')}
                    </Button>
                  </span>
                </Tooltip>
              </>
            )}
          </div>
        </header>
  );
}
