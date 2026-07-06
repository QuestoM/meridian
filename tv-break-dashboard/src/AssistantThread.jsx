import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { pageText } from './surface-helpers';
import { requestJson } from './assistant-stream';

// Companion pieces for the assistant chat column: the single-exchange
// renderer, the live progress row shown while an ask streams, and the
// per-account previous-conversations block backed by the server thread that
// survives restarts. Every surface has honest loading, error and empty states.

const STEP_LABELS = {
  get_settings: ['Reading the saved settings', 'קורא את ההגדרות השמורות'],
  get_day_detail: ['Reading a plan day', 'קורא יום מהתוכנית'],
  list_constraints: ['Reading the placement constraints', 'קורא את אילוצי השיבוץ'],
  list_overrides: ['Reading the manual overrides', 'קורא את העקיפות הידניות'],
  get_pricing: ['Reading the rate card', 'קורא את המחירון'],
  get_net_comparison: ['Comparing the plan to a net-focused plan', 'משווה את התוכנית לתוכנית ממוקדת נטו'],
  get_compliance: ['Checking regulatory compliance', 'בודק עמידה ברגולציה'],
  simulate_settings_change: ['Running a simulation against the optimizer', 'מריץ סימולציה מול האופטימייזר'],
  get_recommendations: ['Reading the recommendations', 'קורא את ההמלצות'],
  get_frontier: ['Reading the balance curve', 'קורא את עקומת האיזון'],
  get_audience_stability: ['Reading audience stability', 'קורא את יציבות הצפייה'],
  get_plan_days: ['Reading the plan days', 'קורא את ימי התוכנית'],
  propose_settings_change: ['Preparing a settings change for approval', 'מכין שינוי הגדרות לאישור'],
  propose_constraint: ['Preparing a constraint for approval', 'מכין אילוץ לאישור'],
  propose_override: ['Preparing an override for approval', 'מכין עקיפה לאישור'],
  propose_pricing_change: ['Preparing a pricing change for approval', 'מכין שינוי מחירון לאישור'],
  propose_recompute: ['Preparing a recompute for approval', 'מכין חישוב מחדש לאישור'],
};

function timeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US', { hour: '2-digit', minute: '2-digit' });
}

function dateTimeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// The live step row under the thinking indicator while an ask streams: names
// the tool the server is running right now in plain language. Unknown tools
// fall back to a generic label plus the raw name, never a fabricated one.
export function StreamProgress({ locale, step }) {
  const tool = step && step.tool ? String(step.tool) : '';
  const pair = STEP_LABELS[tool];
  return (
    <div className="asst-progress" role="status">
      <span className={`asst-progress-dot${step && step.ok === false ? ' fail' : ''}`} aria-hidden="true" />
      <span>{pair ? pageText(locale, pair[0], pair[1]) : pageText(locale, 'Gathering data', 'אוסף נתונים')}</span>
      {!pair && tool ? <code dir="ltr">{tool}</code> : null}
    </div>
  );
}

// One question-and-answer exchange in the live thread, including the quiet
// grounding disclosure, the tool trace with per-tool sources, and the time.
export function AssistantExchange({ entry, locale, proposalCard }) {
  const sources = Array.isArray(entry.sources) ? entry.sources : [];
  const toolTrace = Array.isArray(entry.toolTrace) ? entry.toolTrace : [];
  return (
    <article className="asst-exchange">
      <p className="asst-q" dir="auto">{entry.question}</p>
      {entry.answer ? <div className="asst-a" dir="auto">{entry.answer}</div> : null}
      {entry.truncated ? <p className="asst-truncated">{pageText(locale, 'The answer was shortened by the server.', 'התשובה קוצרה על ידי השרת.')}</p> : null}
      {entry.error ? <div className="asst-a error" dir="auto">{entry.error}</div> : null}
      {proposalCard}
      {entry.disclosure || sources.length || toolTrace.length ? (
        <details className="asst-disclosure">
          <summary>{pageText(locale, 'What data this is based on', 'על בסיס אילו נתונים')}</summary>
          {entry.disclosure ? <p dir="auto">{entry.disclosure}</p> : null}
          {sources.length ? <p dir="ltr">{sources.map(String).join(', ')}</p> : null}
          {toolTrace.length ? (
            <div className="asst-trace">
              {toolTrace.map((step, index) => (
                <code dir="ltr" key={index} className={step && step.ok === false ? 'fail' : ''}>{String((step && step.tool) || '?')}</code>
              ))}
            </div>
          ) : null}
          {toolTrace.some((step) => step && step.source) ? (
            <div className="asst-sources"><span className="asst-sources-head">{pageText(locale, 'Sources', 'מקורות')}</span>
              {toolTrace.filter((step) => step && step.source).map((step, index) => (
                <div className="asst-source-row" dir="ltr" key={index}><code>{String(step.tool || '?')}</code><span className="asst-source-sep" aria-hidden="true">→</span><span className="asst-source-text" dir="auto">{String(step.source)}</span></div>
              ))}
            </div>
          ) : null}
        </details>
      ) : null}
      <footer className="asst-meta"><time dir="ltr">{timeLabel(entry.at, locale)}</time></footer>
    </article>
  );
}

// The previous-conversations block above the live thread: the caller's own
// server-side conversation history, read-only, with a quiet clear action
// behind an inline confirm. Loaded once on mount and again after a clear.
export default function AssistantThread({ locale }) {
  const [state, setState] = useState('loading');
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearError, setClearError] = useState('');

  const load = useCallback(async () => {
    setState('loading');
    try {
      const body = await requestJson('/api/assistant/thread');
      setEntries(Array.isArray(body.entries) ? body.entries : []);
      setState('ready');
      setError('');
    } catch (err) {
      setState('error');
      setError(err.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const clearThread = useCallback(async () => {
    setClearing(true);
    setClearError('');
    try {
      await requestJson('/api/assistant/thread', { method: 'DELETE' });
      setConfirming(false);
      await load();
    } catch (err) {
      setClearError(err.message);
    } finally {
      setClearing(false);
    }
  }, [load]);

  return (
    <section className="asst-prior">
      <button type="button" className="asst-prior-head" aria-expanded={open} onClick={() => setOpen((prev) => !prev)}>
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        <span>{pageText(locale, 'Previous conversations', 'שיחות קודמות')}</span>
        {state === 'ready' ? <span className="asst-prior-count" dir="ltr">{entries.length}</span> : null}
      </button>
      {open ? (
        <div className="asst-prior-body">
          {state === 'loading' ? <div className="asst-loading">{pageText(locale, 'Loading previous conversations', 'טוען שיחות קודמות')}</div> : null}
          {state === 'error' ? <div className="asst-error-note">{pageText(locale, `Previous conversations could not be loaded (${error}).`, `לא ניתן לטעון את השיחות הקודמות (${error}).`)}</div> : null}
          {state === 'ready' && entries.length === 0 ? (
            <div className="asst-empty">{pageText(locale, 'No previous conversations yet. Questions and answers are saved here for your account.', 'אין עדיין שיחות קודמות. שאלות ותשובות נשמרות כאן עבור החשבון שלכם.')}</div>
          ) : null}
          {state === 'ready' ? entries.map((entry, index) => (
            <div className="asst-prior-entry" key={`prior-${index}`}>
              <p className="asst-prior-q" dir="auto">{String((entry && entry.question) || '')}</p>
              {entry && entry.answer ? <p className="asst-prior-a" dir="auto">{String(entry.answer)}</p> : null}
              <time dir="ltr">{dateTimeLabel(entry && entry.at, locale)}</time>
            </div>
          )) : null}
          {state === 'ready' && entries.length > 0 ? (
            confirming ? (
              <div className="asst-confirm" role="alertdialog">
                <p>{pageText(locale, 'This permanently deletes your saved conversations on the server.', 'הפעולה מוחקת לצמיתות את השיחות השמורות שלכם בשרת.')}</p>
                <div className="asst-confirm-actions">
                  <Button variant="contained" size="small" disabled={clearing} onClick={clearThread}>
                    {clearing ? pageText(locale, 'Deleting', 'מוחק') : pageText(locale, 'Delete now', 'מחק עכשיו')}
                  </Button>
                  <Button variant="outlined" size="small" disabled={clearing} onClick={() => setConfirming(false)}>
                    {pageText(locale, 'Cancel', 'ביטול')}
                  </Button>
                </div>
              </div>
            ) : (
              <div className="asst-prior-actions">
                <button type="button" className="asst-prior-clear" onClick={() => setConfirming(true)}>
                  {pageText(locale, 'Clear my history', 'מחיקת ההיסטוריה שלי')}
                </button>
              </div>
            )
          ) : null}
          {clearError ? <div className="asst-error-note">{pageText(locale, `Clearing the history failed (${clearError}).`, `מחיקת ההיסטוריה נכשלה (${clearError}).`)}</div> : null}
        </div>
      ) : null}
    </section>
  );
}
