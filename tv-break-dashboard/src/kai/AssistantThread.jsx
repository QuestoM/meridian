import React from 'react';
import { pageText } from '../shell/surface-helpers';

// Companion pieces for the assistant chat column: the single-exchange renderer and
// the live progress row shown while an ask streams. Both have honest empty and
// error states. The chat loads the saved conversation on open, so returning shows
// the past exchanges.

// Render text as paragraphs, each with its own direction. A single dir="auto" on a
// whole answer sets one direction from the first strong character, so a Hebrew
// answer that opens with a number or a Latin word flips the entire block to LTR.
// Splitting on line breaks and giving each line its own dir="auto" makes every
// line follow its own first character, so Hebrew lines read right-to-left even
// when a neighbour is English.
export function RichText({ text, className }) {
  const value = text === null || text === undefined ? '' : String(text);
  const lines = value.split('\n');
  return (
    <div className={className}>
      {lines.map((line, index) => (
        line.trim()
          ? <p className="asst-para" dir="auto" key={index}>{line}</p>
          : <div className="asst-para-gap" key={index} aria-hidden="true" />
      ))}
    </div>
  );
}

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
  get_schedule_freshness: ['Checking plan freshness', 'בודק את עדכניות התוכנית'],
  get_yield_per_second: ['Reading yield per second', 'קורא תשואה לשנייה'],
  get_gold_breaks: ['Reading gold breaks', 'קורא ברייקי זהב'],
  get_make_good_alerts: ['Checking make-good status', 'בודק סטטוס השלמות'],
  get_run_log_summary: ['Reading the last run summary', 'קורא את סיכום הריצה האחרונה'],
  get_upload_status: ['Checking upload status', 'בודק סטטוס העלאות'],
  get_reports_catalog: ['Reading the reports catalog', 'קורא את קטלוג הדוחות'],
  get_activity_recent: ['Reading recent activity', 'קורא פעילות אחרונה'],
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
      <RichText className="asst-q" text={entry.question} />
      {entry.answer ? <RichText className="asst-a" text={entry.answer} /> : null}
      {entry.truncated ? <p className="asst-truncated">{pageText(locale, 'The answer was shortened by the server.', 'התשובה קוצרה על ידי השרת.')}</p> : null}
      {entry.error ? <RichText className="asst-a error" text={entry.error} /> : null}
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
