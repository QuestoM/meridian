import React from 'react';
import { pageText } from '../shell/surface-helpers';
import { sourceLabel, stepLabel } from './AssistantRunTrace';
import { inApprovedWords } from './kai-vocabulary';

// Companion pieces for the assistant chat column: the single-exchange renderer
// and the paragraph-level text renderer. Both have honest empty and error
// states. The chat loads the saved conversation on open, so returning shows the
// past exchanges. The live progress row lives in AssistantRunTrace, which owns
// the one plain-language name per tool for both the live and the finished view,
// so a step can never be called two different things.

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

// Prose the model wrote, rather than a label the product wrote. Same renderer,
// with the product's own word for the activity put back in place of a retired
// one, in the language the model used. The operator's own question never goes
// through here: it is theirs and it is shown back unchanged.
export function ModelText({ text, className }) {
  return <RichText className={className} text={inApprovedWords(text)} />;
}

function timeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US', { hour: '2-digit', minute: '2-digit' });
}

// One question-and-answer exchange in the live thread, including the quiet
// grounding disclosure, the run trace with per-tool sources, and the time.
export function AssistantExchange({ entry, locale, proposalCard }) {
  const sources = Array.isArray(entry.sources) ? entry.sources : [];
  const toolTrace = Array.isArray(entry.toolTrace) ? entry.toolTrace : [];
  return (
    <article className="asst-exchange">
      <RichText className="asst-q" text={entry.question} />
      {entry.answer ? <ModelText className="asst-a" text={entry.answer} /> : null}
      {entry.answerWithheld ? (
        <p className="asst-truncated" dir="auto">{pageText(locale, 'This question came back as a tool call rather than as an answer, so there is nothing to read here. Ask it again.', 'השאלה הזו חזרה כקריאה לכלי ולא כתשובה, ולכן אין כאן מה לקרוא. אפשר לשאול אותה שוב.')}</p>
      ) : null}
      {entry.truncated ? <p className="asst-truncated">{pageText(locale, 'The answer was shortened by the server.', 'התשובה קוצרה על ידי השרת.')}</p> : null}
      {entry.stoppedAtDeadline ? (
        <p className="asst-truncated" dir="auto">{pageText(locale, 'The answer stopped at the time limit and reports what it had reached by then.', 'התשובה נעצרה במגבלת הזמן ומדווחת על מה שהגיעה אליו עד אז.')}</p>
      ) : null}
      {entry.stoppedAtCeiling ? (
        <p className="asst-truncated" dir="auto">{pageText(locale, 'The search stopped at its limit of model turns and reports what it had reached by then.', 'החיפוש נעצר במגבלת תורות המודל ומדווח על מה שהגיע אליו עד אז.')}</p>
      ) : null}
      {entry.error ? <RichText className="asst-a error" text={entry.error} /> : null}
      {proposalCard}
      {entry.disclosure || sources.length || toolTrace.length ? (
        <details className="asst-disclosure">
          <summary>{pageText(locale, 'What data this is based on', 'על בסיס אילו נתונים')}</summary>
          {entry.disclosure ? <p dir="auto">{entry.disclosure}</p> : null}
          {sources.length ? <p dir="ltr">{sources.map(String).join(', ')}</p> : null}
          {toolTrace.length ? (
            <ol className="asst-run-steps done">
              {toolTrace.map((step, index) => {
                const tool = String((step && step.tool) || '');
                const label = stepLabel(tool, locale);
                return (
                  <li key={index} className={step && step.ok === false ? 'fail' : ''}>
                    <span dir="auto">{label || pageText(locale, 'Read saved data', 'קרא נתונים שמורים')}</span>
                    {label ? null : <code dir="ltr">{tool}</code>}
                    {step && step.source ? <span className="asst-run-source" dir="auto">{sourceLabel(step.source, locale)}</span> : null}
                  </li>
                );
              })}
            </ol>
          ) : null}
        </details>
      ) : null}
      <footer className="asst-meta">
        <time dir="ltr">{timeLabel(entry.at, locale)}</time>
        {Number.isFinite(entry.elapsedSeconds) ? (
          <span dir="ltr">{`${entry.elapsedSeconds.toFixed(1)}s`}</span>
        ) : null}
      </footer>
    </article>
  );
}
