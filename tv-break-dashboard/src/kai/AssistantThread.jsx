import React from 'react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Code } from '../shell/bidi';
import { sourceLabel, stepLabel } from './AssistantRunTrace';
import { inApprovedWords } from './kai-vocabulary';
import { claimSegments } from './kai-claimed-action';
import './kai-claimed-action.css';

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

// The model's own words when the payload contradicts them. Nothing is deleted:
// the sentences that claim a proposal nobody can approve are struck through and
// the rest stays readable, because an answer that lied in one line usually
// carries real figures in the next and the operator still needs those.
function RetractedText({ text, locale }) {
  return (
    <blockquote className="asst-retracted">
      <span className="asst-retracted-label">
        {pageText(locale, 'What Kai wrote, with the unbacked part struck out', 'מה שקאי כתב, כשהחלק שאינו נתמך מסומן במחיקה')}
      </span>
      {inApprovedWords(text).split('\n').map((line, index) => (
        line.trim() ? (
          <p className="asst-para" dir="auto" key={index}>
            {claimSegments(line).map((segment, part) => (
              segment.claim
                ? <del className="asst-struck" key={part}>{segment.text}</del>
                : <span key={part}>{segment.text}</span>
            ))}
          </p>
        ) : <div className="asst-para-gap" key={index} aria-hidden="true" />
      ))}
    </blockquote>
  );
}

function timeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US', { hour: '2-digit', minute: '2-digit' });
}

// One question-and-answer exchange in the live thread, including the quiet
// grounding disclosure, the run trace with per-tool sources, and the time.
export function AssistantExchange({ entry, locale, proposalCard, onAskAgain }) {
  const sources = Array.isArray(entry.sources) ? entry.sources : [];
  const toolTrace = Array.isArray(entry.toolTrace) ? entry.toolTrace : [];
  return (
    <article className="asst-exchange">
      <RichText className="asst-q" text={entry.question} />
      {/* The answer said a proposal is waiting for approval and the payload
          recorded none (kai-claimed-action.js). The honest sentence REPLACES the
          claim as the answer rather than following it: a critic measured the
          operator reading a confident paragraph with the correction under it and
          having to trust the smaller of two contradictory statements. What the
          model wrote is kept below, quoted and struck, so nothing is hidden. */}
      {entry.answer && !entry.unrecordedClaim ? <ModelText className="asst-a" text={entry.answer} /> : null}
      {entry.unrecordedClaim ? (
        <p className="asst-unrecorded">
          <span>{pageText(locale, 'No proposal was recorded for this answer, so there is nothing here to approve.', 'לא נרשמה הצעה לתשובה הזו, ולכן אין כאן מה לאשר.')}</span>
          {onAskAgain ? (
            <button type="button" onClick={onAskAgain}>{pageText(locale, 'Ask again', 'שאלו שוב')}</button>
          ) : <span>{pageText(locale, 'Ask again to have the change recorded.', 'אפשר לשאול שוב כדי שהשינוי יירשם.')}</span>}
        </p>
      ) : null}
      {entry.answer && entry.unrecordedClaim ? <RetractedText text={entry.answer} locale={locale} /> : null}
      {entry.answerWithheld ? (
        <p className="asst-truncated">{pageText(locale, 'This question came back as a tool call rather than as an answer, so there is nothing to read here. Ask it again.', 'השאלה הזו חזרה כקריאה לכלי ולא כתשובה, ולכן אין כאן מה לקרוא. אפשר לשאול אותה שוב.')}</p>
      ) : null}
      {entry.truncated ? <p className="asst-truncated">{pageText(locale, 'The answer was shortened by the server.', 'התשובה קוצרה על ידי השרת.')}</p> : null}
      {entry.stoppedAtDeadline ? (
        <p className="asst-truncated">{pageText(locale, 'The answer stopped at the time limit and reports what it had reached by then.', 'התשובה נעצרה במגבלת הזמן ומדווחת על מה שהגיעה אליו עד אז.')}</p>
      ) : null}
      {entry.stoppedAtCeiling ? (
        <p className="asst-truncated">{pageText(locale, 'The search stopped at its limit of model turns and reports what it had reached by then.', 'החיפוש נעצר במגבלת תורות המודל ומדווח על מה שהגיע אליו עד אז.')}</p>
      ) : null}
      {entry.error ? <RichText className="asst-a error" text={entry.error} /> : null}
      {proposalCard}
      {entry.disclosure || sources.length || toolTrace.length ? (
        <details className="asst-disclosure">
          <summary>{pageText(locale, 'What data this is based on', 'על בסיס אילו נתונים')}</summary>
          {entry.disclosure ? <p>{entry.disclosure}</p> : null}
          {sources.length ? <p><Code>{sources.map(String).join(', ')}</Code></p> : null}
          {toolTrace.length ? (
            <ol className="asst-run-steps done">
              {toolTrace.map((step, index) => {
                const tool = String((step && step.tool) || '');
                const label = stepLabel(tool, locale);
                return (
                  <li key={index} className={step && step.ok === false ? 'fail' : ''}>
                    <span>{label || pageText(locale, 'Read saved data', 'קרא נתונים שמורים')}</span>
                    {label ? null : <Code>{tool}</Code>}
                    {step && step.source ? <span className="asst-run-source">{sourceLabel(step.source, locale)}</span> : null}
                  </li>
                );
              })}
            </ol>
          ) : null}
        </details>
      ) : null}
      <footer className="asst-meta">
        <time><Figure>{timeLabel(entry.at, locale)}</Figure></time>
        {Number.isFinite(entry.elapsedSeconds) ? (
          <Figure>{`${entry.elapsedSeconds.toFixed(1)}s`}</Figure>
        ) : null}
      </footer>
    </article>
  );
}
