import React from 'react';
import { Button } from '@mui/material';
import { Command, Play, RefreshCcw } from 'lucide-react';
import { formatNumber, pageText } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { Code, Figure } from '../../shell/bidi';

// The step strip, and the plan's own state above it.
//
// Google Ads reads a goal and the progress against it on one line, with the
// state named by a published rule and the remedy on the same row. That is the
// device here: the plan's state is one of three named words, the reason it is in
// that state is beside it, and the control that resolves it is on the same strip
// rather than three screens away.
//
// The control on that row starts the run. It used to carry the run panel's own
// label and only scroll to it, which reproduced the dead end the baseline
// recorded on the frozen top bar: measured, two visible buttons with the same
// words, one of which issued no request at all in two minutes. It now fires the
// same act the palette's run row and the R key fire, through one function, and
// it names that act in its own words rather than borrowing the panel's.
//
// The step numbers are JS-2's own sequence, so the destination reads as the job
// rather than as a menu, and each step carries the shortcut that reaches it, the
// way Linear teaches its keyboard on the row that performs the action.

function stateWord(reading, status, locale) {
  if (reading) return pageText(locale, 'Checking', 'בבדיקה');
  if (status === 'stale') return pageText(locale, 'Out of date', 'לא עדכנית');
  if (status === 'fresh') return pageText(locale, 'Current', 'עדכנית');
  return pageText(locale, 'Unknown', 'לא ידוע');
}

// Why the plan is in that state, in one sentence. A read still in flight says so
// rather than borrowing the unknown state's sentence, which is a claim about the
// plan and would be a false one while nobody has asked the server yet.
function stateDetail({ reading, error, status, changed, freshness, words, locale }) {
  if (reading) return pageText(locale, 'Reading the plan state from the server.', 'קורא את מצב התוכנית מהשרת.');
  if (error) {
    return pageText(locale, `The plan state could not be read: ${error}`, `לא ניתן היה לקרוא את מצב התוכנית: ${error}`);
  }
  if (status === 'stale' && changed.length > 0) {
    return pageText(locale, `Changed since the last run: ${changed.join(', ')}`, `השתנה מאז ההרצה האחרונה: ${changed.join(', ')}`);
  }
  if (status === 'stale') return words.planOutOfDate;
  if (status === 'fresh') return `${words.planCurrent} ${formatStamp(freshness?.computed_at)}`.trim();
  return pageText(locale, 'No run stamp was found for the saved plan.', 'לא נמצא חותם הרצה לתוכנית השמורה.');
}

export function PlanWeekHeader({
  locale,
  words,
  sections,
  active,
  onGo,
  freshness,
  freshnessState,
  freshnessError,
  runState,
  elapsed,
  onRun,
  versionCount,
  liveFrozenAs,
  scopeText,
  onOpenPalette,
}) {
  const reading = freshnessState === 'loading';
  const error = freshnessState === 'unavailable' ? (freshnessError || pageText(locale, 'no reason was given', 'לא נמסרה סיבה')) : null;
  const status = reading || error ? '' : String(freshness?.status || '').toLowerCase();
  const changed = Array.isArray(freshness?.changed) ? freshness.changed.filter(Boolean) : [];
  const running = runState === 'running';

  return (
    <header className="plan-header">
      <div className="plan-header-top">
        <div>
          <h1>{words.place}</h1>
          <p className="plan-header-scope">
            {scopeText || pageText(locale, 'The week your channel is planning.', 'השבוע שהערוץ שלכם מתכנן.')}
          </p>
        </div>
        <div className="plan-header-actions">
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={onOpenPalette}>
            <Command size={14} />
            {pageText(locale, 'Commands', 'פקודות')}
            <kbd className="plan-header-kbd"><Code>Cmd K</Code></kbd>
          </Button>
        </div>
      </div>

      <div className={`plan-header-state is-${reading ? 'reading' : (status || 'unknown')}`} role="status">
        <span className="plan-header-state-word">{stateWord(reading, status, locale)}</span>
        <span className="plan-header-state-detail">
          {stateDetail({ reading, error, status, changed, freshness, words, locale })}
        </span>
        {status === 'stale' && (
          <Button
            className="secondary-button compact"
            type="button"
            variant="outlined"
            disabled={running}
            onClick={onRun}
          >
            {running ? <RefreshCcw size={14} className="upload-spinner" /> : <Play size={14} fill="currentColor" />}
            {running ? `${pageText(locale, 'Running', 'רץ')} ${elapsed}s` : words.runShort}
          </Button>
        )}
        <span className="plan-header-versions">
          {versionCount > 0
            ? pageText(
              locale,
              `${formatNumber(versionCount, locale)} frozen ${versionCount === 1 ? 'version' : 'versions'}${liveFrozenAs ? ', this plan is one of them' : ', this plan is not one of them yet'}`,
              `${formatNumber(versionCount, locale)} גרסאות מוקפאות${liveFrozenAs ? ', התוכנית הזאת אחת מהן' : ', התוכנית הזאת עדיין לא אחת מהן'}`,
            )
            : pageText(locale, 'No plan version has been frozen yet', 'עדיין לא הוקפאה אף גרסת תוכנית')}
        </span>
      </div>

      <nav className="plan-steps" aria-label={pageText(locale, 'Plan steps', 'שלבי התוכנית')}>
        {sections.map((section) => {
          const isActive = section.id === active;
          return (
            <button
              key={section.id}
              type="button"
              className={`plan-step${isActive ? ' is-active' : ''}${section.step ? '' : ' is-reference'}`}
              aria-current={isActive ? 'step' : undefined}
              onClick={() => onGo(section.id)}
            >
              {section.step ? <Figure className="plan-step-number">{section.step}</Figure> : null}
              <span className="plan-step-name">{locale === 'he' ? section.he : section.en}</span>
              <kbd className="plan-step-key"><Code>G {section.key.toUpperCase()}</Code></kbd>
            </button>
          );
        })}
      </nav>
    </header>
  );
}

export default PlanWeekHeader;
