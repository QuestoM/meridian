import React from 'react';
import { Button } from '@mui/material';
import { Flag, Target } from 'lucide-react';
import { EMPTY_VALUE, formatCurrency, formatNumber, pageText } from '../../shell/format';
import { Figure } from '../../shell/bidi';

// The goal and the progress against it, read together.
//
// Google Ads is the reference and the device is taken as a mechanic, not as a
// decoration: the state is a named word decided by a published numeric rule
// rather than a chart the reader has to interpret, the rule is printed beside
// the state, and the remedy sits in the same strip as the diagnosis. Linear
// supplies the second device: an empty field is an action, so a window with no
// target renders the control that supplies one rather than a blank or a
// sentence.
//
// Nothing here is computed in the browser. The window, the projection, the
// target and the verdict all arrive from one call, so there is exactly one
// implementation of the threshold in the product and this surface cannot drift
// from the one the general manager reads.

const STATE_WORDS = {
  on_plan: { en: 'On plan', he: 'עומד ביעד' },
  at_risk: { en: 'At risk', he: 'בסיכון' },
  behind: { en: 'Behind', he: 'בפיגור' },
};

export function verdictWord(state, locale) {
  const entry = STATE_WORDS[state];
  if (!entry) return locale === 'he' ? 'אין יעד' : 'No target';
  return locale === 'he' ? entry.he : entry.en;
}

function windowLine(window, locale) {
  const from = window?.date_from;
  const to = window?.date_to;
  if (!from || !to) return null;
  const days = Number(window?.n_dates) || 0;
  return pageText(
    locale,
    `${from} to ${to}, ${formatNumber(days, locale)} broadcast days`,
    `${from} עד ${to}, ${formatNumber(days, locale)} ימי שידור`,
  );
}

function Remedy({ state, reason, locale, onGo, canEdit, scoped }) {
  // Nothing else on the strip matters while the projection is not the
  // operator's, so the one control is the one that makes it theirs.
  if (!scoped) {
    return (
      <Button
        className="secondary-button compact"
        type="button"
        variant="outlined"
        onClick={() => { window.location.hash = 'Settings'; }}
      >
        {pageText(locale, 'Set the operator channel', 'הגדירו ערוץ מפעיל')}
      </Button>
    );
  }
  if (state === 'behind' || state === 'at_risk') {
    return (
      <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => onGo('compare')}>
        {pageText(locale, 'Compare two ways to run it', 'השוו שתי דרכים להריץ')}
      </Button>
    );
  }
  if (state === 'on_plan') {
    return (
      <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => onGo('publish')}>
        {pageText(locale, 'Freeze this plan', 'הקפיאו את התוכנית הזאת')}
      </Button>
    );
  }
  if (reason === 'no_projection') {
    return (
      <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => onGo('run')}>
        {pageText(locale, 'Run the plan', 'הריצו את התוכנית')}
      </Button>
    );
  }
  if (!canEdit) return null;
  return (
    <Button
      className="secondary-button compact"
      type="button"
      variant="outlined"
      onClick={() => { window.location.hash = 'Overview'; }}
    >
      {pageText(locale, 'Set a target on Today', 'קבעו יעד במסך היום')}
    </Button>
  );
}

export function GoalStrip({ progress, locale, words, onGo }) {
  if (!progress || progress.available === false) return null;
  const verdict = progress.verdict || {};
  const target = progress.target || {};
  const state = String(verdict.state || 'unavailable');
  const set = target.state === 'set';
  const scopeText = windowLine(progress.window, locale);
  const channel = String(progress.channel || '').trim();
  // The projection is summed over whatever the route could scope it to. With no
  // operator channel configured that is every channel in the source, so the
  // figure is the market's and not this operator's, and printing it under their
  // own revenue label would be the boundary broken by a heading. Measured on
  // this tree at 2026-08-01: 54,650,165.39 across four channels.
  const scoped = Boolean(channel);
  const threshold = locale === 'he' ? verdict.threshold_he : verdict.threshold_en;
  const others = Array.isArray(progress.other_windows) ? progress.other_windows : [];

  return (
    <div className={`plan-goal is-${state}`} role="status" aria-label={pageText(locale, 'The week against its target', 'השבוע מול היעד שלו')}>
      <span className="plan-goal-icon" aria-hidden="true">{set ? <Flag size={16} /> : <Target size={16} />}</span>

      <div className="plan-goal-figures">
        <div className="plan-goal-pair">
          <span className="plan-goal-label">{words.expectedRevenue}</span>
          <strong className="numeric">
            <Figure>{scoped ? formatCurrency(progress.projected?.revenue, locale) : EMPTY_VALUE}</Figure>
          </strong>
        </div>
        {scoped && set && (
          <div className="plan-goal-pair">
            <span className="plan-goal-label">{pageText(locale, 'Target', 'יעד')}</span>
            <strong className="numeric"><Figure>{formatCurrency(target.amount_ils, locale)}</Figure></strong>
          </div>
        )}
        {scoped && set && (
          <div className="plan-goal-pair">
            <span className="plan-goal-label">{pageText(locale, 'Against the target', 'מול היעד')}</span>
            <strong className={`numeric plan-goal-variance is-${state}`}>
              <Figure>
                {Number(verdict.variance_ils) > 0 ? '+' : ''}{formatCurrency(verdict.variance_ils, locale)}
                {Number.isFinite(Number(verdict.variance_percent))
                  ? ` (${Number(verdict.variance_percent) > 0 ? '+' : ''}${Number(verdict.variance_percent).toFixed(2)}%)`
                  : ''}
              </Figure>
            </strong>
          </div>
        )}
      </div>

      <div className="plan-goal-verdict">
        <span className={`plan-goal-chip is-${state}`}>{verdictWord(state, locale)}</span>
        <span className="plan-goal-basis">
          {scopeText}
          {/* A Hebrew channel name inside an English sentence reorders without
              an isolate, so the name carries its own direction. */}
          {channel ? <>{', '}<bdi>{channel}</bdi></> : ''}
        </span>
        {!scoped && (
          <span className="plan-goal-rule">
            {pageText(
              locale,
              'No operator channel is configured, so this projection would cover every channel in the source rather than yours, and no figure is shown for it. The path to a figure: set the operator channel in Settings.',
              'לא הוגדר ערוץ מפעיל, ולכן התחזית הזאת הייתה מכסה את כל הערוצים שבמקור הנתונים ולא את שלכם, ולכן לא מוצג עבורה מספר. הדרך למספר: הגדירו ערוץ מפעיל במסך ההגדרות.',
            )}
          </span>
        )}
        {set && threshold ? <span className="plan-goal-rule">{threshold}</span> : null}
        {scoped && !set && (
          <span className="plan-goal-rule">
            {pageText(
              locale,
              'No target is set for this window, so there is no honest answer to whether the week is on plan and no figure is shown for it. A target is a number somebody owns: the revenue a week is measured against, its window, and the percentage below it that counts as at risk.',
              'לא נקבע יעד לחלון הזה, ולכן אין תשובה כנה לשאלה אם השבוע עומד ביעד ולא מוצג עבורה מספר. יעד הוא מספר שמישהו אחראי עליו: ההכנסה שמולה נמדד שבוע, החלון שלה, והאחוז מתחתיה שנחשב סיכון.',
            )}
          </span>
        )}
        {scoped && !set && others.length > 0 && (
          <span className="plan-goal-rule">
            {pageText(
              locale,
              `A target exists on ${formatNumber(others.length, locale)} other window${others.length === 1 ? '' : 's'} for this channel, and a target set for a different span is deliberately not read as this one's.`,
              `קיים יעד ל-${formatNumber(others.length, locale)} חלונות אחרים בערוץ הזה, ויעד שנקבע לטווח אחר אינו נקרא במכוון כיעד של החלון הזה.`,
            )}
          </span>
        )}
        {set && target.set_by ? (
          <span className="plan-goal-rule">
            {pageText(locale, `Set by ${target.set_by}`, `נקבע בידי ${target.set_by}`)}
          </span>
        ) : null}
      </div>

      <Remedy
        state={state}
        reason={verdict.reason}
        locale={locale}
        onGo={onGo}
        canEdit={progress.can_edit !== false}
        scoped={scoped}
      />
    </div>
  );
}

export default GoalStrip;
