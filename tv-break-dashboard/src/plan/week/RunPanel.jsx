import React from 'react';
import { Button } from '@mui/material';
import { Play, RefreshCcw } from 'lucide-react';
import { Numeric, finiteNumber, formatCurrency, formatNumber, pageText } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { scopeLine } from './plan-week-model';

// Step two: run the plan, and read what the run produced.
//
// One verb. The product used two retired words for this act, and the same two
// for training the model, so nobody could tell which of the two a button did.
// The canonical word is run, taken from the frozen vocabulary, and it is the
// only word on this control.
//
// Whether the week is on plan is answered once, on the goal strip at the top of
// the destination, so this panel does not carry a second copy of that question.
// Two states for one question on one surface is the duplication this rebuild
// exists to remove.

function Figure({ label, value, sub }) {
  return (
    <div className="plan-figure">
      <span>{label}</span>
      <strong><Numeric>{value}</Numeric></strong>
      {sub ? <small>{sub}</small> : null}
    </div>
  );
}

export function RunPanel({
  locale,
  words,
  freshness,
  freshnessState,
  runState,
  runProgress,
  runResult,
  runError,
  elapsed,
  planScope,
  onRun,
}) {
  // A read still in flight is not a verdict, so it is not drawn as one.
  const reading = freshnessState === 'loading';
  const status = reading ? '' : String(freshness?.status || '').toLowerCase();
  const changed = Array.isArray(freshness?.changed) ? freshness.changed.filter(Boolean) : [];
  const running = runState === 'running';
  const owned = runResult?.owned;
  const ownedScope = scopeLine(owned?.scope, locale);

  const stateLine = reading
    ? pageText(locale, 'Reading the plan state from the server', 'קורא את מצב התוכנית מהשרת')
    : status === 'stale'
      ? words.planOutOfDate
      : status === 'fresh'
        ? `${words.planCurrent} ${formatStamp(freshness?.computed_at)}`.trim()
        : pageText(locale, 'The plan state is unknown', 'מצב התוכנית אינו ידוע');

  return (
    <section className="plan-section" aria-labelledby="plan-run-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-run-title">{pageText(locale, 'Run the plan', 'הרצת התוכנית')}</h2>
          <p>
            {pageText(
              locale,
              'A run reads the saved objective and rewrites the weekly plan every screen reads. It is the only act that moves the plan.',
              'הרצה קוראת את המטרה השמורה וכותבת מחדש את התוכנית השבועית שכל המסכים קוראים. זו הפעולה היחידה שמזיזה את התוכנית.',
            )}
          </p>
        </div>
        <Button className="run-button" type="button" variant="contained" disabled={running} onClick={onRun}>
          {running ? <RefreshCcw size={15} className="upload-spinner" /> : <Play size={15} fill="currentColor" />}
          {running ? `${pageText(locale, 'Running', 'רץ')} ${elapsed}s` : words.run}
        </Button>
      </div>

      <div className={`plan-state plan-state-${status || 'unknown'}`} role="status">
        <strong>{stateLine}</strong>
        {status === 'stale' && changed.length > 0 && (
          <span>
            {pageText(locale, 'What changed since the last run: ', 'מה השתנה מאז ההרצה האחרונה: ')}
            {changed.join(', ')}
          </span>
        )}
        {status === 'fresh' && planScope ? <span>{planScope}</span> : null}
      </div>

      {running && (
        <p className="plan-note" role="status">
          {runProgress
            ? pageText(
              locale,
              `Running the weekly plan: day ${runProgress.done} of ${runProgress.total}. Elapsed ${elapsed}s.`,
              `מריץ את התוכנית השבועית: יום ${runProgress.done} מתוך ${runProgress.total}. חלפו ${elapsed} שניות.`,
            )
            : pageText(
              locale,
              `Running the whole weekly plan. Elapsed ${elapsed}s.`,
              `מריץ את כל התוכנית השבועית. חלפו ${elapsed} שניות.`,
            )}
        </p>
      )}

      {runError && (
        <p className="plan-note plan-note-red" role="alert">
          {pageText(locale, `The run failed and the saved plan is unchanged: ${runError}`, `ההרצה נכשלה והתוכנית השמורה לא השתנתה: ${runError}`)}
        </p>
      )}

      {runResult && owned && (
        <div className="plan-figure-row">
          <Figure
            label={pageText(locale, 'Plan rows written', 'שורות תוכנית שנכתבו')}
            value={formatNumber(owned.rows, locale)}
            sub={ownedScope || undefined}
          />
          <Figure
            label={pageText(locale, 'Breaks on your channel', 'ברייקים בערוץ שלכם')}
            value={formatNumber(owned.total_breaks, locale)}
            sub={pageText(locale, `over ${formatNumber(owned.days, locale)} broadcast days`, `על פני ${formatNumber(owned.days, locale)} ימי שידור`)}
          />
          <Figure
            label={words.expectedRevenue}
            value={formatCurrency(owned.total_revenue, locale)}
            sub={ownedScope || undefined}
          />
          <Figure
            label={pageText(locale, 'Run at', 'הורצה ב')}
            value={formatStamp(runResult.computed_at) || pageText(locale, 'unknown', 'לא ידוע')}
          />
        </div>
      )}

    </section>
  );
}

export function planScopeLine(schedule, locale) {
  const note = schedule?.scope?.plan;
  const line = scopeLine(note, locale);
  if (line) return line;
  const total = finiteNumber(schedule?.break_schedule_total_rows);
  if (total === null) return null;
  return pageText(locale, `${formatNumber(total, locale)} plan rows`, `${formatNumber(total, locale)} שורות תוכנית`);
}

export default RunPanel;
