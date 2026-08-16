import React from 'react';
import { Button } from '../../studio/actions';
import { Check, Circle, Play, RefreshCcw, TriangleAlert } from 'lucide-react';
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

function TraceNode({ state, label, detail }) {
  const Icon = state === 'complete' ? Check : state === 'error' ? TriangleAlert : state === 'active' ? RefreshCcw : Circle;
  return (
    <li className={`plan-trace-node is-${state}`}>
      <span className="plan-trace-marker" aria-hidden="true">
        <Icon size={15} className={state === 'active' ? 'upload-spinner' : undefined} />
      </span>
      <span className="plan-trace-copy">
        <strong>{label}</strong>
        <small>{detail}</small>
      </span>
    </li>
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
  actionDisabled,
  actionDisabledReason,
}) {
  // A read still in flight is not a verdict, so it is not drawn as one.
  const reading = freshnessState === 'loading';
  const status = reading ? '' : String(freshness?.status || '').toLowerCase();
  const changed = Array.isArray(freshness?.changed) ? freshness.changed.filter(Boolean) : [];
  const running = runState === 'running';
  const owned = runResult?.owned;
  const zeroBreaks = Number(owned?.total_breaks) === 0;
  const ownedScope = scopeLine(owned?.scope, locale);
  const progressDone = Number(runProgress?.done);
  const progressTotal = Number(runProgress?.total);
  const progressKnown = Number.isFinite(progressDone) && Number.isFinite(progressTotal) && progressTotal > 0;

  const stateLine = reading
    ? pageText(locale, 'Reading the plan state from the server', 'קורא את מצב התוכנית מהשרת')
    : status === 'stale'
      ? words.planOutOfDate
      : status === 'fresh'
        ? `${words.planCurrent} ${formatStamp(freshness?.computed_at)}`.trim()
        : pageText(locale, 'The plan state is unknown', 'מצב התוכנית אינו ידוע');

  return (
    <section className="card plan-section" aria-labelledby="plan-run-title">
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
        <Button className="run-button" type="button" variant="contained" disabled={running || actionDisabled} title={actionDisabledReason || undefined} onClick={onRun}>
          {running ? <RefreshCcw size={15} className="upload-spinner" /> : <Play size={15} fill="currentColor" />}
          {running ? `${pageText(locale, `Running ${elapsed}s`, `רץ ${elapsed} שנ'`)}` : words.run}
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

      <div className="plan-run-console" aria-label={pageText(locale, 'Weekly plan run stages', 'שלבי הרצת התוכנית השבועית')}>
        <div className="plan-run-console-head">
          <span>{pageText(locale, 'Run stages', 'שלבי הרצה')}</span>
          <Numeric>{running ? `${elapsed}s` : runResult ? formatStamp(runResult.computed_at) : '\u2014'}</Numeric>
        </div>
        <ol className="plan-run-trace">
          <TraceNode
            state={reading ? 'active' : 'complete'}
            label={pageText(locale, 'Read plan state', 'קריאת מצב התוכנית')}
            detail={stateLine}
          />
          <TraceNode
            state={runError ? 'error' : running || runResult ? 'complete' : 'waiting'}
            label={pageText(locale, 'Load the saved objective', 'טעינת המטרה השמורה')}
            detail={running || runResult
              ? pageText(locale, 'The saved objective is the run input', 'המטרה השמורה היא קלט ההרצה')
              : pageText(locale, 'Starts only after the consequence review is confirmed', 'מתחיל רק לאחר אישור בדיקת ההשפעה')}
          />
          <TraceNode
            state={runError ? 'error' : runResult ? 'complete' : running ? 'active' : 'waiting'}
            label={pageText(locale, 'Decide each broadcast day', 'הכרעת כל יום שידור')}
            detail={progressKnown
              ? pageText(locale, `${formatNumber(progressDone, locale)} of ${formatNumber(progressTotal, locale)} days decided`, `${formatNumber(progressDone, locale)} מתוך ${formatNumber(progressTotal, locale)} ימים הוכרעו`)
              : runResult
                ? pageText(locale, `${formatNumber(owned?.days, locale)} broadcast days completed`, `${formatNumber(owned?.days, locale)} ימי שידור הושלמו`)
                : pageText(locale, 'No day result is claimed before it arrives', 'לא מוצגת תוצאת יום לפני שהגיעה')}
          />
          <TraceNode
            state={runError ? 'error' : runResult ? 'complete' : running ? 'active' : 'waiting'}
            label={pageText(locale, 'Write the weekly result', 'כתיבת תוצאת השבוע')}
            detail={runError
              ? pageText(locale, 'The write did not complete; the saved plan is unchanged', 'הכתיבה לא הושלמה; התוכנית השמורה לא השתנתה')
              : runResult
                ? pageText(locale, `${formatNumber(owned?.rows, locale)} plan rows written`, `${formatNumber(owned?.rows, locale)} שורות תוכנית נכתבו`)
                : pageText(locale, 'The live weekly plan moves only at this stage', 'התוכנית השבועית החיה משתנה רק בשלב הזה')}
          />
        </ol>
        {running && progressKnown ? (
          <div className="plan-run-progress" role="progressbar" aria-valuemin="0" aria-valuemax={progressTotal} aria-valuenow={progressDone}>
            <i style={{ '--plan-run-progress': progressDone / progressTotal }} />
          </div>
        ) : null}
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

      {runResult && owned && zeroBreaks && (
        <p className="plan-note plan-note-amber" role="alert">
          {pageText(
            locale,
            'The run finished with zero breaks on your channel. This is not a routine completion. Review the objective and inputs before freezing the plan.',
            'ההרצה הסתיימה עם אפס ברייקים בערוץ שלכם. זו אינה השלמה שגרתית. יש לבדוק את המטרה ואת הקלטים לפני הקפאת התוכנית.',
          )}
        </p>
      )}

      {runResult && owned && (
        <div className="plan-result-ledger">
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
