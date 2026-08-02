import React from 'react';
import { pageText } from '../shell/format';
import { RUNS_REMEDY, RUNS_WITHHELD, runsSourceLine } from './history-runs';

// The run log's own state, rendered. The words and the decision are in
// history-runs.js so they can be executed by a test; this file is the two
// elements that carry them, and it is used both in the provenance footer and in
// the empty list a reader lands on when they open the run filter while the
// source is withheld. Same sentence, same door, one source.

export function RunsRemedyLink({ locale }) {
  return (
    <a className="hist-link" href="#Settings">
      {pageText(locale, RUNS_REMEDY[0], RUNS_REMEDY[1])}
    </a>
  );
}

export default function HistoryRunsSource({ locale, state, records, channel }) {
  const line = runsSourceLine(state, records, channel);
  return (
    <>
      <span className={state === RUNS_WITHHELD ? 'warn' : undefined} dir="auto">
        {pageText(locale, line[0], line[1])}
      </span>
      {state === RUNS_WITHHELD ? <RunsRemedyLink locale={locale} /> : null}
    </>
  );
}
