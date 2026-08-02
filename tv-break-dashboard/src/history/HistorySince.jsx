import React, { useCallback, useEffect, useState } from 'react';
import { ShieldCheck } from 'lucide-react';
import { pageText } from '../shell/format';
import { RunsRemedyLink } from './HistoryRunsSource';
import { fetchSince } from './history-api';
import { todayIso } from './history-labels';
import { attestationStartLine } from './history-reach';
import { RUNS_WITHHELD, runsCountLine, runsCounted, runsSourceState } from './history-runs';

// "Has anything changed since?" with the evidence attached.
//
// The compliance half of this destination. A regulator asks whether the limits
// in force are the current ones, and the only honest answer is the guardrail
// store's own append-only change record: an empty list since a named day is the
// evidence, and the day the record itself starts is printed beside it so nobody
// mistakes "nothing recorded" for "nothing happened before the record existed".
//
// Two records answer on this strip and each one now names itself. The guardrail
// store starts on its own baseline date; the merged change record starts
// wherever pruning left it, which is a different day and usually a much later
// one. Measured before they were told apart: "2,562 changes and points recorded"
// for a window opening on 14 June sat directly beside "and this record starts on
// 2026-06-14", over five hours of surviving change record.
//
// A store that cannot be read answers unknown. It never answers unchanged.
//
// The day it starts on is today in the broadcast zone, which is the same zone
// the server files an entry under, so the two cannot disagree about what "since
// today" means.

export default function HistorySince({ locale, landing, onShow }) {
  const [day, setDay] = useState(todayIso);
  const [state, setState] = useState('loading');
  const [body, setBody] = useState(null);
  const [error, setError] = useState('');

  const load = useCallback(async (wanted) => {
    setState('loading');
    const result = await fetchSince(wanted);
    if (!result.ok) {
      setState('error');
      setError(result.error);
      return;
    }
    setBody(result.data);
    setState('ready');
  }, []);

  // The landing answer rides the timeline read, so opening History costs one
  // request rather than two, and this waits for it rather than racing it. Any
  // other day is a question this asks for itself.
  useEffect(() => {
    if (landing && landing.day === day) {
      setBody(landing);
      setState('ready');
      return;
    }
    if (!landing && day === todayIso()) {
      setState('loading');
      return;
    }
    load(day);
  }, [load, day, landing]);

  const guardrails = (body && body.guardrails) || {};
  const counts = (body && body.counts) || {};
  const changeCount = (counts.change || 0) + (counts.restore || 0) + (counts.restore_point || 0);
  // The runs are counted only when the product may attribute them. Withheld, the
  // tally is zero because no run entry was assembled, not because none ran, and
  // printing that zero here would put a false attestation in the one sentence a
  // compliance owner reads first. The examined sources travel with the verdict
  // for exactly this reason, so the sentence says unknown and names the cause.
  const runsState = runsSourceState(body && body.examined);
  const counted = runsCounted(runsState);
  // The day the counted record starts on, and whether the asked-for day is
  // inside it. Both are read from the payload rather than from the control, so
  // the sentence and the warning it carries can never describe different days
  // while a slower read is still in the air.
  const startLine = attestationStartLine(body);
  const covered = !(body && body.record_starts) || String(body.day || '') >= String(body.record_starts);

  return (
    <section className="hist-since" aria-label={pageText(locale, 'What changed since', 'מה השתנה מאז')}>
      <ShieldCheck size={16} aria-hidden="true" />
      <label className="hist-select">
        <span>{pageText(locale, 'Since', 'מאז')}</span>
        <input type="date" value={day} onChange={(event) => setDay(event.target.value)} dir="ltr" />
      </label>

      {state === 'loading' ? <span className="hist-since-line">{pageText(locale, 'Reading the record', 'קורא את הרישום')}</span> : null}
      {state === 'error' ? <span className="hist-since-line warn" dir="auto">{pageText(locale, `The record could not be read. ${error}`, `לא ניתן לקרוא את הרישום. ${error}`)}</span> : null}

      {state === 'ready' ? (
        <>
          <span className="hist-since-line" dir="auto">
            {changeCount && counted
              ? pageText(locale, `${changeCount} changes and points recorded, and ${counts.run || 0} runs.`, `נרשמו ${changeCount} שינויים ונקודות, ו-${counts.run || 0} הרצות.`)
              : null}
            {changeCount && !counted
              ? pageText(locale, `${changeCount} changes and points recorded.`, `נרשמו ${changeCount} שינויים ונקודות.`)
              : null}
            {!changeCount && counted
              ? pageText(locale, 'Nothing has been recorded since that day.', 'לא נרשם דבר מאז אותו יום.')
              : null}
            {!changeCount && !counted
              ? pageText(locale, 'No change and no restore point has been recorded since that day.', 'לא נרשמו שינויים ולא נקודות שחזור מאז אותו יום.')
              : null}
          </span>
          {/* And the day that count is only as old as. A count of changes since a
              day is evidence for the days the record covers and for no others. */}
          {startLine ? (
            <span className={`hist-since-line${covered ? '' : ' warn'}`} dir="auto">{pageText(locale, startLine[0], startLine[1])}</span>
          ) : null}
          {/* The count opens the entries it counted. It sets the list's own From day
              to this day and drops every filter, so the tabs under it add up to the
              figure in this sentence and a reader can walk from the attestation to
              the record that backs it. */}
          {changeCount && onShow ? (
            <button type="button" className="hist-link" onClick={() => onShow(day)}>
              {pageText(locale, 'Show these in the list', 'הצגה ברשימה')}
            </button>
          ) : null}
          {counted ? null : (
            <span className="hist-since-line warn" dir="auto">{pageText(locale, ...runsCountLine(runsState))}</span>
          )}
          {runsState === RUNS_WITHHELD ? <RunsRemedyLink locale={locale} /> : null}
          {guardrails.state === 'unchanged' ? (
            <span className="hist-since-verdict ok" dir="auto">
              {pageText(locale, `No regulatory limit moved. The limits in force took effect on ${guardrails.effective_date}, and the regulatory limit record starts on ${guardrails.record_starts}.`, `אף מגבלת רגולציה לא זזה. המגבלות שבתוקף נכנסו לתוקף ב-${guardrails.effective_date}, ורישום מגבלות הרגולציה מתחיל ב-${guardrails.record_starts}.`)}
            </span>
          ) : null}
          {guardrails.state === 'changed' ? (
            <span className="hist-since-verdict warn" dir="auto">
              {pageText(locale, `${guardrails.changes.length} regulatory limit changes were recorded since that day.`, `נרשמו ${guardrails.changes.length} שינויים במגבלות הרגולציה מאז אותו יום.`)}
            </span>
          ) : null}
          {guardrails.state === 'unknown' ? (
            <span className="hist-since-verdict warn" dir="auto">
              {pageText(locale, 'The regulatory limit record could not be read, so no attestation can be made from this screen.', 'לא ניתן לקרוא את רישום מגבלות הרגולציה, ולכן אי אפשר להצהיר מהמסך הזה.')}
            </span>
          ) : null}
          {(guardrails.scheduled || []).length ? (
            <span className="hist-since-verdict warn" dir="auto">
              {pageText(locale, `${guardrails.scheduled.length} limit changes are recorded for a future date.`, `${guardrails.scheduled.length} שינויי מגבלות רשומים לתאריך עתידי.`)}
            </span>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
