import React, { useEffect, useState } from 'react';
import { Code, Name, Prose } from '../../shell/bidi';
import { Numeric } from '../../shell/format';
import { readSection, startTraining } from './console-api';
import { Absent, Panel, RecordDrill } from './console-bits';
import { canEditReason, pick, t } from './console-words';
import { Pressable, SelectControl } from '../../studio/dom-controls';

// Training, started from the console and only ever into the console's own
// store. The safety sentence is on the screen and not only in the code: a run
// writes into the releases store and never over the shipped artifact, so
// nothing an operator reads moves until somebody adopts a run deliberately.
//
// The five gate overrides are the provenance hole the console itself reports.
// A run started here records exactly which were used, so for these runs a
// forced gate and a self-activated one are told apart afterwards.
//
// **The panel follows the run it started.** Measured on the shipped tree: the
// start read the route once, 41 ms after the write, got `running`, and never
// asked again. The run finished 7.0 s later with exit code 0 and the screen
// still read "training" three minutes on, with the trainer's own control still
// disabled, so the steward could not start another run either. Leaving the
// section and returning read the truth, which proved the store was right and
// only the screen was wrong. A finish is an event on the server with no click
// behind it, so the only honest way for this screen to learn about it is to
// ask again while a run is open, and to stop asking the moment none is.

const CHOICES = [
  { value: '', key: 'training.auto' },
  { value: 'force-on', key: 'training.force_on' },
  { value: 'force-off', key: 'training.force_off' },
];

// How long the panel waits between two reads while a run is open. The route
// answers warm in about 3 ms, so this costs the server nothing and bounds how
// stale the screen can be at a run's end.
const WATCH_MS = 1500;

// Whether a run is open, read from the payload the route serves and never from
// anything this panel remembers about its own click.
//
// The signal is the server's own flight register rather than the word on a run
// record, and the difference matters in one case: a run whose process died with
// the server leaves a record saying it is running that nothing will ever
// finish. Following the record would leave this panel asking for ever about a
// run nobody is performing. Following the register, the panel follows a run
// exactly while a run is really being performed, and the register is only ever
// cleared after the finished record has been written.
function runIsOpen(payload) {
  return Object.keys(payload.in_flight || {}).length > 0;
}

function Trainer({ trainer, locale, running, locked, onStart }) {
  const [flags, setFlags] = useState({});
  const busy = Boolean(running);
  return (
    <div className="mc-trainer">
      <div className="mc-trainer-head">
        <div>
          <strong>{locale === 'en' ? trainer.label_en : trainer.label_he}</strong>
          <small><Code>{trainer.script}</Code></small>
        </div>
        <Pressable
          type="button"
          className="mc-button mc-primary"
          onClick={() => onStart(trainer.artifact, flags)}
          disabled={busy || locked}
        >
          {busy ? t('training.running', locale) : t('training.start', locale)}
        </Pressable>
      </div>
      <p className="mc-note">
        {t('training.writes', locale)} <code><Code>{trainer.writes}</Code></code>
        {trainer.measured_seconds ? (
          <>
            {' '}
            <Numeric>{`${trainer.measured_seconds}s`}</Numeric>
          </>
        ) : null}
      </p>
      {trainer.flags.length ? (
        <div className="mc-flags">
          <span className="mc-flags-label">{t('training.overrides', locale)}</span>
          {trainer.flags.map((flag) => (
            <label className="mc-flag" key={flag.flag}>
              <code><Code>{flag.flag}</Code></code>
              <SelectControl
                value={flags[flag.flag] || ''}
                onChange={(event) => setFlags((current) => {
                  const next = { ...current };
                  if (event.target.value) next[flag.flag] = event.target.value;
                  else delete next[flag.flag];
                  return next;
                })}
              >
                {CHOICES.map((choice) => (
                  <option value={choice.value} key={choice.key}>{t(choice.key, locale)}</option>
                ))}
              </SelectControl>
            </label>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RunRow({ run, locale }) {
  const [open, setOpen] = useState(false);
  const changed = run.would_change || {};
  return (
    <li className={`mc-run mc-run-${run.state}`}>
      <div className="mc-run-head">
        <Code className="mc-run-id">{run.run_id}</Code>
        <span className="mc-run-artifact">{locale === 'en' ? run.label_en : run.label_he}</span>
        <span className={`mc-verdict mc-${run.state === 'done' ? 'active' : run.state === 'failed' ? 'tested_and_lost' : 'no_contrast'} mc-sm`}>
          {run.state === 'done' ? t('training.done', locale) : run.state === 'failed' ? t('training.failed', locale) : t('training.running', locale)}
        </span>
      </div>
      {/*
        Three facts, three runs, rather than one string forced left-to-right.
        The stamp and the duration are figures and the actor is a name, and an
        actor written in Hebrew inside a figure would have been laid out as
        though it were part of the timestamp beside it.
      */}
      <p className="mc-run-meta">
        <Numeric>{String(run.started_at || '').slice(0, 19)}</Numeric>
        {' '}
        <Name>{run.actor}</Name>
        {run.duration_seconds ? <> <Numeric>{`${run.duration_seconds}s`}</Numeric></> : null}
      </p>
      <p className="mc-run-command"><code><Code>{run.command}</Code></code></p>
      {changed.available ? (
        <p className="mc-note">
          {t('training.would_change', locale)}:{' '}
          <Numeric>{`${(changed.gate_deltas || []).length}`}</Numeric>{' '}
          {locale === 'en' ? 'gate differences' : 'הבדלי שערים'}
          {changed.coefficient_deltas ? (
            <>
              {', '}
              <Numeric>{`${changed.coefficient_deltas.cells_moved} / ${changed.coefficient_deltas.cells_compared}`}</Numeric>{' '}
              {t('candidates.cells_moved', locale)}
            </>
          ) : null}
        </p>
      ) : null}
      <RecordDrill record={run} locale={locale} open={open} onToggle={() => setOpen((v) => !v)} />
    </li>
  );
}

export default function TrainingPanel({ payload, locale, onRefresh }) {
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState(null);
  const [lost, setLost] = useState(false);

  // The console's own read is the authority. What the watch below reads only
  // fills the gap between two of them, so a fresh payload from the console
  // clears it rather than competing with it.
  useEffect(() => { setLive(null); }, [payload]);

  const shown = live || payload;
  const watching = runIsOpen(shown);

  // The watch: while a run is open the route is read again, and the moment none
  // is open the reads stop. A read that does not answer stops the watch too and
  // says so on the screen, because a screen that silently retries a dead route
  // is the same lie in a slower form.
  useEffect(() => {
    if (!watching) return undefined;
    let alive = true;
    setLost(false);
    const timer = setInterval(() => {
      readSection('training').then((result) => {
        if (!alive) return;
        if (result.status === 'ok' && result.payload) {
          setLive(result.payload);
          return;
        }
        clearInterval(timer);
        setLost(true);
      });
    }, WATCH_MS);
    return () => { alive = false; clearInterval(timer); };
  }, [watching]);

  const inFlight = shown.in_flight || {};
  const runs = shown.runs || [];
  // Who may press the buttons below, read from the same wall that would
  // refuse the write. ``can_edit`` absent (a payload built without the route,
  // as the watch's synthetic bodies are) locks nothing, since the surface has
  // made no claim either way.
  const locked = shown.can_edit === false;
  async function start(artifact, flags) {
    setBusy(true);
    await startTraining(artifact, flags);
    setBusy(false);
    onRefresh();
  }
  return (
    <>
      <Panel title={t('training.title', locale)} sub={pick(shown, 'safety', locale)}>
        {locked ? (
          <Prose className="mc-note mc-trainer-locked">
            {canEditReason(shown.can_edit_reason, locale)}
          </Prose>
        ) : null}
        {(shown.trainers || []).map((trainer) => (
          <Trainer
            key={trainer.artifact}
            trainer={trainer}
            locale={locale}
            running={busy || inFlight[trainer.artifact]}
            locked={locked}
            onStart={start}
          />
        ))}
      </Panel>
      <Panel title={t('training.log', locale)} sub={`${runs.length}`}>
        {lost ? (
          <p className="mc-note mc-run-watch">
            {t('training.watch_lost', locale)}{' '}
            <Pressable type="button" className="mc-link" onClick={onRefresh}>
              {t('training.watch_again', locale)}
            </Pressable>
          </p>
        ) : watching ? (
          <p className="mc-note mc-run-watch">{t('training.watching', locale)}</p>
        ) : null}
        {runs.length === 0 ? (
          <Absent title={t('training.no_runs', locale)} />
        ) : (
          <ul className="mc-run-list">
            {runs.map((run) => <RunRow run={run} locale={locale} key={run.run_id} />)}
          </ul>
        )}
      </Panel>
    </>
  );
}
