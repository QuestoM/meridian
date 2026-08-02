import React, { useState } from 'react';
import { Numeric } from '../../shell/format';
import { startTraining } from './console-api';
import { Absent, Panel, RecordDrill } from './console-bits';
import { pick, t } from './console-words';

// Training, started from the console and only ever into the console's own
// store. The safety sentence is on the screen and not only in the code: a run
// writes into the releases store and never over the shipped artifact, so
// nothing an operator reads moves until somebody adopts a run deliberately.
//
// The five gate overrides are the provenance hole the console itself reports.
// A run started here records exactly which were used, so for these runs a
// forced gate and a self-activated one are told apart afterwards.

const CHOICES = [
  { value: '', key: 'training.auto' },
  { value: 'force-on', key: 'training.force_on' },
  { value: 'force-off', key: 'training.force_off' },
];

function Trainer({ trainer, locale, running, onStart }) {
  const [flags, setFlags] = useState({});
  const busy = Boolean(running);
  return (
    <div className="mc-trainer">
      <div className="mc-trainer-head">
        <div>
          <strong>{locale === 'en' ? trainer.label_en : trainer.label_he}</strong>
          <small dir="ltr">{trainer.script}</small>
        </div>
        <button
          type="button"
          className="mc-button mc-primary"
          onClick={() => onStart(trainer.artifact, flags)}
          disabled={busy}
        >
          {busy ? t('training.running', locale) : t('training.start', locale)}
        </button>
      </div>
      <p className="mc-note">
        {t('training.writes', locale)} <code dir="ltr">{trainer.writes}</code>
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
              <code dir="ltr">{flag.flag}</code>
              <select
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
              </select>
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
        <span dir="ltr" className="mc-run-id">{run.run_id}</span>
        <span className="mc-run-artifact">{locale === 'en' ? run.label_en : run.label_he}</span>
        <span className={`mc-verdict mc-${run.state === 'done' ? 'active' : run.state === 'failed' ? 'tested_and_lost' : 'no_contrast'} mc-sm`}>
          {run.state === 'done' ? t('training.done', locale) : run.state === 'failed' ? t('training.failed', locale) : t('training.running', locale)}
        </span>
      </div>
      <p className="mc-run-meta" dir="ltr">
        <Numeric>
          {`${String(run.started_at || '').slice(0, 19)}  ${run.actor}`}
          {run.duration_seconds ? `  ${run.duration_seconds}s` : ''}
        </Numeric>
      </p>
      <p className="mc-run-command" dir="ltr"><code>{run.command}</code></p>
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
  const inFlight = payload.in_flight || {};
  async function start(artifact, flags) {
    setBusy(true);
    await startTraining(artifact, flags);
    setBusy(false);
    onRefresh();
  }
  return (
    <>
      <Panel title={t('training.title', locale)} sub={pick(payload, 'safety', locale)}>
        {(payload.trainers || []).map((trainer) => (
          <Trainer
            key={trainer.artifact}
            trainer={trainer}
            locale={locale}
            running={busy || inFlight[trainer.artifact]}
            onStart={start}
          />
        ))}
      </Panel>
      <Panel title={t('training.log', locale)} sub={`${(payload.runs || []).length}`}>
        {(payload.runs || []).length === 0 ? (
          <Absent title={t('training.no_runs', locale)} />
        ) : (
          <ul className="mc-run-list">
            {payload.runs.map((run) => <RunRow run={run} locale={locale} key={run.run_id} />)}
          </ul>
        )}
      </Panel>
    </>
  );
}
