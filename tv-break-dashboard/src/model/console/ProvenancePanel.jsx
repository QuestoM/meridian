import React, { useState } from 'react';
import { Numeric } from '../../shell/format';
import { Absent, Panel, RecordDrill, Stat } from './console-bits';
import { pick, t } from './console-words';

// What produced these artifacts, and the two things nothing on disk records.
//
// The other five sections answer what the model decided. This one answers who
// and how, which is the question a steward asks second and cannot answer today
// from any surface at all: the seeds, the method, the exact input files with
// their digests, and the five gate-override flags whose use is invisible after
// the fact. That last one is a hole in the artifacts, not in this screen, so it
// is stated as a hole with the flag list beside it rather than left blank.
//
// It also names the two commands that overwrite the shipped artifacts, with the
// consequence of each, so the one act this console deliberately refuses to
// perform is legible rather than merely absent.

function Artifact({ artifact, locale }) {
  const [open, setOpen] = useState(false);
  const fingerprints = artifact.source_fingerprints || {};
  const paths = Object.keys(fingerprints);
  return (
    <li className="mc-artifact">
      <div className="mc-artifact-head">
        <strong>{pick(artifact, 'subject', locale)}</strong>
        <code dir="ltr">{artifact.path}</code>
      </div>
      <div className="mc-stat-row">
        <Stat
          label={t('provenance.digest', locale)}
          value={<code dir="ltr">{artifact.short}</code>}
          sub={<span dir="ltr"><Numeric>{`${Number(artifact.bytes || 0).toLocaleString('en-US')} B`}</Numeric></span>}
        />
        <Stat
          label={t('provenance.trained_at', locale)}
          value={<span dir="ltr"><Numeric>{String(artifact.computed_at || '').slice(0, 19)}</Numeric></span>}
        />
      </div>
      {paths.length === 0 ? (
        <p className="mc-note">{t('provenance.no_fingerprints', locale)}</p>
      ) : (
        <>
          <h4>{t('provenance.read_from', locale)}</h4>
          <ul className="mc-fingerprint-list">
            {paths.map((path) => (
              <li key={path} dir="ltr">
                <code>{path}</code>
                <span className="mc-fingerprint">{String(fingerprints[path]).slice(0, 8)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <RecordDrill record={artifact} locale={locale} open={open} onToggle={() => setOpen((v) => !v)} />
    </li>
  );
}

function methodStats(payload, locale) {
  const method = payload.method || {};
  const seeds = payload.seeds || {};
  const rows = [
    ['provenance.pooling', method.pooling_method],
    ['provenance.interval', method.interval_method],
    ['provenance.detrend', method.detrend_baseline_mode],
    ['provenance.window', method.before_after_window_minutes],
    ['provenance.audience_base', method.audience_base_kind],
    ['provenance.interval_seed', seeds.interval_seed],
    ['provenance.bootstrap', seeds.bootstrap_B],
    ['provenance.placebo_seed', seeds.placebo_seed],
  ];
  return rows
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => (
      <Stat
        key={key}
        label={t(key, locale)}
        value={<span dir="ltr"><Numeric>{String(value)}</Numeric></span>}
      />
    ));
}

export default function ProvenancePanel({ payload, locale }) {
  const version = payload.model_version || {};
  const artifacts = Object.values(version.artifacts || {}).filter((a) => a && a.present);
  const flags = payload.gate_override_flags || [];
  return (
    <>
      <Panel title={t('provenance.artifacts', locale)} sub={pick(version, 'basis', locale)}>
        {artifacts.length === 0 ? (
          <Absent title={pick(version, 'reason', locale) || t('provenance.no_artifacts', locale)} />
        ) : (
          <ul className="mc-artifact-list">
            {artifacts.map((artifact) => (
              <Artifact artifact={artifact} locale={locale} key={artifact.kind} />
            ))}
          </ul>
        )}
      </Panel>

      <Panel title={t('provenance.method', locale)}>
        <div className="mc-stat-row mc-stat-wrap">{methodStats(payload, locale)}</div>
      </Panel>

      <Panel title={t('provenance.flags', locale)} sub={pick(payload, 'override_gap', locale)}>
        <ul className="mc-flag-list">
          {flags.map((flag) => (
            <li key={flag.flag}>
              <code dir="ltr">{flag.flag}</code>
              <code dir="ltr" className="mc-flag-env">{flag.env}</code>
              <span className="mc-verdict mc-no_contrast mc-sm">
                {flag.recorded_in_artifact
                  ? t('provenance.flag_recorded', locale)
                  : t('provenance.flag_not_recorded', locale)}
              </span>
            </li>
          ))}
        </ul>
        <p className="mc-note">
          <strong>{t('provenance.actor', locale)}</strong>{' '}
          {payload.actor_recorded ? '' : pick(payload, 'actor_gap', locale)}
        </p>
      </Panel>

      <Panel title={t('provenance.commands', locale)} sub={t('provenance.commands_sub', locale)}>
        <ul className="mc-command-list">
          {(payload.training_commands || []).map((row) => (
            <li key={row.artifact}>
              <code dir="ltr">{row.command}</code>
              <span className="mc-command-writes" dir="ltr">
                {t('training.writes', locale)} <code>{row.writes}</code>
              </span>
              <p className="mc-note">{pick(row, 'consequence', locale)}</p>
            </li>
          ))}
        </ul>
      </Panel>
    </>
  );
}
