import React, { useEffect, useState } from 'react';
import { Code, Name } from '../../shell/bidi';
import { Numeric } from '../../shell/format';
import { recordDecision } from './console-api';
import { Absent, Money, Panel, RecordDrill } from './console-bits';
import { pick, t } from './console-words';

// The decision surface. A verdict is a record with four required parts: the
// version it is about, whether it ships, why, and, when it ships, the release
// note the operator side will read. The note is the only training-authored text
// that crosses the line, so the server refuses one carrying a gate verdict, a
// p-value or a coefficient and the refusal is shown here verbatim.
//
// Recording a ship verdict never copies an artifact. The form says so before
// the click, and the stored record repeats it afterwards.

export function DecisionForm({ subject, candidateId, locale, onDone, onCancel }) {
  const [decision, setDecision] = useState('not_shipped');
  const [reason, setReason] = useState('');
  const [noteHe, setNoteHe] = useState('');
  const [noteEn, setNoteEn] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError('');
    const result = await recordDecision({
      decision,
      subject,
      candidate_id: candidateId || null,
      reason,
      release_note_he: noteHe,
      release_note_en: noteEn,
      money_direction: 'unknown',
    });
    setBusy(false);
    if (result.status === 'ok') {
      onDone(result.payload);
      return;
    }
    setError(result.detail || t('state.unreachable', locale));
  }

  return (
    <form className="mc-decision-form" onSubmit={submit}>
      <div className="mc-decision-subject">
        {subject === 'candidate'
          ? <>{t('versions.subject_candidate', locale)} <code><Code>{candidateId}</Code></code></>
          : t('versions.subject_current', locale)}
      </div>
      <div className="mc-decision-choice">
        <label>
          <input
            type="radio"
            name="mc-decision"
            checked={decision === 'shipped'}
            onChange={() => setDecision('shipped')}
          />
          {t('versions.ship', locale)}
        </label>
        <label>
          <input
            type="radio"
            name="mc-decision"
            checked={decision === 'not_shipped'}
            onChange={() => setDecision('not_shipped')}
          />
          {t('versions.no_ship', locale)}
        </label>
      </div>
      <label className="mc-field">
        <span>{t('versions.reason', locale)}</span>
        <textarea value={reason} onChange={(event) => setReason(event.target.value)} rows={2} required />
      </label>
      {decision === 'shipped' ? (
        <>
          <label className="mc-field">
            <span>{t('versions.note', locale)}</span>
            <small>{t('versions.note_rule', locale)}</small>
            <textarea value={noteHe} onChange={(event) => setNoteHe(event.target.value)} rows={2} required />
          </label>
          <label className="mc-field">
            <Name>English</Name>
            {/*
              dir on a form control is not alignment. It sets the order the
              characters are INSERTED in as the operator types, which is the one
              thing the isolating primitives cannot do: they render a span and
              there is no element to wrap around a field's own value. The note in
              this field is English by definition, so the insertion order is too.
            */}
            <textarea value={noteEn} onChange={(event) => setNoteEn(event.target.value)} rows={2} dir="ltr" />
          </label>
        </>
      ) : null}
      {error ? <p className="mc-error">{error}</p> : null}
      <div className="mc-decision-actions">
        <button type="submit" className="mc-button mc-primary" disabled={busy}>
          {t('versions.record', locale)}
        </button>
        <button type="button" className="mc-button" onClick={onCancel}>
          {t('versions.cancel', locale)}
        </button>
      </div>
    </form>
  );
}

function DecisionRow({ record, locale }) {
  const [open, setOpen] = useState(false);
  const shipped = record.decision === 'shipped';
  const adoption = record.adoption || {};
  return (
    <li className={`mc-decision mc-${record.decision}`}>
      <div className="mc-decision-head">
        <span className={`mc-verdict ${shipped ? 'mc-active' : 'mc-tested_and_lost'} mc-md`}>
          {shipped ? t('versions.shipped', locale) : t('versions.not_shipped', locale)}
        </span>
        <span className="mc-decision-subject-line">
          {record.subject === 'candidate'
            ? <>{t('versions.subject_candidate', locale)} <code><Code>{record.candidate_id}</Code></code></>
            : t('versions.subject_current', locale)}
        </span>
        <span className="mc-decision-meta">
          <Numeric>{`${String(record.recorded_at || '').slice(0, 19)}`}</Numeric>
        </span>
      </div>
      <p className="mc-decision-reason">{record.reason}</p>
      {record.release_note_he ? (
        <p className="mc-release-note">{record.release_note_he}</p>
      ) : null}
      <p className="mc-decision-meta">
        {t('versions.by', locale)} <Name>{record.actor}</Name>
        {' - '}
        <Numeric>{record.model_version_name}</Numeric>
      </p>
      {adoption.state === 'escalated' ? (
        <p className="mc-escalated">
          <strong>{t('versions.escalated', locale)}</strong>
          <Money value={adoption.measured_revenue_delta} locale={locale} />
          <span>{pick(adoption, 'reason', locale)}</span>
        </p>
      ) : null}
      <RecordDrill record={record.evidence} locale={locale} open={open} onToggle={() => setOpen((v) => !v)} />
    </li>
  );
}

export default function VersionsPanel({ payload, locale, onRefresh, openForm, onCloseForm }) {
  const [formOpen, setFormOpen] = useState(false);
  useEffect(() => {
    if (openForm) setFormOpen(true);
  }, [openForm]);
  const decisions = payload.decisions || [];
  return (
    <>
      <Panel
        title={t('versions.title', locale)}
        sub={<Code>{payload.store_dir}</Code>}
        right={(
          <button type="button" className="mc-button mc-primary" onClick={() => setFormOpen(true)}>
            {t('candidates.decide', locale)}
          </button>
        )}
      >
        {formOpen ? (
          <DecisionForm
            subject={openForm ? 'candidate' : 'current'}
            candidateId={openForm || ''}
            locale={locale}
            onDone={() => {
              setFormOpen(false);
              if (onCloseForm) onCloseForm();
              onRefresh();
            }}
            onCancel={() => {
              setFormOpen(false);
              if (onCloseForm) onCloseForm();
            }}
          />
        ) : null}
        {decisions.length === 0 ? (
          <Absent title={t('versions.none', locale)} />
        ) : (
          <ul className="mc-decision-list">
            {decisions.map((record) => (
              <DecisionRow record={record} locale={locale} key={record.decision_id} />
            ))}
          </ul>
        )}
      </Panel>
      <Panel title={t('versions.recorded', locale)} sub={`${(payload.observed || []).length}`}>
        {(payload.observed || []).length === 0 ? (
          <Absent
            title={locale === 'en'
              ? 'No model version has been recorded yet.'
              : 'טרם נרשמה גרסת מודל.'}
          />
        ) : (
          <ul className="mc-version-list">
            {payload.observed.map((version) => (
              <li key={version.id}>
                <Code className="mc-version-id">{version.id}</Code>
                <span className="mc-version-seen">
                  <Numeric>{String(version.first_seen_at || '').slice(0, 19)}</Numeric>
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </>
  );
}
