import React, { useEffect, useState } from 'react';
import { Numeric } from '../../shell/format';
import { readCandidate } from './console-api';
import { Figure, Money, RecordDrill } from './console-bits';
import { pick, t } from './console-words';

// The verdict already recorded about one candidate, on that candidate's own
// card. Without it a recorded verdict left no trace on the thing it was about:
// the reader recorded a decision, came back to the shelf, and every card read
// exactly as it had before, which is JS-19's "a later reader can see what was
// tried" met on one screen and lost on the one where the question is asked.
//
// It reads `/api/model/candidates/{id}`, which is the only route that answers
// "what was decided about this one". The list route cannot: a verdict is keyed
// by the candidate it is about.
//
// Three states, never conflated. A verdict exists and is shown with the figure
// it was recorded on. No verdict exists against the model version in force, and
// the sentence says which version that is, because a verdict recorded against
// an earlier version is not a verdict about this one. Or the read did not
// answer, and nothing is shown rather than an absence that would read as "no
// verdict was ever taken".

export function useCandidateDecision(candidateId, refreshKey) {
  const [state, setState] = useState({ status: 'loading', payload: null, detail: '' });
  useEffect(() => {
    let active = true;
    readCandidate(candidateId).then((result) => {
      if (active) setState(result);
    });
    return () => { active = false; };
  }, [candidateId, refreshKey]);
  return state;
}

// The money the verdict was taken on, in the state it was in at the time. A
// verdict recorded before anyone measured says so; a verdict recorded on a
// measurement that has since been superseded keeps its figure and says that.
function RecordedOn({ evidence, locale }) {
  const state = evidence.money_state;
  if (state !== 'measured' && state !== 'stale') {
    return <p className="mc-note">{t('candidates.verdict_not_measured', locale)}</p>;
  }
  const scope = evidence.scope || {};
  return (
    <div className={`mc-verdict-money ${state === 'stale' ? 'mc-stale' : ''}`}>
      <span className="mc-verdict-money-label">{t('candidates.verdict_on', locale)}</span>
      <Money value={evidence.revenue_delta} locale={locale} />
      <Figure value={evidence.revenue_delta_pct} unit="percent" />
      <small>
        <Numeric>{`${scope.rows ?? '?'}`}</Numeric>{' '}
        {t('candidates.rows', locale)}{', '}
        {t('candidates.scope_owned', locale)}{', '}
        {t('candidates.measured_at', locale)}{' '}
        <Numeric>{String(evidence.measured_at || '').slice(0, 19)}</Numeric>
      </small>
      {state === 'stale' ? <small>{t('candidates.verdict_stale', locale)}</small> : null}
    </div>
  );
}

function Recorded({ record, locale }) {
  const [open, setOpen] = useState(false);
  const shipped = record.decision === 'shipped';
  const adoption = record.adoption || {};
  return (
    <div className={`mc-verdict-block mc-${record.decision}`}>
      <div className="mc-verdict-head">
        <span className={`mc-verdict ${shipped ? 'mc-active' : 'mc-tested_and_lost'} mc-md`}>
          {shipped ? t('versions.shipped', locale) : t('versions.not_shipped', locale)}
        </span>
        <span className="mc-verdict-actor">
          {t('versions.by', locale)} <span dir="ltr">{record.actor}</span>
        </span>
        <span className="mc-verdict-when" dir="ltr">
          <Numeric>{String(record.recorded_at || '').slice(0, 19)}</Numeric>
        </span>
        <span className="mc-verdict-version" dir="ltr">
          <Numeric>{record.model_version_name || record.model_version_id || ''}</Numeric>
        </span>
      </div>
      <p className="mc-decision-reason">{record.reason}</p>
      {record.release_note_he ? <p className="mc-release-note">{record.release_note_he}</p> : null}
      <RecordedOn evidence={record.evidence || {}} locale={locale} />
      {adoption.state === 'escalated' || adoption.state === 'recorded' ? (
        <p className="mc-escalated">
          <strong>
            {adoption.state === 'escalated'
              ? t('versions.escalated', locale)
              : t('candidates.adoption_recorded', locale)}
          </strong>
          <span>{pick(adoption, 'reason', locale)}</span>
        </p>
      ) : null}
      <RecordDrill
        record={record}
        locale={locale}
        open={open}
        onToggle={() => setOpen((v) => !v)}
        label={t('candidates.verdict_record', locale)}
      />
    </div>
  );
}

export default function CandidateVerdict({ state, locale }) {
  if (state.status === 'loading') {
    return <p className="mc-note">{t('candidates.verdict_reading', locale)}</p>;
  }
  if (state.status !== 'ok') {
    return (
      <p className="mc-note">
        {t('candidates.verdict_unreadable', locale)}
        {state.detail ? <> <span dir="ltr">{state.detail}</span></> : null}
      </p>
    );
  }
  const payload = state.payload || {};
  if (!payload.decision) {
    return (
      <p className="mc-note">
        {t('candidates.no_verdict', locale)}{' '}
        <span dir="ltr"><Numeric>{(payload.model_version || {}).name || ''}</Numeric></span>
      </p>
    );
  }
  return <Recorded record={payload.decision} locale={locale} />;
}
