import React, { useState } from 'react';
import { Numeric } from '../../shell/format';
import CandidateVerdict, { useCandidateDecision } from './CandidateVerdict';
import { measureCandidate } from './console-api';
import { Absent, Figure, Money, Panel, RecordDrill } from './console-bits';
import { pick, t } from './console-words';

// What a new training would change, per candidate artifact: which gates decide
// differently, the figures each of those verdicts was decided on, how far the
// coefficients move, and the money adopting it would move in shekels with the
// scope printed beside it. A candidate that has never been measured says so and
// offers the measurement as a control, because an empty field is an action and
// a guessed figure is a lie.
//
// The evidence block exists because a candidate can agree on every gate flag
// and move no cell while the held-out figures under those flags have moved a
// long way. Without it the screen reads "nothing decides differently", which is
// a different and wrong piece of news.

// Which artifact a value came from, in words, on the value itself. It replaces
// a connector between two bare values: a reader who lands on the second value
// still knows whose figure it is, and it says the same thing in both languages.
function Side({ side, locale }) {
  return <small className="mc-side-label">{t(`candidates.side_${side}`, locale)}</small>;
}

// A gate value the artifact does not carry is an absence, not a value. Printing
// the raw null said "this candidate turned the gate off", which is a different
// and wrong piece of news: the candidate simply predates the key.
function GateValue({ value, absent, locale }) {
  if (absent) {
    return <span className="mc-delta-absent">{t('candidates.key_absent', locale)}</span>;
  }
  if (value === null || value === undefined) {
    return <span className="mc-delta-absent">{t('candidates.value_null', locale)}</span>;
  }
  return <span>{String(value)}</span>;
}

// A figure out of a gate's own record. Three states, never conflated: a number,
// a slot the artifact recorded with no number in it, and a key the artifact
// does not carry at all.
function RecordFigure({ value, absent, digits, locale }) {
  if (absent) {
    return <span className="mc-delta-absent">{t('candidates.key_absent', locale)}</span>;
  }
  if (value === null || value === undefined) {
    return <span className="mc-delta-absent">{t('candidates.value_null', locale)}</span>;
  }
  return <Figure value={value} unit="ratio" digits={digits} />;
}

function FigurePair({ figure, locale }) {
  return (
    <li dir="ltr">
      <code>{figure.key}</code>
      <span className="mc-delta-before">
        <Side side="shipped" locale={locale} />
        <RecordFigure value={figure.shipped} absent={figure.shipped_absent} digits={figure.digits} locale={locale} />
      </span>
      <span className="mc-delta-after">
        <Side side="candidate" locale={locale} />
        <RecordFigure value={figure.candidate} absent={figure.candidate_absent} digits={figure.digits} locale={locale} />
      </span>
    </li>
  );
}

// One gate's evidence, both sides. Where a side records nothing at all that is
// said in one sentence rather than repeated down a column of absences.
function HeldOutGate({ row, locale }) {
  const missing = row.candidate_records_nothing || row.shipped_records_nothing;
  return (
    <li className="mc-holdout">
      <div className="mc-holdout-head">
        <span className="mc-holdout-gate">{locale === 'en' ? row.label_en : row.label_he}</span>
        <span className="mc-holdout-statistic">{pick(row, 'statistic', locale)}</span>
      </div>
      {row.candidate_records_nothing ? (
        <p className="mc-note">{t('candidates.candidate_records_nothing', locale)}</p>
      ) : null}
      {row.shipped_records_nothing ? (
        <p className="mc-note">{t('candidates.shipped_records_nothing', locale)}</p>
      ) : null}
      {missing ? null : (
        <ul className="mc-delta-list">
          {(row.figures || []).filter((figure) => figure.moved).map((figure) => (
            <FigurePair figure={figure} locale={locale} key={figure.key} />
          ))}
        </ul>
      )}
      {row.reason_moved && !missing ? (
        <div className="mc-holdout-reasons">
          <p>
            <Side side="shipped" locale={locale} />
            <span dir="ltr">{row.reason_shipped || t('candidates.no_sentence', locale)}</span>
          </p>
          <p>
            <Side side="candidate" locale={locale} />
            <span dir="ltr">{row.reason_candidate || t('candidates.no_sentence', locale)}</span>
          </p>
        </div>
      ) : null}
    </li>
  );
}

function HeldOutBlock({ rows, locale }) {
  return (
    <div className="mc-holdout-block">
      <h4>{t('candidates.held_out', locale)}</h4>
      {(rows || []).length === 0 ? (
        <p className="mc-note">{t('candidates.no_held_out_moves', locale)}</p>
      ) : (
        <ul className="mc-holdout-list">
          {rows.map((row) => <HeldOutGate row={row} locale={locale} key={row.gate_id} />)}
        </ul>
      )}
    </div>
  );
}

function MoneyBlock({ money, locale, onMeasure, busy }) {
  if (!money || money.state === 'not_measured') {
    return (
      <Absent
        title={t('candidates.not_measured', locale)}
        reason={money ? pick(money, 'reason', locale) : ''}
        action={(
          <button type="button" className="mc-button" onClick={onMeasure} disabled={busy}>
            {t('candidates.measure', locale)}
          </button>
        )}
      />
    );
  }
  if (money.state === 'measuring') {
    const past = money.past_durations_seconds || [];
    return (
      <div className="mc-measuring">
        <span className="mc-spinner" aria-hidden="true" />
        <span>{t('candidates.measuring', locale)}</span>
        {past.length ? (
          <small>
            {t('candidates.past_runs', locale)}{' '}
            <Numeric>{`${past[Math.floor(past.length / 2)]}`}</Numeric> {t('candidates.seconds', locale)}
          </small>
        ) : null}
      </div>
    );
  }
  if (money.state === 'stale') {
    // The superseded figure is still shown, dimmed and labelled with what moved
    // and when it was measured. It was computed from real data against inputs
    // that have since changed, which is a different thing from a number nobody
    // has, and hiding it would lose the only measurement that exists.
    const owned = money.operator_channel_delta || {};
    return (
      <div className="mc-money-block mc-stale">
        <Absent
          title={t('candidates.stale', locale)}
          reason={pick(money, 'reason', locale)}
          action={(
            <button type="button" className="mc-button" onClick={onMeasure} disabled={busy}>
              {t('candidates.remeasure', locale)}
            </button>
          )}
        />
        {owned.revenue_delta === undefined ? null : (
          <div className="mc-money-row">
            <span className="mc-money-scope">
              {t('candidates.scope_owned', locale)}
              <small>
                <Numeric>{String(money.measured_at || '').slice(0, 19)}</Numeric>
              </small>
            </span>
            <Money value={owned.revenue_delta} locale={locale} />
            <Figure value={owned.revenue_delta_pct} unit="percent" />
          </div>
        )}
      </div>
    );
  }
  const owned = money.operator_channel_delta || {};
  const whole = money.whole_plan_delta || {};
  const scope = money.scope || {};
  return (
    <div className="mc-money-block">
      <div className="mc-money-row">
        <span className="mc-money-scope">
          {t('candidates.scope_owned', locale)}
          <small>
            <Numeric>{`${(scope.operator_channel || {}).rows || 0}`}</Numeric>{' '}
            {t('candidates.rows', locale)}
          </small>
        </span>
        <Money value={owned.revenue_delta} locale={locale} />
        <Figure value={owned.revenue_delta_pct} unit="percent" />
      </div>
      <div className="mc-money-row">
        <span className="mc-money-scope">
          {t('candidates.scope_plan', locale)}
          <small>
            <Numeric>{`${(scope.whole_plan || {}).channels || 0}`}</Numeric>{' '}
            {t('candidates.channels', locale)}{', '}
            <Numeric>{`${(scope.whole_plan || {}).rows || 0}`}</Numeric>{' '}
            {t('candidates.rows', locale)}
          </small>
        </span>
        <Money value={whole.revenue_delta} locale={locale} />
        <Figure value={whole.revenue_delta_pct} unit="percent" />
      </div>
      <small className="mc-money-basis">
        {t('candidates.measured_at', locale)}{' '}
        <Numeric>{String(money.measured_at || '').slice(0, 19)}</Numeric>{', '}
        <Numeric>{`${money.duration_seconds || '?'}`}</Numeric>{' '}
        {t('candidates.seconds', locale)}{', '}
        {t('candidates.engine', locale)}{' '}
        <Numeric>{money.engine_version || '?'}</Numeric>
      </small>
    </div>
  );
}

function CandidateCard({ candidate, index, total, locale, onMeasure, onDecide, busy, refreshKey }) {
  const [open, setOpen] = useState(false);
  const deltas = candidate.coefficient_deltas || {};
  // The verdict recorded about this candidate, read per card. The button says
  // which act it is: a first verdict, or another one on top of the one on
  // screen. An unqualified "record a verdict" beside an existing verdict reads
  // as though nothing had been decided.
  const verdict = useCandidateDecision(candidate.id, refreshKey);
  const decided = verdict.status === 'ok' && Boolean((verdict.payload || {}).decision);
  return (
    <li className="mc-candidate">
      <div className="mc-candidate-head">
        <div>
          <strong dir="ltr">{candidate.id}</strong>
          <small className="mc-candidate-position">
            <Numeric>{`${index + 1} / ${total}`}</Numeric>
          </small>
        </div>
        <button type="button" className="mc-button" onClick={() => onDecide(candidate)}>
          {decided ? t('candidates.decide_again', locale) : t('candidates.decide', locale)}
        </button>
      </div>
      <p className="mc-candidate-subject">{pick(candidate, 'subject', locale) || candidate.purpose || ''}</p>
      <div className="mc-candidate-verdict">
        <h4>{t('candidates.verdict', locale)}</h4>
        <CandidateVerdict state={verdict} locale={locale} />
      </div>
      <div className="mc-candidate-grid">
        <div>
          <h4>{t('candidates.gate_deltas', locale)}</h4>
          {(candidate.gate_deltas || []).length === 0 ? (
            <p className="mc-note">{t('candidates.no_gate_deltas', locale)}</p>
          ) : (
            <ul className="mc-delta-list">
              {candidate.gate_deltas.map((delta) => (
                <li key={delta.key} dir="ltr">
                  <code>{delta.key}</code>
                  <span className="mc-delta-before">
                    <Side side="shipped" locale={locale} />
                    <GateValue value={delta.shipped} absent={delta.shipped_absent} locale={locale} />
                  </span>
                  <span className="mc-delta-after">
                    <Side side="candidate" locale={locale} />
                    <GateValue value={delta.candidate} absent={delta.candidate_absent} locale={locale} />
                  </span>
                </li>
              ))}
            </ul>
          )}
          <HeldOutBlock rows={candidate.held_out_deltas} locale={locale} />
          <p className="mc-note">
            <Numeric>{`${deltas.cells_moved ?? 0} / ${deltas.cells_compared ?? 0}`}</Numeric>{' '}
            {t('candidates.cells_moved', locale)}
            {deltas.max_abs_delta ? (
              <>
                {', '}
                {t('candidates.largest_move', locale)}{' '}
                <Figure value={deltas.max_abs_delta} unit="ratio" digits={6} />{' '}
                <code dir="ltr">{deltas.max_abs_delta_cell}</code>
              </>
            ) : null}
          </p>
        </div>
        <div>
          <h4>{t('candidates.money', locale)}</h4>
          <MoneyBlock money={candidate.money} locale={locale} onMeasure={() => onMeasure(candidate.id)} busy={busy} />
        </div>
      </div>
      <RecordDrill
        record={{
          file: candidate.file,
          sha256: candidate.sha256,
          bytes: candidate.bytes,
          computed_at: candidate.computed_at,
          differences: candidate.differences,
          held_out_deltas: candidate.held_out_deltas,
        }}
        locale={locale}
        open={open}
        onToggle={() => setOpen((v) => !v)}
      />
    </li>
  );
}

export default function CandidatesPanel({ payload, locale, onRefresh, onDecide, refreshKey }) {
  const [busy, setBusy] = useState(false);
  const candidates = payload.candidates || [];
  async function measure(id) {
    setBusy(true);
    await measureCandidate(id);
    setBusy(false);
    onRefresh();
  }
  return (
    <Panel
      title={t('candidates.title', locale)}
      sub={<span dir="ltr">{payload.directory}</span>}
    >
      {candidates.length === 0 ? (
        <Absent
          title={locale === 'en' ? 'No candidate artifacts on the shelf.' : 'אין קבצי מועמדים על המדף.'}
          reason={pick(payload, 'measurement_cost', locale)}
        />
      ) : (
        <ul className="mc-candidate-list">
          {candidates.map((candidate, index) => (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              index={index}
              total={candidates.length}
              locale={locale}
              onMeasure={measure}
              onDecide={onDecide}
              busy={busy}
              refreshKey={refreshKey}
            />
          ))}
        </ul>
      )}
    </Panel>
  );
}
