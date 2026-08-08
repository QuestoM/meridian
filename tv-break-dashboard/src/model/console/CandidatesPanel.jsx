import React, { useEffect, useState } from 'react';
import { Code, Name } from '../../shell/bidi';
import { Numeric } from '../../shell/format';
import CandidateVerdict, { useCandidateDecision } from './CandidateVerdict';
import { measureCandidate, readSection } from './console-api';
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
//
// **The shelf follows the measurements it is showing.** Measured on the shipped
// tree: the shelf was opened with two measurements in flight, the page read the
// route once at open and never again, and 60 s after the store had recorded a
// measured figure the screen still carried two blocks reading "the plan is being
// computed twice". Leaving the section and returning read the truth, which
// proved the store was right and only the screen was wrong. A measurement ends
// on the server with no press behind it, so the only honest way for this screen
// to learn about it is to ask again while one is open, and to stop asking the
// moment none is. This is the money story's own screen, so a false state here
// is a false state about shekels.

// How long the shelf waits between two reads while a measurement is open. The
// route answers warm in about 6 ms, measured on this tree, so this costs the
// server nothing and bounds how stale the screen can be at a measurement's end.
const WATCH_MS = 1500;

// Whether a measurement is open, read from the payload the route serves and
// never from anything this panel remembers about its own press. The signal is
// the server's own flight register, which is what the route prints as a card's
// money state, so a measurement started from another window is followed exactly
// like one started here, and a measurement whose thread died with the server is
// not followed for ever.
function measurementIsOpen(payload) {
  return (payload.candidates || []).some((row) => (row.money || {}).state === 'measuring');
}

// Which artifact a value came from, in words, on the value itself. It replaces
// a connector between two bare values: a reader who lands on the second value
// still knows whose figure it is, and it says the same thing in both languages.
// The label carries its own isolation rather than inheriting it from a rule on
// the row. Both the label and the value beside it can be Hebrew inside a row
// whose direction is Latin, and two unisolated Hebrew runs merge and render as
// one unbroken word: measured on this screen, the label read as part of the
// value it labels.
function Side({ side, locale }) {
  return <Name className="mc-side-label">{t(`candidates.side_${side}`, locale)}</Name>;
}

// A gate value the artifact does not carry is an absence, not a value. Printing
// the raw null said "this candidate turned the gate off", which is a different
// and wrong piece of news: the candidate simply predates the key.
function GateValue({ value, absent, locale }) {
  if (absent) {
    return <Name className="mc-delta-absent">{t('candidates.key_absent', locale)}</Name>;
  }
  if (value === null || value === undefined) {
    return <Name className="mc-delta-absent">{t('candidates.value_null', locale)}</Name>;
  }
  return <Code>{String(value)}</Code>;
}

// A figure out of a gate's own record. Three states, never conflated: a number,
// a slot the artifact recorded with no number in it, and a key the artifact
// does not carry at all.
function RecordFigure({ value, absent, digits, locale }) {
  if (absent) {
    return <Name className="mc-delta-absent">{t('candidates.key_absent', locale)}</Name>;
  }
  if (value === null || value === undefined) {
    return <Name className="mc-delta-absent">{t('candidates.value_null', locale)}</Name>;
  }
  return <Figure value={value} unit="ratio" digits={digits} />;
}

function FigurePair({ figure, locale }) {
  return (
    <li>
      <code><Code>{figure.key}</Code></code>
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
            <Name>{row.reason_shipped || t('candidates.no_sentence', locale)}</Name>
          </p>
          <p>
            <Side side="candidate" locale={locale} />
            <Name>{row.reason_candidate || t('candidates.no_sentence', locale)}</Name>
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
          <strong><Code>{candidate.id}</Code></strong>
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
                <li key={delta.key}>
                  <code><Code>{delta.key}</Code></code>
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
                <code><Code>{deltas.max_abs_delta_cell}</Code></code>
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
  const [live, setLive] = useState(null);
  const [lost, setLost] = useState(false);

  // The console's own read is the authority. What the watch below reads only
  // fills the gap between two of them, so a fresh payload from the console
  // clears it rather than competing with it.
  useEffect(() => { setLive(null); }, [payload]);

  const shown = live || payload;
  const watching = measurementIsOpen(shown);

  // The watch: while a measurement is open the route is read again, and the
  // moment none is open the reads stop. A read that does not answer stops the
  // watch too and says so on the screen, because a screen that silently retries
  // a dead route is the same lie in a slower form.
  useEffect(() => {
    if (!watching) return undefined;
    let alive = true;
    setLost(false);
    const timer = setInterval(() => {
      readSection('candidates').then((result) => {
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

  const candidates = shown.candidates || [];
  async function measure(id) {
    setBusy(true);
    await measureCandidate(id);
    setBusy(false);
    onRefresh();
  }
  return (
    <Panel
      title={t('candidates.title', locale)}
      sub={<Code>{shown.directory}</Code>}
    >
      {lost ? (
        <p className="mc-note mc-candidate-watch">
          {t('candidates.watch_lost', locale)}{' '}
          <button type="button" className="mc-link" onClick={onRefresh}>
            {t('candidates.watch_again', locale)}
          </button>
        </p>
      ) : watching ? (
        <p className="mc-note mc-candidate-watch">{t('candidates.watching', locale)}</p>
      ) : null}
      {candidates.length === 0 ? (
        <Absent
          title={locale === 'en' ? 'No candidate artifacts on the shelf.' : 'אין קבצי מועמדים על המדף.'}
          reason={pick(shown, 'measurement_cost', locale)}
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
