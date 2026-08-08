import React from 'react';
import { Numeric } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { Figure as BidiFigure, Code, Name } from '../../shell/bidi';
import History from './board-history.jsx';
import Meter from './board-meter.jsx';
import { Provenance, Purpose } from './board-origin.jsx';
import { pick, t } from './board-words';

// What one artifact was decided on, under the table that ranks them.
//
// Split out of CandidateBoard.jsx, which is where the ranking lives. The
// division is the one the question has: the table answers "which of these", and
// this answers "on what", and the second one is four blocks deep while the first
// is one row per artifact.
//
// Every figure here is measured and carries its denominator. The cell table is
// the ranked head rather than all thirty-six rows, and it says so, because a
// screen that silently shows eight of thirty-six is a screen that has hidden
// twenty-eight.

// What the artifact's own producer recorded about adopting it.
//
// Rendered as its own block and never as a column, because it is not a rank. A
// self-test is the artifact's own split under its own fit, and putting two of
// them in one column is exactly the comparison this whole board exists to
// replace. The sentence saying so travels with it every time it is shown.
function SelfReported({ self, locale }) {
  const state = (self || {}).state;
  if (!state || state === 'absent') return null;
  const tone = state === 'advised_against' ? 'cb-amber' : state === 'recommended' ? 'cb-teal' : 'cb-neutral';
  return (
    <section className="cb-self">
      <h4>{t('self.title', locale)}</h4>
      <p className="cb-self-head">
        <span className={`cb-tag ${tone}`}>{t(`self.${state}`, locale)}</span>
        <span>{pick(self, 'reading', locale)}</span>
      </p>
      {self.reason ? (
        <p className="cb-self-words">
          <span className="cb-label">{t('self.words', locale)}</span>
          <Code>{self.reason}</Code>
        </p>
      ) : null}
      {self.n_test ? (
        <p className="cb-self-n">
          <span className="cb-label">{t('self.n_test', locale)}</span>
          <BidiFigure><Numeric>{Number(self.n_test).toLocaleString('en-US')}</Numeric></BidiFigure>
          <span>{t('self.breaks_own', locale)}</span>
        </p>
      ) : null}
      <p className="cb-self-basis">{t('self.basis', locale)}</p>
    </section>
  );
}

// How much of the evaluation was in this artifact's own fit, beside the count it
// sits next to. A count of breaks fitted on means nothing without the count it
// is scored on.
// The separation between the count and the words is a gap on the box, never a
// trailing space inside the figure. A space at the end of an isolated run has
// nowhere to land in a right-to-left line, and this chip read as one token in
// Hebrew while reading correctly in English.
function FitBasis({ basis, locale }) {
  if (!basis || basis.state !== 'fewer') return null;
  return (
    <span className="cb-detail-shortfall cb-tag cb-amber">
      <BidiFigure><Numeric>{String(basis.not_fitted_on)}</Numeric></BidiFigure>
      <span>{t('basis.never_fitted', locale)}</span>
    </span>
  );
}

// One side of a gate row. A stored value, so it renders as a value and never as
// a translated word: true and bootstrap are what the artifact holds, not prose.
// An absence is a state and gets the tag, with the console's own sentence for it
// under the table rather than repeated on every row.
//
// The text is rendered on the measuring side and carried here. A stored 1.0 is a
// float the terminal prints as 1.0 and a browser prints as 1, because JavaScript
// has one number type, and two surfaces of one piece printing one stored value
// two ways is a divergence a steward walks into inside a session.
function GateValue({ text, absent, locale }) {
  if (absent) return <span className="cb-tag cb-neutral">{t('gates.absent_short', locale)}</span>;
  return <Code>{String(text === null || text === undefined ? '' : text)}</Code>;
}

// What its gates decided, which is the second sentence of the steward's job and
// was on no screen at all.
//
// The sentence comes first and the rows come second, because the row COUNT is
// the thing that misreads. The console's comparison returns every key on which
// the two artifacts do not hold the same value, and a key the candidate does not
// carry comes back as one of those. Measured on this tree, three of the five
// candidates return ten such rows and every one is an absence: reading the count
// gives ten gates decided the other way, when the truth is none.
//
// The held-out amounts sit under it because they are the argument for this whole
// board: the shipped artifact decided its series gate on 2,532 breaks and one
// candidate decided the same gate on 506, and two figures taken on different
// amounts are not comparable.
function Gates({ gates, locale }) {
  if (!gates || !gates.state) return null;
  const rows = gates.rows || [];
  const held = gates.held_out || [];
  const anyAbsent = rows.some((row) => row.candidate_absent);
  const anyShippedAbsent = rows.some((row) => row.shipped_absent);
  return (
    <section className="cb-gates">
      <h4>{t('gates.title', locale)}</h4>
      <p className="cb-gates-reading">{pick(gates, 'reading', locale)}</p>
      {rows.length ? (
        <table className="cb-cells-table">
          <thead>
            <tr>
              <th scope="col">{t('gates.key', locale)}</th>
              <th scope="col">{t('detail.shipped', locale)}</th>
              <th scope="col">{t('detail.candidate', locale)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <th scope="row"><code><Code>{row.key}</Code></code></th>
                <td><GateValue text={row.shipped_text} absent={row.shipped_absent} locale={locale} /></td>
                <td><GateValue text={row.candidate_text} absent={row.candidate_absent} locale={locale} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="cb-note">{t('gates.none', locale)}</p>}
      {anyAbsent ? <p className="cb-note">{t('gates.candidate_absent', locale)}</p> : null}
      {anyShippedAbsent ? <p className="cb-note">{t('gates.shipped_absent', locale)}</p> : null}
      {held.length ? (
        <div className="cb-gates-held">
          <p className="cb-label">{t('gates.held_out', locale)}</p>
          <table className="cb-cells-table">
            <thead>
              <tr>
                <th scope="col">{t('gates.block', locale)}</th>
                <th scope="col">{t('detail.shipped', locale)}</th>
                <th scope="col">{t('detail.candidate', locale)}</th>
                <th scope="col">{t('table.verdict', locale)}</th>
              </tr>
            </thead>
            <tbody>
              {held.map((row) => (
                <tr key={row.block}>
                  <th scope="row"><code><Code>{row.block}</Code></code></th>
                  <td><HeldOut size={row.shipped_size} unit={pick(row, 'shipped_unit', locale)} absent={row.shipped_absent} locale={locale} /></td>
                  <td><HeldOut size={row.candidate_size} unit={pick(row, 'candidate_unit', locale)} absent={row.candidate_absent} locale={locale} /></td>
                  <td>
                    <span className={`cb-tag ${row.comparable ? 'cb-teal' : 'cb-amber'}`}>
                      {t(row.comparable ? 'gates.comparable' : 'gates.not_comparable', locale)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="cb-note">{pick(gates, 'held_out_rule', locale)}</p>
          <p className="cb-note">{pick(gates, 'held_out_basis', locale)}</p>
        </div>
      ) : null}
    </section>
  );
}

// An amount with the noun it counts, because 34,560 minutes and 2,532 breaks are
// two different things and the bare figure reads as one of them. The noun rides
// on the payload in both halves rather than being mapped from a key here.
function HeldOut({ size, unit, absent, locale }) {
  if (absent || size === null || size === undefined) {
    return <span className="cb-tag cb-neutral">{t('gates.absent_short', locale)}</span>;
  }
  return (
    <span className="cb-detail-scope">
      <BidiFigure><Numeric>{Number(size).toLocaleString('en-US')}</Numeric></BidiFigure>
      <span>{unit}</span>
    </span>
  );
}

function Number6({ value, digits = 6, sign = false }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="cb-absent-figure">-</span>;
  }
  const number = Number(value);
  return <Numeric>{`${sign && number > 0 ? '+' : ''}${number.toFixed(digits)}`}</Numeric>;
}

// Exported because the table renders shekels too and carried its own identical
// copy of this. One renderer decides the sign, the separators and the locale.
export function Shekels({ value, locale, digits = 2 }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="cb-absent-figure">-</span>;
  }
  const number = Number(value);
  return <Numeric>{`${number > 0 ? '+' : ''}${number.toLocaleString(locale === 'en' ? 'en-US' : 'he-IL', { minimumFractionDigits: digits, maximumFractionDigits: digits })} ₪`}</Numeric>;
}

function MoneyBlock({ money, locale }) {
  const state = String((money || {}).state || 'not_measured');
  if (state === 'measured') {
    return (
      <div className="cb-detail-money">
        <p className="cb-detail-figure">
          <Shekels value={money.revenue_delta} locale={locale} />
          <span className="cb-detail-scope">
            <BidiFigure><Numeric>{Number(money.rows || 0).toLocaleString('en-US')}</Numeric></BidiFigure>
            <span>{t('money.rows', locale)}</span>
          </span>
        </p>
        <p><span className="cb-label">{t('money.whole_plan', locale)}</span><Shekels value={money.whole_plan_delta} locale={locale} /></p>
        <p><span className="cb-label">{t('money.basis', locale)}</span>{money.basis}</p>
        <p><span className="cb-label">{t('state.measured_at', locale)}</span><BidiFigure>{formatStamp(money.measured_at)}</BidiFigure></p>
      </div>
    );
  }
  return (
    <div className="cb-detail-money">
      <p className="cb-detail-bars">
        <span className={`cb-tag ${state === 'stale' ? 'cb-amber' : 'cb-neutral'}`}>{t(`money.${state}`, locale)}</span>
        {typeof money.last_known_revenue_delta === 'number' ? (
          <span className="cb-detail-scope">
            <span className="cb-label">{t('money.last_known', locale)}</span>
            <Shekels value={money.last_known_revenue_delta} locale={locale} digits={0} />
          </span>
        ) : null}
      </p>
      <p>{pick(money, 'reason', locale)}</p>
    </div>
  );
}

function Cells({ cells, locale }) {
  const rows = cells.top || [];
  return (
    <div className="cb-detail-cells">
      <p>{pick(cells, 'reading', locale)}</p>
      {/* cb-detail-bars is the flex row with a gap that every other fact line
          here already uses. Without it the figure and the cell key sat against
          each other with nothing between them and read as one token,
          "0.012751733PrimeShow2_first_short", in both locales. */}
      {cells.max_abs_delta_at ? (
        <p className="cb-detail-bars">
          <span className="cb-label">{t('detail.largest', locale)}</span>
          <BidiFigure><Number6 value={cells.max_abs_delta} digits={9} /></BidiFigure>
          <Code>{cells.max_abs_delta_at}</Code>
        </p>
      ) : null}
      {rows.length ? (
        <table className="cb-cells-table">
          <thead>
            <tr>
              <th scope="col">{t('detail.cell', locale)}</th>
              <th scope="col">{t('detail.shipped', locale)}</th>
              <th scope="col">{t('detail.candidate', locale)}</th>
              <th scope="col">{t('detail.delta', locale)}</th>
              <th scope="col">{t('detail.breaks', locale)}</th>
              <th scope="col">{t('detail.bought', locale)}</th>
              <th scope="col">{t('detail.share', locale)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.cell}>
                <th scope="row"><code><Code>{row.cell}</Code></code></th>
                <td><BidiFigure><Number6 value={row.shipped} sign /></BidiFigure></td>
                <td><BidiFigure><Number6 value={row.candidate} sign /></BidiFigure></td>
                <td><BidiFigure><Number6 value={row.delta} sign /></BidiFigure></td>
                <td><BidiFigure><Numeric>{Number(row.breaks || 0).toLocaleString('en-US')}</Numeric></BidiFigure></td>
                <td><BidiFigure><Number6 value={row.squared_error_delta} digits={9} sign /></BidiFigure></td>
                <td>
                  <BidiFigure><Numeric>{`${(Number(row.share_of_absolute || 0) * 100).toFixed(2)}%`}</Numeric></BidiFigure>
                  {/* Where the movement concentrated, which is the one question
                      on this table that is a shape and not a digit. The share is
                      of the whole absolute movement, which the sentence above
                      the table states. */}
                  <Meter share={row.share_of_absolute} tone="cb-meter-blue" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <p className="cb-note">
        {t('detail.top_of', locale)}
        <BidiFigure><Numeric>{`${rows.length} / ${cells.top_of}`}</Numeric></BidiFigure>
      </p>
      <p className="cb-note">{pick(cells, 'key_shape', locale)}</p>
    </div>
  );
}

export default function BoardDetail({ candidate, board, locale }) {
  if (!candidate) {
    return <section className="cb-detail cb-empty"><p>{t('detail.pick_one', locale)}</p></section>;
  }
  const cells = candidate.cells || {};
  const decision = candidate.decision || {};
  const evaluation = board.evaluation || {};
  return (
    <section className="cb-detail" aria-live="polite">
      <h3>
        <span>{t('detail.title', locale)}</span>
        <code><Code>{candidate.id}</Code></code>
      </h3>
      {/* What it was built for, first, because a reader who does not know that
          cannot read anything below it. The reference's own run overview leads
          with the same line. */}
      <Purpose origin={candidate.origin} locale={locale} />
      <p className="cb-detail-verdict">{pick(candidate, 'verdict', locale)}</p>
      <p className="cb-detail-rule">
        <span className="cb-label">{t('detail.rule', locale)}</span>
        {pick(candidate, 'rule', locale)}
      </p>
      <p className="cb-detail-bars">
        <span className="cb-label">{t('evaluation.bar', locale)}</span>
        <BidiFigure><Number6 value={candidate.paired_bar} digits={1} /></BidiFigure>
        <span className="cb-dot">.</span>
        <span className="cb-label">{t('evaluation.dispersion', locale)}</span>
        <BidiFigure><Number6 value={candidate.fold_dispersion} /></BidiFigure>
        <span className="cb-dot">.</span>
        <span className="cb-label">{t('evaluation.spread', locale)}</span>
        <BidiFigure><Number6 value={evaluation.target_sd} /></BidiFigure>
      </p>
      <p className="cb-detail-bars">
        <span className="cb-label">{t('detail.breaks_moved', locale)}</span>
        <BidiFigure><Numeric>{`${Number(candidate.breaks_improved || 0).toLocaleString('en-US')} / ${Number(candidate.breaks_worsened || 0).toLocaleString('en-US')}`}</Numeric></BidiFigure>
      </p>
      {candidate.duplicate_of && candidate.duplicate_of.length ? (
        <p>
          <span className="cb-label">{t('detail.duplicate', locale)}</span>
          <Code>{candidate.duplicate_of.join(', ')}</Code>
        </p>
      ) : null}

      {/* The gates before the money, because that is the order JS-19 reads them
          in: what did it decide differently, then what would it move. */}
      <Gates gates={candidate.gates} locale={locale} />

      <h4>{t('table.money', locale)}</h4>
      <MoneyBlock money={candidate.money} locale={locale} />

      <h4>{t('detail.coefficients', locale)}</h4>
      <Cells cells={cells} locale={locale} />

      <h4>{t('table.recorded', locale)}</h4>
      <p className="cb-detail-decision">
        <span className={`cb-tag ${decision.state === 'shipped' ? 'cb-teal' : 'cb-neutral'}`}>
          {decision.state ? t(`decision.${decision.state}`, locale) : t('decision.none', locale)}
        </span>
        {decision.actor ? (
          <span>
            <span className="cb-label">{t('decision.by', locale)}</span>
            <Name>{decision.actor}</Name>
          </span>
        ) : null}
        {decision.recorded_at ? <BidiFigure>{formatStamp(decision.recorded_at)}</BidiFigure> : null}
        {decision.count > 1 ? (
          <span className="cb-detail-scope">
            <BidiFigure><Numeric>{String(decision.count)}</Numeric></BidiFigure>
            <span>{t('decision.count', locale)}</span>
          </span>
        ) : null}
      </p>
      {decision.state ? (
        <p className={`cb-detail-basis ${decision.on_rescore ? '' : 'cb-amber'}`}>
          {decision.on_rescore ? t('decision.on_comparison', locale) : t('decision.before_comparison', locale)}
        </p>
      ) : null}

      {/* And every verdict before that one. The block above is the newest, which
          is what the shelf column shows; this is the second half of JS-19's done
          condition, that a later reader can see what was tried. */}
      <History history={candidate.history} locale={locale} />

      <SelfReported self={candidate.self_reported} locale={locale} />

      <h4>{t('detail.identity', locale)}</h4>
      <p className="cb-detail-identity">
        <code><Code>{candidate.file}</Code></code>
        <code><Code>{candidate.short}</Code></code>
        {/* When the artifact was produced. It has been on every row of this
            payload since the board was built and no surface rendered it, so a
            steward could not tell an artifact made last week from one made a
            year ago. Through the one file that decides what a date looks like,
            never as the stored instant. */}
        {candidate.computed_at ? (
          <span className="cb-detail-scope">
            <span className="cb-label">{t('detail.built', locale)}</span>
            <BidiFigure>{formatStamp(candidate.computed_at)}</BidiFigure>
          </span>
        ) : null}
        <BidiFigure><Numeric>{Number(candidate.bytes || 0).toLocaleString('en-US')}</Numeric></BidiFigure>
        <span>{t('detail.bytes', locale)}</span>
        <span className="cb-label">{t('detail.fitted_on', locale)}</span>
        <BidiFigure><Numeric>{Number(candidate.breaks_fitted_on || 0).toLocaleString('en-US')}</Numeric></BidiFigure>
        <span>{t('evaluation.breaks', locale)}</span>
        <FitBasis basis={candidate.fit_basis} locale={locale} />
      </p>

      {/* And what that file read, checked against the files on disk. The block
          above says which file this row is; this says which data it was fitted
          from, whether that data is still here, and the half of the provenance
          nothing in these artifacts records. */}
      <Provenance origin={candidate.origin} locale={locale} />
    </section>
  );
}
