import React from 'react';
import { Numeric } from '../../shell/format';
import { Figure as BidiFigure, Code, Name } from '../../shell/bidi';
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

function Number6({ value, digits = 6, sign = false }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="cb-absent-figure">-</span>;
  }
  const number = Number(value);
  return <Numeric>{`${sign && number > 0 ? '+' : ''}${number.toFixed(digits)}`}</Numeric>;
}

function Shekels({ value, locale, digits = 2 }) {
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
        <p><span className="cb-label">{t('state.measured_at', locale)}</span><BidiFigure><Numeric>{String(money.measured_at || '').slice(0, 19)}</Numeric></BidiFigure></p>
      </div>
    );
  }
  return (
    <div className="cb-detail-money">
      <p>
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
      {cells.max_abs_delta_at ? (
        <p>
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
                <td><BidiFigure><Numeric>{`${(Number(row.share_of_absolute || 0) * 100).toFixed(2)}%`}</Numeric></BidiFigure></td>
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
        {decision.recorded_at ? <BidiFigure><Numeric>{String(decision.recorded_at).slice(0, 19)}</Numeric></BidiFigure> : null}
      </p>
      {decision.state ? (
        <p className={`cb-detail-basis ${decision.on_rescore ? '' : 'cb-amber'}`}>
          {decision.on_rescore ? t('decision.on_comparison', locale) : t('decision.before_comparison', locale)}
        </p>
      ) : null}

      <h4>{t('detail.identity', locale)}</h4>
      <p className="cb-detail-identity">
        <code><Code>{candidate.file}</Code></code>
        <code><Code>{candidate.short}</Code></code>
        <BidiFigure><Numeric>{`${Number(candidate.bytes || 0).toLocaleString('en-US')} bytes`}</Numeric></BidiFigure>
        <span className="cb-label">{t('detail.fitted_on', locale)}</span>
        <BidiFigure><Numeric>{Number(candidate.breaks_fitted_on || 0).toLocaleString('en-US')}</Numeric></BidiFigure>
        <span>{t('evaluation.breaks', locale)}</span>
      </p>
    </section>
  );
}
