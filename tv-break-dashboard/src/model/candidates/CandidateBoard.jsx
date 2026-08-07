import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Numeric } from '../../shell/format';
import { Figure as BidiFigure, Code, Name } from '../../shell/bidi';
import { read } from '../console/console-api';
import BOARD from './candidate-board.json';
import { freshness, sortRows, SORTS } from './board-compare';
import BoardDetail from './board-detail.jsx';
import { pick, t } from './board-words';
import './candidate-board.css';

// The candidate board: five artifacts and the shipped one, on one basis.
//
// **The defect this exists to close.** The console's candidate shelf shows each
// artifact's own held-out figures, and those come from different splits: on this
// tree the shipped artifact reports its series gate on 2,532 breaks and one
// candidate reports the same gate on 506. Reading them side by side compares two
// experiments, not two models. This board is the comparison instead: every
// artifact scored again on one identical set of breaks with one metric, so the
// difference between two rows is a difference between two predictors.
//
// **Where the figures come from and why that is honest.** They are published by
// this piece's own terminal, into this tree, and they carry the digest of every
// file each figure was measured on. The board asks the console's own candidate
// route what the server is serving now and compares the digests. Current, stale
// or unknown, and the state is stated before any figure is read, because a
// comparison about a file that has since been rebuilt is worse than no
// comparison.
//
// **It records nothing.** Adopting a model and recording a verdict are acts that
// write under models/, which is the definition of training, and they live at a
// terminal with the people who own the model. This screen carries no control
// that performs one and no string that names one.

function useLive() {
  const [state, setState] = useState({ status: 'loading', payload: null, detail: '' });
  useEffect(() => {
    let active = true;
    read('/api/model/candidates').then((result) => {
      if (active) setState(result);
    });
    return () => { active = false; };
  }, []);
  return state;
}

function Figure({ value, digits = 6, sign = false, fallback = '-' }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="cb-absent-figure">{fallback}</span>;
  }
  const number = Number(value);
  const text = `${sign && number > 0 ? '+' : ''}${number.toFixed(digits)}`;
  return <Numeric>{text}</Numeric>;
}

function Shekels({ value, locale, digits = 2 }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="cb-absent-figure">-</span>;
  }
  const number = Number(value);
  const text = `${number > 0 ? '+' : ''}${number.toLocaleString(locale === 'en' ? 'en-US' : 'he-IL', { minimumFractionDigits: digits, maximumFractionDigits: digits })} ₪`;
  return <Numeric>{text}</Numeric>;
}

function Money({ money, locale }) {
  const state = String((money || {}).state || 'not_measured');
  if (state === 'measured') {
    return <Shekels value={money.revenue_delta} locale={locale} digits={2} />;
  }
  if (state === 'stale' && typeof money.last_known_revenue_delta === 'number') {
    return (
      <span className="cb-money-stale">
        <span className="cb-tag cb-amber">{t('money.stale', locale)}</span>
        <Shekels value={money.last_known_revenue_delta} locale={locale} digits={0} />
      </span>
    );
  }
  return <span className="cb-tag cb-neutral">{t(`money.${state}`, locale)}</span>;
}

function StateStrip({ state, board, locale }) {
  const tone = { current: 'cb-teal', stale: 'cb-amber', unknown: 'cb-blue' }[state.state] || 'cb-blue';
  return (
    <section className={`cb-state ${tone}`} aria-live="polite">
      <div className="cb-state-head">
        <span className="cb-state-label">{t('state.title', locale)}</span>
        <strong className="cb-state-word">{t(`state.${state.state}`, locale)}</strong>
      </div>
      <p className="cb-state-reason">{state.state === 'unknown' && state.detail ? `${t('state.unknown_reason', locale)} ${state.detail}` : t(`state.${state.state}_reason`, locale)}</p>
      {state.moved.length ? (
        <ul className="cb-state-moved">
          {state.moved.map((row) => (
            <li key={row.id}>
              <span className="cb-state-moved-label">{t('state.moved', locale)}</span>
              <Code>{row.id}</Code>
              <BidiFigure><Numeric>{`${row.measured_on} -> ${row.served_now}`}</Numeric></BidiFigure>
            </li>
          ))}
        </ul>
      ) : null}
      <p className="cb-state-when">
        <span>{t('state.measured_at', locale)}</span>
        <BidiFigure><Numeric>{String(board.measured_at || '').slice(0, 19)}</Numeric></BidiFigure>
        <span className="cb-dot">.</span>
        <span>{t('state.published_at', locale)}</span>
        <BidiFigure><Numeric>{String(board.published_at || '').slice(0, 19)}</Numeric></BidiFigure>
      </p>
    </section>
  );
}

function Evaluation({ board, locale }) {
  const evaluation = board.evaluation || {};
  const limit = board.limit || {};
  return (
    <section className="cb-evaluation">
      <p className="cb-evaluation-line">
        <BidiFigure><Numeric>{Number(evaluation.breaks || 0).toLocaleString('en-US')}</Numeric></BidiFigure>
        <span>{t('evaluation.breaks', locale)}</span>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(evaluation.cells || '')}</Numeric></BidiFigure>
        <span>{t('evaluation.cells', locale)}</span>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(evaluation.window || '')}</Numeric></BidiFigure>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(evaluation.folds || '')}</Numeric></BidiFigure>
        <span>{t('evaluation.folds', locale)}</span>
      </p>
      <p className="cb-evaluation-metric">
        <span className="cb-label">{t('evaluation.metric', locale)}</span>
        {pick(evaluation, 'metric', locale)}
      </p>
      <p className="cb-evaluation-metric">
        <span className="cb-label">{t('evaluation.spread', locale)}</span>
        <BidiFigure><Figure value={evaluation.target_sd} /></BidiFigure>
        <span className="cb-dot">.</span>
        <span>{pick(evaluation, 'target_sd', locale)}</span>
      </p>
      <div className="cb-limit cb-amber">
        <span className="cb-label">{t('limit.title', locale)}</span>
        <p>{locale === 'en' ? limit.en : limit.he}</p>
        <p className="cb-limit-lifted">
          <span className="cb-label">{t('limit.lifted', locale)}</span>
          {locale === 'en' ? limit.unblocked_by_en : limit.unblocked_by_he}
        </p>
      </div>
    </section>
  );
}

function Recorded({ candidate, locale }) {
  const decision = candidate.decision || {};
  if (!decision.state) return <span className="cb-tag cb-neutral">{t('decision.none', locale)}</span>;
  return (
    <span className="cb-recorded">
      <span className={`cb-tag ${decision.state === 'shipped' ? 'cb-teal' : 'cb-neutral'}`}>
        {t(`decision.${decision.state}`, locale)}
      </span>
      {decision.on_rescore ? null : <span className="cb-mark" title={t('decision.before_comparison', locale)}>*</span>}
    </span>
  );
}

function Head({ id, label, locale, sort, onSort }) {
  const sortable = Object.prototype.hasOwnProperty.call(SORTS, id);
  if (!sortable) return <th scope="col">{label}</th>;
  const on = sort.key === id;
  return (
    <th scope="col" aria-sort={on ? (sort.ascending ? 'ascending' : 'descending') : 'none'}>
      <button type="button" className={`cb-sort ${on ? 'on' : ''}`} onClick={() => onSort(id)}>
        {label}
        <span className="cb-sort-mark" aria-hidden="true">{on ? (sort.ascending ? '▲' : '▼') : '○'}</span>
      </button>
    </th>
  );
}

const COLUMNS = [
  ['artifact', 'table.artifact'],
  ['rmse', 'table.rmse'],
  ['rmse_delta', 'table.delta'],
  ['paired_statistic', 'table.statistic'],
  ['verdict', 'table.verdict'],
  ['cells', 'table.cells'],
  ['money', 'table.money'],
  ['recorded', 'table.recorded'],
];

function Table({ board, rows, locale, selected, onSelect, sort, onSort }) {
  const reference = board.shipped || {};
  return (
    <table className="cb-table">
      <caption className="cb-caption">{t('table.title', locale)}</caption>
      <thead>
        <tr>{COLUMNS.map(([id, key]) => (
          <Head key={id} id={id} label={t(key, locale)} locale={locale} sort={sort} onSort={onSort} />
        ))}</tr>
      </thead>
      <tbody>
        <tr className="cb-row cb-reference">
          <th scope="row">
            <span className="cb-name">{t('table.shipped_row', locale)}</span>
            <code className="cb-digest"><Code>{reference.short}</Code></code>
          </th>
          <td><BidiFigure><Figure value={reference.rmse} /></BidiFigure></td>
          <td colSpan={6} className="cb-reference-note">
            {reference.version_name ? (
              <span>
                <span className="cb-label">{t('table.version', locale)}</span>
                <BidiFigure><Numeric>{String(reference.version_name)}</Numeric></BidiFigure>
              </span>
            ) : null}
          </td>
        </tr>
        {rows.map((row) => (
          <tr key={row.id}
            className={`cb-row ${selected === row.id ? 'on' : ''}`}
            aria-selected={selected === row.id}>
            <th scope="row">
              <button type="button" className="cb-pick" onClick={() => onSelect(row.id)} aria-label={`${t('table.pick', locale)} ${row.id}`}>
                <span className="cb-name"><Code>{row.id}</Code></span>
              </button>
              <code className="cb-digest"><Code>{row.short}</Code></code>
            </th>
            <td><BidiFigure><Figure value={row.rmse} /></BidiFigure></td>
            <td><BidiFigure><Figure value={row.rmse_delta} sign /></BidiFigure></td>
            <td><BidiFigure><Figure value={row.paired_statistic} digits={2} sign /></BidiFigure></td>
            <td><span className={`cb-tag ${row.verdict === 'better' ? 'cb-teal' : row.verdict === 'worse' ? 'cb-red' : 'cb-neutral'}`}>{t(`verdict.${row.verdict}`, locale)}</span></td>
            <td>
              <BidiFigure><Numeric>{`${(row.cells || {}).moved} / ${(row.cells || {}).compared}`}</Numeric></BidiFigure>
              {typeof (row.cells || {}).cancelled_share === 'number' ? (
                <small className="cb-cancels"><BidiFigure><Numeric>{`${((row.cells.cancelled_share) * 100).toFixed(1)}%`}</Numeric></BidiFigure></small>
              ) : null}
            </td>
            <td><Money money={row.money} locale={locale} /></td>
            <td><Recorded candidate={row} locale={locale} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Baselines({ board, locale }) {
  const finding = board.structure_finding || {};
  const structure = board.cell_structure || {};
  return (
    <section className="cb-baselines">
      <h3>{t('baselines.title', locale)}</h3>
      <ul>
        {(board.baselines || []).map((row) => (
          <li key={row.id}>
            <Code>{row.id}</Code>
            <BidiFigure><Figure value={row.rmse} /></BidiFigure>
            <span>{pick(row, 'basis', locale)}</span>
          </li>
        ))}
      </ul>
      {finding.earns_its_place === false ? (
        <div className="cb-finding cb-amber">
          <span className="cb-label">{t('finding.title', locale)}</span>
          <p>{pick(structure, 'reading', locale)}</p>
          <p className="cb-finding-figures">
            <BidiFigure><Numeric>{`${Number(finding.structure_cost_rmse).toFixed(6)} rmse`}</Numeric></BidiFigure>
            <span className="cb-dot">.</span>
            <BidiFigure><Numeric>{`${Number(finding.times_the_largest_candidate_move).toFixed(1)}x`}</Numeric></BidiFigure>
            <span className="cb-dot">.</span>
            <BidiFigure><Numeric>{`${Number(finding.largest_candidate_move_rmse).toFixed(6)} rmse`}</Numeric></BidiFigure>
          </p>
          <p>{t('finding.none_address', locale)}</p>
          <p>
            <span className="cb-label">{t('finding.owner', locale)}</span>
            {pick(finding, 'decision_owner', locale)}
          </p>
        </div>
      ) : null}
      {(board.duplicate_groups || []).map((group) => (
        <p className="cb-duplicates" key={group.join('-')}>
          <span className="cb-label">{t('evaluation.duplicates', locale)}</span>
          <Code>{group.join(', ')}</Code>
        </p>
      ))}
    </section>
  );
}

export default function CandidateBoard({ locale = 'he', board = BOARD }) {
  const [selected, setSelected] = useState('');
  const [sort, setSort] = useState({ key: 'rmse', ascending: true });
  const live = useLive();
  const region = useRef(null);
  const state = useMemo(() => freshness(board, live), [board, live]);
  const rows = useMemo(() => sortRows(board.candidates || [], sort), [board, sort]);
  const current = rows.find((row) => row.id === selected) || null;
  const dir = locale === 'en' ? 'ltr' : 'rtl';

  function onSort(key) {
    setSort((was) => (was.key === key ? { key, ascending: !was.ascending } : { key, ascending: true }));
  }

  // Up and down move the selection inside this board, so reading five artifacts
  // is four keystrokes rather than four round trips through the pointer. Bound
  // on the region rather than on the window, so it cannot fight the console's
  // own section keys.
  function onKeyDown(event) {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    const order = rows.map((row) => row.id);
    const at = order.indexOf(selected);
    const next = event.key === 'ArrowDown'
      ? order[Math.min(at + 1, order.length - 1)] || order[0]
      : order[Math.max(at - 1, 0)] || order[0];
    if (next) {
      setSelected(next);
      event.preventDefault();
    }
  }

  return (
    <div className={`cb-board ${dir}`} dir={dir} lang={locale} ref={region} onKeyDown={onKeyDown} tabIndex={-1}>
      <header className="cb-head">
        <h2>{t('board.title', locale)}</h2>
        <p>{t('board.sub', locale)}</p>
        <p className="cb-read-only">{t('board.read_only', locale)}</p>
      </header>
      <StateStrip state={state} board={board} locale={locale} />
      <Evaluation board={board} locale={locale} />
      <Table board={board} rows={rows} locale={locale} selected={selected}
        onSelect={setSelected} sort={sort} onSort={onSort} />
      <BoardDetail candidate={current} board={board} locale={locale} />
      <Baselines board={board} locale={locale} />
    </div>
  );
}
