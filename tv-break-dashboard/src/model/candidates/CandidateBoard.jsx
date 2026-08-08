import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Numeric } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { Figure as BidiFigure, Code, DirectionRoot } from '../../shell/bidi';
import { read } from '../console/console-api';
import BOARD from './candidate-board.json';
import { freshness, movementShare, reveal, sortRows, SORTS } from './board-compare';
// Shekels comes from there because this file carried a byte-identical copy.
import BoardDetail, { Shekels } from './board-detail.jsx';
import Evaluation from './board-evaluation.jsx';
import { LiveModelVerdict } from './board-history.jsx';
import Meter from './board-meter.jsx';
import { PurposeLine } from './board-origin.jsx';
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
      {/* dd/mm/yyyy, from the one file in the product that decides what a date
          looks like. These two read as ISO instants sliced to nineteen
          characters, which is the machine format the design rules ban in both
          locales, and the date guard's own list names calendar-day fields only
          so it could not see them. */}
      <p className="cb-state-when">
        <span>{t('state.measured_at', locale)}</span>
        <BidiFigure>{formatStamp(board.measured_at)}</BidiFigure>
        <span className="cb-dot">.</span>
        <span>{t('state.published_at', locale)}</span>
        <BidiFigure>{formatStamp(board.published_at)}</BidiFigure>
      </p>
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
      {/* One artifact on this tree was refused twice and this column printed one
          word for it. The terminal states the count and the screen did not, so a
          steward reading only the screen could not see the second refusal. */}
      {decision.count > 1 ? (
        <small className="cb-cancels">
          <BidiFigure><Numeric>{String(decision.count)}</Numeric></BidiFigure>
          <span>{t('decision.count', locale)}</span>
        </small>
      ) : null}
    </span>
  );
}

function Head({ id, label, sort, onSort }) {
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

// The caveat on the row it is a caveat about. The paragraph under the limit
// names the rows too, but a reader comparing two numbers is reading the table,
// and a confound stated only in a paragraph above is a confound the comparison
// is made without.
function BasisMark({ basis, locale }) {
  const state = (basis || {}).state;
  if (state !== 'fewer' && state !== 'unknown') return null;
  const label = state === 'fewer' ? t('basis.mark', locale) : t('basis.unknown_mark', locale);
  return <span className="cb-basis-mark cb-tag cb-amber" title={label}>{label}</span>;
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
          <Head key={id} id={id} label={t(key, locale)} sort={sort} onSort={onSort} />
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
              <BasisMark basis={row.fit_basis} locale={locale} />
              {/* What this artifact was built for, in its producer's own words,
                  where the shelf is read as a shelf. Five identifiers with no
                  note beside them is what this board showed, and the reference
                  leads its own run list with exactly this line. */}
              <PurposeLine origin={row.origin} locale={locale} />
              <code className="cb-digest"><Code>{row.short}</Code></code>
            </th>
            <td><BidiFigure><Figure value={row.rmse} /></BidiFigure></td>
            <td>
              <BidiFigure><Figure value={row.rmse_delta} sign /></BidiFigure>
              {/* The movement against the noise it sits in, which is the figure
                  the verdict is decided against and the one thing a column of
                  six-decimal numerals cannot show. Absent when the row has no
                  dispersion to divide by, rather than drawn as an empty bar. */}
              <Meter share={movementShare(row)}
                tone={row.verdict === 'better' ? 'cb-meter-teal' : 'cb-meter-neutral'} />
            </td>
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

// What the table means, under the table, in ink rather than in a tooltip.
//
// Two things were readable only by hovering or only by inference. The `*` beside
// every verdict on this tree carried its sentence in a `title` attribute, which
// a keyboard reaches never and a screen reader reads never, and all five rows
// carry it. And the recorded-verdict column says "Shipped" of a DECISION, while
// nothing on the screen said whether anything had actually replaced the live
// artifact; the terminal half of this piece states that in its own block and the
// screen did not, so a reader could stop at the word and leave believing the
// candidate was live. `adopted` was on every row of the payload already.
function TableNotes({ board, rows, locale }) {
  const live = rows.filter((row) => row.adopted);
  return (
    <div className="cb-notes">
      <p className="cb-note">
        <span className="cb-mark">*</span>
        <span>{t('decision.mark_legend', locale)}</span>
        <span>{t('decision.before_comparison', locale)}</span>
      </p>
      <p className="cb-note">
        <span className="cb-label">{t('adopted.title', locale)}</span>
        {live.length
          ? live.map((row) => <Code key={row.id}>{row.id}</Code>)
          : <span>{t('adopted.none', locale)}</span>}
      </p>
      <p className="cb-note">
        <BidiFigure><Numeric>{String(rows.length)}</Numeric></BidiFigure>
        <span>{t('table.count', locale)}</span>
      </p>
      {/* How the table is worked, in ink. Four of the eight columns sort and the
          only thing telling them apart was a bare circle glyph with no legend,
          and the arrow keys moved the selection with nothing saying so. */}
      <p className="cb-note">{t('table.how_to_read', locale)}</p>
      {/* What the bar under a movement is a share of. A bar with no denominator
          on screen is the visual form of a figure nobody measured. */}
      <p className="cb-note">{t('meter.movement', locale)}</p>
    </div>
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

  // Opening an artifact left the reader looking at the table. Measured here:
  // the panel starts at 900.5 px in a 907 px viewport, so a click tinted a row
  // and changed nothing visible. Pointer picks reveal it; the arrow keys do not,
  // because a reader walking the table with them is reading the table.
  function onPick(id) {
    setSelected(id);
    reveal(region, '.cb-detail');
  }

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
    // DirectionRoot states the direction from the locale and is the only thing
    // that should. A `rtl` or `ltr` class beside it was a third statement of
    // the same fact and no stylesheet in the product selects on either one.
    <DirectionRoot locale={locale} className="cb-board" lang={locale} ref={region} onKeyDown={onKeyDown} tabIndex={-1}>
      <header className="cb-head">
        <h2>{t('board.title', locale)}</h2>
        <p>{t('board.sub', locale)}</p>
        <p className="cb-read-only">{t('board.read_only', locale)}</p>
      </header>
      <StateStrip state={state} board={board} locale={locale} />
      {/* What is on record about the live artifact itself, before the table that
          is measured against it. A decision record may be about the shipped
          model rather than about a candidate, and every read of the log on this
          piece filtered those out, so the shelf showed five verdicts and said
          nothing about a standing verdict on the artifact all five are compared
          with. */}
      <LiveModelVerdict live={board.live_model} log={board.decision_log} locale={locale} />
      <Evaluation board={board} locale={locale} />
      <Table board={board} rows={rows} locale={locale} selected={selected}
        onSelect={onPick} sort={sort} onSort={onSort} />
      <TableNotes board={board} rows={rows} locale={locale} />
      <BoardDetail candidate={current} board={board} locale={locale} />
      <Baselines board={board} locale={locale} />
    </DirectionRoot>
  );
}
