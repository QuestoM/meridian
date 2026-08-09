import React from 'react';
import { Numeric } from '../../shell/format';
import { formatSpan } from '../../shell/dates';
import { Figure as BidiFigure, Code } from '../../shell/bidi';
import { spreadShare } from './board-compare';
import Meter from './board-meter.jsx';
import { pick, t } from './board-words';

// What every row on this board was measured on, and the limit that decides how
// any of it may be read.
//
// Split out of CandidateBoard.jsx, which had arrived at 448 lines of a 450 cap
// with no room for a three-line change. The seam is the one the question has:
// this block answers "on what", the table beside it answers "which of these",
// and the two are read in that order because a table of six errors means nothing
// until the reader knows what they are errors against.

// The rows the limit sentence is naming, under the sentence that names them.
//
// Only the rows that do not cover the evaluation are listed. When every row does
// cover it the whole block is absent, because the limit sentence in that case
// says so itself and a list of six rows agreeing with it is noise.
function BasisRows({ board, locale }) {
  const rows = ((board.fit_basis || {}).rows || []).filter((row) => row.state !== 'all');
  if (!rows.length) return null;
  return (
    <ul className="cb-basis-rows">
      <li className="cb-basis-head"><span className="cb-label">{t('limit.rows', locale)}</span></li>
      {rows.map((row) => (
        <li key={row.id}>
          <Code>{row.id}</Code>
          {row.state === 'fewer' ? (
            <span className="cb-basis-said">
              {/* Two rules meet on this line and both are kept. Design rule 3
                  puts a middle dot between facts rather than whitespace, and a
                  numeric line may never put two bare figures side by side, where
                  "2532 . 196" can be read as one decimal. So the first group
                  ends on a word and the dot separates the groups. */}
              <span className="cb-basis-fitted">
                <span className="cb-label">{t('basis.title', locale)}</span>
                <BidiFigure><Numeric>{String(row.fitted_on)}</Numeric></BidiFigure>
                <span>{t('basis.of', locale)}</span>
                <BidiFigure><Numeric>{String(row.scored_on)}</Numeric></BidiFigure>
                <span>{t('evaluation.breaks', locale)}</span>
              </span>
              <span className="cb-dot">.</span>
              <span className="cb-basis-short">
                <BidiFigure><Numeric>{String(row.not_fitted_on)}</Numeric></BidiFigure>
                <span>{t('basis.never_fitted', locale)}</span>
              </span>
            </span>
          ) : (
            <span>{t('basis.unknown_mark', locale)}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

// The live model's error against the spread of the thing it is predicting, as a
// bar and as the two figures the bar is made of.
//
// This is the most honest line on the surface and it was two numerals with a
// sentence between them. The bar is the share, and on this tree it fills, which
// is the finding: the live model's error is within a thousandth of the standard
// deviation of the effect it predicts.
// The fill is amber rather than neutral because the state palette says a colour
// means something is true about the data in it, and what is true here is that a
// model whose error is this close to its target's own spread explains little
// beyond the mean. A bar that filled in the neutral ink would read as a progress
// meter reaching its goal, which is the opposite of the finding.
function Spread({ board, locale }) {
  const share = spreadShare(board);
  if (share === null) return null;
  return (
    <div className="cb-spread">
      <p className="cb-detail-bars">
        <span className="cb-label">{t('evaluation.error_share', locale)}</span>
        <BidiFigure><Numeric>{`${(share * 100).toFixed(2)}%`}</Numeric></BidiFigure>
        <span className="cb-dot">.</span>
        <span className="cb-label">{t('table.shipped_row', locale)}</span>
        <BidiFigure><Numeric>{Number(board.shipped.rmse).toFixed(6)}</Numeric></BidiFigure>
      </p>
      <Meter share={share} tone="cb-meter-amber" />
      <p className="cb-note">{t('meter.spread', locale)}</p>
    </div>
  );
}

export default function Evaluation({ board, locale }) {
  const evaluation = board.evaluation || {};
  const limit = board.limit || {};
  return (
    <section className="card cb-evaluation">
      <p className="cb-evaluation-line">
        <BidiFigure><Numeric>{Number(evaluation.breaks || 0).toLocaleString('en-US')}</Numeric></BidiFigure>
        <span>{t('evaluation.breaks', locale)}</span>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(evaluation.cells || '')}</Numeric></BidiFigure>
        <span>{t('evaluation.cells', locale)}</span>
        <span className="cb-dot">.</span>
        {/* Two calendar days through the one file that decides what a date looks
            like, not the pre-joined "2024-11-01 to 2024-11-30" this payload used
            to carry, which put an English preposition inside a machine date on a
            Hebrew line and could not be reformatted by anything on this side. */}
        <span>{t('evaluation.window', locale)}</span>
        <BidiFigure>{formatSpan(evaluation.window_from, evaluation.window_to, locale)}</BidiFigure>
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
        <BidiFigure><Numeric>{Number(evaluation.target_sd || 0).toFixed(6)}</Numeric></BidiFigure>
        <span className="cb-dot">.</span>
        <span>{pick(evaluation, 'target_sd', locale)}</span>
      </p>
      <Spread board={board} locale={locale} />
      <div className="cb-limit cb-amber">
        <span className="cb-label">{t('limit.title', locale)}</span>
        <p>{locale === 'en' ? limit.en : limit.he}</p>
        <BasisRows board={board} locale={locale} />
        <p className="cb-limit-lifted">
          <span className="cb-label">{t('limit.lifted', locale)}</span>
          {locale === 'en' ? limit.unblocked_by_en : limit.unblocked_by_he}
        </p>
      </div>
    </section>
  );
}
