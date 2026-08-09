import React from 'react';
import { Numeric } from '../../shell/format';
import { formatDay, formatStamp, parseDay } from '../../shell/dates';
import { Figure as BidiFigure, Code, Name } from '../../shell/bidi';
import { pick, t } from './board-words';

// Every verdict ever recorded, rather than the newest one.
//
// **What this closes.** JS-19 ends "the verdict is stored and a later reader can
// see what was tried". The shelf column shows one word and a count, and the
// count is the whole of what a later reader gets. On this tree one artifact
// carries two verdicts that hold the same word for two different stated
// reasons, which is a restatement and not a repeat, and a count cannot tell
// those apart.
//
// **What is deliberately not here.** The sentence each verdict was taken for.
// It is unbounded text a steward typed at a terminal, the model console renders
// it from the store, and a second copy inside a bundled file is a second source
// that can disagree with the first. The payload says the sentence exists and
// where it is read, and the block says so on the screen.

// A MODEL VERSION NAME, read.
//
// The stored name on this tree is `2026-07-29`, a calendar day, and three
// surfaces in this piece printed it back exactly as stored: the shipped row of
// the table, the amber tag on a verdict taken against another version, and the
// log's own version line. That is the machine format the date rules ban in both
// locales, and the date guard could not see it, because its raw-ISO rule
// enumerates calendar-day payload FIELDS and a version NAME is not on that list.
//
// So the day is read out through shell/dates.js like every other day in the
// product, and ONLY when it is a day. parseDay decides. A version named anything
// else -- and nothing stops one being named `retention-v3` tomorrow -- is handed
// back unchanged in the identifier face it arrived in, because inventing a date
// for a name would be the worse defect. An absent name is stated as absent
// instead of printed as an empty code span, which is what two of these three
// sites did.
export function VersionName({ value, locale }) {
  const text = String(value ?? '').trim();
  if (!text) return <span className="cb-absent-figure">{t('history.version_none', locale)}</span>;
  if (!parseDay(text)) return <Code>{text}</Code>;
  return <BidiFigure>{formatDay(text)}</BidiFigure>;
}

// A cell is a state and its qualifiers, never a paragraph. The sentence behind
// the short word is printed once under the table, on the states that appear.
function Row({ row, locale }) {
  return (
    <tr>
      <th scope="row"><BidiFigure>{formatStamp(row.recorded_at)}</BidiFigure></th>
      <td><Name>{row.actor || '-'}</Name></td>
      <td>
        <span className={`cb-tag ${row.decision === 'shipped' ? 'cb-teal' : 'cb-neutral'}`}>
          {locale === 'en' ? row.decision_en : row.decision_he}
        </span>
        {row.release_note ? <span className="cb-tag cb-blue">{t('history.note', locale)}</span> : null}
      </td>
      <td>
        <span className={`cb-tag ${row.in_force ? 'cb-teal' : 'cb-neutral'}`}>
          {t(row.in_force ? 'history.standing' : 'history.superseded', locale)}
        </span>
        {/* A verdict recorded against an earlier model version is not a verdict
            about the artifact in force. The adoption act has always matched the
            version and the shelf column never did. */}
        {row.against_version_in_force ? null : (
          <span className="cb-tag cb-amber">
            {t('history.other_version', locale)}
            <VersionName value={row.model_version_name} locale={locale} />
          </span>
        )}
      </td>
      <td>
        <span className={`cb-tag ${row.on_rescore ? 'cb-teal' : 'cb-amber'}`}>
          {t(row.on_rescore ? 'history.on_comparison' : 'history.before_comparison', locale)}
        </span>
      </td>
    </tr>
  );
}

function Table({ rows, locale }) {
  const anyBefore = rows.some((row) => !row.on_rescore);
  return (
    <>
      <table className="cb-cells-table">
        <thead>
          <tr>
            <th scope="col">{t('history.when', locale)}</th>
            <th scope="col">{t('history.who', locale)}</th>
            <th scope="col">{t('history.what', locale)}</th>
            <th scope="col">{t('history.standing', locale)}</th>
            <th scope="col">{t('history.basis', locale)}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => <Row key={row.decision_id} row={row} locale={locale} />)}
        </tbody>
      </table>
      {anyBefore ? <p className="cb-note">{t('decision.before_comparison', locale)}</p> : null}
    </>
  );
}

export default function History({ history, locale }) {
  const rows = (history || {}).rows || [];
  if (!rows.length) return <p className="cb-note">{t('history.none', locale)}</p>;
  return (
    <section className="cb-history">
      <h4>{t('history.title', locale)}</h4>
      <p className="cb-history-reading">{pick(history, 'reading', locale)}</p>
      <Table rows={rows} locale={locale} />
      {history.not_shown_by_the_latest > 0 ? (
        <p className="cb-note cb-fact-line">
          <BidiFigure><Numeric>{String(history.not_shown_by_the_latest)}</Numeric></BidiFigure>
          <span>{t('history.hidden', locale)}</span>
        </p>
      ) : null}
      <p className="cb-note">{pick(history, 'reason', locale)}</p>
    </section>
  );
}

// What is on record about the live artifact itself, above the table that is
// measured against it. Every read of the decision log on this piece filtered to
// the candidate rows, so a no-ship recorded against the version in force
// reached no surface here at all.
export function LiveModelVerdict({ live, log, locale }) {
  const rows = (live || {}).rows || [];
  const tally = (log || {}).tally || {};
  return (
    <section className={`cb-live-verdict ${(live || {}).state === 'standing' ? 'cb-amber' : 'cb-blue'}`}>
      <h3>{t('live.title', locale)}</h3>
      <p className="cb-live-reading">{pick(live, 'reading', locale)}</p>
      {rows.length ? <Table rows={rows} locale={locale} /> : null}
      {rows.length ? <p className="cb-note">{pick(live, 'reason', locale)}</p> : null}
      {/* Where every record in an append-only log went. A surface that shows
          some of one and says nothing about the rest is the defect this block
          was written for, so the three counts are on the screen beside the
          total they add up to. */}
      <p className="cb-note cb-fact-line">
        <BidiFigure><Numeric>{String((log || {}).records || 0)}</Numeric></BidiFigure>
        <span>{t('live.log', locale)}</span>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(tally.on_the_shelf || 0)}</Numeric></BidiFigure>
        <span>{t('live.on_the_shelf', locale)}</span>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(tally.on_the_live_model || 0)}</Numeric></BidiFigure>
        <span>{t('live.on_the_live_model', locale)}</span>
        <span className="cb-dot">.</span>
        <BidiFigure><Numeric>{String(tally.off_the_shelf || 0)}</Numeric></BidiFigure>
        <span>{t('live.off_the_shelf', locale)}</span>
      </p>
      <p className="cb-note cb-fact-line">
        <span className="cb-label">{t('history.version', locale)}</span>
        <VersionName value={(log || {}).version_name} locale={locale} />
      </p>
    </section>
  );
}
