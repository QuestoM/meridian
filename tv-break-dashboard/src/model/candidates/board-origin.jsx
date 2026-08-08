import React from 'react';
import { Code, Name } from '../../shell/bidi';
import { pick, t } from './board-words';

// What an artifact was built for, and what data it was built from.
//
// **The absence these two blocks close.** Every candidate's own metadata may
// carry a one-line purpose, and no surface here carried it. Measured before this
// was written: the sentence one candidate records about itself appeared zero
// times in the published board and zero times on any terminal block. So a
// steward opened a shelf of five opaque identifiers and worked out what each one
// was trying to do from its coefficient table.
//
// The reference this board is measured against puts exactly this first: a run
// overview leads with the notes, then the checkout and the command. Two of those
// three are answerable from these artifacts and the third is not, and the third
// says so rather than being left out.
//
// **The purpose is a stored value and not authored copy.** It is the producer's
// own sentence in the producer's own language, so it renders through the shell
// primitive for prose that arrives as data in either script, with a caption
// saying whose words they are. Two of the five candidates record none, and the
// absence with the field that would fill it, never as an inference from the
// numbers above it.

export function Purpose({ origin, locale }) {
  if (!origin) return null;
  return (
    <div className="cb-purpose-block">
      <p className="cb-detail-bars">
        <span className="cb-label">{t('origin.title', locale)}</span>
        {origin.purpose
          ? <Name>{origin.purpose}</Name>
          : <span className="cb-tag cb-amber">{t('origin.none', locale)}</span>}
      </p>
      <p className="cb-note">{pick(origin, 'purpose_reading', locale)}</p>
    </div>
  );
}

// One source file, as the artifact records it and as the disk answers now.
//
// Tri-state, and the third state is the one that matters: a file that is not on
// disk did not fail the comparison, nobody could make it. The verdict comes from
// the engine's own freshness guard on the measuring side, so this cell and the
// engine cannot disagree about one artifact.
function Matches({ row, locale }) {
  if (!row.on_disk) return <span className="cb-tag cb-amber">{t('origin.missing', locale)}</span>;
  const tone = row.matches ? 'cb-teal' : 'cb-red';
  return <span className={`cb-tag ${tone}`}>{t(row.matches ? 'origin.matches' : 'origin.differs', locale)}</span>;
}

export function Provenance({ origin, locale }) {
  const rows = (origin || {}).sources || [];
  if (!origin) return null;
  return (
    <section className="cb-provenance">
      <h4>{t('origin.sources', locale)}</h4>
      {rows.length ? (
        <table className="cb-cells-table">
          <thead>
            <tr>
              <th scope="col">{t('origin.file', locale)}</th>
              <th scope="col">{t('origin.digest', locale)}</th>
              <th scope="col">{t('origin.on_disk', locale)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.file}>
                <th scope="row"><code><Code>{row.file}</Code></code></th>
                <td><code><Code>{row.short}</Code></code></td>
                <td><Matches row={row} locale={locale} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <p className="cb-note">{pick(origin, 'sources_reading', locale)}</p>
      {origin.agreement_reading_en ? (
        <p className="cb-note">{pick(origin, 'agreement_reading', locale)}</p>
      ) : null}
      {/* The half of the provenance this tree cannot answer, stated where the
          half it can answer is stated. No artifact here records the command that
          produced it, so one can be identified exactly and rebuilt from nothing
          it carries, and the line that would close that is named. */}
      <p className="cb-note cb-recipe">
        <span className="cb-label">{t('origin.recipe', locale)}</span>
        {pick(origin, 'recipe', locale)}
      </p>
      <p className="cb-note">
        <span className="cb-label">{t('limit.lifted', locale)}</span>
        {pick(origin, 'recipe_unblocked_by', locale)}
      </p>
    </section>
  );
}

// The purpose under the artifact name in the ranked table, where the shelf is
// read as a shelf. One short line, muted, and an absence renders as an absence
// rather than as a blank that reads like a row nobody wrote a note for.
export function PurposeLine({ origin, locale }) {
  if (!origin) return null;
  if (!origin.purpose) {
    return <span className="cb-purpose cb-purpose-absent">{t('origin.none', locale)}</span>;
  }
  return <span className="cb-purpose"><Name>{origin.purpose}</Name></span>;
}
