import React from 'react';
import { pageText, formatCurrency, formatNumber, finiteNumber } from '../shell/surface-helpers';

// The measured before and after a settings change would produce, and the basis
// those two figures were computed on. Split out of AssistantProposalCard, which
// was at the size cap, so the basis could be added without compressing the card.
// A settings item may carry a measured effect on the owned channel, so the
// operator sees what a change would do before approving it. The three money
// lines use the same vocabulary as the rest of the product (gross revenue,
// retention cost, net) plus the breaks change, each with a signed delta. A
// missing figure is dropped rather than invented; an unavailable effect shows a
// quiet reason instead.
const EFFECT_METRICS = [
  ['gross', 'Gross revenue', 'הכנסות ברוטו'],
  ['retention_cost', 'Retention cost', 'עלות שימור'],
  ['net', 'Net', 'נטו'],
];

// he-IL currency text carries invisible U+200F marks for a context it cannot
// know. Each figure here sits in a fixed left-to-right isolate, this product's
// money convention, and inside one those marks are what pushes the shekel sign
// across the arrow. Only the marks go: every digit and sign is untouched.
const BIDI_MARKS = /[\u200e\u200f]/g;

function figure(text) {
  return String(text).replace(BIDI_MARKS, '');
}

function signedFigure(value, locale, money) {
  const body = money ? formatCurrency(Math.abs(value), locale) : formatNumber(Math.abs(value), locale);
  return `${value > 0 ? '+' : value < 0 ? '-' : ''}${figure(body)}`;
}

function metricCells(beforeVal, afterVal, deltaVal, locale, money) {
  const fmt = (value) => (money ? formatCurrency(value, locale) : formatNumber(value, locale));
  const before = finiteNumber(beforeVal);
  const after = finiteNumber(afterVal);
  let delta = finiteNumber(deltaVal);
  if (delta === null && before !== null && after !== null) delta = after - before;
  return {
    shown: before !== null || after !== null || delta !== null,
    before: before !== null ? figure(fmt(before)) : '',
    after: after !== null ? figure(fmt(after)) : '',
    delta: delta !== null ? signedFigure(delta, locale, money) : '',
  };
}

// One isolate per figure, never one around the pair. A before, an arrow and an
// after sharing a single isolate reorder into each other, so the numbers weld
// together and the shekel signs drift at the moment of approval. Same isolate
// the run trace puts around each end of the date window.
function EffectRow({ label, cells, net }) {
  const both = cells.before !== '' && cells.after !== '';
  return (
    <div className={`asst-effect-row${net ? ' net' : ''}`}>
      <span className="asst-effect-label" dir="auto">{label}</span>
      <span className="asst-effect-flow" dir="ltr">
        <bdi dir="ltr">{cells.before || cells.after || '-'}</bdi>
        {both ? ' → ' : null}
        {both ? <bdi dir="ltr">{cells.after}</bdi> : null}
      </span>
      <span className="asst-effect-delta" dir="ltr"><bdi dir="ltr">{cells.delta}</bdi></span>
    </div>
  );
}

// The basis of the money above it, printed on the surface where the operator
// presses Apply rather than only in the answer text. The simulation optimizes
// one representative day of the owned channel on both sides, so a reader who is
// not told that reads a one-day figure as a weekly one. An item stored before
// the basis was carried says which part is missing instead of naming a channel
// and a date nobody recorded.
function EffectBasis({ basis, locale }) {
  const channel = basis && basis.channel ? String(basis.channel) : '';
  const day = basis && basis.day ? String(basis.day) : '';
  if (!channel || !day) {
    return (
      <p className="asst-effect-basis unknown" dir="auto">
        {pageText(locale, 'The simulation runs one representative channel-day, not the weekly total. The channel and the date were not recorded on this item. Ask Kai for the proposal again to get them.', 'הסימולציה רצה על יום-ערוץ מייצג אחד, לא על הסך השבועי. הערוץ והתאריך לא נשמרו בפריט הזה. אפשר לבקש מקאי את ההצעה מחדש כדי לקבל אותם.')}
      </p>
    );
  }
  return (
    <p className="asst-effect-basis" dir="auto">
      {pageText(locale, `The simulation runs one representative channel-day (${channel}, ${day}), not the weekly total.`, `הסימולציה רצה על יום-ערוץ מייצג אחד (${channel}, ${day}), לא על הסך השבועי.`)}
    </p>
  );
}

export default function EffectView({ effect, basis, locale }) {
  if (!effect || typeof effect !== 'object') return null;
  const header = pageText(locale, 'What this change would do', 'מה השינוי הזה יעשה');
  if (effect.status === 'unavailable') {
    const reason = effect.reason ? String(effect.reason) : pageText(locale, 'A preview is not available for this change.', 'אין תצוגה מקדימה לשינוי הזה.');
    return <div className="asst-effect"><span className="asst-effect-head">{header}</span><p className="asst-effect-note" dir="auto">{reason}</p></div>;
  }
  const before = effect.before && typeof effect.before === 'object' ? effect.before : {};
  const after = effect.after && typeof effect.after === 'object' ? effect.after : {};
  const delta = effect.delta && typeof effect.delta === 'object' ? effect.delta : {};
  const rows = EFFECT_METRICS.map(([key, en, he]) => ({ key, label: pageText(locale, en, he), ...metricCells(before[key], after[key], delta[key], locale, true) })).filter((row) => row.shown);
  const breaks = metricCells(before.breaks, after.breaks, delta.breaks, locale, false);
  if (!rows.length && !breaks.shown) return null;
  return (
    <div className="asst-effect">
      <span className="asst-effect-head">{header}</span>
      <EffectBasis basis={basis} locale={locale} />
      {rows.map((row) => <EffectRow key={row.key} label={row.label} cells={row} net={row.key === 'net'} />)}
      {breaks.shown ? <EffectRow key="breaks" label={pageText(locale, 'Breaks', 'ברייקים')} cells={breaks} /> : null}
    </div>
  );
}
