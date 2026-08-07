import React from 'react';
import { ArrowRight, Undo2, X } from 'lucide-react';
import { formatNumber, pageText } from '../../shell/format';
import { exactCurrency } from './day-board-model';
import { LIVE_PLAN, scopeWithBasis } from './plan-basis';

// What the last act actually did to the plan, kept on screen after the board has
// re-read the day. A save, a gold change, or the undo of either.
//
// This panel exists because of a measured hole. The readout beside it prices the
// arrangement on screen against the plan's own basis, and after a save the board
// re-reads the day, so that readout re-bases itself and prints a change of zero.
// Measured on רשת 13 / 2024-11-01: writing the restriction the board writes for
// one break moved the committed day from 1,067,845.55 to 1,037,270.00 while the
// readout said zero both before and after, so the money a save cost was
// unobservable anywhere in the product.
//
// Both figures here are the engine's own day totals from the same route, read
// once before the save and once after it. Nothing is modelled and nothing is
// predicted twice.
function DayBoardSettlement({ settlement, locale, onUndo, onDismiss, canUndo }) {
  if (!settlement) return null;
  const label = (en, he) => (locale === 'he' ? he : en);
  const { act, basis, before, after, realised, predicted, verdict } = settlement;
  // Both totals are the engine's own day figures, read before and after the act,
  // so they are the live plan's and every scope line here names it.
  const scopeText = basis ? scopeWithBasis(`${basis.channel} / ${basis.day}`, LIVE_PLAN, locale) : '';
  const total = after && after.breaks ? after.breaks : 0;

  return (
    <div className={`day-settlement is-${verdict}`}>
      <div className="day-settlement-head">
        <strong>{headingOf(act, label)}</strong>
        <button type="button" className="day-settlement-close" onClick={onDismiss}>
          <X size={13} aria-hidden="true" />
          {label('Dismiss', 'סגירה')}
        </button>
      </div>

      <div className="day-readout-figures">
        <div className={`day-figure ${directionOf(realised.revenue)}`}>
          <span className="day-figure-label">{label('Realised change', 'שינוי בפועל')}</span>
          <strong dir="ltr">{exactCurrency(realised.revenue, locale)}</strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{label('Predicted before the act', 'התחזית לפני הפעולה')}</span>
          <strong dir="ltr">{predicted === null ? '-' : exactCurrency(predicted, locale)}</strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</span>
          <strong dir="ltr" className="day-settlement-pair">
            {exactCurrency(before.revenue, locale)}
            <ArrowRight size={12} aria-hidden="true" />
            {exactCurrency(after.revenue, locale)}
          </strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Breaks in the day', 'ברייקים ביום')}</span>
          <strong dir="ltr" className="day-settlement-pair">
            {formatNumber(before.breaks, locale)}
            <ArrowRight size={12} aria-hidden="true" />
            {formatNumber(after.breaks, locale)}
          </strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
      </div>

      <p className="day-readout-note" dir="auto">{verdictSentence(settlement, locale, total)}</p>

      <div className="day-readout-actions">
        {act !== 'undo' && (
          <button type="button" className="day-action" onClick={onUndo} disabled={!canUndo}>
            <Undo2 size={13} aria-hidden="true" />
            {act === 'gold'
              ? label('Undo this gold change', 'ביטול שינוי הזהב')
              : label('Undo this save', 'ביטול השמירה הזו')}
          </button>
        )}
      </div>
    </div>
  );
}

// The act, named in the operator's own language.
//
// Three acts print into this one panel and each of them means something else, so
// the heading says which one it is. Measured before the gold act settled at all:
// its cost was on no surface, and the change tile beside this panel had re-based
// itself to zero.
export function headingOf(act, label) {
  if (act === 'gold') return label('What the gold change did to the plan', 'מה שינוי הזהב עשה לתוכנית');
  if (act === 'undo') return label('What the undo restored', 'מה הביטול החזיר');
  return label('What the save changed on the plan', 'מה השמירה שינתה בתוכנית');
}

// The same naming when an act fails, so the message says which act failed rather
// than reporting a save that was never attempted.
export function failureText(act, message) {
  if (act === 'gold') return [`The gold change failed (${message}).`, `שינוי הזהב נכשל (${message}).`];
  if (act === 'undo') return [`Undo failed (${message}).`, `הביטול נכשל (${message}).`];
  return [`Save failed (${message}).`, `השמירה נכשלה (${message}).`];
}

// Money that fell is not money that rose. The readout's own is-moved colour reads
// as a gain, so a realised loss of 47,444 painted in it would say the opposite of
// what happened.
export function directionOf(revenue) {
  if (!Number.isFinite(revenue) || Math.abs(revenue) <= 0.005) return 'is-flat';
  return revenue > 0 ? 'is-gain' : 'is-loss';
}

// The one sentence that has to be right. It names both figures, the gap between
// them, and what the engine did while it was planning the day again.
export function verdictSentence(settlement, locale, total) {
  const { verdict, predicted, realised, difference, rearranged } = settlement;
  const he = locale === 'he';
  const predictedText = predicted === null ? '-' : exactCurrency(predicted, locale);
  const realisedText = exactCurrency(realised.revenue, locale);
  const gapText = difference === null ? '-' : exactCurrency(Math.abs(difference), locale);
  if (verdict === 'agreed') {
    if (he) return `התחזית אמרה ${predictedText} והתוכנית השמורה חזרה עם ${realisedText}. השתיים תואמות.`;
    return `The preview said ${predictedText} and the saved plan came back with ${realisedText}. The two agree.`;
  }
  if (verdict === 'unknown') {
    if (he) return `לא הייתה תחזית להשוות אליה, ולכן מוצג כאן רק השינוי בפועל של ${realisedText}.`;
    return `There was no prediction to compare this against, so only the realised change of ${realisedText} is shown.`;
  }
  if (he) return `התחזית אמרה ${predictedText} והתוכנית השמורה חזרה עם ${realisedText}, פער של ${gapText}. שמירה כותבת מגבלה, והמנוע מריץ את כל היום מחדש כשהיא בתוקף, ולכן הוא רשאי למקם ברייקים אחרים אחרת. הפעם הוא שינה ${rearranged.changed} ברייקים מתוך ${total}, ב-${rearranged.programmes} רצועות שידור.`;
  return `The preview said ${predictedText} and the saved plan came back with ${realisedText}, a gap of ${gapText}. Saving writes a restriction and the engine then runs the whole day again with it in force, so it may place other breaks differently. This time it changed ${rearranged.changed} breaks of ${total}, across ${rearranged.programmes} programmes.`;
}

export default DayBoardSettlement;
