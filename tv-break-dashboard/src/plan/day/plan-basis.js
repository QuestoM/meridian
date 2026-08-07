// Which plan a figure came from, in one vocabulary, for every surface that shows one.
//
// Two plans answer "how many breaks does this day have" and they disagree.
// output/weekly_break_schedule.csv is the SAVED weekly plan, the artifact the
// week board and every export read, and it holds 80 breaks across 82 programmes
// on רשת 13 / 2024-11-01. GET /api/plan/day re-plans that same day LIVE against
// current settings, constraints and models, and it holds 76. Both are real and
// neither is wrong; a figure that does not say which one it came from is.
//
// A critic measured the cost of that silence four rounds running. The editor's
// coverage sentence said "the day's 80 breaks" while the money tile 344 px below
// it said "Breaks in the day 76", and nothing on the page told a scheduler that
// those two counts were answers about different plans. Each round closed the one
// instance the critic pointed at and the next round found the next instance, so
// the basis lives here now: one label per plan, one composer per shape of figure,
// and every count site on these surfaces reads from this module. A new site that
// forgets to is a test failure, not a fifth round.
//
// The words match the ones the committed-plan note already used, so the sentence
// and the tiles under it say the same thing about the same file.

export const SAVED_PLAN = 'saved';
export const LIVE_PLAN = 'live';

// The plan itself, named. Everything below composes this and nothing re-spells it.
export function planBasisLabel(basis, locale) {
  if (basis === LIVE_PLAN) return locale === 'he' ? 'התוכנית החיה הזו' : 'this live plan';
  return locale === 'he' ? 'התוכנית השבועית השמורה' : 'the saved weekly plan';
}

// The same label at the head of a sentence, where English needs a capital and
// Hebrew needs nothing. Kept here so no caller re-spells the words to get one.
export function planBasisLead(basis, locale) {
  const named = planBasisLabel(basis, locale);
  return locale === 'he' ? named : `${named.charAt(0).toUpperCase()}${named.slice(1)}`;
}

// A figure's scope line with the plan it was computed on named at the end of it.
// The scope stays first because it is what the eye reads to find the figure's
// channel and day; the basis follows it in the same small line, beside the
// figure and never in a tooltip.
export function scopeWithBasis(scopeText, basis, locale) {
  const named = planBasisLabel(basis, locale);
  return scopeText ? `${scopeText} / ${named}` : named;
}

// Any label or row name, with the plan behind its numbers named after it.
export function withBasis(text, basis, locale) {
  const named = planBasisLabel(basis, locale);
  return text ? `${text}, ${named}` : named;
}

// The pointer the change tile carries when the live plan and the saved one have
// come apart and the note below spells the gap out.
export function livePlanPointer(locale) {
  const named = planBasisLabel(LIVE_PLAN, locale);
  return locale === 'he' ? `${named}, ראו למטה` : `${named}, see below`;
}

// A count of breaks, in words, for a lane header that has room for a number and
// little else. Two shapes: a plain count of what a surface holds, and a drawn
// count against the count the plan places in the same programmes. The second
// shape exists because the editor draws a capped slice, so a bare number there
// reads as a claim that the slice is the whole lane.
export function breakCountText(count, locale) {
  return locale === 'he' ? `${count} ברייקים` : `${count} breaks`;
}

export function drawnOfPlannedText(shown, planned, locale) {
  if (!Number.isFinite(planned)) {
    return locale === 'he' ? `${shown} ברייקים מוצגים` : `${shown} breaks drawn`;
  }
  return locale === 'he' ? `${shown} מתוך ${planned} ברייקים` : `${shown} of ${planned} breaks`;
}
