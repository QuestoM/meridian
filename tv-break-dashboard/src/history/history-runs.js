// What the run log is, in the three states it can be in, decided once.
//
// Four places on this destination have to say it: the provenance footer, the run
// filter's own count, the empty list under that filter, and the attestation
// strip at the top. A sentence written four times drifts, and the copy that
// drifts is the one nobody reads, so all four read from here.
//
// The withheld state is the one that matters. With no operator channel set the
// product cannot tell which of the recorded runs are the operator's own, so it
// withholds every one of them. Measured on this instance with the field blank:
// 542 records over four channels, and not one the product may attribute. The
// number of runs is therefore unknown, and unknown is never printed as zero. A
// zero here is an attestation that nothing ran, which is a different claim and
// one nobody on this screen is entitled to make.
//
// The words live in a plain module rather than in the component so they can be
// executed rather than read: a test that greps a surface passes on a rule that
// is wrong, and this rule decides what a compliance owner attests to.

export const RUNS_AVAILABLE = 'available';
export const RUNS_UNREADABLE = 'unreadable';
export const RUNS_WITHHELD = 'withheld_no_operator_channel';

// Tri-state, never a silent absence: the runs are here, the log could not be
// read, or the product may not say which runs are the operator's own. A payload
// that names no state at all is the second of those, because a source that
// cannot say what it is was not read.
export function runsSourceState(sources) {
  return ((sources || {}).runs || {}).state || RUNS_UNREADABLE;
}

export function runsCounted(state) {
  return state === RUNS_AVAILABLE;
}

// The provenance line: what the source is, and what that means for the list.
// Only the available state carries a figure, and it carries the scope with it.
export function runsSourceLine(state, records, channel) {
  if (state === RUNS_AVAILABLE) {
    return [`Runs: ${records || 0}, on ${channel || ''} only.`, `הרצות: ${records || 0}, של ${channel || ''} בלבד.`];
  }
  if (state === RUNS_WITHHELD) {
    return ['Runs: no operator channel is set, so this product cannot tell which runs are yours and none is listed.', 'הרצות: לא הוגדר ערוץ מפעיל, ולכן המוצר לא יכול לדעת אילו הרצות שלכם, ואף אחת לא מוצגת.'];
  }
  return ['Runs: the run log could not be read, so no run is listed.', 'הרצות: לא ניתן לקרוא את יומן ההרצות, ולכן אף הרצה לא מופיעה.'];
}

// Why a count is missing rather than zero, for the two places that print counts.
export function runsCountLine(state) {
  if (state === RUNS_WITHHELD) {
    return ['The runs cannot be counted. No operator channel is set, so this product cannot tell which runs are yours.', 'לא ניתן לספור את ההרצות. לא הוגדר ערוץ מפעיל, ולכן המוצר לא יכול לדעת אילו הרצות שלכם.'];
  }
  return ['The runs cannot be counted. The run log could not be read.', 'לא ניתן לספור את ההרצות. לא ניתן לקרוא את יומן ההרצות.'];
}

// The door the withheld state names, wherever it is named.
export const RUNS_REMEDY = ['Set the operator channel', 'הגדרת ערוץ המפעיל'];
