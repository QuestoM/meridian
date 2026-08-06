// The two sentences on the attestation strip, qualified by the reader's own
// permission scope.
//
// The measured defect. The endpoint answers over exactly one of two sets: every
// account's record, for an admin, or the caller's own slice, for anybody else,
// and it says which in `scope` beside every figure it returns. On one store, in
// the same second: an admin read `changed: 77, verdict: changed`, and a plain
// operator on the same store read `changed: 0, verdict: unchanged`, because the
// activity half of the count is filtered to that account before the strip ever
// sees it. The strip printed the unqualified sentence either way, with the one
// line that names the scope on this destination footer, thereafter measured 859
// pixels below the bottom of the strip and off the landing screen. A compliance
// owner reading only the strip could not tell "nothing happened" from "nothing
// I am shown happened", and the two are not the same attestation.
//
// So both sentences here are chosen by the endpoint's own `scope`, "all" or
// "self", the same string `activity_log.visibility_scope` already produces and
// the footer already reads under the same key. Restore points and runs carry no
// per-account scope, so a self-scoped count still mixes this account's own
// changes with every restore point on record, which is exactly what
// `changesSourceLine` and the footer already say in their own words; the count
// sentence says it again, inline, because the footer's saying it 859 pixels
// away is the defect this file closes.
export const SCOPE_SELF = 'self';

function count(value, locale) {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB');
}

// The non-empty case. `runCount` is `null` while the runs may not be counted,
// which is the caller's own decision and never made here.
export function sinceCountLine(changeCount, runCount, scope) {
  const changed = [count(changeCount, 'en'), count(changeCount, 'he')];
  const withRuns = runCount !== null && runCount !== undefined;
  const runs = withRuns ? [count(runCount, 'en'), count(runCount, 'he')] : null;
  if (scope === SCOPE_SELF) {
    if (withRuns) {
      return [
        `${changed[0]} changes and points were applied (your own changes and every restore point), and ${runs[0]} runs were recorded.`,
        `בוצעו ${changed[1]} שינויים ונקודות (השינויים שלכם וכל נקודות השחזור), ונרשמו ${runs[1]} הרצות.`,
      ];
    }
    return [
      `${changed[0]} changes and points were applied (your own changes and every restore point).`,
      `בוצעו ${changed[1]} שינויים ונקודות (השינויים שלכם וכל נקודות השחזור).`,
    ];
  }
  if (withRuns) {
    return [
      `${changed[0]} changes and points were applied, and ${runs[0]} runs were recorded.`,
      `בוצעו ${changed[1]} שינויים ונקודות, ונרשמו ${runs[1]} הרצות.`,
    ];
  }
  return [
    `${changed[0]} changes and points were applied.`,
    `בוצעו ${changed[1]} שינויים ונקודות.`,
  ];
}

// The empty case. A self-scoped zero is never printed as an unqualified claim
// that nothing changed: it names the set it covers and says in the same
// sentence that another account's changes are not part of it.
export function sinceEmptyLine(scope) {
  if (scope === SCOPE_SELF) {
    return [
      'Nothing has changed since that day, among your own changes and the restore points on record. Changes by other accounts are not shown here.',
      'שום דבר לא השתנה מאז אותו יום, מבין השינויים שלכם ונקודות השחזור שברישום. שינויים של חשבונות אחרים אינם מוצגים כאן.',
    ];
  }
  return [
    'Nothing has changed since that day.',
    'שום דבר לא השתנה מאז אותו יום.',
  ];
}
