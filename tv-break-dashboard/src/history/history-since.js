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
// the footer already reads under the same key.
//
// **And what a self-scoped sentence names is read from the payload, never
// written here.** Three rounds in a row wrote the covered set into this file by
// hand and each one was wrong about a different kind: first only the restore
// points, then all three attested kinds but the exclusion clause still naming
// changes alone, which a live store falsified by 3,222 entries because the
// account filter narrows previews and sign-ins exactly as it narrows changes.
// The words now come from `history-scope.js`, which assembles them from
// `attested_kinds` and `scope_kinds`, so a sentence can only ever name the set
// its own payload proves. The page footer builds its own sentence from the same
// module over the kinds the page renders, which is a different set: one phrase
// cannot be true in two scopes, and that was the round-15 defect.
import { SCOPE_SELF, coveredPhrase, withheldLine } from './history-scope.js';

export { SCOPE_SELF };

function count(value, locale) {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB');
}

// What this reader's own attestation covers, in both languages, over the kinds
// the endpoint says it attested. Never called for an admin: an unscoped count
// covers the record and needs no qualification.
function covered(body) {
  const payload = body || {};
  return coveredPhrase(payload.attested_kinds, payload.scope_kinds);
}

// The non-empty case. `runCount` is `null` while the runs may not be counted,
// which is the caller's own decision and never made here.
export function sinceCountLine(changeCount, runCount, scope, body) {
  const changed = [count(changeCount, 'en'), count(changeCount, 'he')];
  const withRuns = runCount !== null && runCount !== undefined;
  const runs = withRuns ? [count(runCount, 'en'), count(runCount, 'he')] : null;
  if (scope === SCOPE_SELF) {
    const set = covered(body);
    if (withRuns) {
      return [
        `${changed[0]} changes and points were applied (${set[0]}), and ${runs[0]} runs were recorded.`,
        `בוצעו ${changed[1]} שינויים ונקודות (${set[1]}), ונרשמו ${runs[1]} הרצות.`,
      ];
    }
    return [
      `${changed[0]} changes and points were applied (${set[0]}).`,
      `בוצעו ${changed[1]} שינויים ונקודות (${set[1]}).`,
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
// that nothing changed: it names the set it covers, and then names what was
// actually kept out of that set and how much of it, both taken over the same
// window this sentence is printed over.
export function sinceEmptyLine(scope, body) {
  if (scope !== SCOPE_SELF) {
    return ['Nothing has changed since that day.', 'שום דבר לא השתנה מאז אותו יום.'];
  }
  const set = covered(body);
  const withheld = withheldLine((body || {}).scope_kinds);
  const opening = [
    `Nothing has changed since that day, among ${set[0]}.`,
    `שום דבר לא השתנה מאז אותו יום, מבין ${set[1]}.`,
  ];
  if (!withheld) return opening;
  return [`${opening[0]} ${withheld[0]}`, `${opening[1]} ${withheld[1]}`];
}
