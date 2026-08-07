// Which part of the record this reader is shown, in words, built from the
// payload's own scope evidence and never from a set somebody typed.
//
// The measured defect this file exists to end. Four rounds in a row fixed one
// sentence and left the next one wrong, because each sentence carried its own
// hardcoded idea of what a self-scoped reader is shown and what is kept from
// them. The fifth round shared one constant between two of them, and that was
// worse: one phrase was attestation-scoped, where the covered set really is
// change, restore and restore_point, and the other was page-scoped, where six
// kinds render, so the footer contradicted itself two spans apart.
//
// Measured on a real store on 2026-08-07, an admin and a self-scoped operator
// reading the same store in the same second: the operator was withheld 2,533
// changes, 636 previews and 2,586 sign-ins, and not one restore, restore point
// or run, while the strip told them that only changes were withheld. The
// exclusion clause was false by 3,222 entries, and a compliance owner told that
// only changes are withheld reads the three sign-ins beside it as the sign-in
// record and attests to it.
//
// So no sentence here names a kind of its own. The endpoint publishes which
// kinds its account filter narrows (`scope_kinds.account`), which reach every
// reader whole (`scope_kinds.shared`) and what this read actually removed
// (`scope_kinds.withheld`, counted per kind over the window it is printed
// over), and every phrase below is assembled from those three. A kind on
// neither list, which today is `run` because it answers to the competitor
// boundary instead, is named by neither phrase: `run_scope` states that scope
// in its own sentence beside them.
export const SCOPE_SELF = 'self';

// The plural word for each kind, in English, in Hebrew, and in the Hebrew
// definite form the covered phrase needs. The three the request recorder holds
// read exactly as `changesSourceLine` already prints them, so one destination
// has one word for one thing.
const KIND_WORDS = {
  change: ['changes', 'שינויים', 'השינויים'],
  preview: ['previews', 'תצוגות מקדימות', 'התצוגות המקדימות'],
  run: ['runs', 'הרצות', 'ההרצות'],
  restore_point: ['restore points', 'נקודות שחזור', 'נקודות השחזור'],
  restore: ['restores', 'שחזורים', 'השחזורים'],
  sign_in: ['sign-ins', 'כניסות', 'הכניסות'],
};

// The order kinds are read out in: the ones shared with every account first,
// then the ones scoped to this account, which is the order the round-14 fix
// established for a reason that still holds. A component that can be zero on
// its own is never the word a reader meets first.
const KIND_ORDER = ['restore_point', 'restore', 'run', 'change', 'preview', 'sign_in'];

// A true sentence for a payload that carries no kind list at all. It names no
// set, because a set that cannot be read from the payload cannot be claimed.
const COVERED_UNKNOWN = ['the part of the record this account may read',
                         'החלק ברישום שהחשבון הזה רשאי לקרוא'];

function count(value, locale) {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB');
}

// One list of kinds as one phrase. Column 0 is English, 1 is Hebrew and 2 is
// the Hebrew definite form.
function joinKinds(kinds, column) {
  const words = kinds.map((kind) => KIND_WORDS[kind][column]);
  if (words.length < 2) return words[0] || '';
  const last = words[words.length - 1];
  const head = words.slice(0, -1).join(', ');
  return column === 0 ? `${head} and ${last}` : `${head} ו${last}`;
}

// Which of these kinds this reader is shown whole and which are their own
// slice, taken from the endpoint's own rule. A kind the rule puts on neither
// list is on neither of these, so no phrase built here can claim it.
export function coveredSets(kinds, scopeKinds) {
  const rule = scopeKinds || {};
  const shared = new Set(rule.shared || []);
  const account = new Set(rule.account || []);
  const named = (kinds || []).filter((kind) => KIND_WORDS[kind]);
  return {
    shared: KIND_ORDER.filter((kind) => named.includes(kind) && shared.has(kind)),
    own: KIND_ORDER.filter((kind) => named.includes(kind) && account.has(kind)),
  };
}

// The covered set as a phrase, over whichever kinds the caller is describing:
// the attested three on the strip, the rendered six in the page footer. Both
// call this and neither owns the words, which is what stops them drifting.
export function coveredPhrase(kinds, scopeKinds) {
  const sets = coveredSets(kinds, scopeKinds);
  const shared = [joinKinds(sets.shared, 0), joinKinds(sets.shared, 2)];
  const own = [joinKinds(sets.own, 0), joinKinds(sets.own, 2)];
  if (!shared[0] && !own[0]) return COVERED_UNKNOWN;
  if (!own[0]) return [`all ${shared[0]} on record`, `כל ${shared[1]} שברישום`];
  if (!shared[0]) return [`your own ${own[0]}`, `${own[1]} שלכם`];
  return [`all ${shared[0]} on record, plus your own ${own[0]}`,
          `כל ${shared[1]} שברישום, בתוספת ${own[1]} שלכם`];
}

// What this reader is not shown, named from what the read actually removed and
// counted. A reader the endpoint withheld nothing from is told that, because a
// clause that names an exclusion set and a clause that names none are two
// different answers and only one of them is true at a time.
export function withheldLine(scopeKinds) {
  const rule = scopeKinds || {};
  if (String(rule.rule || '') !== SCOPE_SELF) return null;
  const removed = rule.withheld || {};
  const kinds = KIND_ORDER.filter((kind) => KIND_WORDS[kind] && Number(removed[kind] || 0) > 0);
  if (!kinds.length) {
    return ['Nothing by another account is withheld here.', 'שום דבר של חשבונות אחרים אינו מוסתר כאן.'];
  }
  const total = [count(rule.withheld_total, 'en'), count(rule.withheld_total, 'he')];
  return [`Only ${joinKinds(kinds, 0)} by other accounts are withheld here, ${total[0]} of them.`,
          `רק ${joinKinds(kinds, 1)} של חשבונות אחרים אינם מוצגים כאן, ${total[1]} מהם.`];
}

// The page footer's own sentence, over the kinds the list actually renders.
// It reads the rule from the attestation the same read already carries, and
// never the counts beside it: those are taken over the attested window and this
// sentence is about the page.
export function pageCoveredLine(body) {
  const payload = body || {};
  const scope = (payload.attestation || {}).scope_kinds || null;
  const covered = coveredPhrase(payload.kinds, scope);
  return [`You see ${covered[0]}.`, `מוצגים ${covered[1]}.`];
}
