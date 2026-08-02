// The engine's basis, in the operator's words.
//
// The server states how it computed the money in its own field names, and those
// names are the engine's rather than a person's. Printing them on a Hebrew page
// is not a disclosure, it is a wall: measured on the Supply step, seven engine
// identifiers and one untranslated English paragraph reached the screen.
//
// So this module holds the words for the one basis the engine publishes today,
// and it refuses to speak for any other. The sentence renders only when the
// formula is character for character the one these words describe. If the engine
// changes it, the surface says it has no words for this basis and shows the
// engine's own text rather than a translation that has quietly stopped being
// true. Nothing is hidden either way: the engine's own line stays available.

// Measured live on GET /api/yield-per-second, 2026-08-01.
const KNOWN_FORMULA = 'retention_cost_ils = base_rate * baseline_tvr * (1 - retention_share) * (ad_seconds / unit_seconds); revenue_net_ils = revenue_ils - retention_cost_ils';

function normalize(value) {
  return String(value ?? '').replace(/\s+/g, ' ').trim();
}

export function basisFormula(basis) {
  return normalize(basis?.formula) || null;
}

// True when the engine states a basis this surface has no words for, which is a
// third state and not an absence: the figure is real, the words are missing.
export function basisIsUnfamiliar(basis) {
  const formula = basisFormula(basis);
  return Boolean(formula) && formula !== KNOWN_FORMULA;
}

export function yieldFormulaWords(basis, locale) {
  if (basisFormula(basis) !== KNOWN_FORMULA) return null;
  return locale === 'he'
    ? 'עלות השימור של ברייק היא מחיר נקודת רייטינג לשנייה לפי כרטיס התעריפים, כפול הרייטינג שנמדד לאותו ברייק, כפול שיעור הקהל שהברייק אינו משמר, כפול שניות הפרסום שהוא נושא. הנטו הוא ההכנסה הצפויה פחות העלות הזאת.'
    : 'The retention cost of a break is the rate card price of one rating point for one second, times the rating measured for that break, times the share of the audience the break does not keep, times the advertising seconds it carries. Net is expected revenue less that cost.';
}

export function unfamiliarBasisWords(locale) {
  return locale === 'he'
    ? 'המנוע מדווח על בסיס חישוב שאין לו כאן ניסוח בעברית, ולכן הוא מוצג למטה בניסוח של המנוע עצמו.'
    : 'The engine reports a basis this screen has no words for, so it is shown below in the wording the engine itself uses.';
}

// The band. Deliberately claims no interval width: the width is the model's and
// this sentence must stay true if the model publishes a different one.
export function bandWords(locale) {
  return locale === 'he'
    ? 'הטווח מתמחר מחדש את אותה תוכנית בשני קצות טווח אי-הוודאות שהמודל מפרסם לעלות הזאת: הערך הנמוך הוא הקצה המקל והגבוה הוא הקצה המחמיר.'
    : 'The band re-prices the same plan at both ends of the uncertainty interval the model publishes for that cost: the low figure is the forgiving end and the high one is the damaging end.';
}

// The engine's named inputs, passed through untouched for the person who is
// reconciling a figure against the code. Engine wording, so it renders inside
// the disclosure and never as the operator's own sentence.
export function basisInputs(basis) {
  const inputs = basis?.inputs;
  if (!inputs || typeof inputs !== 'object') return [];
  return Object.entries(inputs)
    .filter(([name, note]) => name && typeof note === 'string' && note.trim())
    .map(([name, note]) => ({ name, note: note.trim() }));
}
