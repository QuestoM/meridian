// Whose figures are these, asked once for the whole destination.
//
// The saved plan carries every channel in the market, because the retention
// model is measured against the lineup. So every read this screen makes arrives
// with its own scope disclosure: `summary.scope_channel` on the overview body
// and `scope_channel` on the yield body. Both are null until settings name the
// operator's own channel, and null there does not mean zero and does not mean
// still loading. It means the figures in that same payload are the whole
// market's. Printing them here would report three rivals' revenue as this
// operator's, which is a fabrication and a competitor breach at once.
//
// The endpoint behind the three answers already refuses on exactly this signal,
// and says so in `kairos_api/overview_api_today.py`. The blocks below those
// answers read two other endpoints, which disclose the same fact and withhold
// nothing, so the refusal has to be made here. It is made from one test, in one
// place, so this screen cannot refuse in one panel and print in the next.

// The one cause, and the one control that ends it. The same words the endpoint
// sends, kept here in both languages because the two payloads this module reads
// carry the disclosure without the sentence that explains it.
export const NO_CHANNEL = {
  reason: 'no_operator_channel',
  reason_en: "The operator's own channel has not been declared, and the saved plan carries every channel in the market. Nothing here can be reported as yours until one of them is.",
  reason_he: 'לא הוגדר הערוץ של המפעיל, והתוכנית השמורה כוללת את כל הערוצים בשוק. אי אפשר לדווח כאן על שום מספר כשלכם עד שייבחר אחד מהם.',
  needs_en: "Choose the operator's channel in settings.",
  needs_he: 'בחרו את ערוץ המפעיל בהגדרות.',
  opens: 'settings',
};

// The compact form a block prints when the panel above it has already carried
// the full reason. It is derived rather than written, so a second copy of the
// cause cannot drift away from the first: it is the reason's own first
// sentence, which names the missing input, and the control line that follows it
// names the path to supply it.
export function firstSentence(text) {
  const whole = String(text || '').trim();
  const stop = whole.indexOf('. ');
  return stop === -1 ? whole : whole.slice(0, stop + 1);
}

export function shortReason(locale) {
  return firstSentence(locale === 'he' ? NO_CHANNEL.reason_he : NO_CHANNEL.reason_en);
}

// Three states, never two. A read that has not landed is UNKNOWN, and a screen
// that has read nothing yet must not accuse settings of being unset.
export const ATTRIBUTED = 'attributed';
export const UNATTRIBUTED = 'unattributed';
export const UNKNOWN = 'unknown';

// The channel a payload was scoped to, empty when it was scoped to none.
export function scopeChannel(payload) {
  const value = payload && typeof payload === 'object' ? payload.scope_channel : null;
  return typeof value === 'string' && value.trim() ? value.trim() : '';
}

export function scopeState(payload, answered) {
  if (!answered) return UNKNOWN;
  return scopeChannel(payload) ? ATTRIBUTED : UNATTRIBUTED;
}

// The overview body's own version of the same question. Whether the read landed
// is decided by `source_counts`, which the shell's offline-shaped payload
// carries as null: that is the one field that tells an empty payload apart from
// a real answer that happens to be empty, and Today's fallback already uses it.
export function overviewScope(overview) {
  const body = overview && typeof overview === 'object' ? overview : {};
  const answered = body.source_counts !== null && body.source_counts !== undefined;
  return scopeState(body.summary, answered);
}

// True only when a block may print a figure as the operator's own.
export function attributed(state) {
  return state === ATTRIBUTED;
}

// True only when a block must refuse. Deliberately not the negation of the one
// above: UNKNOWN is neither, and collapsing the two is how a loading screen
// starts accusing settings.
export function unattributed(state) {
  return state === UNATTRIBUTED;
}
