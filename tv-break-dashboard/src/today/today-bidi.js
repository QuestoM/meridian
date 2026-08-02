// Bidi isolation for a name that arrives as data.
//
// The operator's channel name is Hebrew in this market and the interface ships
// an English toggle, so a Hebrew run lands inside an English sentence on every
// basis line this destination prints. Left unisolated, the bidi algorithm pulls
// the neighbouring digits and separators into that run: measured in the browser
// on the English Today screen, the basis line printed the window's start day
// torn off its date and moved to the far left, before the separator. The mirror
// case, a Latin name inside a Hebrew sentence, fails the same way.
//
// U+2068 is the first-strong isolate: it infers the run's direction from the
// run's own first strong character, so one call is correct for a Hebrew channel
// name and for a Latin one, and correct in both locales. U+2069 pops it. These
// are the characters rather than a bdi element because the strings they guard
// are joined and handed on as plain text, into a heading and a single span,
// where an element cannot go.
//
// Written as escapes on purpose. The characters render as nothing, so a literal
// pair in the source is invisible to review and to any editor that trims it.

const FIRST_STRONG_ISOLATE = '\u2068';
const POP_DIRECTIONAL_ISOLATE = '\u2069';

export function isolate(value) {
  const text = String(value ?? '').trim();
  return text ? `${FIRST_STRONG_ISOLATE}${text}${POP_DIRECTIONAL_ISOLATE}` : '';
}
