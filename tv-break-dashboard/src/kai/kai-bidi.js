// Bidi isolation for a value that arrives as data, where an element cannot go.
//
// Kai says the same sentence in two languages and drops live values into it: a
// file name, a server message, the open record's name. In this market those
// values are Hebrew and the interface ships an English toggle, so a Hebrew run
// keeps landing inside an English sentence. Left unisolated the bidirectional
// algorithm resolves the sentence's own punctuation as part of that run and the
// reader gets characters in an order nobody wrote.
//
// Measured in a browser, English reading, the shipped upload notice with the
// file name דוח (1).csv interpolated bare:
//
//     Uploaded )1( חוד.csv.
//
// The extension is torn off the name and printed after it, and the brackets
// mirror. With the isolate the name paints as one block and the sentence's full
// stop stays outside it. Same defect the approval card's provenance line had.
//
// On a surface that renders nodes the isolate is a <bdi> element, which is what
// the cards use. These characters are for the other sink: notify takes two
// plain strings and hands them to the shell's toast and activity feed, where no
// element can go. U+2068 is the first-strong isolate, so it infers the value's
// direction from the value's own first strong character and one call is right
// for a Hebrew name and for a Latin one, in both readings. U+2069 pops it.
//
// Written as escapes on purpose: the characters render as nothing, so a literal
// pair in the source is invisible to review and to any editor that trims it.

const FIRST_STRONG_ISOLATE = '\u2068';
const POP_DIRECTIONAL_ISOLATE = '\u2069';

export function isolate(value) {
  const text = String(value ?? '').trim();
  return text ? `${FIRST_STRONG_ISOLATE}${text}${POP_DIRECTIONAL_ISOLATE}` : '';
}
