// A before and an after, printed as one value that moved.
//
// This market's money is formatted by Intl as he-IL, and that format puts a
// right-to-left mark in front of the digits and another in front of the shekel
// sign: "11,155,641 ₪" is really U+200F, the digits, a no-break space, U+200F
// and U+20AA. Each side of the pair is therefore a strong right-to-left run and
// the arrow between them is a neutral. Setting dir="ltr" on the container does
// not save it: the bidirectional algorithm resolves a neutral sitting between
// two right-to-left runs as right-to-left too, welds all three into one run and
// paints it from the right. Measured in a browser with Range rects on the
// shipped rate card, the before value's box started at x=783 and the after
// value's at x=636, so the after painted 147 px to the LEFT of the before and
// "11,155,641 ₪ to 11,083,016 ₪" reached the reader as the two figures swapped.
// The aria-label was correct throughout, so a screen reader heard a fall while
// the eye read a rise.
//
// The fix is an isolate, which is a container in the text itself rather than an
// attribute on an element: the run inside it cannot merge with anything outside
// it, so the arrow stays neutral and the pair keeps the order it was written
// in. The mark is the first-strong isolate rather than the left-to-right one,
// because each side is a Hebrew-formatted figure whose direction is its own.
// Measured the same way, the left-to-right isolate fixes the order but pushes
// the shekel sign hard against the last digit and strands its space at the far
// end; the first-strong isolate gives each figure the direction Intl formatted
// it for, so it reads "₪ 11,155,641" as Hebrew ordinarily renders money, with
// the pair still running before then after.
// The isolate itself is not defined here. It is the shared one from
// shell/bidi.jsx, which is the single home for direction and isolation across
// the dashboard and uses the same first-strong marks this file's note argues
// for. Re-exported so this module's existing callers keep their import.
import { isolate } from '../shell/bidi';

export { isolate };

// The arrow stays direction-neutral on purpose. It replaced the word "to",
// which was English on a Hebrew screen, and the spoken form is supplied
// separately in the reader's own language by pairLabel.
export function valuePair(before, after) {
  return `${isolate(before)} → ${isolate(after)}`;
}
