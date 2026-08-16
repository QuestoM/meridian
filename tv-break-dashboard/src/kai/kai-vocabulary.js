// The extension is written out so this module is importable by node exactly as
// the bundler imports it, which is what lets a test execute the rename rather
// than grep for it.
import { word } from '../vocabulary.js';

// The product's one word for the activity, applied to prose the model wrote.
//
// Section 8.3 retires four words in both languages. Every other surface obeys
// that by importing its labels, because every other surface prints labels. Mabat
// prints sentences a model composed, so there is no label table to fix: the
// word arrives at render time or not at all.
//
// Measured on the stored assistant data on 2026-08-02: three proposal reasons
// and twelve saved answers carry חישוב מחדש, and two of those reasons render on
// the approval card directly under the approved label for the same act. A
// prompt rule already forbids the word (assistant_prompt.py rule 27, both
// languages, both words) and the word still arrived. That is the difference
// between a hope and a guard.
//
// So the record keeps the model's sentence exactly as written, in the stored
// thread and in the audit trail, and the surface renames the retired term to
// the product's own word for the same activity, in the language the retired
// term was written in. It is a rename of one word to its approved synonym,
// never a change of meaning, and the approved word is read from vocabulary.js
// rather than spelled here, because that file is the single source of a label.
//
// One honest limit, stated rather than hidden. A rename into free prose is
// lexical, so where a Hebrew sentence agreed with the retired noun it now
// disagrees with the approved one: נדרש חישוב מחדש becomes נדרש הרצה, and
// הרצה is feminine. The vocabulary holds no masculine word for the activity
// and inventing one would be a second vocabulary, so the trade is deliberate:
// the right word with the old agreement beats a retired word.
//
// Three things it deliberately leaves alone. The operator's own question, which
// is theirs and is shown back unchanged. An error string, which is a technical
// record that may name a route or a tool. And any occurrence inside an
// identifier or a path, so propose_recompute and /api/recompute survive to the
// character.

const RUN_EN = word('activity.run', 'en');
const RUN_HE = word('activity.run', 'he');

// The retired forms, and the approved reading of each in that form's own
// language. The English entries are the vocabulary's own noun inflected for the
// tense the retired word carried; a test pins every one of them to that noun,
// so the table and vocabulary.js cannot drift apart.
export const RENAMES = [
  { retired: /\b(?:recomputing|rebuilding)\b/gi, approved: 'running', locale: 'en' },
  { retired: /\b(?:recomputes|rebuilds)\b/gi, approved: 'runs', locale: 'en' },
  { retired: /\b(?:recomputed|rebuilt)\b/gi, approved: 'run', locale: 'en' },
  { retired: /\b(?:recompute|rebuild)\b/gi, approved: 'run', locale: 'en' },
  { retired: /חישוב מחדש|בנייה מחדש/g, approved: RUN_HE, locale: 'he' },
];

// The exact pattern section 8.3 names, for a reader that wants to ask whether a
// string still carries one of the four.
export const RETIRED = /recompute|rebuild|חישוב מחדש|בנייה מחדש/i;

export const APPROVED_ACTIVITY = { en: RUN_EN, he: RUN_HE };

// A retired word that sits inside an identifier or a path is a name, not a
// word. The character before may be a separator or a dot; the character after
// may not be a dot, because a sentence ends in one.
const NAME_BEFORE = /[/_\-.]/;
const NAME_AFTER = /[/_]/;

function matchCase(match, approved) {
  const first = match.charAt(0);
  if (first && first === first.toUpperCase() && first !== first.toLowerCase()) {
    return approved.charAt(0).toUpperCase() + approved.slice(1);
  }
  return approved;
}

export function inApprovedWords(text) {
  if (text === null || text === undefined) return text;
  let value = String(text);
  for (const rename of RENAMES) {
    value = value.replace(rename.retired, (match, offset, whole) => {
      const before = offset > 0 ? whole.charAt(offset - 1) : '';
      const after = whole.charAt(offset + match.length);
      if ((before && NAME_BEFORE.test(before)) || (after && NAME_AFTER.test(after))) return match;
      return matchCase(match, rename.approved);
    });
  }
  return value;
}

export default inApprovedWords;
