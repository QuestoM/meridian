import React from 'react';
import { vocabularyLabel, vocabularyMeaning } from './pacing-helpers';

// Split out of MakeGoodLedger.jsx, which passed the 450 line cap when the legend
// landed. Both halves are about the same thing: a controlled vocabulary a reader
// can read without a mouse.

// A chip's own reading: the word plus what it means, for anything that cannot
// hover. Falls back to the word alone when the vocabulary carries no meaning.
export function chipReading(entries, value, locale) {
  const label = vocabularyLabel(entries, value, locale);
  const meaning = vocabularyMeaning(entries, value, locale);
  return meaning ? `${label}: ${meaning}` : label;
}

// The controlled vocabulary, printed once instead of hidden behind every chip.
//
// These sets are two and three values long. A vocabulary that small belongs on
// the screen as a legend, which is what the trade expects and what a person on a
// touch screen can actually read; a title attribute on each chip is a meaning
// that exists only for a mouse.
export function Legend({ entries, locale }) {
  const known = (entries || []).filter((entry) => vocabularyMeaning(entries, entry.value, locale));
  if (!known.length) return null;
  return (
    <dl className="makegood-legend">
      {known.map((entry) => (
        <div key={entry.value}>
          <dt className={`makegood-kindmark ${entry.value}`}>{vocabularyLabel(entries, entry.value, locale)}</dt>
          <dd>{vocabularyMeaning(entries, entry.value, locale)}</dd>
        </div>
      ))}
    </dl>
  );
}
