import React from 'react';
import { pageText } from '../shell/format';
import { Name } from '../shell/bidi';
import { moreProgrammesSentence } from './rules-lib';

// Which programme this restriction is about. A type-ahead rather than a full
// list, because the operator's channel carries 106 programme titles and a
// representative arrives knowing the name of one of them.
//
// It answers to the same rule the night picker does: a list that shows fewer
// things than matched says how many more there are and names the act that
// reaches them. For a type-ahead that act is the typing, so the line says so
// rather than offering a control that would do what the input already does.
// Without it the panel shows eight of a hundred and six with nothing on screen
// to say the other ninety-eight exist.
//
// Each row carries the two facts that decide whether this is the right
// programme: how many times it airs in the plan window and how many breaks the
// plan of record gives it. Both come from the airings route, so the picker and
// the preview cannot disagree about what a programme is.

const VISIBLE = 8;

export default function ProgrammeMatches({ locale, titles, matchCount, onPick }) {
  const list = titles || [];
  const shown = list.slice(0, VISIBLE);
  // The server's own count of what matched, which is larger than the list it
  // serves when the query is wide. Falling back to the list length keeps the
  // sentence honest rather than inventing a total nobody counted.
  const total = Number.isFinite(Number(matchCount)) && Number(matchCount) > 0
    ? Number(matchCount)
    : list.length;
  const hidden = Math.max(0, total - shown.length);
  return (
    <ul className="rules-suggestions">
      {shown.map((row) => (
        <li key={row.title}>
          <button type="button" onClick={() => onPick(row.title)}>
            <Name>{row.title}</Name>
            <small>
              {pageText(
                locale,
                `${row.airings} airings, ${row.planned_breaks} breaks planned`,
                `${row.airings} שידורים, ${row.planned_breaks} ברייקים בתוכנית`,
              )}
            </small>
          </button>
        </li>
      ))}
      {hidden > 0 && (
        <li className="rules-suggestions-more" role="status">
          {moreProgrammesSentence(locale, hidden)}
        </li>
      )}
    </ul>
  );
}
