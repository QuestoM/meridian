import React from 'react';
import { pageText } from '../shell/format';
import { dayLabel, nothingToSaveSentence, widerScopeSentence } from './rules-lib';

// What the composer says when the rule compiles to nothing, which is the state
// a night picker makes reachable and the state the save button cannot leave.
//
// A window rule derives one store row per airing that breaches it, so a night
// the plan already keeps clean produces no row and there is nothing to store.
// That is honest and it is also a dead end: the person came to protect a night,
// not to be told that tonight is fine, and the season finale is exactly the
// night a plan has not been run against yet.
//
// So the same sentence is priced once more with the night dropped. When the run
// as a whole does breach it, this says by how much and offers the wider rule,
// which is the one that can be saved. Both figures come from the preview route;
// nothing here counts anything itself.

export default function WiderScopeNote({ locale, night, wider, onWiden }) {
  const breaching = Number(wider?.compiled_rows || 0);
  const matched = Number(wider?.matched_airings || 0);
  const label = night ? dayLabel(night, locale) : '';
  return (
    <div className="rules-widen" role="status">
      <span className="rules-inline-note">{nothingToSaveSentence(locale, label)}</span>
      {breaching > 0 && (
        <>
          <span className="rules-inline-note">{widerScopeSentence(locale, breaching, matched)}</span>
          <button type="button" className="rules-widen-action" onClick={() => onWiden?.()}>
            {pageText(locale, 'Write it for every airing instead', 'כתיבה לכל השידורים במקום')}
          </button>
        </>
      )}
    </div>
  );
}
