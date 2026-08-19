import React from 'react';
import { Button } from '../../studio/actions';
import { Figure } from '../../shell/bidi';
import { pick } from './pacing-helpers';

// The strip above the two panels: which view is showing, and the command that
// re-reads both. It lives here rather than inside the workspace because the
// workspace is at its size law and because this strip is a whole idea on its
// own: two tabs, one command, and the reading they last returned.

export const BOARD = 'board';
export const LEDGER = 'ledger';

export default function PacingViews({ locale, view, setView, moveView, ledger, readAt, onReload }) {
  const open = ledger.status === 'ready'
    ? ledger.payload.open_count + ledger.payload.accepted_count
    : 0;
  return (
    // The tablist holds tabs and nothing else. Read again is a command and not a
    // view, and inside a role=tablist it made a reader count three tabs and find
    // two. It sits beside the list, in the same group.
    <div className="pacing-views">
      <nav className="pacing-view-tabs" role="tablist" aria-label={pick(locale, 'Pacing views', 'תצוגות קצב')} onKeyDown={moveView}>
        <Button type="button" role="tab" id="pacing-tab-board" aria-controls="pacing-panel-board" data-pacing-view={BOARD}
                tabIndex={view === BOARD ? 0 : -1} aria-selected={view === BOARD}
                className={view === BOARD ? 'active' : ''} onClick={() => setView(BOARD)}>
          {pick(locale, 'Campaign pacing', 'קצב אספקה של הקמפיינים')}
        </Button>
        <Button type="button" role="tab" id="pacing-tab-ledger" aria-controls="pacing-panel-ledger" data-pacing-view={LEDGER}
                tabIndex={view === LEDGER ? 0 : -1} aria-selected={view === LEDGER}
                className={view === LEDGER ? 'active' : ''} onClick={() => setView(LEDGER)}>
          {pick(locale, 'Decision ledger', 'ספר ההחלטות')}
          {open ? <Figure className="pacing-open-count">{open}</Figure> : null}
        </Button>
      </nav>
      {/* A command, not a third tab, and it says what it re-reads and when that
          reading happened. "Read again" beside two views is a question with no
          answer on screen: again since when, and again of what. Both are on the
          line now. */}
      <span className="pacing-read-state">
        {readAt ? <Figure>{pick(locale, `Read at ${readAt}`, `נקרא ב־${readAt}`)}</Figure> : null}
        <Button
          type="button"
          className="pacing-refresh"
          onClick={onReload}
          title={pick(
            locale,
            'Read the pacing board and the decision ledger again from the server',
            'קוראים מחדש מהשרת את לוח קצב האספקה ואת ספר ההחלטות',
          )}
        >
          {pick(locale, 'Read again', 'קראו שוב')}
        </Button>
      </span>
    </div>
  );
}
