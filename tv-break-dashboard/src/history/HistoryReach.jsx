import React from 'react';
import { Button } from '../studio/actions';
import { CalendarDays, ChevronDown } from 'lucide-react';
import { pageText } from '../shell/format';
import { InputControl } from '../studio/dom-controls';
import {
  CLEAR_FILTERS,
  DAYS_CLEAR,
  DAY_JUMP,
  EMPTY_WINDOW,
  FROM_CONTROL,
  FROM_HINT,
  NEWEST_CONTROL,
  OLDER_CONTROL,
  SEARCH_SCOPE,
  UNTIL_CONTROL,
  UNTIL_HINT,
  missedLine,
  olderLine,
  reachLine,
  reachState,
  recordStartLine,
} from './history-reach';
import { byActorControl, byActorLine, emptyPage } from './history-search';

// The two elements that make the record before this page reachable, and the one
// that says what the page does not hold. The words and the arithmetic are in
// history-reach.js so a test can execute them; these are the controls that carry
// them.
//
// The day window is two inclusive days rather than one, because the question a
// compliance owner arrives with is a day: who changed the retention floor on 20
// July. Both days are read in the broadcast zone, which is the zone the list is
// already grouped by, so the day a reader picks and the day heading they land on
// are the same day.

export function ReachDays({ locale, from, until, onDays }) {
  const set = Boolean(from || until);
  return (
    <div className="hist-days">
      <CalendarDays size={14} aria-hidden="true" />
      {/* Both dates below stay native date inputs: the browser's own picker
          is the point, MUI has no drop-in that reopens it the same way, and
          scripts/verify-direction-rules.mjs (frozen, outside this file's
          owned scope) budgets this file at exactly two literal dir attributes. */}
      <label className="hist-select" title={pageText(locale, FROM_HINT[0], FROM_HINT[1])}>
        <span>{pageText(locale, FROM_CONTROL[0], FROM_CONTROL[1])}</span>
        <InputControl
          type="date"
          value={from}
          max={until || undefined}
          dir="ltr"
          aria-label={pageText(locale, FROM_HINT[0], FROM_HINT[1])}
          onChange={(event) => onDays(event.target.value, until)}
        />
      </label>
      <label className="hist-select" title={pageText(locale, UNTIL_HINT[0], UNTIL_HINT[1])}>
        <span>{pageText(locale, UNTIL_CONTROL[0], UNTIL_CONTROL[1])}</span>
        <InputControl
          type="date"
          value={until}
          min={from || undefined}
          dir="ltr"
          aria-label={pageText(locale, UNTIL_HINT[0], UNTIL_HINT[1])}
          onChange={(event) => onDays(from, event.target.value)}
        />
      </label>
      {set ? (
        <Button type="button" variant="text" className="hist-link" onClick={() => onDays('', '')}>
          {pageText(locale, DAYS_CLEAR[0], DAYS_CLEAR[1])}
        </Button>
      ) : null}
    </div>
  );
}

// The provenance of the page itself: where the window sits in the matched set,
// how many entries are older than it, and the step that reaches them. Nothing is
// rendered while the page holds the whole matched set, because then there is
// nothing to disclose and nowhere further to go.
export function ReachPager({ locale, body, searching, onOlder, onNewest }) {
  const reach = reachState(body);
  if (!reach.windowed) return null;
  const position = reachLine(reach);
  const dropped = olderLine(reach);
  return (
    <>
      <span className="hist-reach">{pageText(locale, position[0], position[1])}</span>
      {reach.older ? (
        <span className="hist-reach warn">{pageText(locale, dropped[0], dropped[1])}</span>
      ) : null}
      {searching ? (
        <span className="hist-reach warn">{pageText(locale, SEARCH_SCOPE[0], SEARCH_SCOPE[1])}</span>
      ) : null}
      {reach.cursor ? (
        <Button type="button" variant="text" className="hist-link" onClick={() => onOlder(reach.cursor)}>
          <ChevronDown size={13} aria-hidden="true" />
          {pageText(locale, OLDER_CONTROL[0], OLDER_CONTROL[1])}
        </Button>
      ) : null}
      {reach.paged ? (
        <Button type="button" variant="text" className="hist-link" onClick={onNewest}>
          {pageText(locale, NEWEST_CONTROL[0], NEWEST_CONTROL[1])}
        </Button>
      ) : null}
    </>
  );
}

// Where the record itself starts, and what drops what came before it. Printed
// under every list rather than only under an empty one, because the answer does
// not change with the list: a full page of today's entries is just as silent
// about 20 July as an empty one, and only this line says which. Nothing renders
// while the start is unknown, because a start nobody can name is not evidence.
export function ReachStart({ locale, body }) {
  const line = recordStartLine(body);
  if (!line) return null;
  return <span className="hist-reach">{pageText(locale, line[0], line[1])}</span>;
}

// A link asked for one entry and this page does not hold it. The note names
// which of the four true reasons it is, and carries every control that can
// answer it: dropping the filters, widening the page, and going to the day the
// entry is on, which is the only one that reaches past the newest page at all.
export function ReachMissed({ locale, missed, points, limit, wide, day, onClear, onWide, onDay }) {
  const line = missedLine(missed, points, limit);
  return (
    <p className="hist-block" role="note">
      {pageText(locale, line[0], line[1])}
      {missed === 'filtered' ? (
        <Button type="button" variant="text" className="hist-link" onClick={onClear}>
          {pageText(locale, CLEAR_FILTERS[0], CLEAR_FILTERS[1])}
        </Button>
      ) : null}
      {missed === 'paged_out' && limit < wide ? (
        <Button type="button" variant="text" className="hist-link" onClick={onWide}>
          {pageText(locale, `Load ${wide}`, `טעינת ${wide}`)}
        </Button>
      ) : null}
      {day ? (
        <Button type="button" variant="text" className="hist-link" onClick={onDay}>
          {pageText(locale, `${DAY_JUMP[0]}, ${day}`, `${DAY_JUMP[1]}, ${day}`)}
        </Button>
      ) : null}
    </p>
  );
}

// The empty state a hand-set day window produces. Which of the five it is was
// decided in history-reach.js from the payload's own counts, so this renders the
// sentence and the controls that answer it: dropping the filters, which is what
// a narrowed kind or actor needs, and the way back out of the days, which is the
// only one that helps when the days themselves hold nothing.
export function ReachEmpty({ locale, empty, onClear, onNewest }) {
  const state = empty || {};
  const line = state.line || EMPTY_WINDOW;
  return (
    <>
      {pageText(locale, line[0], line[1])}
      {state.scope ? (
        <span className="hist-reach warn">{pageText(locale, SEARCH_SCOPE[0], SEARCH_SCOPE[1])}</span>
      ) : null}
      {state.clear ? (
        <Button type="button" variant="text" className="hist-link" onClick={onClear}>
          {pageText(locale, CLEAR_FILTERS[0], CLEAR_FILTERS[1])}
        </Button>
      ) : null}
      <Button type="button" variant="text" className="hist-link" onClick={onNewest}>
        {pageText(locale, NEWEST_CONTROL[0], NEWEST_CONTROL[1])}
      </Button>
    </>
  );
}

// The empty state a page with no day window produces, which is the state the
// landing page is in and so the one most readers meet. Which of the six it is
// was decided in history-search.js from the payload's own figures; this renders
// the sentence, the line that says how much of the record the search actually
// read, and every control that reaches the rest.
//
// The controls are in the order of what answers: the operator filter, which is
// served over the whole record; the wider page; the step older through it; the
// filters dropped; and the way back to the newest end. A reader who typed a
// colleague's name and got nothing is one click from all of their entries
// instead of one conclusion away from a false attestation.
export function ReachEmptyPage(props) {
  const { locale, body, kind, actor, needle, limit, wide } = props;
  const { onClear, onActor, onWide, onOlder, onNewest } = props;
  const state = emptyPage(body, { kind, actor, needle, limit, wide });
  const control = state.actor ? byActorControl(state.actor) : null;
  return (
    <>
      {pageText(locale, state.line[0], state.line[1])}
      {state.covers ? (
        <span className="hist-reach warn">{pageText(locale, state.covers[0], state.covers[1])}</span>
      ) : null}
      {control ? (
        <span className="hist-reach">{pageText(locale, ...byActorLine(state.actor))}</span>
      ) : null}
      {control ? (
        <Button type="button" variant="text" className="hist-link" onClick={() => onActor(state.actor)}>
          {pageText(locale, control[0], control[1])}
        </Button>
      ) : null}
      {state.wide ? (
        <Button type="button" variant="text" className="hist-link" onClick={onWide}>
          {pageText(locale, `Load ${wide}`, `טעינת ${wide}`)}
        </Button>
      ) : null}
      {state.older ? (
        <Button type="button" variant="text" className="hist-link" onClick={onOlder}>
          <ChevronDown size={13} aria-hidden="true" />
          {pageText(locale, OLDER_CONTROL[0], OLDER_CONTROL[1])}
        </Button>
      ) : null}
      {state.clear ? (
        <Button type="button" variant="text" className="hist-link" onClick={onClear}>
          {pageText(locale, CLEAR_FILTERS[0], CLEAR_FILTERS[1])}
        </Button>
      ) : null}
      {state.newest ? (
        <Button type="button" variant="text" className="hist-link" onClick={onNewest}>
          {pageText(locale, NEWEST_CONTROL[0], NEWEST_CONTROL[1])}
        </Button>
      ) : null}
    </>
  );
}
