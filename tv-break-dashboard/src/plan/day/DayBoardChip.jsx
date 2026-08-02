import React from 'react';
import { Lock, Star } from 'lucide-react';
import { chipLabels, clockOf } from './day-board-model';
import './day-chip.css';

// One break on the day track. It is a real object you can select, drag, resize
// and address, not a coloured rectangle.
//
// Three things a professional editing timeline does and a schedule chip usually
// does not, all of them here: the exact duration is printed on the object rather
// than in a panel, the selected object is ringed as a whole rather than merely
// tinted, and the drag in flight names its own result so the operator knows
// whether they are moving the break or changing its length before they let go.
//
// The object prints only the numbers it is wide enough to print, and the badge
// carries the rest. A two minute break at the scale this board opens at is 12 px
// wide, which is four pixels of text once the padding and the border are taken,
// so printing the clock inside it produced the character 0 over the character 1
// on every chip at once. The badge is the drawing tool's own answer: the exact
// numbers are drawn beside the object, never squeezed into it, and they are there
// on hover and on selection, which are the two moments a person is asking.
//
// The clock is forced left to right inside a right to left page, because a time
// must never be mirrored.
//
// Gold and the saved placement are states, so they are in the name, not only in
// the glyph. Both glyphs are decorative and hidden, and while they were the only
// carriers a screen reader could not tell a pinned break from a free one, which
// is the difference between a position the plan chose and a position a person
// decided.
function DayBoardChip({
  item,
  live,
  startSeconds,
  selected,
  edited,
  saved,
  locale,
  style,
  widthPx,
  onSelect,
  onMovePointerDown,
  onResizePointerDown,
  onKeyDown,
  onOpen,
}) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const seconds = Math.round(live.durationSeconds);
  const clock = clockOf(startSeconds);
  const lengthText = `${seconds}s`;
  const fits = chipLabels(widthPx, clock, lengthText);
  const className = [
    'day-chip',
    selected ? 'is-selected' : '',
    edited ? 'is-edited' : '',
    saved ? 'is-saved' : '',
    live.isGold ? 'is-gold' : '',
  ].filter(Boolean).join(' ');
  const states = [];
  if (live.isGold) states.push(label('gold break', 'ברייק זהב'));
  if (saved) states.push(label('placement saved by the operator', 'נעיצה שמורה של המפעיל'));
  const identity = `${label('Break', 'ברייק')} ${item.ordinal} ${label('of', 'מתוך')} ${item.breaks_in_segment}, ${clock}, ${seconds} ${label('seconds', 'שניות')}, ${item.programme}`;
  const title = [`${clock} / ${lengthText} / ${item.programme}`, ...states].join(' / ');

  return (
    <div
      className={className}
      style={style}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={[identity, ...states].join(', ')}
      title={title}
      data-break-id={item.break_id}
      onPointerDown={(event) => onMovePointerDown(event, item)}
      onKeyDown={(event) => onKeyDown(event, item)}
      onFocus={() => onSelect(item.break_id)}
      onDoubleClick={(event) => {
        event.preventDefault();
        onOpen(item.break_id);
      }}
    >
      <span className="day-chip-body">
        {live.isGold && <Star className="day-chip-gold" size={11} aria-hidden="true" />}
        {saved && <Lock className="day-chip-lock" size={11} aria-hidden="true" />}
        {fits.clock && <span className="day-chip-clock" dir="ltr">{clock}</span>}
        {fits.length && <span className="day-chip-length" dir="ltr">{lengthText}</span>}
      </span>
      <span className="day-chip-readout" dir="ltr" aria-hidden="true">
        <span className="day-chip-readout-clock">{clock}</span>
        <span className="day-chip-readout-length">{lengthText}</span>
      </span>
      <i
        className="day-chip-resize"
        role="separator"
        aria-label={label('Change the length of this break', 'שינוי אורך הברייק')}
        onPointerDown={(event) => onResizePointerDown(event, item)}
      />
    </div>
  );
}

export default DayBoardChip;
