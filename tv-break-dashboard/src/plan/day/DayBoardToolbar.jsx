import React, { useEffect, useRef, useState } from 'react';
import { Button } from '../../studio/actions';
import { Focus, Maximize, Pin, PinOff, Star } from 'lucide-react';
import { Figure, Code, Name } from '../../shell/bidi';
import { pageText } from '../../shell/format';
import { InputControl, Pressable } from '../../studio/dom-controls';
import { Dialog } from '../../studio/modal';
import { ZoomControl } from './schedule-track-view';
import {
  MIN_DURATION_SECONDS,
  SNAP_CHOICES,
  airingsBound,
  clockOf,
  inversePlacement,
  parseClock,
} from './day-board-model';
import { scopeSentence } from './day-board-actions';

// The board's controls, and the selected break's exact numbers.
//
// Both the start and the length are fields you type into, not readouts you can
// only drag. That is the pair of devices a professional editing timeline has and
// a schedule editor normally does not: the playhead position is an editable
// numeric field rather than a display, and a duration is a target you can state.
// Stating either is often faster and always more exact than dragging to it, and
// a person who knows the clock time they want should never have to hunt for it
// with a pointer.
//
// The two framings beside the scale are the pair a drawing tool publishes, fit
// the whole thing and fit the selection, so a person who zoomed in to read one
// break is never more than one press from the day again. Both are computed from
// the span and the width on screen, so neither claims a fit it did not make.
//
// The scope sentence beside the selection is the answer to the question a
// scheduler asks before every save, in words rather than in a grammar: what
// else will this touch. It says the airing it will bind, and nothing wider.
//
// The inverse of a save is offered here, from the break itself, and it is read
// off the break the server served rather than off this session's undo stack.
// Nothing in this component reads that stack, so the control is there on a break
// that was pinned last month by somebody else exactly as it is on one pinned a
// second ago, which is the property the stack cannot have.
function DayBoardToolbar({
  board,
  locale,
  snapGrid,
  onSnapGrid,
  pxPerMin,
  onZoom,
  onZoomStep,
  zoomFloor,
  onFitDay,
  onFitProgramme,
  selectedItem,
  live,
  programme,
  busy,
  onLength,
  onStart,
  onGold,
  onOpen,
  onRemoveSaved,
}) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const absent = label('Not recorded', 'לא תועד');
  const [lengthDraft, setLengthDraft] = useState('');
  const [startDraft, setStartDraft] = useState('');
  const [pendingRemoval, setPendingRemoval] = useState(false);
  const cancelRemovalRef = useRef(null);
  const startClock = clockOf((programme ? programme.start_seconds : 0) + (live ? live.offsetSeconds : 0));
  const savedPin = inversePlacement(selectedItem);

  useEffect(() => {
    setLengthDraft(live ? String(Math.round(live.durationSeconds)) : '');
    setStartDraft(startClock);
  }, [live, selectedItem, startClock]);

  useEffect(() => {
    setPendingRemoval(false);
  }, [selectedItem?.break_id]);

  function commitLength() {
    const parsed = Number(lengthDraft);
    if (!Number.isFinite(parsed) || parsed < MIN_DURATION_SECONDS) {
      setLengthDraft(live ? String(Math.round(live.durationSeconds)) : '');
      return;
    }
    onLength(parsed);
  }

  function commitStart() {
    const parsed = parseClock(startDraft);
    if (parsed === null) {
      setStartDraft(startClock);
      return;
    }
    onStart(parsed);
  }

  return (
    <div className="day-toolbar">
      <div className="day-toolbar-row">
        <div className="day-snap" role="group" aria-label={label('Snap grid', 'רשת הצמדה')}>
          <span>{label('Snap', 'הצמדה')}</span>
          {SNAP_CHOICES.map((choice) => (
            <Pressable
              key={choice}
              type="button"
              className={snapGrid === choice ? 'day-chip-button is-on' : 'day-chip-button'}
              aria-pressed={snapGrid === choice}
              onClick={() => onSnapGrid(choice)}
            >
              <Figure>{choice}s</Figure>
            </Pressable>
          ))}
        </div>
        <ZoomControl pxPerMin={pxPerMin} onZoom={onZoom} onStep={onZoomStep} locale={locale} min={zoomFloor} />
        <div className="day-fit" role="group" aria-label={label('Frame the view', 'מסגור התצוגה')}>
          <Pressable
            type="button"
            className="day-chip-button is-icon"
            onClick={onFitDay}
            aria-label={label('Fit the whole day on screen', 'התאמת כל היום למסך')}
            title={label('Fit the whole day on screen', 'התאמת כל היום למסך')}
          >
            <Maximize size={13} aria-hidden="true" />
          </Pressable>
          <Pressable
            type="button"
            className="day-chip-button is-icon"
            onClick={onFitProgramme}
            disabled={!selectedItem}
            aria-label={label('Fit the programme of the selected break on screen', 'התאמת התוכנית של הברייק הנבחר למסך')}
            title={label('Fit the programme of the selected break on screen', 'התאמת התוכנית של הברייק הנבחר למסך')}
          >
            <Focus size={13} aria-hidden="true" />
          </Pressable>
        </div>
        <span className="day-toolbar-basis">
          {label('Figures are for', 'הנתונים מתייחסים ל')} {board.operator_channel}, <Figure>{board.day}</Figure>
        </span>
      </div>

      {selectedItem && live ? (
        <div className="day-selection">
          <div className="day-selection-identity">
            <strong><Name>{selectedItem.programme}</Name></strong>
            <span className="day-selection-ordinal">
              {label('Break', 'ברייק')} <Figure>{selectedItem.ordinal}</Figure> {label('of', 'מתוך')} <Figure>{selectedItem.breaks_in_segment}</Figure>
            </span>
          </div>
          <label className="day-field">
            <span>{label('Starts at', 'מתחיל ב')}</span>
            <InputControl
              type="text"
              inputMode="numeric"
              className="day-field-time"
              dir="ltr"
              aria-label={label('Start time of this break, hours minutes seconds', 'שעת ההתחלה של הברייק, שעות דקות שניות')}
              value={startDraft}
              onChange={(event) => setStartDraft(event.target.value)}
              onBlur={commitStart}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  commitStart();
                }
              }}
            />
          </label>
          <label className="day-field">
            <span>{label('Length in seconds', 'אורך בשניות')}</span>
            <InputControl
              type="number"
              min={MIN_DURATION_SECONDS}
              step={1}
              dir="ltr"
              value={lengthDraft}
              onChange={(event) => setLengthDraft(event.target.value)}
              onBlur={commitLength}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  commitLength();
                }
              }}
            />
          </label>
          <Pressable type="button" className={live.isGold ? 'day-chip-button is-on' : 'day-chip-button'} onClick={onGold}>
            <Star size={12} aria-hidden="true" />
            {label('Gold break', 'ברייק זהב')}
          </Pressable>
          <Pressable type="button" className="day-chip-button" onClick={onOpen}>
            {label('Open this break', 'פתיחת הברייק')}
          </Pressable>
          {programme && (
            <span className="day-selection-scope">{scopeSentence(programme, locale, airingsBound(board.programmes, programme))}</span>
          )}
          {savedPin && (
            <div className="day-selection-pin">
              <span className="day-pin-state">
                <Pin size={12} aria-hidden="true" />
                {label('This break is pinned by a saved placement', 'הברייק הזה נעוץ על ידי נעיצה שמורה')}
              </span>
              <Code className="day-pin-rule" title={savedPin.savedAt}>{savedPin.constraintId || label('no restriction on record', 'אין מגבלה רשומה')}</Code>
              <Pressable type="button" className="day-chip-button is-inverse" onClick={() => setPendingRemoval(true)} disabled={busy}>
                <PinOff size={12} aria-hidden="true" />
                {label('Remove the saved placement', 'הסרת הנעיצה השמורה')}
              </Pressable>
              <span className="day-pin-note">
                {label(
                  'Removing it deletes the restriction that carries it, and the plan places this break itself again.',
                  'ההסרה מוחקת את המגבלה שנושאת אותה, והתוכנית חוזרת למקם את הברייק בעצמה.',
                )}
              </span>
            </div>
          )}
        </div>
      ) : (
        <p className="day-selection is-empty">
          {label(
            'No break selected. Select a break on the timeline to edit it.',
            'לא נבחר ברייק. בחרו ברייק בציר הזמן כדי לערוך אותו.',
          )}
        </p>
      )}
      <Dialog
        open={Boolean(pendingRemoval && savedPin)}
        onClose={() => setPendingRemoval(false)}
        title={label('Confirm saved placement removal', 'אישור הסרת נעיצה שמורה')}
        description={label('Review the stored placement and consequence before deleting it.', 'בדקו את הנעיצה השמורה ואת ההשפעה לפני המחיקה.')}
        closeLabel={label('Close placement removal review', 'סגירת בדיקת הסרת הנעיצה')}
        initialFocusRef={cancelRemovalRef}
        dismissOnBackdrop={false}
        footer={<><Button ref={cancelRemovalRef} type="button" variant="outlined" onClick={() => setPendingRemoval(false)}>{label('Cancel', 'ביטול')}</Button><Button type="button" className="is-danger" variant="contained" disabled={busy} onClick={() => { setPendingRemoval(false); onRemoveSaved(); }}>{label('Remove saved placement', 'הסרת הנעיצה השמורה')}</Button></>}
      >
        <dl className="day-removal-ledger">
          <div><dt>{label('Stored record', 'רשומה שמורה')}</dt><dd><Name>{selectedItem?.programme || absent}</Name> · <Code>{selectedItem?.break_id || absent}</Code></dd></div>
          <div><dt>{label('Scope', 'היקף')}</dt><dd><Code>{[board.operator_channel, board.day, selectedItem?.break_id].filter(Boolean).join(' / ')}</Code></dd></div>
          <div><dt>{label('Restriction', 'מגבלה')}</dt><dd><Code>{savedPin?.constraintId || absent}</Code></dd></div>
          <div><dt>{label('Consequence', 'השפעה')}</dt><dd>{label('The saved placement and its restriction are deleted immediately. The engine re-plans this day and may place other breaks differently.', 'הנעיצה השמורה והמגבלה שלה נמחקות מיד. המנוע מתכנן את היום מחדש ועשוי למקם ברייקים אחרים אחרת.')}</dd></div>
        </dl>
      </Dialog>
    </div>
  );
}

export default DayBoardToolbar;
