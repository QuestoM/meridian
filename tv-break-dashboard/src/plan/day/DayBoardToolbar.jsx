import React, { useEffect, useState } from 'react';
import { Focus, Maximize, Pin, PinOff, Star } from 'lucide-react';
import { pageText } from '../../shell/format';
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
  const [lengthDraft, setLengthDraft] = useState('');
  const [startDraft, setStartDraft] = useState('');
  const startClock = clockOf((programme ? programme.start_seconds : 0) + (live ? live.offsetSeconds : 0));
  const savedPin = inversePlacement(selectedItem);

  useEffect(() => {
    setLengthDraft(live ? String(Math.round(live.durationSeconds)) : '');
    setStartDraft(startClock);
  }, [live, selectedItem, startClock]);

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
            <button
              key={choice}
              type="button"
              className={snapGrid === choice ? 'day-chip-button is-on' : 'day-chip-button'}
              aria-pressed={snapGrid === choice}
              onClick={() => onSnapGrid(choice)}
            >
              <span dir="ltr">{choice}s</span>
            </button>
          ))}
        </div>
        <ZoomControl pxPerMin={pxPerMin} onZoom={onZoom} onStep={onZoomStep} locale={locale} min={zoomFloor} />
        <div className="day-fit" role="group" aria-label={label('Frame the view', 'מסגור התצוגה')}>
          <button
            type="button"
            className="day-chip-button is-icon"
            onClick={onFitDay}
            aria-label={label('Fit the whole day on screen', 'התאמת כל היום למסך')}
            title={label('Fit the whole day on screen', 'התאמת כל היום למסך')}
          >
            <Maximize size={13} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="day-chip-button is-icon"
            onClick={onFitProgramme}
            disabled={!selectedItem}
            aria-label={label('Fit the programme of the selected break on screen', 'התאמת התוכנית של הברייק הנבחר למסך')}
            title={label('Fit the programme of the selected break on screen', 'התאמת התוכנית של הברייק הנבחר למסך')}
          >
            <Focus size={13} aria-hidden="true" />
          </button>
        </div>
        <span className="day-toolbar-basis" dir="auto">
          {label('Figures are for', 'הנתונים מתייחסים ל')} {board.operator_channel}, <span dir="ltr">{board.day}</span>
        </span>
      </div>

      {selectedItem && live ? (
        <div className="day-selection">
          <div className="day-selection-identity">
            <strong dir="auto">{selectedItem.programme}</strong>
            <span className="day-selection-ordinal">
              {label('Break', 'ברייק')} <span dir="ltr">{selectedItem.ordinal}</span> {label('of', 'מתוך')} <span dir="ltr">{selectedItem.breaks_in_segment}</span>
            </span>
          </div>
          <label className="day-field">
            <span>{label('Starts at', 'מתחיל ב')}</span>
            <input
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
            <input
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
          <button type="button" className={live.isGold ? 'day-chip-button is-on' : 'day-chip-button'} onClick={onGold}>
            <Star size={12} aria-hidden="true" />
            {label('Gold break', 'ברייק זהב')}
          </button>
          <button type="button" className="day-chip-button" onClick={onOpen}>
            {label('Open this break', 'פתיחת הברייק')}
          </button>
          {programme && (
            <span className="day-selection-scope" dir="auto">{scopeSentence(programme, locale, airingsBound(board.programmes, programme))}</span>
          )}
          {savedPin && (
            <div className="day-selection-pin">
              <span className="day-pin-state">
                <Pin size={12} aria-hidden="true" />
                {label('This break is pinned by a saved placement', 'הברייק הזה נעוץ על ידי נעיצה שמורה')}
              </span>
              <span className="day-pin-rule" dir="ltr" title={savedPin.savedAt}>{savedPin.constraintId || label('no restriction on record', 'אין מגבלה רשומה')}</span>
              <button type="button" className="day-chip-button is-inverse" onClick={onRemoveSaved} disabled={busy}>
                <PinOff size={12} aria-hidden="true" />
                {label('Remove the saved placement', 'הסרת הנעיצה השמורה')}
              </button>
              <span className="day-pin-note" dir="auto">
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
            'Click a break to select it. Arrow keys move it by one snap unit, Shift by five, Alt by one second. Up and down change its length. G marks it gold, Enter opens it. Its exact clock and length are on the break itself while the pointer is on it.',
            'לחצו על ברייק כדי לבחור אותו. מקשי החיצים מזיזים אותו ביחידת הצמדה אחת, Shift בחמש, Alt בשנייה אחת. חיצי מעלה ומטה משנים את אורכו. G מסמן זהב, Enter פותח אותו. השעה המדויקת והאורך מופיעים על הברייק עצמו כשהסמן עליו.',
          )}
          {label(
            ' The two buttons beside the zoom fit the whole day on screen, and fit the programme the selected break sits in.',
            ' שני הכפתורים ליד הזום מתאימים את כל היום למסך, ומתאימים את התוכנית שבה נמצא הברייק הנבחר.',
          )}
        </p>
      )}
    </div>
  );
}

export default DayBoardToolbar;
