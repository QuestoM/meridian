import { inverseOfRecord, inversePlacement } from './day-board-model';
import { clearGold, markGold, saveBreakPlacement, undoBreakPlacement } from './day-board-actions';

// Every act on the day board that changes a store, and the exact inverse of each.
//
// Split out of DayBoard.jsx under the 450-line law, and the seam is the right one
// anyway: below this line are the acts that write, above it is the geometry that
// does not. Each one takes what it needs by name and returns nothing but the
// promise, so the component keeps the state and this module keeps the sequence.
//
// All of them settle through the caller's ``settleAfter``, which holds the totals
// the day had, re-reads the day, and prints the difference beside the prediction
// that was on screen. The gold act was the exception, on the reasoning that it
// changes which breaks are premium rather than where they sit, so the day it
// re-reads is the whole answer. Measured on רשת 13 / 2024-11-01, that reasoning
// does not survive: the re-read day is the whole answer only to a person who
// memorised the number it replaced.

// Mark the selected break's programme gold, or take the mark off again.
//
// The count comes back measured on the plan as it now stands, so it is what the
// engine did and not what the act asked for. Measured before the route read it
// back: it answered four while the plan carried none.
//
// It settles like every other act here, and the reason is a measurement. On
// רשת 13 / 2024-11-01 one press of G on 001~1 moved the day from 1,062,669.88 to
// 1,028,205.58, which is 34,464.30 ILS and 3.24 per cent of the day, and took it
// from 80 breaks to 79. On screen afterwards there was no settlement panel at
// all, the readout had re-based itself on the fresh plan and printed a change of
// zero, and the only trace of the money was a toast counting gold breaks.
//
// It settles against no prediction. There is no cheap preview of a gold change
// the way there is of a move, and a prediction of zero would be an invented
// figure, so the panel prints the realised change alone and says why.
export async function applyGold({ item, live, pendingEditCount = 0, notify, settleAfter, rememberGold }) {
  if (pendingEditCount > 0) {
    notify(
      'Save or discard the pending placement changes before changing gold. Gold re-plans the programme and can replace those break ids.',
      'יש לשמור או לבטל את שינויי המיקום הממתינים לפני שינוי הזהב. זהב מתכנן את התוכנית מחדש ועלול להחליף את מזהי הברייקים האלה.',
    );
    return false;
  }
  await settleAfter('gold', null, async () => {
    if (live.isGold) {
      await clearGold(item.break_id);
    } else {
      const result = await markGold(item.break_id);
      if (result.breaks_marked > 0) {
        notify(
          `Gold applies to the programme, so ${result.breaks_marked} break(s) in ${item.programme} are now gold.`,
          `הזהב חל על התוכנית, ולכן ${result.breaks_marked} ברייקים ב-${item.programme} מסומנים כעת כזהב.`,
        );
      } else {
        notify(
          `The gold mark is stored, but the plan came back with no gold break in ${item.programme}. ${result.reason}`,
          `סימון הזהב נשמר, אך התוכנית חזרה בלי ברייק זהב ב-${item.programme}. ${result.reason_he}`,
        );
      }
    }
    // What the act did and which break it did it to, so the panel that prints
    // the cost can also offer the way back.
    rememberGold({ breakId: item.break_id, wasGold: Boolean(live.isGold) });
  });
  return true;
}

// The inverse of the gold act, taken on the break the act named.
//
// It is addressed by that id rather than by a chip, because a gold mark makes the
// engine plan the day again with the mark in force and the second plan is free to
// give the programme fewer breaks. Measured on רשת 13 / 2024-11-01: marking 001~4
// gold takes that programme from four breaks to three and the day from 80 to 79,
// and the id that stops existing is the one the act named, so there is no chip
// left to press. The route parses the programme out of the break id and never
// looks that id up in the plan, so the inverse holds anyway.
//
// And it is exact. Measured on the same day, both directions: mark then clear
// returns the day to 1,062,669.88 and to 80 breaks, to the cent and to the break.
export async function undoGold({ lastGold, settleAfter, forgetGold, notify }) {
  if (!lastGold) return;
  await settleAfter('undo', null, async () => {
    if (lastGold.wasGold) {
      await markGold(lastGold.breakId);
    } else {
      await clearGold(lastGold.breakId);
    }
    forgetGold();
    notify('The gold change came back off the plan.', 'שינוי הזהב הוסר מהתוכנית.');
  });
}

// Save every break the operator moved, one restriction each, scoped to its own
// airing. The records come back so the session can reverse exactly these.
export async function saveEditedBreaks({
  breaks,
  edits,
  programmes,
  liveOf,
  predicted,
  settleAfter,
  setEdits,
  pushHistory,
  notify,
}) {
  const edited = breaks.filter((item) => edits[item.break_id]);
  if (edited.length !== Object.keys(edits).length) {
    notify(
      'The plan changed and at least one edited break is no longer on this board. No placement was saved. Discard the stale edit and make it on the current plan.',
      'התוכנית השתנתה ולפחות ברייק אחד שנערך כבר אינו בלוח הזה. לא נשמר אף מיקום. יש לבטל את העריכה הישנה ולבצע אותה בתוכנית הנוכחית.',
    );
    return false;
  }
  if (!edited.length) return false;
  await settleAfter('save', predicted, async () => {
    const saved = [];
    for (const item of edited) {
      const programme = programmes.get(item.segment_id);
      if (!programme) continue;
      saved.push(await saveBreakPlacement({ item, programme, live: liveOf(item) }));
    }
    setEdits({});
    pushHistory({ type: 'save', records: saved });
    notify(
      `Saved ${saved.length} break placement(s). Each one is pinned to its own airing.`,
      `נשמרו ${saved.length} מיקומי ברייק. כל אחד נעוץ לשידור שלו בלבד.`,
    );
  });
  return true;
}

// The inverse of the save this browser tab remembers.
export async function undoSave({ lastSave, predicted, settleAfter, forgetAction, notify }) {
  if (!lastSave) return;
  await settleAfter('undo', predicted, async () => {
    for (const record of lastSave.records) {
      await undoBreakPlacement(record);
    }
    forgetAction(lastSave);
    notify('The saved placements were removed.', 'מיקומי הברייק שנשמרו הוסרו.');
  });
}

// The inverse of a save, performed on the break rather than on the session.
//
// The undo above reverses the save this browser tab remembers, and it is the
// right control while the tab is open. It was also the only one there was, and a
// reload emptied it. Measured before this existed, on רשת 13 / 2024-11-01: select
// 003~1, one ArrowRight, Save. The plan fell 25,400 ILS, the settlement panel
// offered to put it back, and pressing reload took that offer away for good,
// leaving the API and another destination as the only routes back from an act
// performed here.
//
// This one is read off the break the server served, so it survives a reload, a
// new tab and a different person. It settles exactly as the save did, which is
// what lets the operator read the money back rather than trust that it returned.
export async function removeSavedPlacement({ item, predicted, settleAfter, forgetRecord, notify }) {
  const inverse = inversePlacement(item);
  if (!inverse) return;
  await settleAfter('undo', predicted, async () => {
    await undoBreakPlacement(inverse);
    // A save this session remembers may have carried this break. Reversing it
    // twice is harmless at the API and dishonest on screen, so the record goes.
    forgetRecord(inverse.breakId);
    notify(
      'The saved placement was removed, and the plan places this break itself again.',
      'הנעיצה השמורה הוסרה, והתוכנית חוזרת למקם את הברייק בעצמה.',
    );
  });
}

// The same inverse for a record the board has no chip for.
//
// The control above lives on the selected break, so it can only be reached while
// the plan still carries a break with that id. A save can end that: measured on
// רשת 13 / 2024-11-01, pinning 001~2 re-planned its programme from four breaks to
// one, the day fell by 47,444.20 ILS, and the id the record names stopped
// existing. Nothing on the board bound the record, no chip rendered as saved, and
// after a reload the only routes back were another destination or the API, which
// is the exact failure the control above set out to close.
//
// So the route serves those records and this reverses one by its own two ids. It
// settles exactly as the save did, so the money comes back on screen rather than
// being taken on trust.
export async function removeUnboundPlacement({ record, predicted, settleAfter, forgetRecord, notify }) {
  const inverse = inverseOfRecord(record);
  if (!inverse) return;
  await settleAfter('undo', predicted, async () => {
    await undoBreakPlacement(inverse);
    forgetRecord(inverse.breakId);
    notify(
      'The saved placement was removed, and the plan places this programme itself again.',
      'הנעיצה השמורה הוסרה, והתוכנית חוזרת למקם את רצועת השידור הזו בעצמה.',
    );
  });
}
