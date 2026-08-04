import React from 'react';
import { X } from 'lucide-react';
import DayBoardSettlement from './DayBoardSettlement';
import ScheduleEditorMoney from './ScheduleEditorMoney';
import ScheduleEditorRow from './ScheduleEditorRow';

// The editor's readout: what the day is worth, every break the operator has
// moved and not yet saved, and what the last save actually did.
//
// Split out of ScheduleEditor.jsx under the 450-line law when the save path grew
// the scope line. The seam is the right one anyway: above it is the track and the
// gesture, below it is the money and the acts. Every value is passed in; this
// component decides nothing.
//
// The three money elements are the day board's own, imported rather than
// rebuilt: the scored figures and the pre-save check in ScheduleEditorMoney, and
// the settlement panel with its Undo this save control straight from the board.
// Before this the panel held zero currency figures while one press of its own
// Save button moved the day by 25,399.88 ILS, measured on
// רשת 13 / 2024-11-01, with no route back from the surface that spent it.

function editorText(locale, en, he) {
  return locale === 'he' ? he : en;
}

function ScheduleEditorReadout({
  lanes,
  edits,
  savingPin,
  locale,
  stateOf,
  pinnedFor,
  scopeFor,
  onSave,
  onDiscard,
  money,
}) {
  const he = locale === 'he';
  return (
    <div className="schedule-editor-readout" dir={he ? 'rtl' : 'ltr'}>
      <ScheduleEditorMoney money={money} locale={locale} editCount={Object.keys(edits).length} />
      <DayBoardSettlement
        settlement={money.settlement}
        locale={locale}
        canUndo={money.canUndo}
        onUndo={money.undoLastSave}
        onDismiss={money.dismiss}
      />
      {Object.keys(edits).length === 0 ? (
        <p>{editorText(locale, 'Drag a break to set its offset, then save it as a pin.', 'גררו ברייק כדי לקבוע את ההיסט שלו, ואז שמרו אותו כנעיצה.')}</p>
      ) : (
        <ul className="schedule-editor-edit-list">
          {lanes.flatMap((lane) => lane.items.filter((item) => edits[item.id]).map((item) => {
            const { startSec, durationSec } = stateOf(item);
            const offsetSeconds = Math.max(0, startSec - item.program_start_sec);
            return (
              <React.Fragment key={item.id}>
                <ScheduleEditorRow
                  item={item}
                  startSec={startSec}
                  durationSec={durationSec}
                  offsetSeconds={offsetSeconds}
                  pinned={pinnedFor(item)}
                  saving={savingPin === item.id}
                  locale={locale}
                  scope={scopeFor(item, startSec, durationSec)}
                  onSave={() => onSave(item)}
                />
                <li className="schedule-editor-discard-row" style={{ listStyle: 'none', display: 'flex', justifyContent: 'flex-end', margin: '2px 0 10px' }}>
                  <button
                    type="button"
                    onClick={() => onDiscard(item)}
                    disabled={savingPin === item.id}
                    aria-label={editorText(locale, `Discard the unsaved change to ${item.program_title}`, `ביטול השינוי שלא נשמר בתוכנית ${item.program_title}`)}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'transparent', border: 'none', cursor: 'pointer', color: 'inherit', opacity: 0.75, fontSize: 12, padding: '2px 4px' }}
                  >
                    <X size={12} aria-hidden="true" />
                    {editorText(locale, 'Discard change', 'ביטול השינוי')}
                  </button>
                </li>
              </React.Fragment>
            );
          }))}
        </ul>
      )}
    </div>
  );
}

export default ScheduleEditorReadout;
