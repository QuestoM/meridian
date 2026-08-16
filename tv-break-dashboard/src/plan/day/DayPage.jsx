import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Popover } from '@mui/material';
import { Button } from '../../studio/actions';
import { CircleHelp } from 'lucide-react';
import { pageText } from '../../shell/format';
import DayBoard from './DayBoard';
import DayPicker from './DayPicker';
import ScheduleInspector from './ScheduleInspector';
import './master-control-broadcast.css';
import BreakInspector from '../break/BreakInspector';
import { fetchDays } from './day-board-actions';
import './day-board.css';

function DayOperationsHeader({ locale }) {
  const [helpAnchor, setHelpAnchor] = useState(null);
  const helpOpen = Boolean(helpAnchor);
  return (
    <header className="broadcast-day-header">
      <h1>{pageText(locale, 'Broadcast day', 'יום שידור')}</h1>
      <Button
        type="button"
        variant="outlined"
        aria-haspopup="dialog"
        aria-expanded={helpOpen}
        onClick={(event) => setHelpAnchor(event.currentTarget)}
      >
        <CircleHelp size={16} aria-hidden="true" />
        {pageText(locale, 'Help', 'עזרה')}
      </Button>
      <Popover
        open={helpOpen}
        anchorEl={helpAnchor}
        onClose={() => setHelpAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'end' }}
        transformOrigin={{ vertical: 'top', horizontal: 'end' }}
      >
        <div className="broadcast-day-help">
          <h2>{pageText(locale, 'Editing the day', 'עריכת יום השידור')}</h2>
          <p>{pageText(locale, 'Select a break on the timeline to move it, change its duration, mark it gold, or open its record. Review the projected effect before saving.', 'בחרו ברייק בציר הזמן כדי להזיז אותו, לשנות את אורכו, לסמן אותו כזהב או לפתוח את הרשומה שלו. בדקו את ההשפעה הצפויה לפני השמירה.')}</p>
          <p>{pageText(locale, 'Arrow keys move the selected break by one snap unit; Shift moves five units and Alt moves one second. Up and down change duration. G marks gold and Enter opens the break.', 'מקשי החיצים מזיזים את הברייק הנבחר ביחידת הצמדה אחת; Shift מזיז חמש יחידות ו־Alt מזיז שנייה אחת. חיצי מעלה ומטה משנים את האורך. G מסמן זהב ו־Enter פותח את הברייק.')}</p>
        </div>
      </Popover>
    </header>
  );
}

// Plan, at day zoom. The scheduler's door.
//
// The whole job lives on one screen: pick the day, see it as a timeline, move a
// break, read what it cost, save it, undo it. Nothing on the path opens another
// page, and nothing on it opens a dialog.
//
// Two records open from the board and both open in place. A break opens its own
// drawer, and a programme band opens the programme record with that segment's
// class, break plan, economics and the decisions in force on it, which is the
// drawer the shipped editor already opened from a break chip. A board whose
// programmes could not be opened would be a step backwards from the surface
// this destination replaces.
//
// When there is no plan or no configured channel this is an honest empty state
// naming the missing input, never an empty grid that looks like a finished day.
function DayPage({ locale, notify, onGlobalRefresh, refreshKey }) {
  const [days, setDays] = useState(null);
  const [day, setDay] = useState('');
  const [error, setError] = useState('');
  const [openBreak, setOpenBreak] = useState(null);
  const [openProgramme, setOpenProgramme] = useState(null);
  // The day's breaks in board order, so the drawer can walk the set it was
  // opened from instead of sending the person back to the board between two.
  const [breakIds, setBreakIds] = useState([]);
  const returnFocusRef = useRef(null);

  useEffect(() => {
    let alive = true;
    fetchDays()
      .then((payload) => {
        if (!alive) return;
        setDays(payload);
        setDay((current) => current || (payload.days && payload.days.length ? payload.days[0] : ''));
      })
      .catch((fetchError) => { if (alive) setError(fetchError.message); });
    return () => { alive = false; };
  }, [refreshKey]);

  const onOpenBreak = useCallback((breakId) => {
    returnFocusRef.current = document.activeElement;
    setOpenProgramme(null);
    setOpenBreak(breakId);
  }, []);
  // A programme and a break are two records at two zooms, so one drawer is open
  // at a time and opening either puts the other away.
  const onOpenProgramme = useCallback((programme) => {
    returnFocusRef.current = document.activeElement;
    setOpenBreak(null);
    setOpenProgramme({ segmentId: programme.segment_id, channel: programme.channel, day: programme.day });
  }, []);

  const closeInspector = useCallback(() => {
    setOpenBreak(null);
    setOpenProgramme(null);
    window.setTimeout(() => returnFocusRef.current?.focus?.(), 0);
  }, []);
  const onDayLoaded = useCallback((payload) => {
    setBreakIds((payload.breaks || []).map((row) => row.break_id));
  }, []);

  if (error) {
    return (
      <section className="page-workspace broadcast-day">
        <DayOperationsHeader locale={locale} />
        <div className="day-board-empty">
          <h3>{pageText(locale, 'The day board is not reachable', 'לוח היום אינו זמין')}</h3>
          <p>{error}</p>
        </div>
      </section>
    );
  }

  if (days && !days.available) {
    return (
      <section className="page-workspace broadcast-day">
        <DayOperationsHeader locale={locale} />
        <div className="day-board-empty">
          <h3>{pageText(locale, 'There is no day to open yet', 'אין עדיין יום לפתיחה')}</h3>
          <p>{(locale === 'he' && days.reason_he) || days.reason}</p>
          <p>
            {pageText(
              locale,
              'Run the weekly plan, or set the channel this operator owns, and the day board fills itself.',
              'הריצו את התוכנית השבועית, או קבעו את הערוץ שבבעלות המפעיל, ולוח היום יתמלא מעצמו.',
            )}
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="page-workspace day-page broadcast-day">
      <DayOperationsHeader locale={locale} />
      <DayPicker
        days={days ? days.days : []}
        value={day}
        onChange={setDay}
        locale={locale}
        channel={days ? days.operator_channel : ''}
      />
      {day && (
        <DayBoard
          day={day}
          locale={locale}
          notify={notify}
          onGlobalRefresh={onGlobalRefresh}
          onOpenBreak={onOpenBreak}
          onOpenProgramme={onOpenProgramme}
          onDayLoaded={onDayLoaded}
        />
      )}
      {openBreak && (
        <BreakInspector
          breakId={openBreak}
          locale={locale}
          siblings={breakIds}
          onNavigate={setOpenBreak}
          onClose={closeInspector}
          notify={notify}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
      {openProgramme && (
        <ScheduleInspector
          segmentId={openProgramme.segmentId}
          channel={openProgramme.channel}
          day={openProgramme.day}
          locale={locale}
          notify={notify}
          onClose={closeInspector}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
    </section>
  );
}

export default DayPage;
