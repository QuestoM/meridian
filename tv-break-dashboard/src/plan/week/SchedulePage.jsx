import React from 'react';
import PlanWeek from './PlanWeek';

// The Schedule entrance to Plan, the week.
//
// It lands on the week board, which is where this entry always went, with the
// same four views over the same rows and the day editor as the next zoom level
// rather than a tab of its own page. The drag, its 30 and 60 second snap and the
// shared time scale are the day editor's own and are untouched.

export function SchedulePage({ schedule, inventory, copy, locale, notify, planEvents, onGlobalRefresh }) {
  return (
    <PlanWeek
      entrance="Schedule"
      schedule={schedule}
      inventory={inventory}
      copy={copy}
      locale={locale}
      notify={notify}
      planEvents={planEvents}
      onGlobalRefresh={onGlobalRefresh}
    />
  );
}

export default SchedulePage;
