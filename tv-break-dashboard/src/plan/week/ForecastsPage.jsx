import React from 'react';
import PlanWeek from './PlanWeek';

// The Forecasts entrance to Plan, the week.
//
// Comparing is a step of planning, so this entry lands on the comparison step of
// the one destination. The comparison it lands on is the whole point of the
// move: it now reports revenue net of retention cost, which is the quantity JS-2
// is defined on and the one the old panel printed "Not exposed" for.
//
// The top bar's own Compare button, which is the shell's and frozen, still
// navigates here. It now arrives at a working comparison instead of a page whose
// forecast table read "No forecast rows were found" until somebody ran the A/B.

export function ForecastsPage({ schedule, inventory, copy, locale, notify, planEvents, onGlobalRefresh }) {
  return (
    <PlanWeek
      entrance="Forecasts"
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

export default ForecastsPage;
