import React from 'react';
import PlanWeek from './PlanWeek';

// The Optimizer entrance to Plan, the week.
//
// Discovery measured Optimizer and Schedule as duplicates: the same four tiles,
// the same compliance ledger with the same prop, the same frontier, and the same
// canvas, daypart and timeline views over the same schedule rows. Section 3.5 of
// the specification merges Optimizer into Plan, week, and section 3.3 makes zoom
// a control in the content rather than a second destination.
//
// The navigation list and the shell router are frozen for this run, so the entry
// stays and what it opens changes: this file is now the door, and the
// destination behind all four doors is one component. A planner arriving here
// lands on step one, the objective, because that is where their job starts.

export function OptimizerWorkspace({ schedule, inventory, copy, locale, notify, planEvents, onGlobalRefresh }) {
  return (
    <PlanWeek
      entrance="Optimizer"
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

export default OptimizerWorkspace;
