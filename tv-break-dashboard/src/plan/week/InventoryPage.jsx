import React from 'react';
import PlanWeek from './PlanWeek';

// The Inventory entrance to Plan, the week.
//
// Sellable supply is a question a planner asks while planning, not a place they
// visit, so this entry lands on the supply view of the one destination. Section
// 3.5 merges Inventory into Plan, week for exactly that reason.

export function InventoryPage({ schedule, inventory, copy, locale, notify, planEvents, onGlobalRefresh }) {
  return (
    <PlanWeek
      entrance="Inventory"
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

export default InventoryPage;
