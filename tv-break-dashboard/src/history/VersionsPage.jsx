import React, { useState } from 'react';
import { History } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import AssistantVersions from './AssistantVersions';
import '../kai/assistant-console.css';

// Change restore as its own full-width page instead of a cramped tab inside the
// assistant rail. It hosts the AssistantVersions timeline of restore points, with
// room for the diffs and restore controls to breathe. A local tick reloads the list
// after a save or restore.
export default function VersionsPage({ locale, notify }) {
  const [tick, setTick] = useState(0);
  return (
    <section className="page-workspace versions-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Restore changes', 'שחזור שינויים')}</h1>
          <p>{pageText(locale, 'See every change made to the operating state and roll back to any earlier point. A restore point is saved automatically before every change and every restore, so nothing is ever lost.', 'כאן אפשר לראות כל שינוי שנעשה במצב התפעול ולחזור לכל נקודה קודמת. נקודת שחזור נשמרת אוטומטית לפני כל שינוי ולפני כל שחזור, כך ששום דבר לא אובד.')}</p>
        </div>
        <History size={18} />
      </div>
      <section className="page-panel versions-panel">
        <AssistantVersions locale={locale} notify={notify} reloadKey={tick} onChanged={() => setTick((value) => value + 1)} />
      </section>
    </section>
  );
}
