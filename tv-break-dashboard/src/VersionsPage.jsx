import React, { useState } from 'react';
import { History } from 'lucide-react';
import { pageText } from './surface-helpers';
import AssistantVersions from './AssistantVersions';
import './assistant-console.css';

// Version management as its own full-width page instead of a cramped tab inside the
// assistant rail. It hosts the same AssistantVersions timeline, with room for the
// diffs and restore controls to breathe. A local tick reloads the list after a
// snapshot or restore.
export default function VersionsPage({ locale, notify }) {
  const [tick, setTick] = useState(0);
  return (
    <section className="page-workspace versions-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Version management', 'ניהול גרסאות')}</h1>
          <p>{pageText(locale, 'Review and roll back saved versions of the operating state. A version is captured automatically before every change and every restore, so nothing is ever lost.', 'עיון ושחזור של גרסאות שמורות של מצב התפעול. גרסה נלכדת אוטומטית לפני כל שינוי ולפני כל שחזור, כך ששום דבר לא אובד.')}</p>
        </div>
        <History size={18} />
      </div>
      <section className="page-panel versions-panel">
        <AssistantVersions locale={locale} notify={notify} reloadKey={tick} onChanged={() => setTick((value) => value + 1)} />
      </section>
    </section>
  );
}
