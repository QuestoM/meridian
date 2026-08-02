import React from 'react';
import HistoryPage from './HistoryPage';

// The destination the shell routes to. It keeps this module name because the
// shell's router and its navigation entry are frozen for this wave; the
// surface behind it is History, one timeline over every record of what changed,
// who changed it and how to put it back.
//
// What used to live here was a restore-point list alone. The list is still
// here, inside the timeline, with its diff, its rename, its manual point and
// its viewer write-lock, and it is now beside the changes, the runs and the
// restores it was always missing.
export default function VersionsPage({ locale, notify }) {
  return <HistoryPage locale={locale} notify={notify} />;
}
