import React from 'react';
import SourcesPage from './SourcesPage';

// The second door into Sources, opening on the reports. It is the same
// destination the inputs open on, with the view the entry names selected, so
// nothing that was one click away is two clicks away now.
export function ReportsPage({ reports, files, overview, locale, notify }) {
  return (
    <SourcesPage
      view="downloads"
      reports={reports}
      files={files}
      overview={overview}
      locale={locale}
      notify={notify}
    />
  );
}

export default ReportsPage;
