import React from 'react';
import SourcesPage from './SourcesPage';

// One of the two doors into Sources. This one opens on the inputs: the state
// of every file a run reads, which is the data steward's first screen.
//
// The model and parameter tables this page used to carry are training output.
// They belong to the model console on the company side, where a verdict can be
// read beside the evidence for it, and they are not on an operator surface any
// more. What an operator needs from the model is on the inputs view: which
// version the plan's numbers rest on, and whether the sources it was measured
// on still match the files on disk.
export function DataPage({ files, overview, locale, notify, onGlobalRefresh }) {
  return (
    <SourcesPage
      view="inputs"
      files={files}
      overview={overview}
      locale={locale}
      notify={notify}
      onGlobalRefresh={onGlobalRefresh}
    />
  );
}

export default DataPage;
