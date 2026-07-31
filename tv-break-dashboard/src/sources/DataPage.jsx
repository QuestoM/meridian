import React, { useState } from 'react';
import { Button } from '@mui/material';
import { pageText } from '../shell/format';
import { PageHeader } from '../shell/primitives';
import UploadCenter from './UploadCenter';
import SourceFilesView from './SourceFilesView';
import ModelView from '../model/ModelView';

// The Data page has one identity and three jobs, each its own tab so no tab is a
// grab-bag: bring the source files in (Upload), check they are present and current
// (Source files), and read what the model learned and the rules that govern the
// plan (Model and parameters). One page header sits above the tabs; each tab is
// header-less content.
export function DataPage({ files, impact, parameters, overview, copy, locale, notify, onGlobalRefresh }) {
  const [dataTab, setDataTab] = useState('upload');
  const TABS = [
    ['upload', pageText(locale, 'Upload', 'העלאה')],
    ['sources', pageText(locale, 'Source files', 'קבצי מקור')],
    ['model', pageText(locale, 'Model and parameters', 'מודל ופרמטרים')],
  ];
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Data and model"
        titleHe="נתונים ומודל"
        bodyEn="Upload the source files, check they are present and current, and see what the model learned and the parameters that drive the plan."
        bodyHe="העלאת קבצי המקור, בדיקה שהם קיימים ומעודכנים, וצפייה במה שהמודל למד ובפרמטרים שמניעים את התוכנית."
      />
      <div className="surface-toolbar no-print">
        <div className="toolbar-left" role="tablist">
          {TABS.map(([key, label]) => (
            <Button
              key={key}
              className={dataTab === key ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              role="tab"
              aria-selected={dataTab === key}
              onClick={() => setDataTab(key)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>
      {dataTab === 'upload' ? (
        <UploadCenter copy={copy} locale={locale} notify={notify} onGlobalRefresh={onGlobalRefresh} embedded />
      ) : dataTab === 'sources' ? (
        <SourceFilesView files={files} overview={overview} locale={locale} />
      ) : (
        <ModelView impact={impact} parameters={parameters} locale={locale} />
      )}
    </section>
  );
}

export default DataPage;
