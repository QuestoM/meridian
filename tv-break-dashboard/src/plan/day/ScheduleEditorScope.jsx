import React from 'react';
import { pageText } from '../../shell/format';
import { coverageSentence } from './schedule-editor-scope';
import './schedule-editor.css';

// The one honest sentence the timeline was missing: what it draws against what
// the day and the plan actually hold, with a live route to the surface that
// holds the rest. See schedule-editor-scope.js for the measurement this closes.
function ScheduleEditorScope({ coverage, locale }) {
  return (
    <p className="schedule-editor-scope" dir="auto">
      <span>{coverageSentence(coverage, locale)}</span>
      <a href="#Overrides">{pageText(locale, 'Open the full day board', 'פתחו את לוח היום המלא')}</a>
    </p>
  );
}

export default ScheduleEditorScope;
