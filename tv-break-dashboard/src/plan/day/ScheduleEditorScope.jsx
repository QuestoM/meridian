import React from 'react';
import { pageText } from '../../shell/format';
import { coverageSentence, routeSentence } from './schedule-editor-scope';
import { SAVED_PLAN, drawnOfPlannedText, planBasisLabel } from './plan-basis';
import './schedule-editor.css';

// The one honest sentence the timeline was missing: what it draws against what
// the saved weekly plan actually holds, and what is really behind the link.
// See schedule-editor-scope.js for the four rounds of measurement this closes.
//
// The link keeps the destination's own name beside the description, because the
// rail calls that entry Overrides and a person who follows "the full day board"
// should recognise where they land.
function ScheduleEditorScope({ coverage, locale }) {
  return (
    <p className="schedule-editor-scope">
      <span>{coverageSentence(coverage, locale)}</span>
      <span className="schedule-editor-route">{routeSentence(coverage, locale)}</span>
      <a href="#Overrides">{pageText(locale, 'Open the full day board (Overrides)', 'פתחו את לוח היום המלא (עקיפות)')}</a>
    </p>
  );
}

// A lane's own count, drawn against what the saved plan places in that lane's
// programmes, with the plan named under it.
//
// A bare count here read as a claim that the lane holds that many breaks. It
// holds what the capped board drew, and the plan places more. Both numbers, and
// the file they come from, or the honest drawn-only form while the plan's own
// counts are still being read.
export function LaneCount({ shown, planned, locale }) {
  return (
    <span className="timeline-lane-count">
      <span>{drawnOfPlannedText(shown, planned, locale)}</span>
      <small className="timeline-lane-basis">{planBasisLabel(SAVED_PLAN, locale)}</small>
    </span>
  );
}

export default ScheduleEditorScope;
