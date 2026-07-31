import React from 'react';
import { Button, MenuItem, Select } from '@mui/material';
import { Send } from 'lucide-react';
import { ZoomControl } from './schedule-track-view';

// The editor toolbar: snap grid, pin scope, recompute and the shared zoom
// control. Pure presentation: every value and handler is passed in from the
// editor, so the drag, pin and recompute flow is unchanged. Split out to keep
// the editor module lean.
function ScheduleEditorToolbar({
  locale,
  snapGrid,
  onSnapGrid,
  scopeChoice,
  onScopeChoice,
  recomputeState,
  onRecompute,
  pxPerMin,
  onZoom,
  onZoomStep,
}) {
  const he = locale === 'he';
  const label = (en, heText) => (he ? heText : en);
  return (
    <div className="schedule-editor-toolbar" dir={he ? 'rtl' : 'ltr'}>
      <div className="schedule-editor-snap" role="group" aria-label={label('Snap grid', 'רשת הצמדה')}>
        <span>{label('Snap', 'הצמדה')}</span>
        <Button
          type="button"
          variant="outlined"
          className={snapGrid === 30 ? 'segmented active' : 'segmented'}
          aria-pressed={snapGrid === 30}
          onClick={() => onSnapGrid(30)}
        >
          30s
        </Button>
        <Button
          type="button"
          variant="outlined"
          className={snapGrid === 60 ? 'segmented active' : 'segmented'}
          aria-pressed={snapGrid === 60}
          onClick={() => onSnapGrid(60)}
        >
          60s
        </Button>
      </div>
      <div className="schedule-editor-scope">
        <span>{label('Pin scope', 'היקף הנעיצה')}</span>
        <Select
          size="small"
          value={scopeChoice}
          onChange={(event) => onScopeChoice(event.target.value)}
          aria-label={label('Pin scope', 'היקף הנעיצה')}
        >
          <MenuItem value="date">{label('This date', 'תאריך זה')}</MenuItem>
          <MenuItem value="programme">{label('Every airing of this programme', 'כל שידור של התוכנית')}</MenuItem>
        </Select>
      </div>
      <Button
        type="button"
        variant="outlined"
        className="run-button"
        disabled={recomputeState === 'running'}
        onClick={() => onRecompute && onRecompute()}
      >
        <Send size={14} />
        {recomputeState === 'running'
          ? label('Recomputing...', 'מחשב מחדש...')
          : label('Recompute weekly schedule', 'חישוב מחדש של הלוח השבועי')}
      </Button>
      <ZoomControl pxPerMin={pxPerMin} onZoom={onZoom} onStep={onZoomStep} locale={locale} />
    </div>
  );
}

export default ScheduleEditorToolbar;
