import React from 'react';
import { Button } from '@mui/material';
import { Send } from 'lucide-react';
import { ZoomControl } from './schedule-track-view';

// The editor toolbar: snap grid, the run control and the shared zoom control.
// Pure presentation: every value and handler is passed in from the editor, so the
// drag, pin and run flow is unchanged. Split out to keep the editor module lean.
//
// The pin-scope selector was here and is gone. Its default was the whole
// broadcast date, which the restriction resolver matches against every segment on
// that date: measured on 2024-11-01 it bound 82 of 82, and one click of it cost
// 789,576.18 ILS, 74.3 per cent of the day. A saved move names the airing it was
// dragged on now, so there is nothing to choose, and what it binds is printed on
// the row that carries the Save button.
function ScheduleEditorToolbar({
  locale,
  snapGrid,
  onSnapGrid,
  recomputeState,
  onRecompute,
  pxPerMin,
  onZoom,
  onZoomStep,
}) {
  const he = locale === 'he';
  const label = (en, heText) => (he ? heText : en);
  return (
    <div className="schedule-editor-toolbar">
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
      <Button
        type="button"
        variant="outlined"
        className="run-button"
        disabled={recomputeState === 'running'}
        onClick={() => onRecompute && onRecompute()}
      >
        <Send size={14} />
        {recomputeState === 'running'
          ? label('Running the weekly plan', 'מריץ את הלוח השבועי')
          : label('Run the weekly plan', 'הרצת הלוח השבועי')}
      </Button>
      <ZoomControl pxPerMin={pxPerMin} onZoom={onZoom} onStep={onZoomStep} locale={locale} />
    </div>
  );
}

export default ScheduleEditorToolbar;
