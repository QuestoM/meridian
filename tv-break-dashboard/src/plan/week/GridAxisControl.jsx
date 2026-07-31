import React from 'react';
import { Button } from '@mui/material';
import { pageText } from '../../shell/format';
import { gridAxisLabel } from '../../shell/labels';

export function GridAxisControl({ value, onChange, locale }) {
  const options = ['day', 'daypart', 'hour', 'type'];
  return (
    <div className="axis-control" aria-label={pageText(locale, 'Grid split', 'חלוקת גריד')}>
      {options.map((axis) => (
        <Button
          key={axis}
          className={value === axis ? 'axis-segment active' : 'axis-segment'}
          type="button"
          variant="outlined"
          aria-pressed={value === axis}
          onClick={() => onChange(axis)}
        >
          {gridAxisLabel(axis, locale)}
        </Button>
      ))}
    </div>
  );
}

export default GridAxisControl;
