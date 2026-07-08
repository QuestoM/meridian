import React from 'react';
import { TextField } from '@mui/material';

// A native date input only opens its picker from the calendar icon by default.
// Opening it from a click anywhere on the field is the behaviour operators expect,
// so the whole control calls showPicker(). It is guarded: a browser without
// showPicker keeps the icon, and the shrink label keeps the placeholder clear of
// the floating label. Shared across the settings and constraint-builder date fields.
export default function DateField({ label, value, onChange }) {
  return (
    <TextField
      label={label}
      type="date"
      size="small"
      value={value ?? ''}
      onChange={(event) => onChange(event.target.value)}
      slotProps={{ inputLabel: { shrink: true } }}
      onClick={(event) => {
        const input = event.currentTarget.querySelector('input');
        if (input && typeof input.showPicker === 'function') input.showPicker();
      }}
    />
  );
}
