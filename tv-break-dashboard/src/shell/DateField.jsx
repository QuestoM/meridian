import React from 'react';
import { TextField } from '@mui/material';

// A native date input only opens its picker from the calendar icon by default.
// Opening it from a click anywhere on the field is the behaviour operators expect,
// so the whole control calls showPicker(). It is guarded: a browser without
// showPicker keeps the icon, and the shrink label keeps the placeholder clear of
// the floating label. Shared across the settings and constraint-builder date fields.
export default function DateField({ id, className = '', label, ariaLabel, value, onChange, helperText, fullWidth = false }) {
  return (
    <TextField
      id={id}
      className={`studio-date-field${className ? ` ${className}` : ''}`}
      label={label}
      type="date"
      size="small"
      fullWidth={fullWidth}
      value={value ?? ''}
      onChange={(event) => onChange(event.target.value)}
      helperText={helperText}
      slotProps={{
        inputLabel: { shrink: true },
        htmlInput: ariaLabel ? { 'aria-label': ariaLabel } : undefined,
      }}
      onClick={(event) => {
        const input = event.currentTarget.querySelector('input');
        if (input && typeof input.showPicker === 'function') {
          try { input.showPicker(); } catch { /* The native icon remains usable. */ }
        }
      }}
    />
  );
}
