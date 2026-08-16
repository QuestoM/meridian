import React from 'react';
import { InputAdornment, Switch, TextField } from '@mui/material';

// The unit (min, /day, %, ...) sits INSIDE the field as an end adornment, so every
// settings field is one full-width frame at the same width whether it carries a unit
// or not. Native number spinners are hidden in CSS (they only showed on hover and
// looked out of place); the value stays fully typeable.
export function NumberControl({ label, value, onChange, suffix, helperText }) {
  return (
    <TextField
      className="settings-number"
      label={label}
      type="number"
      size="small"
      fullWidth
      value={value ?? 0}
      onChange={(event) => onChange(event.target.value)}
      helperText={helperText}
      slotProps={suffix ? {
        input: { endAdornment: <InputAdornment position="end">{suffix}</InputAdornment> },
      } : undefined}
    />
  );
}

export function ToggleControl({ label, checked, onChange, helperText }) {
  return (
    <div className="toggle-field">
      <label className="toggle-control">
        <span>{label}</span>
        <Switch size="small" checked={Boolean(checked)} onChange={(event) => onChange(event.target.checked)} />
      </label>
      {helperText ? <p className="settings-field-help">{helperText}</p> : null}
    </div>
  );
}
