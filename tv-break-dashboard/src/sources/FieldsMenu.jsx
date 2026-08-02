import React, { useState } from 'react';
import { Button, Checkbox, FormControlLabel, Menu } from '@mui/material';
import { SlidersHorizontal } from 'lucide-react';
import { FIELDS, fieldLabel } from './sources-fields';
import { text } from './sources-copy';

// Which facts print on every source card. Frame.io's Fields dropdown, with the
// choice remembered, so a person who cares about row counts and a person who
// cares about file sizes are both looking at the card they wanted.
export function FieldsMenu({ fields, onChange, locale }) {
  const [anchor, setAnchor] = useState(null);

  function toggle(key) {
    const next = fields.includes(key) ? fields.filter((entry) => entry !== key) : [...fields, key];
    onChange(FIELDS.map((field) => field.key).filter((key2) => next.includes(key2)));
  }

  return (
    <>
      <Button
        className="secondary-button compact"
        type="button"
        variant="outlined"
        onClick={(event) => setAnchor(event.currentTarget)}
        aria-haspopup="true"
      >
        <SlidersHorizontal size={14} />
        {text('fields', locale)}
      </Button>
      <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)} className="fields-menu">
        <p className="fields-menu-hint">{text('fieldsHint', locale)}</p>
        {FIELDS.map((field) => (
          <FormControlLabel
            key={field.key}
            className="fields-menu-row"
            control={<Checkbox size="small" checked={fields.includes(field.key)} onChange={() => toggle(field.key)} />}
            label={fieldLabel(field.key, locale)}
          />
        ))}
      </Menu>
    </>
  );
}

export default FieldsMenu;
