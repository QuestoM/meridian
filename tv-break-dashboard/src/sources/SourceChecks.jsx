import React, { useState } from 'react';
import { Button } from '@mui/material';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { text } from './sources-copy';

// What the door runs on a file of this kind, and the short list of things it
// genuinely cannot answer.
//
// The reference is Frame.io publishing its proxy ladder exactly, down to the
// statement that files above eight audio channels will not play at all. A
// product that names its own limits in exact terms is trusted on the rest. The
// dotted names below are the real code that runs, sent by the server, so a
// reader who wants to know what was checked can go and read it.
export function SourceChecks({ checks, locale }) {
  const [open, setOpen] = useState(false);
  if (!checks) return null;
  const required = Array.isArray(checks.required_columns) ? checks.required_columns : [];
  const cannot = Array.isArray(checks.cannot_verify) ? checks.cannot_verify : [];
  return (
    <div className="source-checks">
      <Button className="source-checks-toggle" type="button" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {text('whatIsChecked', locale)}
      </Button>
      {open ? (
        <div className="source-checks-body">
          <p>{locale === 'he' ? checks.checked_he : checks.checked_en}</p>
          {required.length ? (
            <p className="source-checks-line">
              <span>{text('checksRequired', locale)}</span>
              <span className="source-checks-columns" dir="ltr">{required.join(', ')}</span>
            </p>
          ) : null}
          {checks.loader ? (
            <p className="source-checks-line">
              <span>{text('checksLoader', locale)}</span>
              <span dir="ltr">{checks.loader}</span>
            </p>
          ) : null}
          {checks.contract ? (
            <p className="source-checks-line">
              <span>{text('checksContract', locale)}</span>
              <span dir="ltr">{checks.contract}</span>
            </p>
          ) : null}
          <p className="source-checks-cannot-head">{text('checksCannot', locale)}</p>
          <ul className="source-checks-cannot">
            {cannot.map((entry) => (
              <li key={entry.code}>{locale === 'he' ? entry.he : entry.en}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export default SourceChecks;
