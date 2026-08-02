import React from 'react';
import { Tooltip } from '@mui/material';
import { pageText } from '../shell/format';

// The one place every campaign surface marks a demo-seed row. `demo` is the
// backend's own bilingual block (kairos_api.campaigns_commitment.demo_block):
// {is_demo, label_en, label_he, meaning_en, meaning_he, replace_en, replace_he}.
// A row with `is_demo` false renders nothing, which is the honest state for
// every campaign booked through the clients flow.
//
// The chip reuses .clients-flag, the same amber inline marker the record
// already uses for "not seen on air yet" and "suspended", so a demo row reads
// as the same class of caveat rather than a new visual language.
export default function DemoBadge({ demo, locale }) {
  if (!demo || !demo.is_demo) {
    return null;
  }
  const label = locale === 'he' ? demo.label_he : demo.label_en;
  const title = [
    locale === 'he' ? demo.meaning_he : demo.meaning_en,
    locale === 'he' ? demo.replace_he : demo.replace_en,
  ].filter(Boolean).join(' ');
  return (
    <Tooltip title={title} arrow placement="top">
      <span className="clients-flag clients-demo-flag" role="img" aria-label={title || label}>
        {label || pageText(locale, 'Demo', 'הדגמה')}
      </span>
    </Tooltip>
  );
}
