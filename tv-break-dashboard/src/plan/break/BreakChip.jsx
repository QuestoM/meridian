import React from 'react';

// The shared break chip body used by both the editor lane and the read-only
// timeline. It is a pure presentation shell: three stacked lines with the exact
// clock second as the prominent first line, a secondary detail line and a small
// tertiary line, all kept left to right so a time is never mirrored in a right
// to left layout. Wrappers own their own positioning, interaction and outer
// classes; this component only guarantees the lines stay legible and never clip.
//
// clock       the HH:MM:SS (or HH:MM) hero string, required
// detail      the middle line: a human offset on the editor, the break position
//             on the timeline
// meta        the small tertiary line: the break length on the editor
// gold        an optional gold marker shown in place of meta on the timeline
// goldLabel   the localized word for a gold break
function BreakChip({ clock, detail, meta, gold, goldLabel, children }) {
  return (
    <>
      <span className="break-chip-clock" dir="ltr">{clock}</span>
      {detail != null && <strong className="break-chip-detail">{detail}</strong>}
      {gold ? (
        <em className="break-chip-gold">{goldLabel}</em>
      ) : (
        meta != null && <em className="break-chip-meta" dir="ltr">{meta}</em>
      )}
      {children}
    </>
  );
}

export default BreakChip;
