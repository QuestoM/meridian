import React from 'react';

// One bar, three call sites, one rule: a bar is a share of a stated denominator.
//
// Every figure on this board was a numeral and nothing else. The reference this
// board is measured against is built on plots, and two questions here are
// genuinely shape questions rather than digit questions: how large is a movement
// against the noise it sits in, and where did a movement concentrate.
//
// The rule this component exists to hold is that a bar is never a free-floating
// impression. It renders a share of something the surface has already named in
// words beside it, so the reader can check the picture against the figure. A bar
// with no denominator on screen would be the visual form of a number nobody
// measured.
//
// It grows from the reading edge, through `inline-size` alone, so it mirrors
// with the document rather than restating a direction. It carries no text and is
// hidden from assistive technology on purpose: every bar here sits beside the
// figure it draws, so announcing it twice would be noise, and a reader who
// cannot see it has lost nothing.
export default function Meter({ share, tone = 'cb-meter-neutral' }) {
  const value = Number(share);
  if (share === null || share === undefined || !Number.isFinite(value)) return null;
  const filled = Math.max(0, Math.min(1, value));
  const over = value > 1;
  return (
    <span className={`cb-meter ${over ? 'cb-meter-over' : ''}`} aria-hidden="true">
      <span className={`cb-meter-fill ${tone}`} style={{ inlineSize: `${(filled * 100).toFixed(1)}%` }} />
    </span>
  );
}
