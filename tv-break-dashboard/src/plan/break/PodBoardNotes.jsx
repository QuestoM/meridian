import React from 'react';
import { Figure, Name } from '../../shell/bidi';

// Two readouts lifted out of PodBoard, which had reached exactly 450 lines and
// could not take another line without breaking the size law. Splitting rather
// than compressing is the rule, and the split is along a real seam: both of
// these describe what is WRONG or MISSING inside the pod, while what remains in
// PodBoard is the pod's own shape and its order.
//
// The extraction was forced by a real need. The technical verdict on a
// commercial's file had nowhere to render, and a component nothing draws is the
// inert-lever class: computed, carried, and structurally unable to reach anyone.

// Where the uncovered seconds sit: the dead air before the first spot plus the
// holes between spots. These sit INSIDE the uncovered figure above them rather
// than beside it, which the basis line says so a reader does not add them twice.
export function ContinuityNote({ continuity, label }) {
  if (!continuity.length) return null;
  return (
    <p className="pod-continuity">
      <span className="pod-continuity-label">{label('Where the uncovered seconds sit', 'היכן יושבות השניות הלא מכוסות')}</span>
      {continuity.map((row) => (
        <span key={row.key} className="pod-continuity-part">
          {row.count}, <Figure>{row.seconds}</Figure>
        </span>
      ))}
      <span className="pod-figure-basis">
        {label(
          'the dead air before the first spot plus the holes between spots, from the times the traffic file declares. These sit inside the uncovered figure above.',
          'האוויר המת לפני התשדיר הראשון בתוספת הרווחים בין התשדירים, לפי השעות שקובץ הטראפיק מצהיר עליהן. אלה נמצאים בתוך המספר הלא מכוסה שלמעלה.',
        )}
      </span>
    </p>
  );
}

// Every disagreement in the pod, each one opening the spot it is about. The
// button is the point: a fault a reader cannot navigate to is a fault they
// cannot fix.
export function PodErrors({ errors, openSpot }) {
  if (!errors.length) return null;
  return (
    <ul className="pod-errors">
      {errors.map((error) => (
        <li key={error.key} className={`pod-error pod-error-${error.kind}`}>
          <button type="button" className="pod-error-open" onClick={() => openSpot(error.spotKey)}>
            <span className="pod-error-advertiser"><Name>{error.advertiser}</Name></span>
            <span>{error.detail}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
