import React from 'react';
import { createRoot } from 'react-dom/client';
import CandidateBoard from './CandidateBoard.jsx';
import '../../tokens.css';
import '../../shell/styles.css';

// The one function that puts this board on a page, and nothing else.
//
// It has no side effect on import, so nothing mounts by loading this file. The
// board's home is the model console, whose rail, section list and panel imports
// are P7's and frozen; until two lines land there, this is how the panel is put
// on a page, and it is what the browser measurement in
// tests/test_p12_board_page.py drives.
//
// It exists because a module outside the frontend tree cannot resolve `react`,
// so the measurement's own entry point cannot be the thing that renders. The
// render belongs inside the tree either way: this piece's frontend row is where
// its screen code lives.
//
// The two sheets are imported here and not in the panel, which is the product's
// own division: `src/index.jsx` imports both once for the application and no
// panel imports either for itself. A mount point loads the sheets; a panel reads
// them. Both, not only the tokens: `.bidi-figure`, `.bidi-code` and `.bidi-name`
// are defined in shell/styles.css, and the panel's every figure is wrapped in
// one of them. Measured with only the tokens loaded, the bundle carried zero
// occurrences of `bidi-figure` and the board rendered a candidate that moves no
// coefficient as "36 / 0" where the figure is 0 moved of 36 compared, and every
// negative movement with its minus at the far end of the run.

export function mountBoard(node, locale = 'he') {
  if (!node) return null;
  const root = createRoot(node);
  root.render(<CandidateBoard locale={locale} />);
  return root;
}

export default mountBoard;
