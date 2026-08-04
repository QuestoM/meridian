import { useCallback, useEffect, useState } from 'react';
// Explicit extensions, so a test can execute this module in node rather than
// assert a copy of it. See schedule-editor-pin.js for the defect that rule closed.
import { movesFrom } from './day-board-model.js';
import { saveEffect } from './day-board-actions.js';

// The measured answer to what saving would do, held for exactly the edits it was
// measured on.
//
// The readout beside it prices the arrangement on screen against the plan's own
// basis while holding the break counts the plan already chose. That is what makes
// it answer in under a millisecond, and it is also what it cannot see: a save
// writes a restriction, and the engine then plans the whole day again with it in
// force and is free to place the rest of the day differently.
//
// Measured on רשת 13 / 2024-11-01, driven over HTTP against the shipped route:
// pinning a break at exactly the offset and duration the plan had already given
// it reads 0.00 on the cheap score and -30,575.55 here, and the real save then
// lands on -30,575.55 to the cent. One ArrowRight on 001~2 reads 0.00 cheap and
// -47,444.20 here, with 80 breaks falling to 78, and the real save lands on
// -47,444.20 and 78. So the figure is a prediction of the same thing the
// settlement afterwards measures, not a second opinion about it.
//
// A forecast belongs to one arrangement. The moment any edit changes it is
// dropped, because a figure that described the board a keystroke ago is a wrong
// figure, and stale money is exactly the defect this destination keeps closing.
export function useSaveForecast({ board, edits, notify }) {
  const [forecast, setForecast] = useState(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => { setForecast(null); }, [edits, board]);

  const check = useCallback(async () => {
    if (!board) return;
    const moves = movesFrom(edits);
    if (!moves.length) return;
    setChecking(true);
    try {
      const measured = await saveEffect(board.day, moves);
      setForecast(measured);
    } catch (error) {
      setForecast(null);
      notify(
        `The check of what saving would do failed (${error.message}).`,
        `הבדיקה של מה תעשה השמירה נכשלה (${error.message}).`,
      );
    } finally {
      setChecking(false);
    }
  }, [board, edits, notify]);

  return { forecast, checking, check };
}

// The prediction a save should be settled against: the measured one when the
// operator asked for it, and otherwise the cheap score. Never a zero stood in for
// a missing figure, because an absent prediction is not a prediction of zero.
export function predictionFor(forecast, score) {
  if (forecast && forecast.delta && Number.isFinite(forecast.delta.revenue)) return forecast.delta.revenue;
  if (score && score.delta && Number.isFinite(score.delta.revenue)) return score.delta.revenue;
  return null;
}
