// Saving a dragged break from the schedule editor: the target it addresses, the
// body it writes, and why both are now the day board's own.
//
// The scope this file used to send was the whole broadcast date, and the
// restriction resolver matches a date scope against every segment on that date.
// Measured through the engine's own resolve_constraints on 2024-11-01: the date
// scope binds 82 of 82 segments, the programme scope binds 1, and the predicate
// the day board sends binds 1. Driven through the live product with one break
// dragged one snap unit right, the same drag either way: at the date scope the
// day fell from 1,062,669.88 to 273,093.70 and from 80 breaks to 23, which is
// 789,576.18 ILS and 74.3 per cent of the day for one click; at the predicate it
// fell to 1,032,180.49 and 77 breaks, 30,489.39 ILS, and the inverse put the day
// back to 1,062,669.88 with a gap of 0.0.
//
// So there is no scope choice here any more. A saved move names one airing, the
// body is built by the day board's own ``placementBody`` rather than a second
// copy of it, and the write goes through the day board's own two-step save so the
// break carries a record, the chip renders as saved, and the Remove control that
// reverses it exists. Before this, the editor wrote a restriction and no record,
// so the money it spent had no route back from any surface.

// The extension is explicit because the test executes this module in node rather
// than through the bundler, and node resolves a relative import literally. A
// module the test cannot run is a module the test can only assert a copy of.
import { placementBody, saveBreakPlacement } from './day-board-actions.js';

// The day board's break and programme, read off one editor row.
//
// ``segmentId`` comes from the segment anchors the editor already resolves for
// the programme inspector, and it is what turns an editor row into an addressable
// break: the plan's identity is <segment_id>~<ordinal>, and the ordinal is the
// break's own 1-based position inside its programme, which both surfaces number
// the same way. Without it there is no break to record a placement against, so
// the caller refuses the save rather than writing half a transaction.
export function pinTarget(item, startSec, durationSec, segmentId) {
  const ordinal = Math.max(1, Number(item.break_num_in_program || 1));
  return {
    item: { break_id: `${segmentId}~${ordinal}`, ordinal },
    programme: {
      day: item.date || '',
      title: item.program_title || '',
      start_seconds: Number(item.program_start_sec || 0),
    },
    live: {
      offsetSeconds: Math.max(0, startSec - item.program_start_sec),
      durationSeconds: durationSec,
      isGold: Boolean(item.is_gold),
    },
  };
}

// The exact constraint body this surface posts, so it can be measured on its own.
export function pinBody(target) {
  return placementBody(target);
}

// Write the restriction and the break's own record of it, in that order.
export function savePinPlacement(target) {
  return saveBreakPlacement(target);
}
