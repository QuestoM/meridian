import { warmContext } from './assistant-stream';

// Keep the model's cached prefix written for as long as a question is being
// typed, not only for the moment the dock opened.
//
// The server writes that prefix on POST /api/assistant/context/warm and holds
// the record for PREFIX_TTL_SECONDS, which is 240 s and sits inside the API's
// own five-minute cache (kairos_api/assistant_warm.py). The panel has warmed on
// mount since round five, and that covers the dock that is opened and used
// straight away. It does not cover the dock that is opened and left open, which
// is the ordinary case: Kai is docked beside the work, so it is open while the
// operator reads a page, and the question comes minutes later.
//
// What the write actually costs, measured on this machine on 2026-08-05 as a
// controlled pair: the same request sent twice, first against an unwritten
// prefix and then against the written one, four times over. Cold 2.478, 1.634,
// 4.472 and 2.718 s to the first token, warm 1.624, 2.907, 1.938 and 2.288 s,
// with the API's own usage record showing about 16,740 tokens written on the
// first of each pair and read on the second. So at this hour the write is worth
// roughly half a second out of the two-second budget in job-stories.md. An
// earlier session on the same machine measured it at 9.032 s against 1.804 s
// (kairos_api/assistant_warm.py), and a blind critic called the worst turn of
// their session 9,359 ms, so the cost of paying it is not stable and the point
// of this file is that an ask never pays it.
//
// So the composer reports that a question is being written and this asks for
// the prefix then. Two guards keep it honest about cost. The interval is under
// the server's own hold, so somebody typing never lets the prefix lapse, and one
// request is in flight at a time. The server dedupes as well and independently:
// a prefix already inside its lifetime returns the state warm and spends no
// model call at all, so the worst this can cost is one cheap round trip per
// interval of actual typing. Idle costs nothing, because no event fires.

// Under the server's 240 s hold, so an active composer always finds it written.
export const MIN_INTERVAL_MS = 120000;

let lastAskedAt = 0;
let inFlight = null;

// Ask for the prefix now, unless it was asked for recently or a request is
// already open. Returns the request in flight, or null when this call was the
// one that was skipped, so a caller may await it and a caller may ignore it.
export function keepPrefixWarm(signal) {
  const now = Date.now();
  if (inFlight) return inFlight;
  if (lastAskedAt && now - lastAskedAt < MIN_INTERVAL_MS) return null;
  lastAskedAt = now;
  inFlight = warmContext(signal).then((body) => {
    // An aborted request never reached the server, so it must not count as the
    // one attempt for this interval. Unmounting the dock aborts on purpose.
    if (signal && signal.aborted) lastAskedAt = 0;
    inFlight = null;
    return body;
  });
  return inFlight;
}

// Forget the throttle. Tests call this; nothing in the product does.
export function resetPrefixWarm() {
  lastAskedAt = 0;
  inFlight = null;
}
