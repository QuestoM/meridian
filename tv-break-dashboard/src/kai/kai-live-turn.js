// What the dock does with each stage frame while an answer is still running.
//
// Split out of AssistantPanel.jsx, which sits at the file-size cap, and because
// this is a rule about turns rather than a piece of render.
//
// The server streams text as the model produces it, and a run has several model
// turns. Only the LAST turn is the answer: the ask body's answer field is that
// turn's text (kairos_api/assistant_pipeline.py), so anything a previous turn
// wrote was the model saying what it was about to do before calling a tool. The
// dock used to concatenate all of it and then swap the whole thing for the real
// answer when the run finished, which read as the answer changing under the
// operator. Each new turn now clears the live text, so what is on screen is
// always the turn that is being written.
//
// The verifying stage is the same rule with a sharper reason. It is sent when
// the server has caught the answer claiming a recorded proposal that nothing
// backs and is spending one more turn on it. The text already painted is that
// false claim, and it does not stay on the screen for a second longer than the
// server takes to notice.

// The two stages that end a turn's text. Every other stage leaves it alone.
const TURN_START = new Set(['thinking', 'verifying']);

// Limits the run hit, captured off the stage channel because the ask body's key
// set is frozen and carries neither. The caller owns the object.
export function noteStageLimits(stage, measured) {
  if (!stage || !measured) return measured;
  if (stage.stage === 'deadline') measured.stoppedAtDeadline = true;
  if (stage.stage === 'ceiling') measured.stoppedAtCeiling = true;
  return measured;
}

// One stage frame folded into the live exchange. Returns the previous state
// unchanged when there is nothing live, so it is safe to call at any time.
export function applyStage(prev, stage) {
  if (!prev) return prev;
  if (!stage || typeof stage !== 'object') return prev;
  const next = { ...prev, stage };
  if (stage.facts && typeof stage.facts === 'object') next.facts = stage.facts;
  if (Number.isFinite(stage.deadline_seconds)) next.deadlineSeconds = stage.deadline_seconds;
  if (TURN_START.has(stage.stage)) next.text = '';
  if (stage.stage === 'verifying') next.verifying = true;
  return next;
}
