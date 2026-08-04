// An answer that says a proposal was recorded, on an exchange where nothing was.
//
// Measured twice in one day, same phrasing, by a blind critic. Asked to raise
// the retention floor as a proposal only, the dock printed
//
//     ההצעה נרשמה (settings: min_retention_floor 0.78 ← 0.80), במצב ממתין לאישור.
//
// and nothing else. The ask body carried proposals null and an empty tool_trace,
// the audit line recorded tools [], the proposal store did not grow, and no card
// rendered. The operator was told a change was waiting for their approval and
// there was nothing anywhere to approve.
//
// The prompt now forbids that sentence without a propose tool, but a rule the
// model can ignore is not a guarantee. This module is the half that does not
// depend on the model complying: it reads the payload, decides whether anything
// was actually recorded, and lets the surface print the honest note in place of
// the claim. The payload is the authority, never the prose.
//
// Since round 5 this rule is also enforced on the server, ported verb for verb
// into kairos_api/assistant_claimed_action.py, where a claim nothing backs buys
// one more model turn to record the change or restate it honestly. So this
// module is the last line rather than the only one, and it is still needed: the
// extra turn can fail, run out of clock, or come back with the claim intact, and
// a saved conversation from before any of it exists is replayed through here.
//
// It carries one more job than the server's copy. claimSegments cuts an answer
// into sentences with a verdict on each, so the surface can strike the sentence
// that lies and keep the ones that carry real figures.
//
// What counts as recorded, and why it is provable. A proposal item exists only
// when a propose_* tool returns captured, and that call lands on the trace as
// {tool, ok} (kairos_api/assistant_tools.py:418-444) while the batch rides on
// the response's proposals key (assistant_pipeline.py:412-416). So no successful
// propose step and no batch means no proposal was recorded in this turn, which
// is a fact about the payload and not a guess about the answer.
//
// The Hebrew half is grammar, measured on the stored exchanges on this machine.
// הצעה is feminine, so a claim about one reads נרשמה, הוגשה or נשמרה. The
// masculine נרשם in this corpus is never a claim: it appears only in the
// model's honest explanation of the mechanism, "שינוי כזה הוא הצעה בלבד: הוא
// נרשם, ואתה מאשר או דוחה", which is an offer and not a report. Including it
// would have flagged three truthful answers, so it is out, and the rule is
// stated here rather than left as a silent regex.
//
// The verb list is the part that has to keep growing, and it was measured to be
// too narrow. A blind critic put ten phrasings of the same false claim through
// this module and it caught three: the three whose verbs came from the corpus it
// was built on. The seven it missed are below, in the list, because the model's
// own wording moves between runs. On the same day it answered one ask with
// ההצעה נרשמה ונמצאת במצב pending and another with ההצעה במצב pending, and only
// the first had a verb this module knew.
//
// The blunt alternative was measured and rejected. Annotating every answer that
// merely mentions a proposal, with the same payload gate in front of it, fires
// on 22 of the 150 stored exchanges here and 19 of those are truthful. So the
// payload stays the authority, the offer and negation filters stay exactly as
// they were, and only the verb list widens. Measured after widening: the same
// three unbacked exchanges are annotated, 964e40275eeb#6 and bbb47ee1dc76#30
// and #31, and nothing else moves.
//
// Where the widened list is deliberately loose, said out loud. הכנתי is the
// broadest verb in it, and a sentence such as הכנתי סיכום של ההצעות שאושרו would
// be annotated even though it claims nothing. Nothing of that shape exists in
// the corpus, and the cost if it appears is a note that is still true, since the
// gate only opens when this answer really did record nothing. A missed claim
// costs the operator a change they think is waiting and is not, which is the
// worse of the two, so the list stays wide.

// Digits keep their decimal point through the sentence split, so a claim is not
// cut in half by the very number it quotes.
const DECIMAL_POINT = /(\d)\.(\d)/g;
const SENTENCE = /[\n.?!]+/;

const PROPOSAL = /הצע|proposal/i;

// Recorded, registered, submitted, saved or pending: the words the prompt rule
// names, in the forms this market's answers actually use. Three groups, so each
// addition can be read against the phrasing that forced it.
//
// Something was done to the proposal, in the passive third person the model
// prefers and in the first person it sometimes uses instead.
const RECORDED_DONE = /נרשמה|נרשמו|הוגשה|הוגשו|נשמרה|נשמרו|נוצרה|נוצרו|נשלחה|נשלחו|רשמתי|הגשתי|שמרתי|יצרתי|הכנתי|שלחתי/;
// The proposal is described as sitting in a queue, waiting for an approval.
const RECORDED_WAITING = /ממתינה לאישור|ממתינות לאישור|ממתין לאישור|מחכה לאישור|מחכות לאישור|בתור לאישור|במצב pending|במצב ממתין/;
// The same two readings in English, since the surface answers in both.
const RECORDED_EN = /recorded|registered|submitted|logged|created a proposal|pending approval|pending your approval|awaiting approval|awaiting your approval|is pending|are pending|waiting in the pending/;
const RECORDED = new RegExp([RECORDED_DONE, RECORDED_WAITING, RECORDED_EN].map((part) => part.source).join('|'), 'i');

// A denial is not a claim. Only the words just before the verb count, so a
// sentence that reports one thing and denies another is read correctly.
const NEGATION_WINDOW = 24;
const NEGATED_HE = /(^|[^֐-׿])(לא|אין|אינה|אינו|טרם|ללא|בלי)([^֐-׿]|$)/;
const NEGATED_EN = /\b(no|not|nothing|never|without|cannot)\b/i;

// An offer to record something is not a report that it was recorded.
const OFFER = /רוצה ש|תרצה ש|אם תרצה|האם|אוכל ל|אני יכול|אפשר ש|would you like|shall i|if you want/i;

function negatedBefore(sentence, index) {
  const before = sentence.slice(Math.max(0, index - NEGATION_WINDOW), index);
  return NEGATED_HE.test(before) || NEGATED_EN.test(before);
}

// One sentence, one verdict. The same three questions in the same order as the
// whole-text reading, so a sentence never classifies one way alone and another
// way inside the paragraph it sits in.
export function sentenceClaims(sentence) {
  if (!PROPOSAL.test(sentence) || OFFER.test(sentence)) return false;
  const found = RECORDED.exec(sentence);
  return Boolean(found) && !negatedBefore(sentence, found.index);
}

// The text cut into sentences, each carrying its own verdict, together covering
// the original exactly: joining the pieces back gives the string that came in.
// It exists so the surface can strike the one sentence that carries the claim
// and leave the rest of the answer readable, rather than demoting a paragraph
// of real figures because one line in it was false.
//
// The probe is the decimal-normalised copy, and its replacement is exactly as
// long as what it replaces, so an index found on the probe is the same index in
// the original and no slice is ever shifted by a number.
export function claimSegments(text) {
  const value = String(text === null || text === undefined ? '' : text);
  const probe = value.replace(DECIMAL_POINT, '$1 $2');
  const breaks = new RegExp(SENTENCE.source, 'g');
  const segments = [];
  let start = 0;
  let match = breaks.exec(probe);
  while (match !== null) {
    const end = match.index + match[0].length;
    segments.push({ text: value.slice(start, end), claim: sentenceClaims(probe.slice(start, match.index)) });
    start = end;
    match = breaks.exec(probe);
  }
  if (start < value.length) segments.push({ text: value.slice(start), claim: sentenceClaims(probe.slice(start)) });
  return segments;
}

// True when the text asserts, somewhere in it, that a proposal is recorded,
// registered, submitted or pending approval.
export function claimsRecordedProposal(text) {
  return claimSegments(text).some((segment) => segment.claim);
}

// Did this turn actually record a proposal. A batch on the exchange is the
// direct proof; a propose_* step that returned ok is the same fact from the
// trace, and it stands even when the batch failed to normalize.
export function recordedProposal(body, batch) {
  if (batch) return true;
  if (body && body.proposals) return true;
  const trace = body && Array.isArray(body.tool_trace) ? body.tool_trace : [];
  return trace.some((step) => step && step.ok === true && String(step.tool || '').startsWith('propose'));
}

// The one question the surface asks: was the operator told a change is waiting
// for their approval when nothing is. batch is the batch this exchange carries,
// or null. Both halves must hold, so a truthful answer is never annotated and a
// recorded proposal never gets a note that contradicts its own card.
export function unrecordedProposalClaim(body, batch) {
  if (recordedProposal(body, batch)) return false;
  return claimsRecordedProposal(body && body.answer);
}
