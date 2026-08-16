import { useCallback, useRef, useState } from 'react';
import { pageText } from '../shell/surface-helpers';
import { postJson, streamAsk } from './assistant-stream';
import { normalizeBatch, asArray } from './assistant-panel-state';
import { buildPageContext } from '../shell/assistant-page-context';
import { unrecordedProposalClaim } from './kai-claimed-action';
import { applyStage, noteStageLimits } from './kai-live-turn';
import { liveRefs } from './mention-refs';

// THE ASK, LIFTED OUT OF THE PANEL.
//
// AssistantPanel.jsx stood at 448 lines against a 450-line law and
// assistant-console.css stood at exactly 450, so neither could take one more
// line. The split was made BEFORE the first edit of this round rather than
// discovered by a failing count, and what moved is the whole of one idea: the
// question being written, the references attached to it, sending it, and the
// entry the finished turn leaves behind. The panel keeps the layout and the
// conversation; this keeps the turn.
//
// WHAT IS NEW HERE AND WHY IT COULD NOT LIVE ANYWHERE ELSE.
//
// A mention is a TYPED reference, {type, id}, and it has to travel beside the
// prose because the prose alone cannot say which of two same-named things was
// meant. The composer is where a reference is made; the panel is where a
// question is sent; and the two were joined by exactly one state variable, the
// question string, which has no room in it for a type or an identifier. So the
// references live beside the question, in the same hook, and they are cleared,
// restored and sent as one thing with it.
//
// NOTHING HERE MAKES A REFERENCE REQUIRED. The measurement this whole piece
// turns on is that the other product's structured mention path was exercised
// zero times in 10,952 recorded turns, which says plainly that a mention system
// that is the only way to name a thing does not get used. So the prose still
// carries the human-readable label, the free-text routes still resolve it, and
// an ask with no references is byte-identical to the ask that shipped before.

export function useAsk({ locale, conv, pageState, mergeBatches, refreshRail, appendEntry, markAdopted, onUnavailable }) {
  const [question, setQuestion] = useState('');
  // The typed references the question carries, as {start, len, type, id, label}.
  // The offsets are the composer's business and never leave the browser; what
  // is sent is the type, the identifier and the label that was inserted.
  const [refs, setRefs] = useState([]);
  const [asking, setAsking] = useState(false);
  const [live, setLive] = useState(null);
  const abortRef = useRef(null);

  const finishAsk = useCallback((trimmed, body, measured, sent) => {
    if (body.available === false) {
      // The panel owns the status cluster, so it is told rather than reached
      // into: the console must say Mabat is unavailable the moment an ask says so.
      if (onUnavailable) onUnavailable(body.reason || body.error);
      appendEntry({ question: trimmed, error: String(body.reason || body.error || '') });
      return body;
    }
    // Adopt the conversation id the ask landed in: a new conversation minted
    // by the server appears in the list, and the id rides on the next ask.
    const returnedConv = body.conversation_id ? String(body.conversation_id) : null;
    if (returnedConv) {
      if (returnedConv !== conv.activeId) {
        markAdopted(returnedConv);
        conv.adopt(returnedConv);
      }
      if (conv.supported) conv.refreshList();
    }
    const batch = body.proposals ? normalizeBatch(body.proposals) : null;
    if (batch) mergeBatches([batch], false);
    appendEntry({
      question: trimmed,
      answer: body.answer ? String(body.answer) : null,
      error: body.error ? String(body.error) : !body.answer && !batch ? pageText(locale, 'The server returned no answer.', 'השרת לא החזיר תשובה.') : null,
      // Said a proposal is pending when this payload recorded none.
      unrecordedClaim: unrecordedProposalClaim(body, batch),
      disclosure: typeof body.context_disclosure === 'string' ? body.context_disclosure : '',
      sources: asArray(body.grounding && body.grounding.sources),
      toolTrace: asArray(body.tool_trace),
      truncated: Boolean(body.truncated),
      // WHAT EACH REFERENCE BOUND TO, from the server's own resolution at send
      // time. The silent drop is the failure this replaces: a reference that
      // did not come back clean has to be visible, or the operator reads an
      // answer about a thing that was not read. An older server that returns
      // nothing here falls back to what was sent, which shows the references
      // without a state rather than showing nothing at all.
      mentions: asArray(body.mentions).length ? asArray(body.mentions) : sent,
      stoppedAtDeadline: Boolean(measured && measured.stoppedAtDeadline),
      stoppedAtCeiling: Boolean(measured && measured.stoppedAtCeiling),
      elapsedSeconds: measured && Number.isFinite(measured.elapsedSeconds) ? measured.elapsedSeconds : null,
      batchId: batch ? batch.batch_id : null,
      at: (body.grounding && body.grounding.generated_at) || new Date().toISOString(),
    });
    if (batch) refreshRail();
    return body;
  }, [locale, appendEntry, mergeBatches, refreshRail, conv, markAdopted, onUnavailable]);

  const stopAsk = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
  }, []);

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || asking) return null;
    setAsking(true);
    // The wire form of a reference: the type and the identifier that say WHICH
    // object, and the label that is already in the prose. The offsets stay here.
    //
    // liveRefs is the last gate before anything leaves the browser. A span
    // survives only while the characters it covers are still the label it was
    // made from, so a reference the operator edited out of the sentence does not
    // travel: sending {type, id} for one object while the prose names another is
    // the silent-drop failure inverted, and it would be worse.
    const sent = liveRefs(question, refs).map((ref) => ({ type: ref.type, id: ref.id, label: ref.label }));
    // Clear at send time, not on completion: the composer stays typeable while
    // the answer streams, so a late clear would wipe whatever has already been
    // typed for the next question.
    setQuestion('');
    setRefs([]);
    const startedAt = Date.now();
    setLive({ question: trimmed, text: '', stage: null, steps: [], startedAt });
    const controller = new AbortController();
    abortRef.current = controller;
    const conversationId = conv.supported && conv.activeId ? conv.activeId : null;
    // Advisory grounding only, per the frozen contract: where the operator is
    // right now, plus the focused entity when a record is open. Absent context
    // sends nothing and the ask behaves exactly as before.
    const pageContext = buildPageContext(pageState);
    const mentions = sent.length ? sent : null;
    // Stage frames the finished exchange keeps: the scope Mabat grounded on, and
    // whether the run stopped at one of its own limits, the time limit or the
    // turn budget. None is in the ask body, whose key set is the frozen
    // contract, so each is captured as it streams.
    const measured = { stoppedAtDeadline: false, stoppedAtCeiling: false, elapsedSeconds: null };
    try {
      let body;
      try {
        body = await streamAsk(trimmed, {
          conversationId,
          pageContext,
          mentions,
          signal: controller.signal,
          onStage: (stage) => {
            noteStageLimits(stage, measured);
            setLive((prev) => applyStage(prev, stage));
          },
          onStep: (step) => setLive((prev) => (prev ? { ...prev, steps: [...prev.steps, step] } : prev)),
          onDelta: (text) => setLive((prev) => (prev ? { ...prev, text: prev.text + text } : prev)),
        });
      } catch (streamError) {
        // A stop is the person's own decision, so it ends here rather than
        // silently starting the same ask again on the plain endpoint.
        if (controller.signal.aborted) throw streamError;
        // The stream endpoint is unavailable or broke mid-flight; the plain
        // ask returns the same answer without live updates, so retry there.
        setLive({ question: trimmed, text: '', stage: null, steps: [], startedAt });
        body = await postJson('/api/assistant/ask', {
          question: trimmed,
          ...(conversationId ? { conversation_id: conversationId } : {}),
          ...(pageContext ? { page_context: pageContext } : {}),
          ...(mentions ? { mentions } : {}),
        }, { signal: controller.signal });
      }
      measured.elapsedSeconds = (Date.now() - startedAt) / 1000;
      return finishAsk(trimmed, body, measured, sent);
    } catch (error) {
      const stopped = controller.signal.aborted;
      appendEntry({
        question: trimmed,
        error: stopped
          ? pageText(locale, 'You stopped this question. Nothing was changed.', 'עצרתם את השאלה הזו. שום דבר לא שונה.')
          : error.message,
      });
      // Put the question back for an easy retry, but never overwrite text
      // typed while the ask was in flight. The references go back with it, or
      // the restored sentence would carry labels with nothing bound behind them.
      let restored = false;
      setQuestion((current) => {
        restored = !current;
        return current || trimmed;
      });
      setRefs((current) => (restored && !current.length ? refs : current));
      return null;
    } finally {
      abortRef.current = null;
      setLive(null);
      setAsking(false);
    }
  }, [question, refs, asking, finishAsk, appendEntry, locale, conv.supported, conv.activeId, pageState]);

  function onComposerKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  }

  return { question, setQuestion, refs, setRefs, asking, live, ask, stopAsk, onComposerKeyDown };
}
