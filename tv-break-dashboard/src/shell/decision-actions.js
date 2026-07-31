import { postBreakDecision, mapProposedKind } from './api';
import { normalizeRows } from './plan-model';

// The recommendation decision handlers. A factory rather than a hook: it calls
// no React hook, and it is rebuilt on every render exactly as the inline
// function declarations were, so every closure sees the same fresh state.
export function createDecisionActions({
  overview,
  activeRec,
  selectedProgram,
  scenario,
  notify,
  setApproved,
  setRejected,
  setRefreshKey,
  setOverridePrefill,
  setActiveView,
}) {
  function markApprovedLocal(id) {
    setApproved((current) => new Set(current).add(id));
    setRejected((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  }

  // Send an actionable recommendation to the Overrides workspace with a prefill so
  // the operator sets the exact break count against the live segment state and the
  // projected-delta preview, instead of the model guessing a target it cannot know.
  function openRecommendationInOverrides(rec) {
    if (!rec?.segment_id) return;
    setOverridePrefill({
      segment_id: rec.segment_id,
      kind: mapProposedKind(rec.proposed_kind) || 'pin',
      anchor: rec.anchor || null,
      rec_id: rec.id || '',
    });
    setActiveView('Overrides');
  }

  async function approveRecommendation(id) {
    const rec = normalizeRows(overview.recommendations).find((item) => item.id === id) || (activeRec?.id === id ? activeRec : null);
    const kind = rec && rec.actionable ? mapProposedKind(rec.proposed_kind) : '';
    const anchor = rec?.anchor || {};

    // A forced break count needs a target the recommendation does not carry, so route
    // it to Overrides where the live segment state and preview are available rather
    // than committing a silent no-op. Everything else creates a real override inline.
    if (rec && rec.actionable && rec.segment_id && kind === 'force') {
      openRecommendationInOverrides(rec);
      notify('Set the break count in overrides, where the live segment and preview are available.',
        'קבעו את מספר הברייקים בעקיפות, שם זמינים המשבצת החיה והתצוגה המקדימה.');
      return;
    }

    if (rec && rec.actionable && rec.segment_id && kind) {
      const payload = {
        action: 'approve',
        recommendation_id: id,
        break_id: selectedProgram?.selected_break?.id,
        program_type: rec.program_type || selectedProgram?.program_type,
        scenario,
        target_id: rec.segment_id,
        kind,
        anchor_date: anchor.date,
        anchor_start: anchor.start_clock,
        anchor_title: anchor.program,
      };
      if (kind === 'gold') payload.gold = true;
      const result = await postBreakDecision(payload);
      if (result.status === 404) {
        // Older backend without the anchored decision route: keep the honest log-only
        // behavior so approvals still register on the command surface.
        markApprovedLocal(id);
        notify('Approval recorded in the decision log.', 'האישור נרשם ביומן ההחלטות.');
        return;
      }
      if (!result.ok) {
        notify(`Approval failed (${result.error}).`, `האישור נכשל (${result.error}).`);
        return;
      }
      markApprovedLocal(id);
      setRefreshKey((current) => current + 1);
      notify('Override created from this recommendation. The schedule is now marked stale; recompute when ready.',
        'נוצרה עקיפה מההמלצה הזו. לוח השידורים מסומן כעת כלא מעודכן; הריצו חישוב מחדש כשתרצו.');
      return;
    }

    // Non-actionable recommendation: annotate the decision log only, no override.
    markApprovedLocal(id);
    await postBreakDecision({
      action: 'approve',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: selectedProgram?.program_type || rec?.program_type,
      scenario,
    });
    notify('Approval recorded in the decision log.', 'האישור נרשם ביומן ההחלטות.');
  }

  function markRejectedLocal(id) {
    setRejected((current) => new Set(current).add(id));
    setApproved((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  }

  async function rejectRecommendation(id) {
    const rec = normalizeRows(overview.recommendations).find((item) => item.id === id) || (activeRec?.id === id ? activeRec : null);
    const anchor = rec?.anchor || {};
    const actionable = Boolean(rec && rec.actionable && rec.segment_id);
    const payload = {
      action: 'reject',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: rec?.program_type || selectedProgram?.program_type,
      scenario,
    };
    if (actionable) {
      // A rejection is a dismissed, anchored record. Kind is left for the backend to
      // default (forbid), since rejecting means "do not do this", not dismissing the
      // rec's specific proposed kind.
      payload.target_id = rec.segment_id;
      payload.anchor_date = anchor.date;
      payload.anchor_start = anchor.start_clock;
      payload.anchor_title = anchor.program;
    }
    const result = await postBreakDecision(payload);
    // Only an actionable rejection can create an anchored record, so only it surfaces a
    // real server error and stays unmarked on failure. A non-actionable rejection is a
    // decision-log annotation, and a 400 (no target to anchor) is expected there.
    if (actionable && !result.ok) {
      notify(`Rejection failed (${result.error}).`, `הדחייה נכשלה (${result.error}).`);
      return;
    }
    markRejectedLocal(id);
    notify('Rejection recorded in the decision log.', 'הדחייה נרשמה ביומן ההחלטות.');
  }

  function applySimilarRecommendations() {
    const targetType = activeRec?.program_type;
    const matching = normalizeRows(overview.recommendations).filter((rec) => !targetType || rec.program_type === targetType);
    setApproved((current) => {
      const next = new Set(current);
      matching.forEach((rec) => next.add(rec.id));
      return next;
    });
    setRejected((current) => {
      const next = new Set(current);
      matching.forEach((rec) => next.delete(rec.id));
      return next;
    });
    postBreakDecision({
      action: 'apply_similar',
      recommendation_id: activeRec?.id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: targetType || selectedProgram?.program_type,
      scenario,
    });
    notify('Similar recommendations recorded as approved in the decision log.', 'המלצות דומות נרשמו כמאושרות ביומן ההחלטות.');
  }

  return {
    markApprovedLocal,
    openRecommendationInOverrides,
    approveRecommendation,
    markRejectedLocal,
    rejectRecommendation,
    applySimilarRecommendations,
  };
}
