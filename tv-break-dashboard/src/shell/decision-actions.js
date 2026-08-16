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
  function reportUnrecorded(result, labelEn, labelHe, feminine = false) {
    const notRecordedHe = feminine ? 'לא נרשמה' : 'לא נרשם';
    const failedHe = feminine ? 'נכשלה' : 'נכשל';
    if (result?.offline) {
      notify(
        `${labelEn} was not recorded because the decision service is unreachable. Nothing changed.`,
        `${labelHe} ${notRecordedHe} כי שירות ההחלטות אינו זמין. דבר לא השתנה.`,
      );
    } else if (result?.status === 404) {
      notify(
        `${labelEn} was not recorded because this server does not support the decision route. Nothing changed.`,
        `${labelHe} ${notRecordedHe} כי השרת הזה אינו תומך בנתיב ההחלטות. דבר לא השתנה.`,
      );
    } else {
      notify(`${labelEn} failed (${result?.error || 'unknown error'}).`, `${labelHe} ${failedHe} (${result?.error || 'שגיאה לא ידועה'}).`);
    }
  }

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
      return true;
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
      if (!result.ok || result.offline || result.status === 404) {
        reportUnrecorded(result, 'Approval', 'האישור');
        return false;
      }
      markApprovedLocal(id);
      setRefreshKey((current) => current + 1);
      notify('Override created from this recommendation. The schedule is now marked stale; recompute when ready.',
        'נוצרה עקיפה מההמלצה הזו. לוח השידורים מסומן כעת כלא מעודכן; הריצו חישוב מחדש כשתרצו.');
      return true;
    }

    // Non-actionable recommendation: annotate the decision log only, no override.
    const result = await postBreakDecision({
      action: 'approve',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: selectedProgram?.program_type || rec?.program_type,
      scenario,
    });
    if (!result.ok || result.offline || result.status === 404) {
      reportUnrecorded(result, 'Approval', 'האישור');
      return false;
    }
    markApprovedLocal(id);
    notify('Approval recorded in the decision log.', 'האישור נרשם ביומן ההחלטות.');
    return true;
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
    if (!result.ok || result.offline || result.status === 404) {
      reportUnrecorded(result, 'Rejection', 'הדחייה', true);
      return false;
    }
    markRejectedLocal(id);
    notify('Rejection recorded in the decision log.', 'הדחייה נרשמה ביומן ההחלטות.');
    return true;
  }

  async function applySimilarRecommendations(id = activeRec?.id) {
    const selectedRec = normalizeRows(overview.recommendations).find((rec) => rec.id === id) || activeRec;
    const targetType = selectedRec?.program_type;
    const matching = normalizeRows(overview.recommendations).filter((rec) => !targetType || rec.program_type === targetType);
    const result = await postBreakDecision({
      action: 'apply_similar',
      recommendation_id: selectedRec?.id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: targetType || selectedProgram?.program_type,
      scenario,
    });
    if (!result.ok || result.offline || result.status === 404) {
      reportUnrecorded(result, 'Similar approvals', 'אישור ההמלצות הדומות');
      return false;
    }
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
    notify('Similar recommendations recorded as approved in the decision log.', 'המלצות דומות נרשמו כמאושרות ביומן ההחלטות.');
    return true;
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
