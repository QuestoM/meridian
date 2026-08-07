import React, { useEffect, useState } from 'react';
import { API_BASE, pageText } from '../shell/surface-helpers';
import './calendar-audience.css';
import './console-mount.js';

// The audience-model block on the calendar, which is a run surface.
//
// **What changed, and why it is not a loss.** This block used to render the
// eight held-out gate verdicts, with their reasons and their held-out deltas,
// to every account on the product. Those are training internals: section 4.2's
// lexicon test fails on any run surface that carries a gate, a held-out delta
// or a coefficient, and this block failed it eight times over. Every one of
// those verdicts is now in the model console, with more behind it than the
// calendar ever showed: the basis each was decided on, the bar it was measured
// against, and, for the ones that could not run, what would end the block.
//
// So this block keeps exactly one fact, and it is a run-side fact: whether runs
// are currently using the audience model, which is what tells an operator
// whether a forward-dated rating came from the model or from the historical
// baseline. It carries no verdict, no delta and no reason.
//
// A channel account sees nothing here at all. The route is walled on
// affiliation, and the refusal renders as absence rather than as a message,
// because a message would tell a channel account that the other side exists.

export function useAudienceModel(refreshKey) {
  const [state, setState] = useState({ status: 'loading', payload: null });
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/api/model/audience`, { credentials: 'include' });
        if (response.status === 403) {
          if (active) setState({ status: 'refused', payload: null });
          return;
        }
        if (!response.ok) throw new Error(String(response.status));
        const payload = await response.json();
        if (active) setState({ status: 'loaded', payload: payload && typeof payload === 'object' ? payload : null });
      } catch {
        if (active) setState({ status: 'unreachable', payload: null });
      }
    })();
    return () => { active = false; };
  }, [refreshKey]);
  return state;
}

function activationState(payload) {
  if (!payload || payload.available !== true) return payload && payload.activation === true ? 'on_no_artifact' : 'off';
  return payload.activation === true ? 'on' : 'off';
}

export function AudienceModelBlock({ locale, refreshKey }) {
  const { status, payload } = useAudienceModel(refreshKey);
  // Loading, refused and unreachable all render nothing: the first because
  // there is nothing yet, the second because a channel account must not learn
  // the other side exists, and the third because an invented state is worse
  // than an absent one on a surface where this is a side note.
  if (status !== 'loaded') return null;
  const state = activationState(payload);
  const label = state === 'on'
    ? pageText(locale, 'On', 'דלוק')
    : state === 'on_no_artifact'
      ? pageText(locale, 'On, nothing trained', 'דלוק, אין מודל מאומן')
      : pageText(locale, 'Off', 'כבוי');
  return (
    <div className="aud-block">
      <span className="cal-context-label aud-title">
        {pageText(locale, 'Audience model in runs', 'מודל הקהל בהרצות')}
      </span>
      <div className="aud-facts">
        <span className="aud-fact">
          {pageText(locale, 'Activation:', 'הפעלה:')}
          <span className={state === 'on' ? 'cal-chip aud-on' : 'cal-chip'}>{label}</span>
        </span>
        {typeof payload.computed_at === 'string' && payload.computed_at ? (
          <span className="aud-fact">
            {pageText(locale, 'Model version, trained at:', 'גרסת מודל, אומנה ב:')}
            <span className="bidi-figure figure-nowrap">{new Date(payload.computed_at).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB')}</span>
          </span>
        ) : null}
      </div>
      <small className="cal-context-source">
        {pageText(locale, 'While this is off, forward-dated ratings are the historical baseline. While it is on, they come from the model version named above.', 'כשזה כבוי, רייטינג לתאריכים עתידיים הוא קו הבסיס ההיסטורי. כשזה דלוק, הוא מגיע מגרסת המודל שצוינה למעלה.')}
      </small>
    </div>
  );
}

export default AudienceModelBlock;
