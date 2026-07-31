import React, { useEffect, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Info } from 'lucide-react';
import { API_BASE, pageText, finiteNumber } from '../shell/surface-helpers';
import './calendar-audience.css';

// Audience-model disclosure block for the Calendar tab's model panel
// (CalendarEventsModel.jsx). Renders GET /api/model/audience faithfully: the
// activation state, when the artifact was computed, and the per-family
// training-gate verdicts, each measured on a held-out gate, never asserted.
// Tri-state honesty: an unreachable endpoint, an absent artifact and a family
// without a gate record each get their own honest state; nothing is invented.

// The frozen factor-family vocabulary of the rebuild artifact, in disclosure order.
const FAMILY_LABELS = {
  weekday_slot: { en: 'Weekday and slot', he: 'יום ורצועה' },
  series: { en: 'Series', he: 'סדרה' },
  calendar_school_and_chol_hamoed: { en: 'School holidays and Chol HaMoed', he: 'חול המועד וחופשות' },
  calendar_hanukkah: { en: 'Hanukkah', he: 'חנוכה' },
  calendar_religious_blackout: { en: 'Shabbat and holy days', he: 'שבתות וימים טובים' },
  season: { en: 'Season', he: 'עונה' },
  operator_events: { en: 'Operator events', he: 'אירועי מפעיל' },
  competitor_lineup: { en: 'Competitor lineup', he: 'ליינאפ מתחרים' },
};
const FAMILY_ORDER = Object.keys(FAMILY_LABELS);

function verdictChip(verdict, locale) {
  const value = String(verdict || 'unknown');
  if (value === 'on') return { className: 'cal-chip aud-on', label: pageText(locale, 'Active', 'פעיל') };
  if (value === 'off') return { className: 'cal-chip', label: pageText(locale, 'Off', 'כבוי') };
  return { className: 'cal-chip warn', label: pageText(locale, 'Unknown', 'לא ידוע') };
}

// Fetches the audience-model disclosure once per mount (and per refreshKey).
// A failed fetch resolves to an explicit unreachable state, never a fabricated one.
export function useAudienceModel(refreshKey) {
  const [state, setState] = useState({ status: 'loading', payload: null });
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/api/model/audience`);
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

function familyRows(gates) {
  const map = gates && typeof gates === 'object' ? gates : {};
  const known = FAMILY_ORDER.map((family) => ({ family, gate: map[family] && typeof map[family] === 'object' ? map[family] : null }));
  // Unknown families the backend may add later render after the known set, never dropped.
  const extras = Object.keys(map)
    .filter((family) => !FAMILY_ORDER.includes(family) && map[family] && typeof map[family] === 'object')
    .map((family) => ({ family, gate: map[family] }));
  return [...known, ...extras];
}

function GateRow({ family, gate, locale }) {
  const label = FAMILY_LABELS[family];
  const chip = verdictChip(gate ? gate.verdict : 'unknown', locale);
  const delta = finiteNumber(gate ? gate.held_out_delta_pct : null);
  const reason = gate && typeof gate.reason === 'string' && gate.reason.trim()
    ? gate.reason
    : pageText(locale, 'The artifact carries no gate record for this family.', 'הקובץ אינו נושא רישום שער עבור הגורם הזה.');
  const hint = delta === null ? reason : `${reason} (${pageText(locale, `held-out delta ${delta.toFixed(2)}%`, `דלתא במבחן מוחזק ${delta.toFixed(2)}%`)})`;
  return (
    <tr>
      <td>{label ? pageText(locale, label.en, label.he) : <span className="ltr-run">{family}</span>}</td>
      <td>
        <Tooltip title={hint} arrow placement="bottom">
          <span className={chip.className}>{chip.label}</span>
        </Tooltip>
      </td>
    </tr>
  );
}

function AudienceModelBody({ payload, locale }) {
  const activationOn = payload.activation === true || (payload.activation && payload.activation.enabled === true);
  const activationChip = activationOn
    ? { className: 'cal-chip aud-on', label: pageText(locale, 'Active', 'פעיל') }
    : { className: 'cal-chip', label: pageText(locale, 'Off', 'כבוי') };
  const rows = familyRows(payload.gates);
  const anyOn = rows.some(({ gate }) => gate && gate.verdict === 'on');
  return (
    <>
      <div className="aud-facts">
        <Tooltip title={pageText(locale, 'While off, forward numbers stay byte-identical to the historical baseline. When on, baseline_tvr for forward-dated segments is replaced by the model prediction with the basis recorded per segment; historical measurement paths never use predictions.', 'כשהמתג כבוי, המספרים קדימה נשארים זהים לחלוטין לקו הבסיס ההיסטורי. כשהוא פעיל, baseline_tvr למקטעים עתידיים מוחלף בתחזית המודל עם רישום הבסיס לכל מקטע; נתיבי המדידה ההיסטוריים לעולם אינם משתמשים בתחזיות.')} arrow placement="bottom">
          <span className="aud-fact">
            {pageText(locale, 'Activation:', 'הפעלה:')}
            <span className={activationChip.className}>{activationChip.label}</span>
          </span>
        </Tooltip>
        {typeof payload.computed_at === 'string' && payload.computed_at && (
          <span className="aud-fact">
            {pageText(locale, 'Computed at:', 'חושב בתאריך:')}
            <span className="ltr-run">{new Date(payload.computed_at).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB')}</span>
          </span>
        )}
      </div>
      <table className="cal-weekly-table aud-gate-table">
        <thead>
          <tr>
            <th>{pageText(locale, 'Factor family', 'משפחת גורמים')}</th>
            <th>{pageText(locale, 'Gate verdict', 'הכרעת השער')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ family, gate }) => (
            <GateRow family={family} gate={gate} key={family} locale={locale} />
          ))}
        </tbody>
      </table>
      {!anyOn && (
        <p className="aud-headline">
          {pageText(locale, 'The model has not yet learned calendar factors from the data; the current history is too short, and the two-year training will decide.', 'המודל טרם למד גורמי לוח מהנתונים; ההיסטוריה הנוכחית קצרה מדי, האימון הדו-שנתי יכריע.')}
        </p>
      )}
      <small className="cal-context-source">
        {pageText(locale, 'Expected rating (this audience model) and predicted retention (the break coefficient model) are different models; these gates never touch a retention coefficient.', 'רייטינג צפוי (מודל הקהל הזה) ושימור חזוי (מודל מקדמי הברייקים) הם מודלים שונים; ההכרעות כאן אינן נוגעות באף מקדם שימור.')}
      </small>
    </>
  );
}

// The block itself: header with a bottom-placed tooltip, then the honest state.
export function AudienceModelBlock({ locale, refreshKey }) {
  const { status, payload } = useAudienceModel(refreshKey);
  if (status === 'loading') return null;
  let body;
  if (status === 'unreachable') {
    body = <p className="cal-empty">{pageText(locale, 'The backend did not serve the audience model state, so nothing is shown rather than an invented state.', 'השרת לא הגיש את מצב מודל הקהל, ולכן לא מוצג דבר במקום מצב מומצא.')}</p>;
  } else if (!payload || payload.available !== true) {
    const reason = payload && typeof payload.reason === 'string' && payload.reason.trim() ? payload.reason : null;
    body = (
      <p className="cal-empty">
        {pageText(locale, 'The audience model artifact has not been built yet, so no gate verdicts exist to show.', 'קובץ מודל הקהל טרם נבנה, ולכן אין הכרעות שער להצגה.')}
        {reason && <span className="ltr-run aud-reason">{reason}</span>}
      </p>
    );
  } else {
    body = <AudienceModelBody locale={locale} payload={payload} />;
  }
  return (
    <div className="aud-block">
      <Tooltip title={pageText(locale, 'What the audience model conditions on. Each factor family ships only after passing its measured held-out gate on a rebuild; an operator never asserts a factor.', 'על מה מודל הקהל מתנה. כל משפחת גורמים נכנסת רק לאחר שעברה את שער ההחזקה הנמדד בבנייה מחדש; מפעיל לעולם אינו מצהיר גורם.')} arrow placement="bottom">
        <span className="cal-context-label aud-title">
          <Info size={12} aria-hidden="true" />
          {pageText(locale, 'Audience model (expected rating)', 'מודל הקהל (רייטינג צפוי)')}
        </span>
      </Tooltip>
      {body}
    </div>
  );
}
