import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Download, RefreshCcw, SlidersHorizontal, X } from 'lucide-react';
import { pageText } from './advertisers-helpers';
import { programTypeLabel } from './surface-helpers';
import { KINDS, kindLabel, runDayRecomputeJob, isNum } from './override-console-lib';
import './schedule-inspector.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// Click-to-inspect drawer for one owned-channel programme (segment). It shows the
// full saved-plan detail the engine holds (identity, break plan, economics, and
// retention with its credible interval), the segment's current manual overrides,
// and the edit actions the optimizer honors (pin, force a count, forbid, mark
// gold) with a live WITH-vs-WITHOUT preview before anything is saved. Saving marks
// the plan stale; the operator then recomputes this one day (a fast incremental
// job) and can download the corrected schedule. Individual ad assignment is a
// downstream daily step and is not editable here; the drawer says so honestly.

const fmtMoney = (value, locale) => (isNum(value)
  ? `${Math.round(value).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US')} ${'₪'}`
  : '-');
const fmtNum2 = (value, locale) => (isNum(value)
  ? value.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 })
  : '-');

function Row({ label, value }) {
  return (
    <div className="si-row">
      <span className="si-row-label">{label}</span>
      <span className="si-row-value" dir="ltr">{value}</span>
    </div>
  );
}

export default function ScheduleInspector({ segmentId, channel, day, onClose, locale, notify, onGlobalRefresh }) {
  const he = locale === 'he';
  const [detail, setDetail] = useState(null);
  const [state, setState] = useState('loading'); // loading | ready | error
  const [kind, setKind] = useState('pin');
  const [countValue, setCountValue] = useState('');
  const [preview, setPreview] = useState(null);
  const [previewState, setPreviewState] = useState('idle'); // idle | loading | ready | unavailable
  const [dayJobState, setDayJobState] = useState('idle');
  const [dirty, setDirty] = useState(false);

  const loadDetail = useCallback(async () => {
    if (!segmentId) return;
    setState('loading');
    try {
      const response = await fetch(`${API_BASE}/api/schedule/segment/${encodeURIComponent(segmentId)}`);
      if (!response.ok) throw new Error(`${response.status}`);
      setDetail(await response.json());
      setState('ready');
    } catch {
      setState('error');
    }
  }, [segmentId]);

  useEffect(() => { loadDetail(); }, [loadDetail]);

  // Ask the engine for the WITH/WITHOUT delta of the candidate edit before it is
  // saved. Real numbers only; anything unparseable reads as honest empty.
  useEffect(() => {
    if (!segmentId) { setPreview(null); setPreviewState('idle'); return undefined; }
    if (kind === 'force' && !(Number(countValue) >= 0)) { setPreview(null); setPreviewState('idle'); return undefined; }
    let cancelled = false;
    setPreviewState('loading');
    const params = new URLSearchParams({ target_id: segmentId, scope: 'segment', kind });
    if (kind === 'force') params.set('value', String(Number(countValue)));
    if (kind === 'gold') params.set('gold', 'true');
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/api/overrides/effect?${params.toString()}`);
        if (!response.ok) throw new Error(`${response.status}`);
        const json = await response.json();
        if (cancelled) return;
        if (json && json.available === false) { setPreview(null); setPreviewState('unavailable'); return; }
        setPreview(json);
        setPreviewState('ready');
      } catch {
        if (!cancelled) { setPreview(null); setPreviewState('unavailable'); }
      }
    })();
    return () => { cancelled = true; };
  }, [segmentId, kind, countValue]);

  const previewRows = useMemo(() => {
    const summary = preview && preview.summary;
    if (!summary) return [];
    const descriptors = [
      { before: 'before_revenue', after: 'after_revenue', en: 'Day revenue', he: 'הכנסת היום', money: true },
      { before: 'before_total_breaks', after: 'after_total_breaks', en: 'Day breaks', he: 'ברייקים ביום', money: false },
    ];
    return descriptors.map((d) => {
      const a = isNum(summary[d.before]) ? summary[d.before] : undefined;
      const b = isNum(summary[d.after]) ? summary[d.after] : undefined;
      if (!isNum(a) && !isNum(b)) return null;
      const diff = isNum(a) && isNum(b) ? b - a : undefined;
      return { label: pageText(locale, d.en, d.he), a, b, diff, money: d.money };
    }).filter(Boolean);
  }, [preview, locale]);

  async function handleSave() {
    if (!detail) return;
    if (kind === 'force' && !(Number(countValue) >= 0)) {
      notify('Enter a break count of 0 or more.', 'הזינו מספר ברייקים אפס ומעלה.');
      return;
    }
    const anchor = detail.anchor || {};
    const body = {
      scope: 'segment',
      target_id: segmentId,
      kind,
      source: 'manual',
      anchor_date: anchor.date,
      anchor_start: anchor.start_clock,
      anchor_title: anchor.program,
    };
    if (kind === 'force') body.value = Number(countValue);
    if (kind === 'gold') body.gold = true;
    try {
      const response = await fetch(`${API_BASE}/api/overrides`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setDirty(true);
      notify('Decision saved. The plan is now marked stale; recompute this day to apply it.',
        'ההחלטה נשמרה. התוכנית מסומנת כלא מעודכנת; חשבו מחדש את היום כדי להחיל אותה.');
      await loadDetail();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Save failed (${error.message}).`, `השמירה נכשלה (${error.message}).`);
    }
  }

  async function handleRemove(overrideId) {
    try {
      const response = await fetch(`${API_BASE}/api/overrides/${encodeURIComponent(overrideId)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(`${response.status}`);
      setDirty(true);
      notify('Decision removed. Recompute this day to apply.', 'ההחלטה הוסרה. חשבו מחדש את היום כדי להחיל.');
      await loadDetail();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Remove failed (${error.message}).`, `ההסרה נכשלה (${error.message}).`);
    }
  }

  async function handleDayRecompute() {
    const ch = channel || detail?.identity?.channel;
    const d = day || detail?.identity?.date;
    if (!ch || !d) return;
    setDayJobState('running');
    try {
      const result = await runDayRecomputeJob(API_BASE, [{ channel: ch, day: d }]);
      setDayJobState('idle');
      if (result.status === 'done') {
        setDirty(false);
        notify('Day recomputed. The plan reflects your decision.', 'היום חושב מחדש. התוכנית משקפת את ההחלטה.');
        await loadDetail();
        onGlobalRefresh?.();
      } else if (result.status === 'missing') {
        notify('Day recompute needs the updated backend. Use the full recompute instead.',
          'חישוב יום דורש שרת מעודכן. השתמשו בחישוב המלא במקום.');
      } else {
        const reason = result.error || (result.status === 'timeout' ? 'timed out' : 'unknown error');
        notify(`Day recompute failed: ${reason}.`, `חישוב היום נכשל: ${reason}.`);
      }
    } catch (error) {
      setDayJobState('idle');
      notify(`Day recompute failed (${error.message}).`, `חישוב היום נכשל (${error.message}).`);
    }
  }

  async function handleDownload() {
    try {
      const response = await fetch(`${API_BASE}/api/export/schedule.csv`);
      if (!response.ok) throw new Error(`${response.status}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'kairos-weekly-schedule.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      notify('Downloaded the current saved schedule.', 'הורד לוח השידורים השמור הנוכחי.');
    } catch (error) {
      notify(`Download failed (${error.message}).`, `ההורדה נכשלה (${error.message}).`);
    }
  }

  const id = detail?.identity || {};
  const plan = detail?.plan || {};
  const eco = detail?.economics || {};
  const ret = detail?.retention || {};
  const overrides = Array.isArray(detail?.overrides) ? detail.overrides : [];

  return (
    <aside className="schedule-inspector" role="dialog" aria-label={pageText(locale, 'Programme inspector', 'מפקח תוכנית')} dir={he ? 'rtl' : 'ltr'}>
      <div className="si-head">
        <div>
          <span className="si-kicker">{pageText(locale, 'Programme inspector', 'מפקח תוכנית')}</span>
          <h3>{programTypeLabel(id.program_type, locale) || pageText(locale, 'Programme', 'תוכנית')}</h3>
        </div>
        <button type="button" className="si-close" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
          <X size={18} />
        </button>
      </div>

      {state === 'loading' && <p className="si-note">{pageText(locale, 'Loading the full detail...', 'טוען את הפרטים המלאים...')}</p>}
      {state === 'error' && <p className="si-note">{pageText(locale, 'This programme detail is unavailable right now.', 'פרטי התוכנית אינם זמינים כרגע.')}</p>}

      {state === 'ready' && detail && (
        <div className="si-body">
          <section className="si-section">
            <h4>{pageText(locale, 'Identity', 'זיהוי')}</h4>
            <Row label={pageText(locale, 'Channel', 'ערוץ')} value={id.channel || '-'} />
            <Row label={pageText(locale, 'Date', 'תאריך')} value={`${id.date || '-'} ${id.day ? `(${id.day})` : ''}`} />
            <Row label={pageText(locale, 'Start', 'התחלה')} value={id.start_clock || '-'} />
            <Row label={pageText(locale, 'Class', 'מחלקה')} value={programTypeLabel(id.program_type, locale) || '-'} />
          </section>

          <section className="si-section">
            <h4>{pageText(locale, 'Break plan', 'תוכנית ברייקים')}</h4>
            <Row label={pageText(locale, 'Breaks', 'ברייקים')} value={fmtNum2(plan.num_breaks, locale)} />
            <Row label={pageText(locale, 'Break length', 'אורך ברייק')} value={isNum(plan.break_length_seconds) ? `${Math.round(plan.break_length_seconds)}s` : '-'} />
            <Row label={pageText(locale, 'Total ad time', 'זמן פרסום כולל')} value={isNum(plan.total_break_seconds) ? `${Math.round(plan.total_break_seconds)}s` : '-'} />
            <Row label={pageText(locale, 'Gold', 'זהב')} value={plan.is_gold ? pageText(locale, 'Yes', 'כן') : pageText(locale, 'No', 'לא')} />
          </section>

          <section className="si-section">
            <h4>{pageText(locale, 'Economics', 'כלכלה')}</h4>
            <Row label={pageText(locale, 'Plan revenue (this segment)', 'הכנסת תוכנית (סגמנט זה)')} value={fmtMoney(eco.predicted_revenue, locale)} />
            <Row label={pageText(locale, 'Base rate', 'תעריף בסיס')} value={fmtNum2(eco.base_rate, locale)} />
            <Row label={pageText(locale, 'Baseline audience (TVR)', 'קהל בסיס (TVR)')} value={fmtNum2(eco.baseline_tvr, locale)} />
          </section>

          <section className="si-section">
            <h4>{pageText(locale, 'Retention', 'שימור')}</h4>
            <Row label={pageText(locale, 'Plan retention (this segment)', 'שימור בתוכנית (סגמנט זה)')} value={isNum(ret.predicted_retention) ? `${fmtNum2(ret.predicted_retention, locale)}%` : '-'} />
            <Row
              label={pageText(locale, 'Credible interval', 'רווח סמך')}
              value={isNum(ret.ci_low) && isNum(ret.ci_high) ? `${fmtNum2(ret.ci_low, locale)}% - ${fmtNum2(ret.ci_high, locale)}%` : pageText(locale, 'not measured', 'לא נמדד')}
            />
            <Row label={pageText(locale, 'Sample size', 'גודל מדגם')} value={isNum(ret.sample_n) ? `n=${ret.sample_n}` : '-'} />
            <Row label={pageText(locale, 'Confidence', 'ביטחון')} value={ret.confidence || '-'} />
          </section>

          <section className="si-section">
            <h4>{pageText(locale, 'Active decisions', 'החלטות פעילות')}</h4>
            {overrides.length === 0 ? (
              <p className="si-empty">{pageText(locale, 'No manual decision on this programme yet.', 'אין עדיין החלטה ידנית על התוכנית הזו.')}</p>
            ) : (
              overrides.map((o) => (
                <div className="si-override" key={o.override_id}>
                  <span className="si-override-kind">{kindLabel(o.kind, locale)}{o.value ? ` (${o.value})` : ''}</span>
                  <span className="si-override-src">{o.source === 'recommendation' ? pageText(locale, 'from recommendation', 'מהמלצה') : pageText(locale, 'manual', 'ידני')}</span>
                  <button type="button" className="si-override-remove" onClick={() => handleRemove(o.override_id)}>
                    {pageText(locale, 'Remove', 'הסרה')}
                  </button>
                </div>
              ))
            )}
          </section>

          <section className="si-section si-edit">
            <h4>{pageText(locale, 'Make a decision', 'קבלת החלטה')}</h4>
            <p className="si-sub">{pageText(locale, 'The optimizer honors these on the next recompute. Read the projected delta before saving.', 'האופטימייזר מכבד אותן בחישוב הבא. קראו את הדלתא הצפויה לפני השמירה.')}</p>
            <label className="si-field">
              <span>{pageText(locale, 'Decision', 'החלטה')}</span>
              <select value={kind} onChange={(e) => setKind(e.target.value)}>
                {KINDS.map((entry) => (
                  <option key={entry.key} value={entry.key}>{kindLabel(entry.key, locale)}</option>
                ))}
              </select>
            </label>
            {kind === 'force' && (
              <label className="si-field">
                <span>{pageText(locale, 'Break count', 'מספר ברייקים')}</span>
                <input type="number" min="0" step="1" dir="ltr" value={countValue} onChange={(e) => setCountValue(e.target.value)} />
              </label>
            )}

            <div className="si-preview">
              {previewState === 'loading' && <p className="si-sub">{pageText(locale, 'Reading the projected delta...', 'קורא את הדלתא הצפויה...')}</p>}
              {previewState === 'unavailable' && <p className="si-sub">{pageText(locale, 'The preview is unavailable, so no projected delta is shown. The decision can still be saved.', 'התצוגה המקדימה אינה זמינה. עדיין ניתן לשמור.')}</p>}
              {previewState === 'ready' && previewRows.length > 0 && (
                <>
                  <div className="si-preview-row head">
                    <span>{pageText(locale, 'Metric', 'מדד')}</span>
                    <span className="num">{pageText(locale, 'Without', 'בלי')}</span>
                    <span className="num">{pageText(locale, 'With', 'עם')}</span>
                    <span className="num">{pageText(locale, 'Delta', 'דלתא')}</span>
                  </div>
                  {previewRows.map((r) => (
                    <div className="si-preview-row" key={r.label}>
                      <span>{r.label}</span>
                      <span className="num" dir="ltr">{r.money ? fmtMoney(r.a, locale) : fmtNum2(r.a, locale)}</span>
                      <span className="num" dir="ltr">{r.money ? fmtMoney(r.b, locale) : fmtNum2(r.b, locale)}</span>
                      <span className={`num si-delta ${isNum(r.diff) && r.diff > 0 ? 'up' : isNum(r.diff) && r.diff < 0 ? 'down' : ''}`} dir="ltr">
                        {isNum(r.diff) ? `${r.diff > 0 ? '+' : ''}${r.money ? fmtMoney(r.diff, locale) : fmtNum2(r.diff, locale)}` : '-'}
                      </span>
                    </div>
                  ))}
                </>
              )}
            </div>

            <Button className="primary-button compact" type="button" variant="contained" onClick={handleSave}>
              <SlidersHorizontal size={14} />
              {pageText(locale, 'Save decision', 'שמירת החלטה')}
            </Button>
          </section>

          <section className="si-section si-apply">
            <Button className="compact" type="button" variant="outlined" disabled={dayJobState === 'running'} onClick={handleDayRecompute}>
              <RefreshCcw size={14} className={dayJobState === 'running' ? 'spin' : undefined} />
              {dayJobState === 'running' ? pageText(locale, 'Recomputing this day', 'מחשב את היום') : pageText(locale, 'Recompute this day', 'חישוב מחדש ליום זה')}
            </Button>
            <Button className="compact" type="button" variant="text" onClick={handleDownload}>
              <Download size={14} />
              {pageText(locale, 'Download schedule', 'הורדת הלוח')}
            </Button>
            {dirty && <span className="si-stale">{pageText(locale, 'Saved schedule is stale, recompute to apply.', 'הלוח השמור אינו מעודכן, חשבו מחדש להחלה.')}</span>}
          </section>

          <p className="si-footnote">
            {pageText(
              locale,
              'This inspector edits break placement and counts. Which advertiser fills each slot is decided in the daily pricing step downstream, not here.',
              'המפקח עורך מיקום ומספר ברייקים. איזה מפרסם ממלא כל משבצת נקבע בשלב התמחור היומי במורד הזרם, לא כאן.',
            )}
          </p>
        </div>
      )}
    </aside>
  );
}
