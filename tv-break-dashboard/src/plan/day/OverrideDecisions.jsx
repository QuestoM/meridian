import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '../../studio/actions';
import { Info, RefreshCcw, SlidersHorizontal, Trash2 } from 'lucide-react';
import { InputControl, SelectControl, TextAreaControl } from '../../studio/dom-controls';
import { formatCurrency, formatPercent, pageText } from '../../shell/surface-helpers';
import { Figure, Code, Name } from '../../shell/bidi';
import { asList, isNum, fmtNum, anchorText, isStale, runDayPlanJob, KINDS, kindLabel } from './override-console-lib';
import { LIVE_PLAN, withBasis } from './plan-basis';
import OverrideDecisionDialogs from './OverrideDecisionDialogs';
import DayRunSafetyNotice from './DayRunSafetyNotice';
import { useScopedDayRun } from './use-scoped-day-run';
import './override-console.css';
import './override-console-studio.css';
import './master-control-broadcast.css';
const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

function OverrideDecisions({ copy, locale, notify, onGlobalRefresh, prefill, onPrefillConsumed }) {
  const [overrides, setOverrides] = useState([]);
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [segOnline, setSegOnline] = useState(true);
  const [search, setSearch] = useState('');
  const [segId, setSegId] = useState('');
  const [kind, setKind] = useState('pin');
  const [countValue, setCountValue] = useState('');
  const [notes, setNotes] = useState('');
  const [prefillRecId, setPrefillRecId] = useState('');
  const [preview, setPreview] = useState(null);
  const [lastCreated, setLastCreated] = useState(null);
  const [previewState, setPreviewState] = useState('idle'); // idle | loading | ready | unavailable
  const [visibleOverrideCount, setVisibleOverrideCount] = useState(12);
  const [pendingDelete, setPendingDelete] = useState(null);
  const dayRun = useScopedDayRun({
    scope: lastCreated ? [lastCreated] : null,
    runner: runDayPlanJob,
    locale,
    notify,
    success: {
      en: 'The day ran. The plan now reflects the override.',
      he: 'היום הורץ. התוכנית משקפת כעת את העקיפה.',
    },
    onDone: async () => {
      setLastCreated(null);
      onGlobalRefresh?.();
    },
  });
  const loadOverrides = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/overrides`);
      if (!response.ok) throw new Error(`${response.status}`);
      const payload = await response.json();
      const grouped = payload && payload.overrides ? payload.overrides : payload;
      setOverrides(Array.isArray(grouped) ? grouped : [...asList(grouped, 'segment'), ...asList(grouped, 'spot')]);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);
  const loadSegments = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/schedule/segments`);
      if (!response.ok) throw new Error(`${response.status}`);
      setSegments(asList(await response.json(), 'segments'));
      setSegOnline(true);
    } catch {
      setSegOnline(false);
    }
  }, []);
  const loadAll = useCallback(async () => {
    setLoading(true);
    await Promise.all([loadOverrides(), loadSegments()]);
    setLoading(false);
  }, [loadOverrides, loadSegments]);
  useEffect(() => { loadAll(); }, [loadAll]);
  useEffect(() => {
    if (!prefill || !prefill.segment_id) return;
    setSegId(prefill.segment_id);
    if (prefill.kind) setKind(prefill.kind);
    setPrefillRecId(prefill.rec_id || '');
    setSearch(prefill.segment_id);
    onPrefillConsumed?.();
  }, [prefill, onPrefillConsumed]);
  const segById = useMemo(() => {
    const map = new Map();
    segments.forEach((s) => map.set(s.segment_id, s));
    return map;
  }, [segments]);
  const selectedSeg = segById.get(segId) || null;
  const pendingDeleteOverride = overrides.find((item) => (item.override_id || item.id) === pendingDelete) || null;
  const visibleSegments = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return segments.slice(0, 200);
    return segments.filter((s) => {
      const a = s.anchor || {};
      return [s.segment_id, s.day, s.channel, a.program || a.title, a.date, a.start_clock]
        .filter(Boolean).some((f) => String(f).toLowerCase().includes(q));
    }).slice(0, 200);
  }, [segments, search]);

  useEffect(() => {
    if (!selectedSeg) { setPreview(null); setPreviewState('idle'); return; }
    if (kind === 'force' && !(Number(countValue) >= 0)) { setPreview(null); setPreviewState('idle'); return; }
    let cancelled = false;
    setPreviewState('loading');
    const params = new URLSearchParams({ target_id: selectedSeg.segment_id, scope: 'segment', kind });
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
  }, [selectedSeg, kind, countValue]);

  const previewRows = useMemo(() => {
    const summary = preview && preview.summary;
    if (!summary) return [];
    const descriptors = [
      { before: 'before_revenue', after: 'after_revenue', en: 'Day revenue', he: 'הכנסת היום' },
      { before: 'before_total_breaks', after: 'after_total_breaks', en: 'Day breaks', he: 'ברייקים ביום' },
    ];
    return descriptors.map((d) => {
      const a = isNum(summary[d.before]) ? summary[d.before] : undefined;
      const b = isNum(summary[d.after]) ? summary[d.after] : undefined;
      if (!isNum(a) && !isNum(b)) return null;
      const diff = isNum(a) && isNum(b) ? b - a : undefined;
      return { label: withBasis(pageText(locale, d.en, d.he), LIVE_PLAN, locale), a, b, diff };
    }).filter(Boolean);
  }, [preview, locale]);

  async function handleCreate() {
    if (!selectedSeg) return;
    if (kind === 'force' && !(Number(countValue) >= 0)) {
      notify('Enter a break count of 0 or more.', 'הזינו מספר ברייקים אפס ומעלה.');
      return;
    }
    const seg = selectedSeg;
    const fromRecommendation = Boolean(prefillRecId);
    const body = {
      scope: 'segment',
      target_id: seg.segment_id,
      kind,
      source: fromRecommendation ? 'recommendation' : 'manual',
      notes: notes.trim() || undefined,
      anchor_date: seg.anchor?.date,
      anchor_start: seg.anchor?.start_clock,
      anchor_title: seg.anchor?.program || seg.anchor?.title || '',
    };
    if (fromRecommendation) body.rec_id = prefillRecId;
    if (kind === 'force') body.value = Number(countValue);
    if (kind === 'gold') body.gold = true;
    try {
      const response = await fetch(`${API_BASE}/api/overrides`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      notify('Override saved. Run the plan when you are ready to apply it.',
        'העקיפה נשמרה. הריצו את התוכנית כשתרצו להחיל אותה.');
      setNotes('');
      setPrefillRecId('');
      const day = seg.day || seg.anchor?.date || '';
      if (seg.channel && day) setLastCreated({ channel: seg.channel, day });
      await loadOverrides();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Override save failed (${error.message}).`, `שמירת העקיפה נכשלה (${error.message}).`);
    }
  }

  async function handleDelete(id) {
    try {
      const response = await fetch(`${API_BASE}/api/overrides/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setPendingDelete(null);
      notify('Override removed. Run the plan when ready.', 'העקיפה הוסרה. הריצו את התוכנית כשתרצו.');
      await loadOverrides();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Remove failed (${error.message}).`, `ההסרה נכשלה (${error.message}).`);
    }
  }

  return (
    <section className="page-workspace broadcast-decisions" aria-busy={loading}>
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Overrides', 'עקיפות')}</h1>
          <p>{pageText(locale,
            'Plan input: manual segment decisions. Saving changes the input record; only a confirmed run rewrites the plan.',
            'קלט לתוכנית: החלטות ידניות ברמת משבצת. שמירה משנה את רשומת הקלט; רק הרצה מאושרת משכתבת את התוכנית.')}</p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadAll}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      <div className="oc-banner">
        <Info size={16} aria-hidden="true" />
        <p>{pageText(locale,
          'An override changes a fingerprinted input, so the plan reads out of date until you run it. Saving here never starts the run on its own.',
          'עקיפה משנה קלט מזוהה, ולכן התוכנית נקראת כלא מעודכנת עד שתריצו אותה. שמירה כאן לעולם אינה מפעילה את ההרצה מעצמה.')}</p>
      </div>

      <div className="oc-grid">
        <div className="card oc-card">
          <h3>{pageText(locale, 'Create an override', 'יצירת עקיפה')}</h3>
          <p className="oc-sub">{pageText(locale,
            'Pick an owned-channel segment, choose the decision, and read the projected change before you commit.',
            'בחרו משבצת בערוץ שבבעלותכם, בחרו את ההחלטה וקראו את השינוי הצפוי לפני האישור.')}</p>
          {prefillRecId && (
            <p className="oc-sub">
              <span className="oc-chip rec">{pageText(locale, 'From recommendation', 'מהמלצה')}</span>
              {' '}
              {pageText(locale,
                'This segment and decision came from an approved recommendation. Saving records the override with that provenance.',
                'המשבצת וההחלטה הגיעו מהמלצה שאושרה. שמירה תרשום את העקיפה עם ייחוס זה.')}
            </p>
          )}

          {!segOnline && (
            <div className="oc-empty">{pageText(locale,
              'The segments service is unreachable, so no segment can be chosen. No list is shown rather than a fabricated one.',
              'שירות המשבצות אינו זמין, ולכן לא ניתן לבחור משבצת. לא מוצגת רשימה במקום להמציא נתון.')}</div>
          )}

          {segOnline && (
            <>
              <label className="oc-field">
                <span>{pageText(locale, 'Find a segment', 'חיפוש משבצת')}</span>
                <InputControl type="search" value={search} onChange={(e) => setSearch(e.target.value)}
                  placeholder={pageText(locale, 'Search by day, title or id', 'חיפוש לפי יום, כותרת או מזהה')} />
              </label>
              <label className="oc-field">
                <span>{pageText(locale, 'Segment', 'משבצת')}</span>
                <SelectControl value={segId} onChange={(e) => setSegId(e.target.value)}>
                  <option value="">{pageText(locale, 'Select a segment', 'בחרו משבצת')}</option>
                  {visibleSegments.map((s) => {
                    const a = s.anchor || {};
                    const label = [a.date, a.start_clock, a.program || a.title || s.segment_id].filter(Boolean).join(' - ');
                    return <option key={s.segment_id} value={s.segment_id}>{label}</option>;
                  })}
                </SelectControl>
              </label>

              {selectedSeg && (
                <div className="oc-seg-current">
                  <span><b>{pageText(locale, 'Channel', 'ערוץ')}:</b> {selectedSeg.channel || '-'}</span>
                  <span><b>{pageText(locale, 'Breaks', 'ברייקים')}:</b> <Figure>{fmtNum(selectedSeg.state?.num_breaks, locale)}</Figure></span>
                  <span><b>{pageText(locale, 'Gold', 'זהב')}:</b> {selectedSeg.state?.is_gold ? pageText(locale, 'Yes', 'כן') : pageText(locale, 'No', 'לא')}</span>
                  <span><b>{pageText(locale, 'Revenue', 'הכנסה')}:</b> <Figure>{formatCurrency(selectedSeg.state?.predicted_revenue, locale)}</Figure></span>
                  <span><b>{pageText(locale, 'Retention', 'שימור')}:</b> <Figure>{formatPercent(selectedSeg.state?.retention, locale)}</Figure></span>
                </div>
              )}

              <label className="oc-field">
                <span>{pageText(locale, 'Decision', 'החלטה')}</span>
                <SelectControl value={kind} onChange={(e) => setKind(e.target.value)}>
                  {KINDS.map((entry) => (
                    <option key={entry.key} value={entry.key}>{pageText(locale, entry.en, entry.he)}</option>
                  ))}
                </SelectControl>
              </label>

              {kind === 'force' && (
                <label className="oc-field">
                  <span>{pageText(locale, 'Break count', 'מספר ברייקים')}</span>
                  <InputControl type="number" min="0" step="1" dir="ltr" value={countValue}
                    onChange={(e) => setCountValue(e.target.value)} />
                </label>
              )}

              <label className="oc-field">
                <span>{pageText(locale, 'Notes (optional)', 'הערות (רשות)')}</span>
                <TextAreaControl value={notes} onChange={(e) => setNotes(e.target.value)} />
              </label>

              {selectedSeg && (
                <div className="oc-preview">
                  {previewState === 'loading' && (
                    <p className="oc-sub">{pageText(locale, 'Reading the projected change...', 'קורא את השינוי הצפוי...')}</p>
                  )}
                  {previewState === 'unavailable' && (
                    <p className="oc-sub">{pageText(locale,
                      'The preview is unavailable, so no projected change is shown. The override can still be saved.',
                      'התצוגה המקדימה אינה זמינה, ולכן לא מוצג שינוי צפוי. עדיין ניתן לשמור את העקיפה.')}</p>
                  )}
                  {previewState === 'ready' && previewRows.length === 0 && (
                    <p className="oc-sub">{pageText(locale,
                      'The preview returned no comparable numbers.',
                      'התצוגה המקדימה לא החזירה מספרים להשוואה.')}</p>
                  )}
                  {previewState === 'ready' && previewRows.length > 0 && (
                    <>
                      <div className="oc-preview-row head">
                        <span>{pageText(locale, 'Metric', 'מדד')}</span>
                        <span className="num">{pageText(locale, 'Without', 'בלי')}</span>
                        <span className="num">{pageText(locale, 'With', 'עם')}</span>
                        <span className="num">{pageText(locale, 'Change', 'הפרש')}</span>
                      </div>
                      {previewRows.map((row) => (
                        <div className="oc-preview-row" key={row.label}>
                          <span>{row.label}</span>
                          <Figure className="num">{fmtNum(row.a, locale)}</Figure>
                          <Figure className="num">{fmtNum(row.b, locale)}</Figure>
                          <Figure className={`num oc-delta ${isNum(row.diff) && row.diff > 0 ? 'up' : isNum(row.diff) && row.diff < 0 ? 'down' : ''}`}>{isNum(row.diff) ? `${row.diff > 0 ? '+' : ''}${fmtNum(row.diff, locale)}` : '-'}</Figure>
                        </div>
                      ))}
                      {Array.isArray(preview?.rejected_overrides) && preview.rejected_overrides.length > 0 && (
                        <p className="oc-sub oc-rejected">{pageText(locale,
                          `The optimizer rejected ${preview.rejected_overrides.length} override(s) as infeasible: ${preview.rejected_overrides.map((r) => r.reason).filter(Boolean).join('; ') || 'no reason given'}.`,
                          `האופטימייזר דחה ${preview.rejected_overrides.length} עקיפות כלא ישימות: ${preview.rejected_overrides.map((r) => r.reason).filter(Boolean).join('; ') || 'ללא נימוק'}.`)}</p>
                      )}
                    </>
                  )}
                </div>
              )}

              <div className="oc-actions">
                <Button className="primary-button compact" type="button" variant="contained"
                  disabled={!selectedSeg} onClick={handleCreate}>
                  <SlidersHorizontal size={14} />
                  {pageText(locale, 'Save override', 'שמירת עקיפה')}
                </Button>
                {lastCreated && (
                  <Button className="compact" type="button" variant="outlined"
                    disabled={dayRun.jobState === 'running' || dayRun.safety.status === 'checking'} onClick={dayRun.requestReview}>
                    <RefreshCcw size={14} className={dayRun.jobState === 'running' ? 'spin' : undefined} />
                    {dayRun.jobState === 'running' ? pageText(locale, 'Running the day', 'מריץ את היום') : pageText(locale, 'Review this day run', 'בדיקת הרצת היום')}
                  </Button>
                )}
              </div>
              <DayRunSafetyNotice safety={dayRun.safety} locale={locale} />
            </>
          )}
        </div>

        <div className="card oc-card">
          <h3>{pageText(locale, 'Current overrides', 'עקיפות נוכחיות')}</h3>
          <p className="oc-sub">{pageText(locale,
            'Every override the optimizer will honor on the next run. A stale marker means the anchor no longer matches the live segment.',
            'כל עקיפה שהאופטימייזר יכבד בריצה הבאה. סימון ״לא מעודכן״ פירושו שהעוגן אינו תואם עוד את המשבצת החיה.')}</p>

          {loading && <p className="oc-sub" role="status">{pageText(locale, 'Loading overrides...', 'טוען עקיפות...')}</p>}

          {!loading && !online && (
            <div className="oc-empty" role="alert">{pageText(locale,
              'The overrides service is unreachable. No list is shown rather than a fabricated one.',
              'שירות העקיפות אינו זמין. לא מוצגת רשימה במקום להמציא נתון.')}</div>
          )}

          {!loading && online && overrides.length === 0 && (
            <div className="oc-empty">
              <span>{pageText(locale, 'No overrides yet.', 'אין עדיין עקיפות.')}</span>
              <span>{pageText(locale,
                'Create one on the left to steer a specific segment away from the model default.',
                'צרו עקיפה בצד כדי להסיט משבצת מסוימת מברירת המחדל של המודל.')}</span>
            </div>
          )}

          {!loading && online && overrides.length > 0 && (
            <div className="oc-list">
              {overrides.slice(0, visibleOverrideCount).map((o) => {
                const id = o.override_id || o.id;
                const forceCount = String(o.value ?? '').trim();
                const stale = isStale(o, segById);
                const anchor = anchorText(o);
                const fromRec = o.source && o.source !== 'manual';
                return (
                  <div className={`oc-row${stale ? ' stale' : ''}`} key={id}>
                    <div className="oc-row-main">
                      <p className="oc-row-title"><Name>{o.anchor_title || o.target_id}</Name></p>
                      <div className="oc-row-chips">
                        <span className="oc-chip kind">{kindLabel(o.kind, locale)}{o.kind === 'force' && forceCount ? ` (${forceCount})` : ''}</span>
                        {stale
                          ? <span className="oc-chip staleflag">{pageText(locale, 'Stale', 'לא מעודכנת')}</span>
                          : o.status === 'dismissed'
                            ? <span className="oc-chip dismissed">{pageText(locale, 'Dismissed', 'בוטלה')}</span>
                            : <span className="oc-chip active">{pageText(locale, 'Active', 'פעילה')}</span>}
                        {fromRec && <span className="oc-chip rec">{pageText(locale, 'From recommendation', 'מהמלצה')}</span>}
                      </div>
                      <div className="oc-row-meta">
                        {anchor && <span><Code>{anchor}</Code></span>}
                        <span>
                          {pageText(locale, 'Segment', 'משבצת')}: <Code>{o.target_id}</Code>
                          {o.notes ? <> · <Code>{o.notes}</Code></> : null}
                        </span>
                        {stale && (
                          <span className="oc-row-stale-copy">
                            {pageText(locale,
                              'The anchor no longer matches a live segment. Review it before the next run.',
                              'העוגן אינו תואם עוד משבצת חיה. בדקו לפני החישוב הבא.')}
                          </span>
                        )}
                      </div>
                    </div>
                    <Button className="secondary-button compact" type="button" variant="outlined"
                      onClick={() => setPendingDelete(id)}
                      aria-label={pageText(locale, `Remove override for ${o.anchor_title || o.target_id}`, `הסרת עקיפה עבור ${o.anchor_title || o.target_id}`)}>
                      <Trash2 size={16} aria-hidden="true" />
                    </Button>
                  </div>
                );
              })}
              {visibleOverrideCount < overrides.length && (
                <Button type="button" className="oc-show-more" variant="outlined" onClick={() => setVisibleOverrideCount((count) => count + 12)}>
                  {pageText(locale, `Show 12 more · ${overrides.length - visibleOverrideCount} remaining`, `הצגת 12 נוספות · נותרו ${overrides.length - visibleOverrideCount}`)}
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      <OverrideDecisionDialogs
        locale={locale}
        pendingDeleteOverride={pendingDeleteOverride}
        dayRun={dayRun}
        onCancelDelete={() => setPendingDelete(null)}
        onConfirmDelete={() => handleDelete(pendingDelete)}
        onCancelDayRun={dayRun.cancelReview}
        onConfirmDayRun={dayRun.confirmReview}
      />
    </section>
  );
}
export default OverrideDecisions;
