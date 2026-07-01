import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Info, RefreshCcw, SlidersHorizontal, Trash2 } from 'lucide-react';
import { pageText } from './advertisers-helpers';
import './override-console.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

// Backend override kinds are pin | force | forbid | gold. "force" carries a
// target break count (the lower-count lever); "gold" carries gold:true.
const KINDS = [
  { key: 'pin', en: 'Pin current plan', he: 'נעילת התוכנית הנוכחית' },
  { key: 'force', en: 'Force a break count', he: 'קיבוע מספר ברייקים' },
  { key: 'forbid', en: 'Forbid breaks here', he: 'מניעת ברייקים כאן' },
  { key: 'gold', en: 'Mark as gold', he: 'סימון כזהב' },
];

const asList = (json, key) => (Array.isArray(json) ? json : (json && json[key]) || []);
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

function pick(obj, keys) {
  if (!obj) return undefined;
  for (const k of keys) {
    if (isNum(obj[k])) return obj[k];
  }
  return undefined;
}

function fmtNum(value, locale) {
  if (!isNum(value)) return '-';
  return value.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 });
}

function anchorText(o) {
  const parts = [o.anchor_date, o.anchor_start, o.anchor_title].filter(Boolean);
  return parts.length ? parts.join(' - ') : '';
}

// An override reads stale when the backend says so, or when its anchor no
// longer matches the live segment carrying the same id (a re-ingest drifted).
function isStale(o, segById) {
  if (o.status === 'stale') return true;
  const seg = segById.get(o.target_id);
  if (!seg || !seg.anchor) return false;
  const a = seg.anchor;
  const drift = (o.anchor_date && a.date && o.anchor_date !== a.date)
    || (o.anchor_start && a.start_clock && o.anchor_start !== a.start_clock)
    || (o.anchor_title && a.title && o.anchor_title !== a.title);
  return Boolean(drift);
}

function OverrideConsole({ copy, locale, notify, onGlobalRefresh }) {
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
  const [preview, setPreview] = useState(null);
  const [lastCreated, setLastCreated] = useState(null);
  const [dayJobState, setDayJobState] = useState('idle');
  const [previewState, setPreviewState] = useState('idle'); // idle | loading | ready | unavailable

  const loadOverrides = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/overrides`);
      if (!response.ok) throw new Error(`${response.status}`);
      setOverrides(asList(await response.json(), 'overrides'));
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

  const segById = useMemo(() => {
    const map = new Map();
    segments.forEach((s) => map.set(s.segment_id, s));
    return map;
  }, [segments]);

  const selectedSeg = segById.get(segId) || null;

  const visibleSegments = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return segments.slice(0, 200);
    return segments.filter((s) => {
      const a = s.anchor || {};
      return [s.segment_id, s.day, s.channel, a.title, a.date, a.start_clock]
        .filter(Boolean).some((f) => String(f).toLowerCase().includes(q));
    }).slice(0, 200);
  }, [segments, search]);

  // Ask the engine for the WITH/WITHOUT delta of the candidate override before
  // it is committed. Real numbers only; anything unparseable reads as honest empty.
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
    if (!preview) return [];
    const wo = preview.without || preview.baseline || preview.before;
    const wi = preview.with || preview.candidate || preview.after;
    const descriptors = [
      { keys: ['predicted_revenue', 'revenue'], en: 'Predicted revenue', he: 'הכנסה חזויה' },
      { keys: ['retention'], en: 'Retention', he: 'שימור' },
      { keys: ['num_breaks', 'breaks'], en: 'Breaks', he: 'ברייקים' },
    ];
    return descriptors.map((d) => {
      const a = pick(wo, d.keys);
      const b = pick(wi, d.keys);
      const delta = pick(preview.delta, d.keys);
      if (!isNum(a) && !isNum(b) && !isNum(delta)) return null;
      const diff = isNum(delta) ? delta : (isNum(a) && isNum(b) ? b - a : undefined);
      return { label: pageText(locale, d.en, d.he), a, b, diff };
    }).filter(Boolean);
  }, [preview, locale]);

  async function handleCreate() {
    if (!selectedSeg) return;
    if (kind === 'force' && !(Number(countValue) >= 0)) {
      notify('Enter a break count of 0 or more.', 'הזינו מספר ברייקים אפס ומעלה.');
      return;
    }
    const seg = selectedSeg;
    const body = {
      scope: 'segment',
      target_id: seg.segment_id,
      kind,
      source: 'manual',
      notes: notes.trim() || undefined,
      anchor_date: seg.anchor?.date,
      anchor_start: seg.anchor?.start_clock,
      anchor_title: seg.anchor?.title,
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
      notify('Override saved. Recompute when ready to apply it to the plan.',
        'העקיפה נשמרה. הריצו חישוב מחדש כשתרצו להחיל אותה על התוכנית.');
      setNotes('');
      const day = seg.day || seg.anchor?.date || '';
      if (seg.channel && day) setLastCreated({ channel: seg.channel, day });
      await loadOverrides();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Override save failed (${error.message}).`, `שמירת העקיפה נכשלה (${error.message}).`);
    }
  }

  async function handleDayRecompute() {
    if (!lastCreated) return;
    setDayJobState('running');
    const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
    try {
      const startResponse = await fetch(`${API_BASE}/api/jobs/recompute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scope: [lastCreated] }),
      });
      if (startResponse.status === 404) {
        setDayJobState('idle');
        notify('Day recompute needs the updated backend. Use the full recompute instead.',
          'חישוב מחדש ליום דורש שרת מעודכן. השתמשו בחישוב המלא במקום.');
        return;
      }
      if (!startResponse.ok) throw new Error(`${startResponse.status} ${startResponse.statusText}`);
      const { job_id: jobId } = await startResponse.json();
      for (let attempt = 0; attempt < 120; attempt += 1) {
        await sleep(1000);
        const statusResponse = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        if (!statusResponse.ok) throw new Error(`${statusResponse.status} ${statusResponse.statusText}`);
        const record = await statusResponse.json();
        if (record.status === 'done') {
          setDayJobState('idle');
          setLastCreated(null);
          notify('Day recomputed. The plan now reflects the override.',
            'היום חושב מחדש. התוכנית משקפת כעת את העקיפה.');
          onGlobalRefresh?.();
          return;
        }
        if (record.status === 'failed') {
          setDayJobState('idle');
          notify(`Day recompute failed: ${record.error || 'unknown error'}.`,
            `חישוב היום מחדש נכשל: ${record.error || 'שגיאה לא ידועה'}.`);
          return;
        }
      }
      throw new Error('timed out');
    } catch (error) {
      setDayJobState('idle');
      notify(`Day recompute failed (${error.message}).`, `חישוב היום מחדש נכשל (${error.message}).`);
    }
  }

  async function handleDelete(id) {
    try {
      const response = await fetch(`${API_BASE}/api/overrides/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      notify('Override removed. Recompute when ready.', 'העקיפה הוסרה. הריצו חישוב מחדש כשתרצו.');
      await loadOverrides();
      onGlobalRefresh?.();
    } catch (error) {
      notify(`Remove failed (${error.message}).`, `ההסרה נכשלה (${error.message}).`);
    }
  }

  const kindLabel = (k) => {
    const found = KINDS.find((entry) => entry.key === k);
    return found ? pageText(locale, found.en, found.he) : k;
  };

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Overrides', 'עקיפות')}</h1>
          <p>{pageText(locale,
            'Manual decisions the optimizer honors. Pin, forbid, force a break count or mark a segment gold, carrying the segment anchor so the decision survives a re-ingest. Saving marks the plan stale; recompute when ready.',
            'החלטות ידניות שהאופטימייזר מכבד. נעלו, מנעו, קבעו מספר ברייקים או סמנו משבצת כזהב, תוך נשיאת עוגן המשבצת כך שההחלטה שורדת קליטה מחדש. שמירה מסמנת את התוכנית כלא מעודכנת; הריצו חישוב מחדש כשתרצו.')}</p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadAll}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      <div className="oc-banner">
        <Info size={16} aria-hidden="true" />
        <p>{pageText(locale,
          'An override changes a fingerprinted input, so the schedule reads stale until you run a recompute. Saving here never triggers the recompute on its own.',
          'עקיפה משנה קלט מזוהה, ולכן לוח השידורים נקרא כלא מעודכן עד שתריצו חישוב מחדש. שמירה כאן לעולם אינה מפעילה את החישוב מעצמה.')}</p>
      </div>

      <div className="oc-grid">
        <div className="oc-card">
          <h3>{pageText(locale, 'Create an override', 'יצירת עקיפה')}</h3>
          <p className="oc-sub">{pageText(locale,
            'Pick an owned-channel segment, choose the decision, and read the projected delta before you commit.',
            'בחרו משבצת בערוץ שבבעלותכם, בחרו את ההחלטה וקראו את הדלתא הצפויה לפני האישור.')}</p>

          {!segOnline && (
            <div className="oc-empty">{pageText(locale,
              'The segments service is unreachable, so no segment can be chosen. No list is shown rather than a fabricated one.',
              'שירות המשבצות אינו זמין, ולכן לא ניתן לבחור משבצת. לא מוצגת רשימה במקום להמציא נתון.')}</div>
          )}

          {segOnline && (
            <>
              <label className="oc-field">
                <span>{pageText(locale, 'Find a segment', 'חיפוש משבצת')}</span>
                <input type="search" value={search} onChange={(e) => setSearch(e.target.value)}
                  placeholder={pageText(locale, 'Search by day, title or id', 'חיפוש לפי יום, כותרת או מזהה')} />
              </label>
              <label className="oc-field">
                <span>{pageText(locale, 'Segment', 'משבצת')}</span>
                <select value={segId} onChange={(e) => setSegId(e.target.value)}>
                  <option value="">{pageText(locale, 'Select a segment', 'בחרו משבצת')}</option>
                  {visibleSegments.map((s) => {
                    const a = s.anchor || {};
                    const label = [a.date, a.start_clock, a.title || s.segment_id].filter(Boolean).join(' - ');
                    return <option key={s.segment_id} value={s.segment_id}>{label}</option>;
                  })}
                </select>
              </label>

              {selectedSeg && (
                <div className="oc-seg-current">
                  <span><b>{pageText(locale, 'Channel', 'ערוץ')}:</b> {selectedSeg.channel || '-'}</span>
                  <span><b>{pageText(locale, 'Breaks', 'ברייקים')}:</b> <span dir="ltr">{fmtNum(selectedSeg.current?.num_breaks, locale)}</span></span>
                  <span><b>{pageText(locale, 'Gold', 'זהב')}:</b> {selectedSeg.current?.is_gold ? pageText(locale, 'Yes', 'כן') : pageText(locale, 'No', 'לא')}</span>
                  <span><b>{pageText(locale, 'Revenue', 'הכנסה')}:</b> <span dir="ltr">{fmtNum(selectedSeg.current?.predicted_revenue, locale)}</span></span>
                  <span><b>{pageText(locale, 'Retention', 'שימור')}:</b> <span dir="ltr">{fmtNum(selectedSeg.current?.retention, locale)}</span></span>
                </div>
              )}

              <label className="oc-field">
                <span>{pageText(locale, 'Decision', 'החלטה')}</span>
                <select value={kind} onChange={(e) => setKind(e.target.value)}>
                  {KINDS.map((entry) => (
                    <option key={entry.key} value={entry.key}>{pageText(locale, entry.en, entry.he)}</option>
                  ))}
                </select>
              </label>

              {kind === 'force' && (
                <label className="oc-field">
                  <span>{pageText(locale, 'Break count', 'מספר ברייקים')}</span>
                  <input type="number" min="0" step="1" dir="ltr" value={countValue}
                    onChange={(e) => setCountValue(e.target.value)} />
                </label>
              )}

              <label className="oc-field">
                <span>{pageText(locale, 'Notes (optional)', 'הערות (רשות)')}</span>
                <textarea value={notes} onChange={(e) => setNotes(e.target.value)} />
              </label>

              {selectedSeg && (
                <div className="oc-preview">
                  {previewState === 'loading' && (
                    <p className="oc-sub">{pageText(locale, 'Reading the projected delta...', 'קורא את הדלתא הצפויה...')}</p>
                  )}
                  {previewState === 'unavailable' && (
                    <p className="oc-sub">{pageText(locale,
                      'The preview is unavailable, so no projected delta is shown. The override can still be saved.',
                      'התצוגה המקדימה אינה זמינה, ולכן לא מוצגת דלתא צפויה. עדיין ניתן לשמור את העקיפה.')}</p>
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
                        <span className="num">{pageText(locale, 'Delta', 'דלתא')}</span>
                      </div>
                      {previewRows.map((row) => (
                        <div className="oc-preview-row" key={row.label}>
                          <span>{row.label}</span>
                          <span className="num" dir="ltr">{fmtNum(row.a, locale)}</span>
                          <span className="num" dir="ltr">{fmtNum(row.b, locale)}</span>
                          <span className={`num oc-delta ${isNum(row.diff) && row.diff > 0 ? 'up' : isNum(row.diff) && row.diff < 0 ? 'down' : ''}`} dir="ltr">
                            {isNum(row.diff) ? `${row.diff > 0 ? '+' : ''}${fmtNum(row.diff, locale)}` : '-'}
                          </span>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}

              <div style={{ marginTop: 14, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <Button className="primary-button compact" type="button" variant="contained"
                  disabled={!selectedSeg} onClick={handleCreate}>
                  <SlidersHorizontal size={14} />
                  {pageText(locale, 'Save override', 'שמירת עקיפה')}
                </Button>
                {lastCreated && (
                  <Button className="compact" type="button" variant="outlined"
                    disabled={dayJobState === 'running'} onClick={handleDayRecompute}>
                    <RefreshCcw size={14} className={dayJobState === 'running' ? 'spin' : undefined} />
                    {dayJobState === 'running' ? pageText(locale, 'Recomputing day', 'מחשב את היום') : pageText(locale, 'Recompute this day', 'חישוב מחדש ליום זה')}
                  </Button>
                )}
              </div>
            </>
          )}
        </div>

        <div className="oc-card">
          <h3>{pageText(locale, 'Current overrides', 'עקיפות פעילות')}</h3>
          <p className="oc-sub">{pageText(locale,
            'Every override the optimizer will honor on the next run. A stale marker means the anchor no longer matches the live segment.',
            'כל עקיפה שהאופטימייזר יכבד בריצה הבאה. סימון ״לא מעודכן״ פירושו שהעוגן אינו תואם עוד את המשבצת החיה.')}</p>

          {loading && <p className="oc-sub">{pageText(locale, 'Loading overrides...', 'טוען עקיפות...')}</p>}

          {!loading && !online && (
            <div className="oc-empty">{pageText(locale,
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
              {overrides.map((o) => {
                const stale = isStale(o, segById);
                const anchor = anchorText(o);
                const fromRec = o.source && o.source !== 'manual';
                return (
                  <div className={`oc-row${stale ? ' stale' : ''}`} key={o.id}>
                    <div className="oc-row-main">
                      <p className="oc-row-title">{o.anchor_title || o.target_id}</p>
                      <div style={{ marginBottom: 4 }}>
                        <span className="oc-chip kind">{kindLabel(o.kind)}{o.kind === 'force' && isNum(o.value) ? ` (${o.value})` : ''}</span>
                        {stale
                          ? <span className="oc-chip staleflag">{pageText(locale, 'Stale', 'לא מעודכן')}</span>
                          : o.status === 'dismissed'
                            ? <span className="oc-chip dismissed">{pageText(locale, 'Dismissed', 'בוטלה')}</span>
                            : <span className="oc-chip active">{pageText(locale, 'Active', 'פעילה')}</span>}
                        {fromRec && <span className="oc-chip rec">{pageText(locale, 'From recommendation', 'מהמלצה')}</span>}
                      </div>
                      <div className="oc-row-meta">
                        {anchor && <span dir="ltr">{anchor}</span>}
                        {anchor && <br />}
                        {pageText(locale, 'Segment', 'משבצת')}: <span dir="ltr">{o.target_id}</span>
                        {o.notes ? ` - ${o.notes}` : ''}
                        {stale && (
                          <>
                            <br />
                            {pageText(locale,
                              'The anchor no longer matches a live segment. Review before the next recompute.',
                              'העוגן אינו תואם עוד משבצת חיה. בדקו לפני החישוב הבא.')}
                          </>
                        )}
                      </div>
                    </div>
                    <Button className="secondary-button compact" type="button" variant="outlined"
                      onClick={() => handleDelete(o.id)}
                      aria-label={pageText(locale, 'Delete override', 'מחיקת עקיפה')}>
                      <Trash2 size={14} />
                    </Button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default OverrideConsole;
