import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Info, RefreshCcw, RotateCcw } from 'lucide-react';
import { pageText } from './advertisers-helpers';
import './pricing-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

// Maps a layer's display name to the YAML override key the engine reads.
const LAYER_TO_YAML = {
  program: 'program_type',
  day: 'day_of_week',
  show: 'show',
  position: 'position_in_break',
  ad_type: 'ad_type',
};

const DAY_NAMES = {
  1: ['Mon', 'שני'], 2: ['Tue', 'שלישי'], 3: ['Wed', 'רביעי'], 4: ['Thu', 'חמישי'],
  5: ['Fri', 'שישי'], 6: ['Sat', 'שבת'], 7: ['Sun', 'ראשון'],
};

function layerLabel(name) {
  if (name === 'ad_type') return 'Ad type';
  return name.charAt(0).toUpperCase() + name.slice(1);
}

// A day layer key is an ISO weekday; everything else shows its own key verbatim.
function keyLabel(layerName, key, locale) {
  if (layerName === 'day' && DAY_NAMES[key]) {
    return pageText(locale, DAY_NAMES[key][0], DAY_NAMES[key][1]);
  }
  return key;
}

function PricingManager({ copy, locale, notify }) {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [slot, setSlot] = useState({
    pricing_class: 'News', weekday_iso: 1, show: '', position: '', break_size: '', ad_type: '', advertiser_base: '',
  });
  const [breakdown, setBreakdown] = useState(null);
  const [testerError, setTesterError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/pricing`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      setState(await response.json());
      setOnline(true);
    } catch {
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Deep-merge a partial override onto the saved rate card and refresh.
  const applyOverride = useCallback(async (overrides, reset = false) => {
    try {
      const response = await fetch(`${API_BASE}/api/pricing`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ overrides, reset }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `${response.status} ${response.statusText}`);
      }
      setState(await response.json());
      notify('Rate card saved. It is live in the next optimizer run and forecast.',
        'כרטיס התעריפים נשמר. הוא פעיל בריצת האופטימייזר והתחזית הבאות.');
      return true;
    } catch (error) {
      notify(`Rate-card save failed (${error.message}).`, `שמירת כרטיס התעריפים נכשלה (${error.message}).`);
      return false;
    }
  }, [notify]);

  function saveBase(value) {
    const num = Number(value);
    if (!Number.isFinite(num) || num < 0) {
      notify('Base price must be a number of 0 or more.', 'מחיר הבסיס חייב להיות מספר אפס ומעלה.');
      return;
    }
    if (state && num === state.base.value) return;
    applyOverride({ base_price_per_second_per_tvr_point: num });
  }

  function saveMultiplier(layerName, key, value) {
    const num = Number(value);
    if (!Number.isFinite(num) || num < 0) {
      notify('A premium must be a number of 0 or more.', 'מקדם חייב להיות מספר אפס ומעלה.');
      return;
    }
    applyOverride({ premiums: { [LAYER_TO_YAML[layerName]]: { [key]: num } } });
  }

  function toggleLayer(layerName, enabled) {
    applyOverride({ pricing_activation: { [layerName]: enabled } });
  }

  function resetCard() {
    applyOverride({}, true);
  }

  const runTester = useCallback(async () => {
    const body = {
      pricing_class: slot.pricing_class || 'Other',
      weekday_iso: Number(slot.weekday_iso) || 1,
    };
    if (slot.show) body.show = slot.show;
    if (slot.position) body.position = Number(slot.position);
    if (slot.break_size) body.break_size = Number(slot.break_size);
    if (slot.ad_type) body.ad_type = slot.ad_type;
    if (slot.advertiser_base) body.advertiser_base = Number(slot.advertiser_base);
    try {
      const response = await fetch(`${API_BASE}/api/pricing/price-slot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      setBreakdown(await response.json());
      setTesterError(null);
    } catch (error) {
      setBreakdown(null);
      setTesterError(error.message);
      notify(`Price tester failed (${error.message}).`, `בודק המחיר נכשל (${error.message}).`);
    }
  }, [slot, notify]);

  // Recompute the tester whenever the inputs or the saved rate card change.
  useEffect(() => {
    if (online && state) runTester();
  }, [online, state, runTester]);

  const currency = state?.currency || 'ILS';
  const premiumLayers = useMemo(
    () => (state?.layers || []).filter((layer) => layer.kind === 'premium'),
    [state],
  );

  if (loading) {
    return (
      <section className="page-workspace">
        <div className="page-header"><h1>{pageText(locale, 'Pricing', 'תמחור')}</h1></div>
        <p>{pageText(locale, 'Loading the rate card...', 'טוען את כרטיס התעריפים...')}</p>
      </section>
    );
  }

  if (!online || !state) {
    return (
      <section className="page-workspace">
        <div className="page-header"><h1>{pageText(locale, 'Pricing', 'תמחור')}</h1></div>
        <div className="pricing-banner">
          <Info size={16} aria-hidden="true" />
          <p>{pageText(locale,
            'The pricing service is unreachable. No rate card is shown rather than a fabricated one.',
            'שירות התמחור אינו זמין. לא מוצג כרטיס תעריפים במקום להמציא נתון.')}</p>
        </div>
      </section>
    );
  }

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Pricing', 'תמחור')}</h1>
          <p>{pageText(locale,
            'The rate card: base price per rating point and the named premium layers that stack on top. Edit any value and watch the price recompute in the tester. Every number traces to base times named layers.',
            'כרטיס התעריפים: מחיר בסיס לנקודת רייטינג והשכבות הנקובות שמצטברות מעליו. ערכו כל ערך וצפו במחיר מתעדכן בבודק. כל מספר נגזר מבסיס כפול שכבות נקובות.')}</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={load}>
            <RefreshCcw size={14} />
            {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
          </Button>
          {state.has_overrides && (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={resetCard}>
              <RotateCcw size={14} />
              {pageText(locale, 'Reset to rate card', 'איפוס לתעריף')}
            </Button>
          )}
        </div>
      </div>

      <div className="pricing-banner">
        <Info size={16} aria-hidden="true" />
        <p>{state.has_overrides
          ? pageText(locale,
            'Operator edits applied. Every value traces to base times named layers. Saved edits are live in the next optimizer run, forecast and spot export.',
            'עריכות מפעיל הוחלו. כל ערך נגזר מבסיס כפול שכבות נקובות. עריכות שנשמרו פעילות בריצת האופטימייזר, התחזית וייצוא הספוטים הבאים.')
          : pageText(locale,
            'Rate card only. No operator edits yet. Position, ad-type and show layers ship activation-off, so revenue is unchanged until you turn a layer on here.',
            'כרטיס תעריפים בלבד. אין עדיין עריכות מפעיל. שכבות המיקום, סוג הפרסומת והתוכנית מסופקות כבויות, כך שההכנסה אינה משתנה עד שתפעילו שכבה כאן.')}</p>
      </div>

      <div className="pricing-grid">
        <div>
          <div className="pricing-base-card">
            <div className="pricing-base-row">
              <span className="pricing-layer-title">{pageText(locale, 'Base CPP', 'מחיר בסיס')}</span>
              <span className="pricing-base-value">
                <input
                  type="number" min="0" step="1" dir="ltr"
                  defaultValue={state.base.value}
                  key={`base-${state.base.value}`}
                  onBlur={(event) => saveBase(event.target.value)}
                  aria-label={pageText(locale, 'Base price per rating point per second', 'מחיר בסיס לנקודת רייטינג לשנייה')}
                />
              </span>
              <span className="pricing-base-note">
                {currency} / {pageText(locale, 'second / rating point. Base, not a premium.', 'שנייה / נקודת רייטינג. בסיס, לא מקדם.')}
              </span>
            </div>
          </div>

          <div className="pricing-layer-stack">
            {premiumLayers.map((layer) => {
              const entries = Object.entries(layer.values || {});
              const defaults = layer.defaults || {};
              const isEmpty = entries.length === 0;
              const chip = layer.live_today ? 'live' : (isEmpty ? 'empty' : 'off');
              const chipText = layer.live_today
                ? pageText(locale, 'Live', 'פעיל')
                : (isEmpty ? pageText(locale, 'Empty', 'ריק') : pageText(locale, 'Wired off', 'כבוי'));
              return (
                <div className="pricing-layer-card" key={layer.name}>
                  <div className="pricing-layer-head">
                    <div>
                      <span className="pricing-layer-title">{layerLabel(layer.name)}</span>
                      <p className="pricing-layer-desc">{layer.description}</p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span className={`pricing-chip ${chip}`}>{chipText}</span>
                      {layer.activatable && (
                        <label className="pricing-toggle">
                          <input
                            type="checkbox"
                            checked={!!layer.enabled}
                            onChange={(event) => toggleLayer(layer.name, event.target.checked)}
                          />
                          {pageText(locale, 'On', 'הפעלה')}
                        </label>
                      )}
                    </div>
                  </div>
                  {isEmpty ? (
                    <p className="pricing-empty">{pageText(locale,
                      'No values yet; defaults to 1.0 (no effect).',
                      'אין עדיין ערכים; ברירת המחדל 1.0 (ללא השפעה).')}</p>
                  ) : (
                    <div className="pricing-multipliers">
                      {entries.map(([key, value]) => {
                        const edited = defaults[key] !== undefined && Number(defaults[key]) !== Number(value);
                        return (
                          <div className={`pricing-mult${edited ? ' edited' : ''}`} key={key}>
                            <span>{keyLabel(layer.name, key, locale)}</span>
                            <input
                              type="number" min="0" step="0.01" dir="ltr"
                              defaultValue={value}
                              key={`${layer.name}-${key}-${value}`}
                              onBlur={(event) => saveMultiplier(layer.name, key, event.target.value)}
                              aria-label={`${layerLabel(layer.name)} ${key}`}
                            />
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="pricing-tester">
          <h3>{pageText(locale, 'Price any slot', 'תמחור משבצת')}</h3>
          <p className="pricing-base-note">{pageText(locale,
            'Pick a slot and read the full per-layer breakdown. Wired-off layers show struck-through, never multiplied into the live total.',
            'בחרו משבצת וקראו את הפירוט המלא לפי שכבה. שכבות כבויות מוצגות עם קו חוצה, ולעולם אינן נכפלות בסך החי.')}</p>
          <div className="pricing-tester-form">
            <label>
              {pageText(locale, 'Program class', 'מחלקת תוכנית')}
              <input value={slot.pricing_class} onChange={(e) => setSlot({ ...slot, pricing_class: e.target.value })} />
            </label>
            <label>
              {pageText(locale, 'Weekday', 'יום')}
              <select value={slot.weekday_iso} onChange={(e) => setSlot({ ...slot, weekday_iso: e.target.value })}>
                {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                  <option key={d} value={d}>{pageText(locale, DAY_NAMES[d][0], DAY_NAMES[d][1])}</option>
                ))}
              </select>
            </label>
            <label>
              {pageText(locale, 'Show', 'תוכנית')}
              <input value={slot.show} onChange={(e) => setSlot({ ...slot, show: e.target.value })} />
            </label>
            <label>
              {pageText(locale, 'Position', 'מיקום')}
              <input type="number" min="1" dir="ltr" value={slot.position} onChange={(e) => setSlot({ ...slot, position: e.target.value })} />
            </label>
            <label>
              {pageText(locale, 'Break size', 'גודל ברייק')}
              <input type="number" min="1" dir="ltr" value={slot.break_size} onChange={(e) => setSlot({ ...slot, break_size: e.target.value })} />
            </label>
            <label>
              {pageText(locale, 'Ad type', 'סוג פרסומת')}
              <input value={slot.ad_type} onChange={(e) => setSlot({ ...slot, ad_type: e.target.value })} />
            </label>
            <label>
              {pageText(locale, 'Advertiser base', 'בסיס מפרסם')}
              <input type="number" min="0" dir="ltr" value={slot.advertiser_base} onChange={(e) => setSlot({ ...slot, advertiser_base: e.target.value })} />
            </label>
          </div>

          {testerError && (
            <div className="pricing-breakdown">
              <p className="pricing-empty">{pageText(locale,
                `Could not price this slot (${testerError}). No breakdown is shown rather than a stale one.`,
                `לא ניתן לתמחר את המשבצת (${testerError}). לא מוצג פירוט במקום פירוט ישן.`)}</p>
            </div>
          )}

          {breakdown && !testerError && (
            <div className="pricing-breakdown">
              <div className="pricing-break-row">
                <span>{pageText(locale, 'Base CPP', 'מחיר בסיס')}</span>
                <span className="mult" dir="ltr">{Number(breakdown.base_cpp ?? 0).toFixed(2)}</span>
              </div>
              {(breakdown.layers || []).map((layer, idx) => (
                <div className="pricing-break-row" key={`live-${layer.name}-${idx}`}>
                  <span>x {layerLabel(layer.name)} <span className="src">({layer.source})</span></span>
                  <span className="mult" dir="ltr">{Number.isFinite(layer.multiplier) ? Number(layer.multiplier).toFixed(3) : '-'}</span>
                </div>
              ))}
              {(breakdown.wired_off_layers || []).map((layer, idx) => (
                <div className="pricing-break-row off" key={`off-${layer.name}-${idx}`}>
                  <span>x {layerLabel(layer.name)} <span className="src">({pageText(locale, 'wired off', 'כבוי')})</span></span>
                  <span className="mult" dir="ltr">{Number.isFinite(layer.multiplier) ? Number(layer.multiplier).toFixed(3) : '-'}</span>
                </div>
              ))}
              <div className="pricing-break-row total">
                <span>= {pageText(locale, 'Final CPP', 'מחיר סופי')} ({currency})</span>
                <span dir="ltr">{Number.isFinite(breakdown.final_cpp) ? Number(breakdown.final_cpp).toFixed(2) : '-'}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default PricingManager;
