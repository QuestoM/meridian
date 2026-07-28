import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Info, RefreshCcw, RotateCcw } from 'lucide-react';
import { pageText } from './advertisers-helpers';
import { LAYER_TEXT, LAYER_TO_YAML, keyLabel, layerLabel } from './pricing-layers-lib';
import PricingEventsLayer from './PricingEventsLayer';
import PricingSlotTester from './PricingSlotTester';
import './pricing-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

function PricingManager({ copy, locale, notify, onGlobalRefresh }) {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  // Inline confirm step for the destructive reset: the first click only arms
  // it, the explicit confirm click performs it.
  const [confirmReset, setConfirmReset] = useState(false);

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
      if (reset) {
        notify('Rate card reset to its defaults. It can be restored from the Restore changes page.',
          'כרטיס התעריפים אופס לברירת המחדל. ניתן לשחזר מעמוד שחזור שינויים.');
      } else {
        notify('Rate card saved. It is live in the next optimizer run and forecast.',
          'כרטיס התעריפים נשמר. הוא פעיל בריצת האופטימייזר והתחזית הבאות.');
      }
      onGlobalRefresh?.();
      return true;
    } catch (error) {
      notify(`Rate-card save failed (${error.message}).`, `שמירת כרטיס התעריפים נכשלה (${error.message}).`);
      return false;
    }
  }, [notify, onGlobalRefresh]);

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
    setConfirmReset(false);
    applyOverride({}, true);
  }

  const currency = state?.currency || 'ILS';
  // The events layer renders through its own card, so it is filtered out of the
  // generic per-key multiplier stack (it has no key table, only a toggle).
  const premiumLayers = useMemo(
    () => (state?.layers || []).filter((layer) => layer.kind === 'premium' && layer.name !== 'events' && layer.name !== 'event'),
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
          {state.has_overrides && !confirmReset && (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmReset(true)}>
              <RotateCcw size={14} />
              {pageText(locale, 'Reset to rate card', 'איפוס לתעריף')}
            </Button>
          )}
          {state.has_overrides && confirmReset && (
            <span role="alertdialog" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12 }}>{pageText(locale, 'Reset deletes every operator edit on the rate card.', 'האיפוס ימחק את כל עריכות המפעיל בכרטיס התעריפים.')}</span>
              <Button className="secondary-button compact" type="button" variant="outlined" onClick={resetCard}>
                {pageText(locale, 'Confirm reset', 'אישור איפוס')}
              </Button>
              <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmReset(false)}>
                {pageText(locale, 'Cancel', 'ביטול')}
              </Button>
            </span>
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
            'Rate card only. No operator edits yet. Position, ad-type, show and events layers ship activation-off, so revenue is unchanged until you turn a layer on here.',
            'כרטיס תעריפים בלבד. אין עדיין עריכות מפעיל. שכבות המיקום, סוג הפרסומת, התוכנית והאירועים מסופקות כבויות, כך שההכנסה אינה משתנה עד שתפעילו שכבה כאן.')}</p>
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
                      <span className="pricing-layer-title">{layerLabel(layer.name, locale)}</span>
                      <p className="pricing-layer-desc">{(locale === 'he' && LAYER_TEXT[layer.name] && LAYER_TEXT[layer.name].descHe) || layer.description}</p>
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
                  {Array.isArray(layer.warnings) && layer.warnings.map((warning, index) => {
                    const categories = Array.isArray(warning.categories) ? warning.categories.join(', ') : '';
                    return (
                      <p className="pricing-layer-warning" key={`${layer.name}-warn-${index}`} role="status">
                        {pageText(
                          locale,
                          `Turning this layer on would zero the price for ${categories}, because its configured multiplier is 0. That category would earn no revenue until you change the multiplier.`,
                          `הפעלת השכבה תאפס את המחיר עבור ${categories}, מכיוון שהמכפיל שהוגדר הוא 0. הקטגוריה הזו לא תניב הכנסה עד לשינוי המכפיל.`,
                        )}
                      </p>
                    );
                  })}
                  {isEmpty ? (
                    <p className="pricing-empty">{pageText(locale,
                      'No values yet; defaults to 1.0 (no effect).',
                      'אין עדיין ערכים; ברירת המחדל 1.0 (ללא השפעה).')}</p>
                  ) : (
                    <div className="pricing-multipliers">
                      {entries.map(([key, value]) => {
                        const edited = defaults[key] !== undefined && Number(defaults[key]) !== Number(value);
                        const label = keyLabel(layer.name, key, locale);
                        return (
                          <div className={`pricing-mult${edited ? ' edited' : ''}`} key={key}>
                            <span className="pricing-mult-label" title={label}>{label}</span>
                            <input
                              type="number" min="0" step="0.01" dir="ltr"
                              defaultValue={value}
                              key={`${layer.name}-${key}-${value}`}
                              onBlur={(event) => saveMultiplier(layer.name, key, event.target.value)}
                              aria-label={`${layerLabel(layer.name, locale)} ${label}`}
                            />
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
            <PricingEventsLayer
              state={state}
              locale={locale}
              onToggle={(enabled) => toggleLayer('events', enabled)}
            />
          </div>
        </div>

        <PricingSlotTester state={state} locale={locale} notify={notify} currency={currency} />
      </div>
    </section>
  );
}

export default PricingManager;
