import React, { useCallback, useEffect, useMemo, useState } from 'react';
import DateField from './DateField';
import { pageText } from './advertisers-helpers';
import { DAY_NAMES, layerLabel, sourceLabel } from './pricing-layers-lib';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// The price-any-slot tester panel of the Pricing page. It re-prices on every
// input change and on every saved rate-card change, and renders the full
// per-layer breakdown, including the events layer when a date is given and the
// backend applies it. Wired-off layers render struck through, never multiplied.
function PricingSlotTester({ state, locale, notify, currency }) {
  const [slot, setSlot] = useState({
    pricing_class: 'News', weekday_iso: 1, day: '', show: '', position: '', break_size: '', ad_type: '', advertiser_base: '',
  });
  const [breakdown, setBreakdown] = useState(null);
  const [testerError, setTesterError] = useState(null);

  const premiumLayers = useMemo(
    () => (state?.layers || []).filter((layer) => layer.kind === 'premium'),
    [state],
  );

  // Known program classes and show names, sourced from the live rate card so the
  // tester's typeahead offers exactly what the engine prices. Free entry stays open;
  // anything off-list resolves to the Other/base rate on the backend.
  const layerKeys = useCallback(
    (name) => Object.keys(premiumLayers.find((layer) => layer.name === name)?.values || {}),
    [premiumLayers],
  );
  const classOptions = useMemo(
    () => Array.from(new Set([...layerKeys('program'), 'Other'])),
    [layerKeys],
  );
  const showOptions = useMemo(() => layerKeys('show'), [layerKeys]);
  const adTypeOptions = useMemo(() => layerKeys('ad_type'), [layerKeys]);

  const runTester = useCallback(async () => {
    const body = {
      pricing_class: slot.pricing_class || 'Other',
      weekday_iso: Number(slot.weekday_iso) || 1,
    };
    if (slot.day) body.day = slot.day;
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
    if (state) runTester();
  }, [state, runTester]);

  return (
    <div className="pricing-tester">
      <h3>{pageText(locale, 'Price any slot', 'תמחור משבצת')}</h3>
      <p className="pricing-base-note">{pageText(locale,
        'Pick a slot and read the full per-layer breakdown. Wired-off layers show struck-through, never multiplied into the live total.',
        'בחרו משבצת וקראו את הפירוט המלא לפי שכבה. שכבות כבויות מוצגות עם קו חוצה, ולעולם אינן נכפלות בסך החי.')}</p>
      <div className="pricing-tester-form">
        <label>
          {pageText(locale, 'Program class', 'מחלקת תוכנית')}
          <input
            list="pricing-class-options"
            value={slot.pricing_class}
            onChange={(e) => setSlot({ ...slot, pricing_class: e.target.value })}
          />
          <datalist id="pricing-class-options">
            {classOptions.map((option) => <option key={option} value={option} />)}
          </datalist>
        </label>
        <label>
          {pageText(locale, 'Weekday', 'יום')}
          <select value={slot.weekday_iso} onChange={(e) => setSlot({ ...slot, weekday_iso: e.target.value })}>
            {[1, 2, 3, 4, 5, 6, 7].map((d) => (
              <option key={d} value={d}>{pageText(locale, DAY_NAMES[d][0], DAY_NAMES[d][1])}</option>
            ))}
          </select>
        </label>
        <DateField
          label={pageText(locale, 'Date (optional)', 'תאריך (לא חובה)')}
          value={slot.day}
          onChange={(value) => setSlot({ ...slot, day: value })}
          helperText={pageText(locale, 'Used by the events layer when it is activated', 'משמש את שכבת האירועים כשהיא מופעלת')}
        />
        <label>
          {pageText(locale, 'Show', 'תוכנית')}
          <input
            list={showOptions.length ? 'pricing-show-options' : undefined}
            value={slot.show}
            onChange={(e) => setSlot({ ...slot, show: e.target.value })}
          />
          {showOptions.length > 0 && (
            <datalist id="pricing-show-options">
              {showOptions.map((option) => <option key={option} value={option} />)}
            </datalist>
          )}
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
          <input
            list={adTypeOptions.length ? 'pricing-ad-type-options' : undefined}
            value={slot.ad_type}
            onChange={(e) => setSlot({ ...slot, ad_type: e.target.value })}
          />
          {adTypeOptions.length > 0 && (
            <datalist id="pricing-ad-type-options">
              {adTypeOptions.map((option) => <option key={option} value={option} />)}
            </datalist>
          )}
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
              <span>x {layerLabel(layer.name, locale)} <span className="src">({sourceLabel(layer.source, locale)})</span></span>
              <span className="mult" dir="ltr">{Number.isFinite(layer.multiplier) ? Number(layer.multiplier).toFixed(3) : '-'}</span>
            </div>
          ))}
          {(breakdown.wired_off_layers || []).map((layer, idx) => (
            <div className="pricing-break-row off" key={`off-${layer.name}-${idx}`}>
              <span>x {layerLabel(layer.name, locale)} <span className="src">({pageText(locale, 'wired off', 'כבוי')})</span></span>
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
  );
}

export default PricingSlotTester;
