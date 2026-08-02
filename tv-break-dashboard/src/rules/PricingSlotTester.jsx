import React, { useCallback, useEffect, useMemo, useState } from 'react';
import DateField from '../shell/DateField';
import { pageText } from '../shell/surface-helpers';
import { DAY_NAMES, layerLabel, sourceLabel } from './pricing-layers-lib';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// The price-any-slot tester panel of the Pricing page. It re-prices on every
// input change and on every saved rate-card change, and renders the full
// per-layer breakdown, including the events layer when a date is given and the
// backend applies it. Wired-off layers render struck through, never multiplied.
function PricingSlotTester({ state, locale, notify, currency }) {
  const [slot, setSlot] = useState({
    pricing_class: 'News', weekday_iso: 1, day: '', show: '', position: '', break_size: '', ad_type: '', advertiser_base: '', advertiser: '', campaign: '',
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
    // Naming an advertiser opts the slot into the personal-pricing path: the
    // backend resolves that advertiser's (and campaign's) targeted overrides.
    if (slot.advertiser.trim()) {
      body.advertiser = slot.advertiser.trim();
      if (slot.campaign.trim()) body.campaign = slot.campaign.trim();
    }
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

  // Re-price the tester whenever the inputs or the saved rate card change.
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
          <span className="pricing-base-note">{pageText(locale,
            'Positions are 1 to 5 and L for last. A spot equal to the break size is the L position, so set the break size to test L.',
            'המיקומים הם 1 עד 5 ו-L לאחרון. ספוט ששווה לגודל הברייק הוא מיקום L, לכן הגדירו גודל ברייק כדי לבדוק את L.')}</span>
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
        <label>
          {pageText(locale, 'Advertiser (optional)', 'מפרסם (לא חובה)')}
          <input dir="ltr" value={slot.advertiser} placeholder="ADV_01" onChange={(e) => setSlot({ ...slot, advertiser: e.target.value })} />
        </label>
        <label>
          {pageText(locale, 'Campaign (optional)', 'קמפיין (לא חובה)')}
          <input dir="ltr" value={slot.campaign} onChange={(e) => setSlot({ ...slot, campaign: e.target.value })} />
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
          {breakdown.position_key && (
            <div className="pricing-break-row">
              <span className="src">{pageText(locale,
                `Position resolved to ${breakdown.position_label_en || breakdown.position_key}`,
                `המיקום נקבע כ${breakdown.position_label_he || breakdown.position_key}`)}</span>
              <span className="mult" dir="ltr">{breakdown.position_key}</span>
            </div>
          )}
          <div className="pricing-break-row total">
            <span>= {pageText(locale, 'Final CPP', 'מחיר סופי')} ({currency})</span>
            <span dir="ltr">{Number.isFinite(breakdown.final_cpp) ? Number(breakdown.final_cpp).toFixed(2) : '-'}</span>
          </div>
          <OverrideBlocks breakdown={breakdown} advertiser={slot.advertiser.trim()} locale={locale} />
        </div>
      )}
    </div>
  );
}

// The personal-pricing blocks of a priced slot: which advertiser overrides were
// applied, which were shadowed by a more specific rule, and any final-price
// guardrail breaches. Each block renders only what the backend reported; an
// advertiser with no matching rule gets a plain no-match line, never silence.
function OverrideBlocks({ breakdown, advertiser, locale }) {
  const applied = Array.isArray(breakdown.applied_overrides) ? breakdown.applied_overrides : [];
  const shadowed = Array.isArray(breakdown.shadowed_overrides) ? breakdown.shadowed_overrides : [];
  const warnings = Array.isArray(breakdown.guardrail_warnings) ? breakdown.guardrail_warnings : [];
  return (
    <>
      {advertiser && applied.length > 0 && (
        <>
          <p className="pricing-base-note">{pageText(locale, `Personal pricing rules of ${advertiser} applied to this slot:`, `כללי תמחור אישיים של ${advertiser} שהוחלו על המשבצת:`)}</p>
          {applied.map((entry, idx) => (
            <div className="pricing-break-row" key={`applied-${entry.rule_id}-${idx}`}>
              <span dir="auto">{layerLabel(entry.target_layer || 'final', locale)} <span className="src" dir="ltr">({entry.rule_id})</span></span>
              <span className="mult" dir="ltr">{Number.isFinite(entry.multiplier) ? `x ${Number(entry.multiplier).toFixed(3)}` : '-'}</span>
            </div>
          ))}
        </>
      )}
      {advertiser && applied.length === 0 && (
        <p className="pricing-base-note">{pageText(locale, `No personal pricing rule of ${advertiser} matches this slot, so the rate-card price stands.`, `אף כלל תמחור אישי של ${advertiser} אינו תואם למשבצת זו, ולכן מחיר המחירון נשאר בתוקף.`)}</p>
      )}
      {advertiser && shadowed.length > 0 && (
        <>
          <p className="pricing-base-note">{pageText(locale, 'Rules not applied because a more specific rule wins the layer:', 'כללים שלא הוחלו כי כלל ממוקד יותר גובר באותה שכבה:')}</p>
          {shadowed.map((entry, idx) => (
            <div className="pricing-break-row off" key={`shadowed-${entry.rule_id}-${idx}`}>
              <span dir="auto">{layerLabel(entry.target_layer || 'final', locale)} <span className="src" dir="ltr">({entry.rule_id})</span></span>
              <span className="src" dir="auto">{pageText(locale, `${entry.winner_rule_id} wins`, `${entry.winner_rule_id} גובר`)}</span>
            </div>
          ))}
        </>
      )}
      {warnings.map((warning, idx) => (
        <p className="pricing-layer-warning" key={`guardrail-${warning.code || idx}`} dir="auto">
          {pageText(locale, 'Price guardrail breached: ', 'חריגה מגבול מחיר: ')}
          {warning.message || warning.code}
        </p>
      ))}
    </>
  );
}

export default PricingSlotTester;
