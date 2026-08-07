import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Info, RefreshCcw, RotateCcw } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Name } from '../shell/bidi';
import { LAYER_TO_YAML, categoryList, keyLabel, layerDescription, layerEntries, layerLabel } from './pricing-layers-lib';
import PricingEventsLayer from './PricingEventsLayer';
import PricingPreferredPositions from './PricingPreferredPositions';
import PricingSlotTester from './PricingSlotTester';
import RateCardEffect from './RateCardEffect';
import { detailWords, draftValueAt, dropOverride, mergeOverrides } from './rules-lib';
import './pricing-management.css';
import './rules-workspace.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

function PricingManager({ copy, locale, notify, onGlobalRefresh, embedded }) {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  // Inline confirm step for the destructive reset: the first click only arms
  // it, the explicit confirm click performs it.
  const [confirmReset, setConfirmReset] = useState(false);
  // An edit is a draft until it is saved. It used to land on blur, which meant
  // the revenue owner's own question, what does this do to the money, could only
  // be answered after the answer had already changed. The draft is priced
  // against the saved card and the save is a separate, deliberate act.
  const [pending, setPending] = useState(null);
  const [saving, setSaving] = useState(false);

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
        const body = await response.json().catch(() => ({}));
        const raw = body && body.detail;
        const words = raw && typeof raw === 'object' ? raw : null;
        const failure = new Error(words ? String(words.en || words.he || '') : (raw ? String(raw) : `${response.status} ${response.statusText}`));
        failure.words = words;
        throw failure;
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
      notify(`Rate-card save failed (${detailWords(error, 'en')}).`, `שמירת כרטיס התעריפים נכשלה (${detailWords(error, 'he')}).`);
      return false;
    }
  }, [notify, onGlobalRefresh]);

  function stage(patch) {
    setPending((current) => mergeOverrides(current || {}, patch));
  }

  // Typing the saved figure back into a box is a revert, not a no-op. Without
  // the drop, the draft kept the earlier edit while the box showed the saved
  // value, and the effect panel below priced a figure that was on nobody's
  // screen.
  function unstage(path) {
    setPending((current) => dropOverride(current, path));
  }

  function saveBase(value) {
    const num = Number(value);
    if (!Number.isFinite(num) || num < 0) {
      notify('Base price must be a number of 0 or more.', 'מחיר הבסיס חייב להיות מספר אפס ומעלה.');
      return;
    }
    if (state && num === state.base.value) {
      unstage(['base_price_per_second_per_tvr_point']);
      return;
    }
    stage({ base_price_per_second_per_tvr_point: num });
  }

  function saveMultiplier(layerName, key, value) {
    // An empty box means the key is UNSET, which is a real state on the position
    // layer (nobody has priced position 4 or 5). Blurring an untouched empty box
    // must leave it unset, never stage a premium of 0.
    if (String(value).trim() === '') {
      unstage(['premiums', LAYER_TO_YAML[layerName], key]);
      return;
    }
    const num = Number(value);
    if (!Number.isFinite(num) || num < 0) {
      notify('A premium must be a number of 0 or more.', 'מקדם חייב להיות מספר אפס ומעלה.');
      return;
    }
    const saved = Number(((state?.layers || []).find((entry) => entry.name === layerName) || {}).values?.[key]);
    if (Number.isFinite(saved) && num === saved) {
      unstage(['premiums', LAYER_TO_YAML[layerName], key]);
      return;
    }
    stage({ premiums: { [LAYER_TO_YAML[layerName]]: { [key]: num } } });
  }

  function toggleLayer(layerName, enabled) {
    stage({ pricing_activation: { [layerName]: enabled } });
  }

  async function commitPending() {
    setSaving(true);
    const ok = await applyOverride(pending || {});
    setSaving(false);
    if (ok) setPending(null);
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

  // What a draft-bound control shows: the staged edit while one exists, the
  // saved card the moment it is discarded. Measured on the shipped surface
  // before this, the base box was bound with defaultValue and a key made only of
  // the saved value, so discarding an edit cleared the draft, left the key
  // unchanged and left the discarded 80 in the box, while the price tester two
  // columns away read 60 and the server held 60. One screen, one rate card.
  const shownBase = draftValueAt(pending, ['base_price_per_second_per_tvr_point']) ?? state.base.value;

  return (
    <section className={embedded ? 'rules-section' : 'page-workspace'}>
      <div className="page-header">
        <div>
          {!embedded && <h1>{pageText(locale, 'Pricing', 'תמחור')}</h1>}
          <p>{pageText(locale,
            'The rate card: base price per rating point and the named premium layers that stack on top. Edit any value and see what it does to the worth of a second before you save. Every number traces to base times named layers.',
            'כרטיס התעריפים: מחיר בסיס לנקודת רייטינג והשכבות הנקובות שמצטברות מעליו. ערכו כל ערך וראו מה זה עושה לשווי של שנייה לפני השמירה. כל מספר נגזר מבסיס כפול שכבות נקובות.')}</p>
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
                  type="number" min="0" step="1"
                  defaultValue={shownBase}
                  key={`base-${shownBase}`}
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
              const entries = layerEntries(layer);
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
                      <p className="pricing-layer-desc">{layerDescription(layer, locale)}</p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span className={`pricing-chip ${chip}`}>{chipText}</span>
                      {layer.activatable && (
                        <label className="pricing-toggle">
                          <input
                            type="checkbox"
                            checked={draftValueAt(pending, ['pricing_activation', layer.name]) ?? !!layer.enabled}
                            onChange={(event) => toggleLayer(layer.name, event.target.checked)}
                          />
                          {pageText(locale, 'On', 'הפעלה')}
                        </label>
                      )}
                    </div>
                  </div>
                  {/* The categories are named through the same table the rows
                      below print, so the sentence and the row cannot call one
                      category two things. Before this the warning read "would
                      zero the price for פרומו" inside an English sentence. */}
                  {Array.isArray(layer.warnings) && layer.warnings.map((warning, index) => {
                    const categories = categoryList(layer.name, warning.categories, locale);
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
                        // A null value is a key nobody has priced (the trade's
                        // positions 4 and 5 ship unset). It shows empty rather
                        // than as a premium of 1 somebody chose, and the box,
                        // its remount key and the draft all read one value.
                        const shown = draftValueAt(pending, ['premiums', LAYER_TO_YAML[layer.name], key]) ?? value ?? '';
                        const unset = shown === '';
                        return (
                          <div className={`pricing-mult${edited ? ' edited' : ''}${unset ? ' unset' : ''}`} key={key}>
                            <Name className="pricing-mult-label" title={label}>{label}</Name>
                            <input
                              type="number" min="0" step="0.01"
                              defaultValue={shown}
                              placeholder={pageText(locale, 'not set', 'לא הוגדר')}
                              key={`${layer.name}-${key}-${shown}`}
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
              stagedEnabled={draftValueAt(pending, ['pricing_activation', 'events'])}
              onToggle={(enabled) => toggleLayer('events', enabled)}
            />
            <PricingPreferredPositions state={state} locale={locale} />
          </div>
        </div>

        <PricingSlotTester state={state} locale={locale} notify={notify} currency={currency} />
      </div>

      <RateCardEffect
        locale={locale}
        overrides={pending || {}}
        dirty={Boolean(pending)}
        saving={saving}
        onSave={commitPending}
        onDiscard={() => setPending(null)}
      />
    </section>
  );
}

export default PricingManager;
