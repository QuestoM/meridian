import React, { useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { Loader2 } from 'lucide-react';
import { pageText } from '../shell/format';
import { fetchPricingEffect, isolate, money, pairLabel, rate, valuePair } from './rules-lib';

// What a rate-card edit does, before it is saved. The two figures are the ones
// the revenue owner's question actually needs: what a second of airtime is worth
// under the card as saved, and what it is worth under the edit in front of them.
//
// Both are the saved plan re-priced through the engine's own pricing seam, on
// the operator's own channel, and the payload carries the check that makes them
// trustworthy: re-pricing under the card as saved reproduces the plan's own
// revenue exactly. When it does not, that is stated rather than smoothed over,
// because it means the plan was built on a card nobody has any more.

const WARM_DELAY_MS = 2500;

export default function RateCardEffect({ locale, overrides, dirty, onSave, onDiscard, saving }) {
  const [effect, setEffect] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // The saved side of the comparison is the same build for every edit, so it
  // is fetched once when the card opens rather than inside the wait after the
  // first change. It waits for the page's own load to finish first: the API runs
  // one worker and a request that joins eight others each take as long as all of
  // them, so a warm-up fired into that is a warm-up that makes things worse.
  useEffect(() => {
    const timer = setTimeout(() => { fetchPricingEffect({}, false).catch(() => {}); }, WARM_DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!dirty) {
      setEffect(null);
      return undefined;
    }
    let alive = true;
    setLoading(true);
    setError('');
    fetchPricingEffect(overrides, false)
      .then((body) => { if (alive) { setEffect(body); setLoading(false); } })
      .catch((problem) => { if (alive) { setError(problem.message); setLoading(false); } });
    return () => { alive = false; };
  }, [overrides, dirty]);

  if (!dirty) return null;

  const scope = effect?.scope;
  const where = scope && !scope.scoped
    ? pageText(
      locale,
      `all ${scope.channels_priced} channels in the loaded plan`,
      `כל ${scope.channels_priced} הערוצים בתוכנית שנטענה`,
    )
    : scope?.channel;
  const scopeText = scope
    ? pageText(
      locale,
      `${where}, ${scope.date_from} to ${scope.date_to}, ${scope.rows} planned segments`,
      `${where}, ${scope.date_from} עד ${scope.date_to}, ${scope.rows} מקטעים בתוכנית`,
    )
    : '';

  return (
    <section className="rules-card rules-ratecard-effect">
      <h3>{pageText(locale, 'What this edit does, before you save it', 'מה העריכה הזו עושה, לפני השמירה')}</h3>

      {loading && !effect && (
        <p className="rules-effect-idle" role="status">
          <Loader2 size={15} className="rules-spin" aria-hidden="true" />
          <span>{pageText(locale, 'Re-pricing the plan', 'מתמחר מחדש את התוכנית')}</span>
        </p>
      )}
      {error && <p className="rules-inline-error" role="status">{error}</p>}

      {effect && effect.available === false && (
        <p className="rules-figure-reason">{effect.reason}</p>
      )}

      {effect && effect.available && (
        <>
          <div className="rules-figures">
            <div className="rules-figure">
              <span className="rules-figure-label">{pageText(locale, 'A second of airtime is worth', 'שנייה של זמן שידור שווה')}</span>
              <strong
                className="rules-figure-delta"
                dir="ltr"
                aria-label={pairLabel(locale, rate(effect.saved.yield_per_second, locale), rate(effect.draft.yield_per_second, locale))}
              >
                {valuePair(rate(effect.saved.yield_per_second, locale), rate(effect.draft.yield_per_second, locale))}
              </strong>
              <span className="rules-figure-scope">{scopeText}</span>
              <span className="rules-figure-basis">
                {pageText(
                  locale,
                  'The saved plan re-priced through the engine pricing seam. Per second of ad time.',
                  'התוכנית השמורה מתומחרת מחדש דרך מנוע התמחור. לשנייה של זמן פרסום.',
                )}
              </span>
            </div>
            <div className="rules-figure">
              <span className="rules-figure-label">{pageText(locale, 'Projected revenue on this plan', 'הכנסה צפויה בתוכנית הזו')}</span>
              <strong className={`rules-figure-delta${Number(effect.delta.revenue) < 0 ? ' negative' : ' positive'}`} dir="ltr">
                {isolate(money(effect.delta.revenue, locale))}
              </strong>
              <span
                className="rules-figure-pair"
                dir="ltr"
                aria-label={pairLabel(locale, money(effect.saved.revenue, locale), money(effect.draft.revenue, locale))}
              >
                {valuePair(money(effect.saved.revenue, locale), money(effect.draft.revenue, locale))}
              </span>
              <span className="rules-figure-scope">{scopeText}</span>
              <span className="rules-figure-basis">
                {effect.delta.percent === null
                  ? ''
                  : pageText(locale, `${effect.delta.percent}% of the plan`, `${effect.delta.percent}% מהתוכנית`)}
              </span>
            </div>
          </div>

          {effect.scope && !effect.scope.scoped && (
            <p className="rules-inline-error" role="status">
              {pageText(
                locale,
                `No operator channel is declared, so this prices every channel in the loaded plan and the figures are not your channel's money. Declare your channel under Channel and model.`,
                `לא הוצהר ערוץ מפעיל, ולכן התמחור כאן מכסה את כל הערוצים בתוכנית שנטענה והמספרים אינם הכסף של הערוץ שלכם. הצהירו על הערוץ במדור ערוץ ומודל.`,
              )}
            </p>
          )}

          {!effect.reproduces_plan && (
            <p className="rules-inline-error" role="status">
              {pageText(
                locale,
                'Re-pricing under the card as saved does not reproduce the plan on record, so the plan was built with a different card. Run the plan before reading this delta as settlement.',
                'תמחור מחדש לפי הכרטיס השמור אינו משחזר את התוכנית הרשומה, ולכן התוכנית נבנתה עם כרטיס אחר. הריצו את התוכנית לפני שתקראו את ההפרש כהתחשבנות.',
              )}
            </p>
          )}

          {(effect.changed_layers || []).some((layer) => layer.moves_plan === false) && (
            <p className="rules-inline-note">
              {pageText(
                locale,
                'Position, ad type and show price an individual spot inside a break, so they cannot move a per-break projection. Their money shows up in the spot ledger.',
                'מיקום, סוג פרסומת ותוכנית מתמחרים תשדיר בודד בתוך ברייק, ולכן אינם משנים תחזית ברמת הברייק. הכסף שלהם מופיע בדוח התשדירים.',
              )}
            </p>
          )}
        </>
      )}

      <div className="rules-composer-actions">
        <Button className="run-button" type="button" variant="contained" disabled={saving || loading} onClick={onSave}>
          {saving ? <Loader2 size={14} className="rules-spin" /> : null}
          {pageText(locale, 'Save the rate card', 'שמירת כרטיס התעריפים')}
        </Button>
        <Button className="secondary-button" type="button" variant="outlined" disabled={saving} onClick={onDiscard}>
          {pageText(locale, 'Discard the edit', 'ביטול העריכה')}
        </Button>
      </div>
    </section>
  );
}
