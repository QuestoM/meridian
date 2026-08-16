import React, { useEffect, useMemo, useState } from 'react';
import { Button } from '../studio/actions';
import { Pencil, Plus, Search } from 'lucide-react';
import {
  WEEKDAY_OPTIONS,
  isAnySelected,
  normalizeConditions,
  normalizeMode,
  pageText,
  parseTokens,
  serializeTokens,
  toggleToken,
} from './advertisers-helpers';
import { InputControl } from '../studio/dom-controls';

// The personal-pricing surface: the shared pricing vocabulary (scope labels, the
// keyboard-operable scope multi-select used by the conditions builder) and the
// plain-language "personal pricing" summary section shown in the advertiser
// drawer, including the live worked-price example against the price-slot tester.

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// Bilingual label for a scope token, looked up against the options list the
// backend serves; falls back to the raw token so engine data is never dropped.
export function tokenLabel(token, optionMap, locale) {
  const entry = optionMap.get(String(token));
  if (entry) {
    return locale === 'he' ? entry.he : entry.en;
  }
  return token;
}

// Normalize the various option shapes the /options endpoint returns into a
// uniform [{ value, he, en }] list. Strings (genres/programmes) become
// value=label; objects (positions/dayparts) carry he/en labels.
export function normalizeOptions(raw) {
  return (raw || []).map((item) => {
    if (typeof item === 'string') {
      return { value: item, he: item, en: item };
    }
    return { value: item.key, he: item.he || item.key, en: item.en || item.key };
  });
}

// Bilingual label for an effect.
export function effectLabel(effect, locale) {
  const labels = {
    premium: ['Coefficient', 'מקדם'],
    require: ['Require', 'חובה'],
    forbid: ['Forbid', 'איסור'],
    pressure: ['Placement preference', 'העדפת שיבוץ'],
  };
  const pair = labels[effect] || [effect, effect];
  return pageText(locale, pair[0], pair[1]);
}

// Bilingual label for a coefficient mode.
export function modeLabel(mode, locale) {
  const labels = {
    multiplier: ['Multiplier (x)', 'מכפיל (×)'],
    percent: ['Percent (+/-%)', 'אחוז (+/-%)'],
    cpp_absolute: ['CPP absolute', 'נקודה מוחלטת'],
    cpp_add: ['CPP add', 'תוספת לנקודה'],
    cpp_discount: ['CPP discount', 'הנחה מהנקודה'],
    premium_discount: ['Surcharge discount (%)', 'הנחה על תוספת המחיר'],
  };
  const pair = labels[mode] || [mode, mode];
  return pageText(locale, pair[0], pair[1]);
}

// Keyboard-operable scope multi-select. ANY first, then the backend option list,
// then any stored tokens not in that list (engine data is never dropped). Long
// lists (programmes) get a filter box so the operator can find a show fast.
export function ScopeMultiSelect({ label, options, value, onChange, locale, filterable = false }) {
  const [query, setQuery] = useState('');
  const tokens = parseTokens(value);
  const anyActive = isAnySelected(tokens);
  const optionMap = useMemo(() => {
    const map = new Map();
    (options || []).forEach((option) => map.set(String(option.value), option));
    return map;
  }, [options]);

  // Build the visible option set: ANY, the backend options, then stored unknowns.
  const visibleOptions = useMemo(() => {
    const values = ['ANY', ...(options || []).map((option) => option.value)];
    tokens.forEach((token) => {
      if (token.toUpperCase() !== 'ANY' && !values.includes(token)) {
        values.push(token);
      }
    });
    if (!filterable || !query.trim()) {
      return values;
    }
    const term = query.trim().toLowerCase();
    return values.filter((token) => {
      if (token.toUpperCase() === 'ANY') {
        return true;
      }
      const text = `${token} ${tokenLabel(token, optionMap, locale)}`.toLowerCase();
      return text.includes(term);
    });
  }, [options, tokens, filterable, query, optionMap, locale]);

  return (
    <div className="adv-chip-field adv-cond-scope">
      <span className="adv-field-label">{label}</span>
      {filterable && (
        <div className="adv-chip-filter">
          <Search size={12} className="adv-chip-filter-icon" />
          <InputControl
            className="adv-chip-filter-input"
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={pageText(locale, 'Filter...', 'סינון...')}
            aria-label={pageText(locale, `Filter ${label}`, `סינון ${label}`)}
          />
        </div>
      )}
      <div className={`adv-chip-row${filterable ? ' adv-chip-row-scroll' : ''}`} role="group" aria-label={label}>
        {visibleOptions.map((token) => {
          const isAny = token.toUpperCase() === 'ANY';
          const active = isAny ? anyActive : tokens.includes(token);
          return (
            <Button
              key={token}
              type="button"
              className={`adv-chip${active ? ' active' : ''}${isAny ? ' any' : ''}`}
              aria-pressed={active}
              onClick={() => onChange(serializeTokens(toggleToken(tokens, token)))}
            >
              <span>{isAny ? pageText(locale, 'Any', 'הכול') : tokenLabel(token, optionMap, locale)}</span>
            </Button>
          );
        })}
        {visibleOptions.length === 1 && filterable && query.trim() && (
          <span className="adv-chip-empty">{pageText(locale, 'no match', 'אין התאמה')}</span>
        )}
      </div>
    </div>
  );
}

// Full weekday names for accessible labels on the single-letter chips.
const WEEKDAY_FULL = {
  7: ['Sunday', 'יום ראשון'], 1: ['Monday', 'יום שני'], 2: ['Tuesday', 'יום שלישי'], 3: ['Wednesday', 'יום רביעי'],
  4: ['Thursday', 'יום חמישי'], 5: ['Friday', 'יום שישי'], 6: ['Saturday', 'שבת'],
};

// The weekday scope chips of the conditions builder, in Israeli week order
// (Sunday first, שבת last). Chip labels are the Hebrew day letters; the stored
// tokens stay ISO weekday numbers (Monday=1 .. Sunday=7) per the data contract.
// ANY is the default and means the rule ignores the weekday entirely.
export function WeekdayScope({ value, onChange, locale }) {
  const tokens = parseTokens(value);
  const anyActive = isAnySelected(tokens);
  const label = pageText(locale, 'Weekdays', 'ימים בשבוע');
  return (
    <div className="adv-chip-field adv-cond-scope">
      <span className="adv-field-label">{label}</span>
      <div className="adv-chip-row" role="group" aria-label={label}>
        <Button type="button" className={`adv-chip any${anyActive ? ' active' : ''}`} aria-pressed={anyActive} onClick={() => onChange('ANY')}>
          <span>{pageText(locale, 'Any', 'הכול')}</span>
        </Button>
        {WEEKDAY_OPTIONS.map((day) => {
          const active = !anyActive && tokens.includes(day.value);
          const full = WEEKDAY_FULL[day.value] || [day.en, day.he];
          return (
            <Button
              key={day.value}
              type="button"
              className={`adv-chip${active ? ' active' : ''}`}
              aria-pressed={active}
              aria-label={pageText(locale, full[0], full[1])}
              onClick={() => onChange(serializeTokens(toggleToken(tokens, day.value)))}
            >
              <span>{pageText(locale, day.en, day.he)}</span>
            </Button>
          );
        })}
      </div>
    </div>
  );
}

// The specific (non-ANY) tokens of one scope field.
function scopeTokens(value) {
  const tokens = parseTokens(value);
  return isAnySelected(tokens) ? [] : tokens;
}

// Trim trailing zeros off a number for sentence display (1.50 -> 1.5).
function compactNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? String(parseFloat(num.toFixed(3))) : String(value);
}

// The effect half of a rule sentence, per coefficient mode.
function effectPhrase(rule, locale) {
  const value = Number(rule.value);
  const mode = normalizeMode(rule.mode);
  if (mode === 'percent') {
    if (value > 0) return pageText(locale, `a price increase of ${compactNumber(value)} percent`, `תוספת מחיר של ${compactNumber(value)} אחוז`);
    if (value < 0) return pageText(locale, `a price discount of ${compactNumber(Math.abs(value))} percent`, `הנחה של ${compactNumber(Math.abs(value))} אחוז מהמחיר`);
    return pageText(locale, 'the rate-card price', 'מחיר מחירון');
  }
  if (mode === 'cpp_absolute') return pageText(locale, `a fixed price of ${compactNumber(value)} per rating point`, `מחיר קבוע של ${compactNumber(value)} לנקודת רייטינג`);
  if (mode === 'cpp_add') return pageText(locale, `an addition of ${compactNumber(value)} per rating point`, `תוספת של ${compactNumber(value)} לנקודת רייטינג`);
  if (mode === 'cpp_discount') return pageText(locale, `a discount of ${compactNumber(value)} per rating point`, `הנחה של ${compactNumber(value)} לנקודת רייטינג`);
  if (mode === 'premium_discount') return pageText(locale, `a discount of ${compactNumber(value)} percent on the price surcharge`, `הנחה של ${compactNumber(value)} אחוז על תוספת המחיר`);
  // multiplier
  const pct = Math.round((value - 1) * 100);
  if (pct > 0) return pageText(locale, `a price increase of ${pct} percent (multiplier ${compactNumber(value)})`, `תוספת מחיר של ${pct} אחוז (מקדם ${compactNumber(value)})`);
  if (pct < 0) return pageText(locale, `a price discount of ${Math.abs(pct)} percent (multiplier ${compactNumber(value)})`, `הנחה של ${Math.abs(pct)} אחוז מהמחיר (מקדם ${compactNumber(value)})`);
  return pageText(locale, 'the rate-card price (multiplier 1)', 'מחיר מחירון (מקדם 1)');
}

// One scope phrase like "in programme X" with singular/plural Hebrew wording.
function scopePhrase(tokens, optionMap, locale, single, plural) {
  if (tokens.length === 0) {
    return '';
  }
  const labels = tokens.map((token) => tokenLabel(token, optionMap, locale)).join(', ');
  return `${tokens.length === 1 ? single : plural} ${labels}`;
}

// The weekday phrase: שבת alone reads "on Saturdays only"; letters get a geresh.
// Days are listed in Israeli week order (Sunday first, Saturday last) no matter
// how the stored scope orders its ISO tokens; unknown tokens keep their place after.
function weekdayPhrase(tokens, locale) {
  if (tokens.length === 0) {
    return '';
  }
  if (tokens.length === 1 && tokens[0] === '6') {
    return pageText(locale, 'on Saturdays only', 'בשבתות בלבד');
  }
  const israeliRank = (token) => (/^[1-7]$/.test(token) ? Number(token) % 7 : 7 + tokens.indexOf(token));
  const ordered = [...tokens].sort((a, b) => israeliRank(a) - israeliRank(b));
  const map = new Map(WEEKDAY_OPTIONS.map((day) => [day.value, day]));
  const labels = ordered.map((token) => {
    const entry = map.get(String(token));
    if (!entry) return String(token);
    if (locale !== 'he') return entry.en;
    return entry.value === '6' ? entry.he : `${entry.he}׳`;
  }).join(', ');
  return pageText(locale, `on ${labels} only`, `בימי ${labels} בלבד`);
}

// Build the full plain-language sentence for one premium rule.
export function ruleSentence(rule, maps, locale) {
  const parts = [effectPhrase(rule, locale)];
  const positions = scopePhrase(scopeTokens(rule.scope_positions), maps.positions, locale, pageText(locale, 'in position', 'במיקום'), pageText(locale, 'in positions', 'במיקומים'));
  const programmes = scopePhrase(scopeTokens(rule.scope_programmes), maps.programmes, locale, pageText(locale, 'in the programme', 'בתוכנית'), pageText(locale, 'in the programmes', 'בתוכניות'));
  const genres = scopePhrase(scopeTokens(rule.scope_genres), maps.genres, locale, pageText(locale, 'in the genre', 'בז׳אנר'), pageText(locale, 'in the genres', 'בז׳אנרים'));
  const dayparts = scopePhrase(scopeTokens(rule.scope_dayparts), maps.dayparts, locale, pageText(locale, 'in the daypart', 'ברצועת השידור'), pageText(locale, 'in the dayparts', 'ברצועות השידור'));
  const weekdays = weekdayPhrase(scopeTokens(rule.scope_weekdays), locale);
  const scopes = [positions, programmes, genres, dayparts, weekdays].filter(Boolean);
  if (scopes.length === 0) {
    parts.push(pageText(locale, 'on all airings', 'בכל השידורים'));
  } else {
    parts.push(...scopes);
  }
  return parts.join(' ');
}

// Specificity = how many scope dimensions a rule narrows. The most specific
// premium rule drives the live example; ties keep store order.
function specificity(rule) {
  return ['scope_positions', 'scope_genres', 'scope_dayparts', 'scope_programmes', 'scope_weekdays'].filter((field) => scopeTokens(rule[field]).length > 0).length;
}

export function pickMostSpecific(rules) {
  return rules.reduce((best, rule) => (best === null || specificity(rule) > specificity(best) ? rule : best), null);
}

// Map a position scope token to the numeric position the tester understands.
// The vocabulary is 1 to 5 and L. L is the tail of a break rather than an
// ordinal, so it has no single numeric slot to price and the worked example
// omits the position instead of inventing one; the gold break is a break tag,
// not a position, and is omitted for the same reason. "first" is the legacy
// word form and still reads.
function positionNumber(tokens) {
  if (tokens.length === 0) return null;
  if (tokens[0] === 'first') return 1;
  return /^\d+$/.test(tokens[0]) ? Number(tokens[0]) : null;
}

// One worked price line for the most specific rule, priced by the real
// price-slot tester endpoint. Never fabricates: while unreachable it shows an
// honest unavailable note instead of a number.
function LiveExample({ advertiserId, rule, maps, locale }) {
  const [state, setState] = useState({ status: 'loading', data: null });
  const requestKey = JSON.stringify([advertiserId, rule.rule_id, rule.scope_programmes, rule.scope_genres, rule.scope_dayparts, rule.scope_positions, rule.scope_weekdays, rule.mode, rule.value]);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      setState({ status: 'loading', data: null });
      const programmes = scopeTokens(rule.scope_programmes);
      const genres = scopeTokens(rule.scope_genres);
      const dayparts = scopeTokens(rule.scope_dayparts);
      const weekdays = scopeTokens(rule.scope_weekdays).filter((token) => /^[1-7]$/.test(token));
      const body = { pricing_class: genres[0] || 'Other', weekday_iso: Number(weekdays[0]) || 1, advertiser: advertiserId };
      if (programmes[0]) body.show = programmes[0];
      if (genres[0]) body.genre = genres[0];
      if (dayparts[0]) body.daypart = dayparts[0];
      const position = positionNumber(scopeTokens(rule.scope_positions));
      if (position) body.position = position;
      try {
        const response = await fetch(`${API_BASE}/api/pricing/price-slot`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          throw new Error(`${response.status}`);
        }
        const data = await response.json();
        if (!cancelled) setState({ status: 'ok', data });
      } catch (error) {
        if (!cancelled) setState({ status: 'unavailable', data: null });
      }
    }
    run();
    return () => { cancelled = true; };
    // requestKey covers every input the request body is built from.
  }, [requestKey]); // eslint-disable-line react-hooks/exhaustive-deps

  if (state.status === 'loading') {
    return <p className="apz-example muted">{pageText(locale, 'Computing a worked example...', 'מחשב דוגמה חיה...')}</p>;
  }
  if (state.status === 'unavailable' || !state.data) {
    return <p className="apz-example muted">{pageText(locale, 'The live example is unavailable right now (the price tester could not be reached), so no example price is shown.', 'הדוגמה החיה אינה זמינה כרגע (בודק המחיר אינו נגיש), ולכן לא מוצג מחיר לדוגמה.')}</p>;
  }
  const base = Number(state.data.base_cpp);
  const final = Number(state.data.final_cpp);
  if (!Number.isFinite(base) || !Number.isFinite(final)) {
    return <p className="apz-example muted">{pageText(locale, 'The price tester returned no usable numbers, so no example price is shown.', 'בודק המחיר לא החזיר מספרים תקינים, ולכן לא מוצג מחיר לדוגמה.')}</p>;
  }
  const programmes = scopeTokens(rule.scope_programmes);
  const slotName = programmes[0] ? tokenLabel(programmes[0], maps.programmes, locale) : pageText(locale, 'a matching slot', 'משבצת תואמת');
  const applied = Array.isArray(state.data.applied_overrides) && state.data.applied_overrides.length > 0;
  return (
    <div className="apz-example">
      <span className="apz-example-line">{pageText(locale, `Worked example for the most specific rule (${slotName}): base price ${base.toFixed(2)}, final price ${final.toFixed(2)}.`, `דוגמה חיה לפי הכלל הממוקד ביותר (${slotName}): מחיר בסיס ${base.toFixed(2)}, מחיר סופי ${final.toFixed(2)}.`)}</span>
      <span className="apz-example-note">{applied ? pageText(locale, 'Includes the targeted layer rules of this advertiser that match the slot.', 'כולל כללים ממוקדי שכבה של המפרסם החלים על המשבצת.') : pageText(locale, 'Rate-card layers only; scoped coefficient rules apply on the per-spot daily pricing path, which this tester does not simulate.', 'שכבות המחירון בלבד; כללי מקדם ממוקדים חלים בנתיב התמחור היומי לכל תשדיר, שהבודק אינו מדמה.')}</span>
    </div>
  );
}

// The "personal pricing" section body for the advertiser drawer: every
// price-affecting (premium) rule as a plain-language sentence, click to edit in
// the builder, an add button preset to a premium rule, and the live example.
function AdvertiserPricingSummary({ advertiserId, conditions, scopeOptions, locale, onEditRule, onAddRule }) {
  const options = scopeOptions || {};
  const maps = useMemo(() => {
    const toMap = (raw) => new Map(normalizeOptions(raw).map((option) => [String(option.value), option]));
    return { positions: toMap(options.positions), genres: toMap(options.genres), dayparts: toMap(options.dayparts), programmes: toMap(options.programmes) };
  }, [options.positions, options.genres, options.dayparts, options.programmes]);
  const moneyRules = normalizeConditions(conditions).filter((rule) => rule.effect === 'premium');
  const mostSpecific = pickMostSpecific(moneyRules);

  return (
    <div className="apz-summary">
      <p className="apz-note">{pageText(locale, 'This is where the price is personalized for this advertiser: increases, discounts and rating-point prices, scoped to programmes, genres, dayparts, positions and weekdays. Require and forbid rules live under scoped rules; this section is money only.', 'כאן קובעים מחיר אישי למפרסם: תוספות, הנחות ומחירים לנקודת רייטינג, לפי תוכנית, ז׳אנר, רצועת שידור, מיקום וימים בשבוע. כללי חובה ואיסור נמצאים תחת כללים ממוקדים; החלק הזה עוסק בכסף בלבד.')}</p>

      {moneyRules.length === 0 ? (
        <p className="apz-empty">{pageText(locale, 'No personal pricing rules yet for this advertiser. Add one to shape the price.', 'אין עדיין כללי תמחור אישיים למפרסם זה. הוסיפו כלל כדי להתאים את המחיר.')}</p>
      ) : (
        <ul className="apz-rules">
          {moneyRules.map((rule) => (
            <li key={rule.rule_id}>
              <Button type="button" className="apz-rule" onClick={() => onEditRule(rule.rule_id)} aria-label={pageText(locale, `Edit pricing rule ${rule.rule_id}`, `עריכת כלל תמחור ${rule.rule_id}`)}>
                <span className="apz-rule-sentence">{ruleSentence(rule, maps, locale)}</span>
                <Pencil size={13} className="apz-rule-icon" />
              </Button>
            </li>
          ))}
        </ul>
      )}

      <div className="apz-actions">
        <Button className="run-button compact" type="button" variant="contained" onClick={onAddRule}>
          <Plus size={14} />
          {pageText(locale, 'Add a pricing rule', 'הוספת כלל תמחור')}
        </Button>
      </div>

      {mostSpecific && (
        <LiveExample advertiserId={advertiserId} rule={mostSpecific} maps={maps} locale={locale} />
      )}
    </div>
  );
}

export default AdvertiserPricingSummary;
