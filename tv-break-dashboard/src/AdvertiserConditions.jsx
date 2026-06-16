import React, { useEffect, useMemo, useState } from 'react';
import { Button, TextField, Tooltip } from '@mui/material';
import { ChevronDown, Info, Plus, RotateCcw, Save, Search, Trash2, TriangleAlert } from 'lucide-react';
import {
  CONDITION_EFFECTS,
  PREMIUM_MODES,
  coefficientHint,
  emptyCondition,
  isAnySelected,
  isConditionDirty,
  normalizeConditions,
  normalizeOverlaps,
  overlapMessage,
  overlapTone,
  pageText,
  parseCondition,
  parseTokens,
  pressureHint,
  scopedRulesBadge,
  serializeTokens,
  toggleToken,
} from './advertisers-helpers';

// Bilingual label for a scope token, looked up against the options list the
// backend serves; falls back to the raw token so engine data is never dropped.
function tokenLabel(token, optionMap, locale) {
  const entry = optionMap.get(String(token));
  if (entry) {
    return locale === 'he' ? entry.he : entry.en;
  }
  return token;
}

// Normalize the various option shapes the /options endpoint returns into a
// uniform [{ value, he, en }] list. Strings (genres/programmes) become
// value=label; objects (positions/dayparts) carry he/en labels.
function normalizeOptions(raw) {
  return (raw || []).map((item) => {
    if (typeof item === 'string') {
      return { value: item, he: item, en: item };
    }
    return { value: item.key, he: item.he || item.key, en: item.en || item.key };
  });
}

// Keyboard-operable scope multi-select. ANY first, then the backend option list,
// then any stored tokens not in that list (engine data is never dropped). Long
// lists (programmes) get a filter box so the operator can find a show fast.
function ScopeMultiSelect({ label, options, value, onChange, locale, filterable = false }) {
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
          <input
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
            <button
              key={token}
              type="button"
              className={`adv-chip${active ? ' active' : ''}${isAny ? ' any' : ''}`}
              aria-pressed={active}
              onClick={() => onChange(serializeTokens(toggleToken(tokens, token)))}
            >
              <span dir="auto">{isAny ? pageText(locale, 'Any', 'הכול') : tokenLabel(token, optionMap, locale)}</span>
            </button>
          );
        })}
        {visibleOptions.length === 1 && filterable && query.trim() && (
          <span className="adv-chip-empty">{pageText(locale, 'no match', 'אין התאמה')}</span>
        )}
      </div>
    </div>
  );
}

// Bilingual label for an effect.
function effectLabel(effect, locale) {
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
function modeLabel(mode, locale) {
  const labels = {
    multiplier: ['Multiplier (x)', 'מכפיל (×)'],
    percent: ['Percent (+/-%)', 'אחוז (+/-%)'],
    cpp_absolute: ['CPP absolute', 'נקודה מוחלטת'],
    cpp_add: ['CPP add', 'תוספת לנקודה'],
    cpp_discount: ['CPP discount', 'הנחה מהנקודה'],
  };
  const pair = labels[mode] || [mode, mode];
  return pageText(locale, pair[0], pair[1]);
}

// The shared scope + effect + value editor used by both the inline edit row and
// the add-a-rule form, so the two never drift apart.
function ConditionFields({ draft, update, locale, scopeOptions }) {
  const positionOptions = normalizeOptions(scopeOptions.positions);
  const genreOptions = normalizeOptions(scopeOptions.genres);
  const daypartOptions = normalizeOptions(scopeOptions.dayparts);
  const programmeOptions = normalizeOptions(scopeOptions.programmes);
  const modes = scopeOptions.modes && scopeOptions.modes.length ? scopeOptions.modes : PREMIUM_MODES;
  const hint = draft.effect === 'pressure'
    ? pressureHint(draft.value, locale)
    : coefficientHint(draft.value, draft.mode, locale);

  return (
    <>
      <div className="adv-cond-scopes">
        <ScopeMultiSelect
          label={pageText(locale, 'Positions', 'מיקומים')}
          options={positionOptions}
          value={draft.scope_positions}
          onChange={(value) => update('scope_positions', value)}
          locale={locale}
        />
        <ScopeMultiSelect
          label={pageText(locale, 'Genres', 'ז׳אנרים')}
          options={genreOptions}
          value={draft.scope_genres}
          onChange={(value) => update('scope_genres', value)}
          locale={locale}
          filterable
        />
        <ScopeMultiSelect
          label={pageText(locale, 'Dayparts', 'חלקי יום')}
          options={daypartOptions}
          value={draft.scope_dayparts}
          onChange={(value) => update('scope_dayparts', value)}
          locale={locale}
        />
        <ScopeMultiSelect
          label={pageText(locale, 'Programmes', 'תוכניות')}
          options={programmeOptions}
          value={draft.scope_programmes}
          onChange={(value) => update('scope_programmes', value)}
          locale={locale}
          filterable
        />
      </div>

      <div className="adv-cond-effect-block">
        <div className="adv-cond-effect">
          <span className="adv-field-label">{pageText(locale, 'Effect', 'השפעה')}</span>
          <select
            value={draft.effect}
            onChange={(event) => update('effect', event.target.value)}
            aria-label={pageText(locale, 'Rule effect', 'השפעת הכלל')}
          >
            {CONDITION_EFFECTS.map((effect) => (
              <option key={effect} value={effect}>{effectLabel(effect, locale)}</option>
            ))}
          </select>
        </div>

        {draft.effect === 'premium' && (
          <>
            <div className="adv-cond-effect">
              <span className="adv-field-label">
                {pageText(locale, 'Mode', 'אופן')}
                <Tooltip
                  title={pageText(
                    locale,
                    'How the coefficient value is read: a multiplier, a +/- percent, or a cost-per-point amount (absolute, added, or discounted).',
                    'כיצד נקרא ערך המקדם: מכפיל, אחוז +/-, או סכום מחיר-לנקודה (מוחלט, תוספת, או הנחה).',
                  )}
                  arrow
                >
                  <Info size={12} className="adv-field-info" />
                </Tooltip>
              </span>
              <select
                value={draft.mode}
                onChange={(event) => update('mode', event.target.value)}
                aria-label={pageText(locale, 'Coefficient mode', 'אופן המקדם')}
              >
                {modes.map((mode) => (
                  <option key={mode} value={mode}>{modeLabel(mode, locale)}</option>
                ))}
              </select>
            </div>
            <div className="adv-premium-field">
              <span className="adv-field-label">{pageText(locale, 'Coefficient value', 'ערך המקדם')}</span>
              <div className="adv-premium-input">
                <TextField
                  type="number"
                  size="small"
                  inputProps={{ step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Coefficient value', 'ערך המקדם') }}
                  value={draft.value ?? 1}
                  onChange={(event) => update('value', event.target.value === '' ? '' : Number(event.target.value))}
                />
                <span className={`adv-premium-hint ${hint.tone}`} dir="auto">{hint.text}</span>
              </div>
            </div>
          </>
        )}

        {draft.effect === 'pressure' && (
          <div className="adv-premium-field">
            <span className="adv-field-label">
              {pageText(locale, 'Placement preference (%)', 'העדפת שיבוץ (%)')}
              <Tooltip
                title={pageText(
                  locale,
                  'Steers where the optimizer wants to place the ad (a +10% preference ranks the slot as if it paid 10% more), but is never charged: the reported revenue is unchanged.',
                  'מטה את המיטוב לכיוון שיבוץ מסוים (העדפה של +10% מדרגת את הסלוט כאילו שילם 10% יותר), אך לעולם לא נגבית: ההכנסה המדווחת אינה משתנה.',
                )}
                arrow
              >
                <Info size={12} className="adv-field-info" />
              </Tooltip>
            </span>
            <div className="adv-premium-input">
              <TextField
                type="number"
                size="small"
                inputProps={{ step: 5, dir: 'ltr', 'aria-label': pageText(locale, 'Placement preference percent', 'אחוז העדפת שיבוץ') }}
                value={draft.value ?? 0}
                onChange={(event) => update('value', event.target.value === '' ? '' : Number(event.target.value))}
              />
              <span className={`adv-premium-hint ${hint.tone}`} dir="auto">{hint.text}</span>
            </div>
          </div>
        )}
      </div>

      <div className="adv-cond-notes">
        <span className="adv-field-label">{pageText(locale, 'Notes', 'הערות')}</span>
        <TextField
          size="small"
          fullWidth
          value={draft.notes || ''}
          onChange={(event) => update('notes', event.target.value)}
          inputProps={{ 'aria-label': pageText(locale, 'Rule notes', 'הערות לכלל') }}
        />
      </div>
    </>
  );
}

// A single editable condition row. Save is disabled until changed; Save and
// Delete are fixed anchors; the optional Revert renders last (no layout shift).
function ConditionRow({ condition, locale, scopeOptions, onSave, onDelete }) {
  const original = useMemo(() => parseCondition(condition), [condition]);
  const [draft, setDraft] = useState(original);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    setDraft(original);
  }, [original]);

  const dirty = isConditionDirty(original, draft);

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function handleSave() {
    setSaving(true);
    await onSave(original.rule_id, draft);
    setSaving(false);
  }

  return (
    <div className="adv-cond-row">
      <ConditionFields draft={draft} update={update} locale={locale} scopeOptions={scopeOptions} />

      <div className="adv-cell-actions adv-cond-actions">
        <Button className="secondary-button compact" type="button" variant="outlined" disabled={!dirty || saving} onClick={handleSave}>
          <Save size={14} />
          {saving ? pageText(locale, 'Saving...', 'שומר...') : pageText(locale, 'Save', 'שמירה')}
        </Button>
        {confirmDelete ? (
          <>
            <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => onDelete(original.rule_id)}>
              <Trash2 size={14} />
              {pageText(locale, 'Confirm', 'אישור')}
            </Button>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmDelete(false)}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </>
        ) : (
          <Button
            className="secondary-button compact"
            type="button"
            variant="outlined"
            onClick={() => setConfirmDelete(true)}
            aria-label={pageText(locale, 'Delete rule', 'מחיקת כלל')}
          >
            <Trash2 size={14} />
            {pageText(locale, 'Delete?', 'מחיקה?')}
          </Button>
        )}
        {dirty && (
          <Button
            className="secondary-button compact adv-revert"
            type="button"
            variant="outlined"
            onClick={() => setDraft(original)}
            aria-label={pageText(locale, 'Revert changes', 'ביטול שינויים')}
          >
            <RotateCcw size={14} />
            {pageText(locale, 'Revert', 'שחזור')}
          </Button>
        )}
      </div>
    </div>
  );
}

// The add-a-rule mini form, mirrors the inline row but POSTs a new condition.
function AddConditionForm({ locale, scopeOptions, onCreate }) {
  const [draft, setDraft] = useState(emptyCondition());
  const [creating, setCreating] = useState(false);

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function handleCreate() {
    setCreating(true);
    const ok = await onCreate(draft);
    setCreating(false);
    if (ok) {
      setDraft(emptyCondition());
    }
  }

  return (
    <div className="adv-cond-row adv-cond-add">
      <ConditionFields draft={draft} update={update} locale={locale} scopeOptions={scopeOptions} />

      <div className="adv-cell-actions adv-cond-actions">
        <Button className="run-button compact" type="button" variant="contained" disabled={creating} onClick={handleCreate}>
          <Plus size={14} />
          {creating ? pageText(locale, 'Adding...', 'מוסיף...') : pageText(locale, 'Add rule', 'הוספת כלל')}
        </Button>
      </div>
    </div>
  );
}

// The collapsible "Scoped rules" section attached to one advertiser row.
function AdvertiserConditions({ advertiserId, conditions, overlaps, locale, scopeOptions, onCreate, onUpdate, onDelete }) {
  const [open, setOpen] = useState(false);
  const rules = normalizeConditions(conditions);
  const findings = normalizeOverlaps(overlaps);
  const badges = scopedRulesBadge(rules, findings, locale);
  const options = scopeOptions || {};

  return (
    <div className={`adv-scoped${open ? ' open' : ''}`}>
      <button
        type="button"
        className="adv-scoped-toggle"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <ChevronDown size={14} className="adv-scoped-caret" />
        <span className="adv-scoped-title">{pageText(locale, 'Scoped rules', 'כללים ממוקדים')}</span>
        {badges.length === 0 ? (
          <span className="adv-scoped-badge muted">{pageText(locale, 'none', 'אין')}</span>
        ) : (
          badges.map((text, index) => (
            <span
              key={text}
              className={`adv-scoped-badge${index === 1 ? ' conflict' : ''}`}
            >
              {text}
            </span>
          ))
        )}
      </button>

      {open && (
        <div className="adv-scoped-body">
          <p className="adv-scoped-note">
            {pageText(
              locale,
              'Scoped rules layer on top of the baseline premium. They apply on the per-spot daily pricing path only - not yet in the weekly break-count plan. Placement preference steers where ads go without changing the reported revenue.',
              'כללים ממוקדים מתווספים מעל המקדם הבסיסי. הם חלים על נתיב התמחור היומי לכל תשדיר בלבד - עדיין לא בתכנון מספר ההפסקות השבועי. העדפת שיבוץ מטה את מיקום התשדירים מבלי לשנות את ההכנסה המדווחת.',
            )}
          </p>

          {findings.length > 0 && (
            <div className="adv-overlaps">
              {findings.map((finding, index) => {
                const tone = overlapTone(finding.kind);
                return (
                  <div key={`${finding.kind}-${index}`} className={`adv-overlap ${tone}`}>
                    <TriangleAlert size={14} className="adv-overlap-icon" />
                    <span className="adv-overlap-text">{overlapMessage(finding)}</span>
                  </div>
                );
              })}
            </div>
          )}

          {rules.length === 0 ? (
            <p className="adv-scoped-empty">
              {pageText(
                locale,
                'No scoped rules yet. Add one to apply a coefficient, a constraint, or a placement preference to specific positions, genres, dayparts, or programmes.',
                'אין עדיין כללים ממוקדים. הוסף כלל כדי להחיל מקדם, אילוץ או העדפת שיבוץ על מיקומים, ז׳אנרים, חלקי יום או תוכניות מסוימים.',
              )}
            </p>
          ) : (
            <div className="adv-cond-list">
              {rules.map((rule) => (
                <ConditionRow
                  key={rule.rule_id}
                  condition={rule}
                  locale={locale}
                  scopeOptions={options}
                  onSave={(ruleId, draft) => onUpdate(advertiserId, ruleId, draft)}
                  onDelete={(ruleId) => onDelete(advertiserId, ruleId)}
                />
              ))}
            </div>
          )}

          <AddConditionForm locale={locale} scopeOptions={options} onCreate={(draft) => onCreate(advertiserId, draft)} />
        </div>
      )}
    </div>
  );
}

export default AdvertiserConditions;
