import React, { useEffect, useMemo, useState } from 'react';
import { Button, TextField } from '@mui/material';
import { ChevronDown, Plus, RotateCcw, Save, Trash2, TriangleAlert } from 'lucide-react';
import {
  CONDITION_EFFECTS,
  GENRE_PRESETS,
  POSITION_PRESETS,
  chipOptions,
  collectDaypartTokens,
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
  premiumHint,
  scopedRulesBadge,
  serializeTokens,
  toggleToken,
} from './advertisers-helpers';

// Keyboard-operable scope chip multi-select. ANY first, then presets, then any
// stored tokens (so engine data and unknown daypart tokens are never dropped).
// A daypart field additionally offers a free-text "add token" affordance.
function ScopeChips({ label, presets, value, onChange, locale, allowAdd = false }) {
  const [adding, setAdding] = useState(false);
  const [draftToken, setDraftToken] = useState('');
  const tokens = parseTokens(value);
  const options = chipOptions(presets, tokens);
  const anyActive = isAnySelected(tokens);

  function commitToken() {
    const trimmed = draftToken.trim();
    if (trimmed && !tokens.includes(trimmed)) {
      onChange(serializeTokens(toggleToken(tokens, trimmed)));
    }
    setDraftToken('');
    setAdding(false);
  }

  return (
    <div className="adv-chip-field adv-cond-scope">
      <span className="adv-field-label">{label}</span>
      <div className="adv-chip-row" role="group" aria-label={label}>
        {options.map((option) => {
          const isAny = option.toUpperCase() === 'ANY';
          const active = isAny ? anyActive : tokens.includes(option);
          return (
            <button
              key={option}
              type="button"
              className={`adv-chip${active ? ' active' : ''}${isAny ? ' any' : ''}`}
              aria-pressed={active}
              onClick={() => onChange(serializeTokens(toggleToken(tokens, option)))}
            >
              <span dir="ltr">{isAny ? pageText(locale, 'Any', 'הכול') : option}</span>
            </button>
          );
        })}
        {allowAdd && !adding && (
          <button
            type="button"
            className="adv-chip adv-chip-add"
            onClick={() => setAdding(true)}
            aria-label={pageText(locale, 'Add daypart token', 'הוספת חלק יום')}
          >
            <Plus size={12} />
          </button>
        )}
        {allowAdd && adding && (
          <input
            className="adv-chip-add-input"
            dir="ltr"
            autoFocus
            value={draftToken}
            onChange={(event) => setDraftToken(event.target.value)}
            onBlur={commitToken}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                commitToken();
              } else if (event.key === 'Escape') {
                setDraftToken('');
                setAdding(false);
              }
            }}
            aria-label={pageText(locale, 'New daypart token', 'חלק יום חדש')}
          />
        )}
      </div>
    </div>
  );
}

// A single editable condition row. Save is disabled until changed; Save and
// Delete are fixed anchors; the optional Revert renders last (no layout shift).
function ConditionRow({ condition, locale, onSave, onDelete }) {
  const original = useMemo(() => parseCondition(condition), [condition]);
  const [draft, setDraft] = useState(original);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    setDraft(original);
  }, [original]);

  const dirty = isConditionDirty(original, draft);
  const hint = premiumHint(draft.value, locale);

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
      <div className="adv-cond-scopes">
        <ScopeChips
          label={pageText(locale, 'Positions', 'מיקומים')}
          presets={POSITION_PRESETS}
          value={draft.scope_positions}
          onChange={(value) => update('scope_positions', value)}
          locale={locale}
        />
        <ScopeChips
          label={pageText(locale, 'Genres', 'ז׳אנרים')}
          presets={GENRE_PRESETS}
          value={draft.scope_genres}
          onChange={(value) => update('scope_genres', value)}
          locale={locale}
        />
        <ScopeChips
          label={pageText(locale, 'Dayparts', 'חלקי יום')}
          presets={['ANY']}
          value={draft.scope_dayparts}
          onChange={(value) => update('scope_dayparts', value)}
          locale={locale}
          allowAdd
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
              <option key={effect} value={effect}>
                {pageText(
                  locale,
                  effect === 'premium' ? 'Premium' : effect === 'require' ? 'Require' : 'Forbid',
                  effect === 'premium' ? 'מקדם' : effect === 'require' ? 'חובה' : 'איסור',
                )}
              </option>
            ))}
          </select>
        </div>

        {draft.effect === 'premium' && (
          <div className="adv-premium-field">
            <span className="adv-field-label">{pageText(locale, 'Premium (x rate card)', 'מקדם (× מחירון)')}</span>
            <div className="adv-premium-input">
              <TextField
                type="number"
                size="small"
                inputProps={{ min: 0, step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Scoped premium multiplier', 'מקדם תוספת ממוקד') }}
                value={draft.value ?? 1}
                onChange={(event) => update('value', event.target.value === '' ? '' : Number(event.target.value))}
              />
              <span className={`adv-premium-hint ${hint.tone}`} dir="ltr">{hint.text}</span>
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
function AddConditionForm({ locale, onCreate }) {
  const [draft, setDraft] = useState(emptyCondition());
  const [creating, setCreating] = useState(false);
  const hint = premiumHint(draft.value, locale);

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
      <div className="adv-cond-scopes">
        <ScopeChips
          label={pageText(locale, 'Positions', 'מיקומים')}
          presets={POSITION_PRESETS}
          value={draft.scope_positions}
          onChange={(value) => update('scope_positions', value)}
          locale={locale}
        />
        <ScopeChips
          label={pageText(locale, 'Genres', 'ז׳אנרים')}
          presets={GENRE_PRESETS}
          value={draft.scope_genres}
          onChange={(value) => update('scope_genres', value)}
          locale={locale}
        />
        <ScopeChips
          label={pageText(locale, 'Dayparts', 'חלקי יום')}
          presets={['ANY']}
          value={draft.scope_dayparts}
          onChange={(value) => update('scope_dayparts', value)}
          locale={locale}
          allowAdd
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
              <option key={effect} value={effect}>
                {pageText(
                  locale,
                  effect === 'premium' ? 'Premium' : effect === 'require' ? 'Require' : 'Forbid',
                  effect === 'premium' ? 'מקדם' : effect === 'require' ? 'חובה' : 'איסור',
                )}
              </option>
            ))}
          </select>
        </div>
        {draft.effect === 'premium' && (
          <div className="adv-premium-field">
            <span className="adv-field-label">{pageText(locale, 'Premium (x rate card)', 'מקדם (× מחירון)')}</span>
            <div className="adv-premium-input">
              <TextField
                type="number"
                size="small"
                inputProps={{ min: 0, step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Scoped premium multiplier', 'מקדם תוספת ממוקד') }}
                value={draft.value ?? 1}
                onChange={(event) => update('value', event.target.value === '' ? '' : Number(event.target.value))}
              />
              <span className={`adv-premium-hint ${hint.tone}`} dir="ltr">{hint.text}</span>
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
function AdvertiserConditions({ advertiserId, conditions, overlaps, locale, onCreate, onUpdate, onDelete }) {
  const [open, setOpen] = useState(false);
  const rules = normalizeConditions(conditions);
  const findings = normalizeOverlaps(overlaps);
  const badges = scopedRulesBadge(rules, findings, locale);

  // Unused but documents that unknown daypart tokens stay visible in chips.
  collectDaypartTokens(rules);

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
              'Scoped rules layer on top of the baseline premium. They apply on the per-spot daily pricing path only - not yet in the weekly break-count plan.',
              'כללים ממוקדים מתווספים מעל המקדם הבסיסי. הם חלים על נתיב התמחור היומי לכל תשדיר בלבד - עדיין לא בתכנון מספר ההפסקות השבועי.',
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
                'No scoped rules yet. Add one to apply a premium or constraint to specific positions, genres, or dayparts.',
                'אין עדיין כללים ממוקדים. הוסף כלל כדי להחיל מקדם או אילוץ על מיקומים, ז׳אנרים או חלקי יום מסוימים.',
              )}
            </p>
          ) : (
            <div className="adv-cond-list">
              {rules.map((rule) => (
                <ConditionRow
                  key={rule.rule_id}
                  condition={rule}
                  locale={locale}
                  onSave={(ruleId, draft) => onUpdate(advertiserId, ruleId, draft)}
                  onDelete={(ruleId) => onDelete(advertiserId, ruleId)}
                />
              ))}
            </div>
          )}

          <AddConditionForm locale={locale} onCreate={(draft) => onCreate(advertiserId, draft)} />
        </div>
      )}
    </div>
  );
}

export default AdvertiserConditions;
