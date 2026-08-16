import React, { useEffect, useMemo, useRef, useState } from 'react';
import { TextField, Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { ChevronDown, Info, Plus, RotateCcw, Save, Trash2, TriangleAlert } from 'lucide-react';
import { leverReasons } from '../shell/lever-state';
import {
  CONDITION_EFFECTS,
  PREMIUM_MODES,
  coefficientHint,
  emptyCondition,
  isConditionDirty,
  normalizeConditions,
  normalizeOverlaps,
  overlapMessage,
  overlapTone,
  pageText,
  parseCondition,
  pressureHint,
  scopedRulesBadge,
} from './advertisers-helpers';
import { ScopeMultiSelect, WeekdayScope, effectLabel, modeLabel, normalizeOptions } from './AdvertiserPricingSummary';
import { SelectControl } from '../studio/dom-controls';

// Escape a rule id for use inside a querySelector attribute selector.
function cssEscape(value) {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(String(value));
  }
  return String(value).replace(/["\\]/g, '\\$&');
}

// The shared scope + effect + value editor used by both the inline edit row and
// the add-a-rule form, so the two never drift apart.
function ConditionFields({ draft, update, locale, scopeOptions }) {
  const positionOptions = normalizeOptions(scopeOptions.positions);
  const genreOptions = normalizeOptions(scopeOptions.genres);
  const daypartOptions = normalizeOptions(scopeOptions.dayparts);
  const programmeOptions = normalizeOptions(scopeOptions.programmes);
  const serverModes = scopeOptions.modes && scopeOptions.modes.length ? scopeOptions.modes : PREMIUM_MODES;
  // A stored rule's mode is always selectable even if this server's option list
  // does not carry it, so engine data is never dropped from the editor.
  const modes = draft.mode && !serverModes.includes(draft.mode) ? [...serverModes, draft.mode] : serverModes;
  const isSurchargeDiscount = draft.mode === 'premium_discount';
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
        <WeekdayScope
          value={draft.scope_weekdays}
          onChange={(value) => update('scope_weekdays', value)}
          locale={locale}
        />
      </div>

      <div className="adv-cond-effect-block">
        <div className="adv-cond-effect">
          <span className="adv-field-label">{pageText(locale, 'Effect', 'השפעה')}</span>
          <SelectControl
            value={draft.effect}
            onChange={(event) => update('effect', event.target.value)}
            aria-label={pageText(locale, 'Rule effect', 'השפעת הכלל')}
          >
            {CONDITION_EFFECTS.map((effect) => (
              <option key={effect} value={effect}>{effectLabel(effect, locale)}</option>
            ))}
          </SelectControl>
        </div>

        {draft.effect === 'premium' && (
          <>
            <div className="adv-cond-effect">
              <span className="adv-field-label">
                {pageText(locale, 'Mode', 'אופן')}
                <Tooltip
                  title={pageText(
                    locale,
                    'How the value you enter changes the price: multiplies it, adds or subtracts a percent, sets a price per rating point (fixed, added, or discounted), or takes a percent off only the surcharge above the base price.',
                    'איך הערך שמזינים משנה את המחיר: מכפיל אותו, מוסיף או גורע אחוזים, קובע מחיר לנקודת רייטינג (קבוע, תוספת או הנחה), או גורע אחוזים מתוספת המחיר שמעל מחיר הבסיס בלבד.',
                  )}
                  arrow
                >
                  <Info size={12} className="adv-field-info" />
                </Tooltip>
              </span>
              <SelectControl
                value={draft.mode}
                onChange={(event) => update('mode', event.target.value)}
                aria-label={pageText(locale, 'Coefficient mode', 'אופן המקדם')}
              >
                {modes.map((mode) => (
                  <option key={mode} value={mode}>{modeLabel(mode, locale)}</option>
                ))}
              </SelectControl>
            </div>
            <div className="adv-premium-field">
              <span className="adv-field-label">
                {isSurchargeDiscount
                  ? pageText(locale, 'Discount percent (0-100)', 'אחוז ההנחה (0-100)')
                  : pageText(locale, 'Coefficient value', 'ערך המקדם')}
                {isSurchargeDiscount && (
                  <Tooltip
                    title={pageText(
                      locale,
                      'The discount applies only to the surcharge above the base price, never to the base itself. A 100 percent discount removes the whole surcharge and returns the slot to the base price; the price never drops below the base.',
                      'ההנחה חלה רק על תוספת המחיר שמעל מחיר הבסיס, לעולם לא על הבסיס עצמו. הנחה של 100 אחוז מבטלת את כל התוספת ומחזירה את המשבצת למחיר הבסיס; המחיר לעולם אינו יורד מתחת למחיר הבסיס.',
                    )}
                    arrow
                    placement="bottom"
                  >
                    <Info size={12} className="adv-field-info" />
                  </Tooltip>
                )}
              </span>
              <div className="adv-premium-input">
                <TextField
                  type="number"
                  size="small"
                  slotProps={{
                    htmlInput: isSurchargeDiscount
                      ? { min: 0, max: 100, step: 5, dir: 'ltr', 'aria-label': pageText(locale, 'Surcharge discount percent', 'אחוז ההנחה על תוספת המחיר') }
                      : { step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Coefficient value', 'ערך המקדם') },
                  }}
                  value={draft.value ?? 1}
                  onChange={(event) => update('value', event.target.value === '' ? '' : Number(event.target.value))}
                />
                <span className={`adv-premium-hint ${hint.tone}`}>{hint.text}</span>
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
                slotProps={{ htmlInput: { step: 5, dir: 'ltr', 'aria-label': pageText(locale, 'Placement preference percent', 'אחוז העדפת שיבוץ') } }}
                value={draft.value ?? 0}
                onChange={(event) => update('value', event.target.value === '' ? '' : Number(event.target.value))}
              />
              <span className={`adv-premium-hint ${hint.tone}`}>{hint.text}</span>
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
          slotProps={{ htmlInput: { 'aria-label': pageText(locale, 'Rule notes', 'הערות לכלל') } }}
        />
      </div>
    </>
  );
}

// A single editable condition row. Save is disabled until changed; Save and
// Delete are fixed anchors; the optional Revert renders last (no layout shift).
function ConditionRow({ condition, locale, scopeOptions, onSave, onDelete, highlight }) {
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
    <div className={`card adv-cond-row${highlight ? ' focus' : ''}`} data-rule-id={original.rule_id}>
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
// A blank draft is already a premium (money) rule, so a jump from the personal
// pricing section lands on a pricing rule with no extra clicks.
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
    <div className="card adv-cond-row adv-cond-add" data-cond-add="true">
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
// focusRequest ({ seq, ruleId }) is an optional one-shot command from the host:
// it opens the section and scrolls to the named rule (ruleId null targets the
// add-a-rule form), so the personal pricing section can jump straight to edit.
function AdvertiserConditions({ advertiserId, conditions, overlaps, locale, scopeOptions, onCreate, onUpdate, onDelete, focusRequest }) {
  const [open, setOpen] = useState(false);
  const [highlightId, setHighlightId] = useState(null);
  const bodyRef = useRef(null);
  const rules = normalizeConditions(conditions);
  const findings = normalizeOverlaps(overlaps);
  const badges = scopedRulesBadge(rules, findings, locale);
  const options = scopeOptions || {};

  useEffect(() => {
    if (!focusRequest || !focusRequest.seq) {
      return undefined;
    }
    setOpen(true);
    setHighlightId(focusRequest.ruleId || null);
    // Wait one tick so the section body exists before scrolling to the target.
    const timer = setTimeout(() => {
      const root = bodyRef.current;
      if (!root) {
        return;
      }
      const selector = focusRequest.ruleId
        ? `[data-rule-id="${cssEscape(focusRequest.ruleId)}"]`
        : '[data-cond-add]';
      const target = root.querySelector(selector);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        const control = target.querySelector('select, input, button');
        if (control) {
          control.focus({ preventScroll: true });
        }
      }
    }, 120);
    return () => clearTimeout(timer);
  }, [focusRequest]);

  return (
    <div className={`adv-scoped${open ? ' open' : ''}`}>
      <Button
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
      </Button>

      {open && (
        <div className="adv-scoped-body" ref={bodyRef}>
          <p className="adv-scoped-note">
            {pageText(
              locale,
              'Scoped rules layer on top of the baseline premium. They price on the per-spot daily path.',
              'כללים ממוקדים מתווספים מעל המקדם הבסיסי. הם מתמחרים בנתיב היומי לכל תשדיר.',
            )}
          </p>
          {/* The placement-preference verdict is COMPUTED, not written here. The
              sentence this replaced said scoped rules were "not yet in the weekly
              break-count plan"; that was a static string, true only while the
              conditions file was empty, and it would have gone silently false on
              the first rule added. */}
          {leverReasons('advertiser', rules.length > 0, locale).map((reason) => (
            <p className="adv-scoped-note" key={reason}>{reason}</p>
          ))}

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
                'No scoped rules yet. Add one to apply a coefficient, a constraint, or a placement preference to specific positions, genres, dayparts, programmes, or weekdays.',
                'אין עדיין כללים ממוקדים. הוסף כלל כדי להחיל מקדם, אילוץ או העדפת שיבוץ על מיקומים, ז׳אנרים, חלקי יום, תוכניות או ימים מסוימים בשבוע.',
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
                  highlight={highlightId !== null && String(rule.rule_id) === String(highlightId)}
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
