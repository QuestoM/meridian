import React, { useEffect, useState } from 'react';
import { Figure, Code, Name } from '../shell/bidi';
import { Drawer, Switch, TextField, Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { Info, RotateCcw, Save, Trash2, TriangleAlert, X } from 'lucide-react';
import { leverReasons } from '../shell/lever-state';
import {
  GENRE_PRESETS,
  POSITION_PRESETS,
  isDirty,
  pageText,
  parseTokens,
  premiumHint,
  serializeTokens,
  toggleToken,
} from './advertisers-helpers';
import {
  conflictCount,
  formatPremium,
  premiumDelta,
  revenuePendingTooltip,
  revenueProvenance,
  totalRules,
} from './advertiser-stats-helpers';
import { exactMoney } from './clients-money-helpers';
import AdvertiserConditions from './AdvertiserConditions';
import AdvertiserPricingSummary from './AdvertiserPricingSummary';
import { normalizeOverlaps, overlapMessage, overlapTone } from './advertisers-helpers';
import { boundName, identityName, operatorName } from './advertiser-name-helpers';
import { useAssistantEntity } from '../shell/assistant-page-context';
import './advertiser-drawer.css';

// A compact chip multi-select reused inside the drawer baseline editor. Mirrors
// the inline chips used elsewhere so behaviour stays identical (ANY exclusivity).
function ChipField({ label, presets, value, onChange, locale }) {
  const tokens = parseTokens(value);
  const anyActive = tokens.length === 1 && tokens[0].toUpperCase() === 'ANY';
  const options = [...presets];
  tokens.forEach((token) => {
    if (token.toUpperCase() !== 'ANY' && !options.includes(token)) {
      options.push(token);
    }
  });
  return (
    <div className="amz-drawer-field">
      <span className="adv-field-label">{label}</span>
      <div className="adv-chip-row" role="group" aria-label={label}>
        {options.map((option) => {
          const isAny = option.toUpperCase() === 'ANY';
          const active = isAny ? anyActive : tokens.includes(option);
          return (
            <Button
              key={option}
              type="button"
              className={`adv-chip${active ? ' active' : ''}${isAny ? ' any' : ''}`}
              aria-pressed={active}
              onClick={() => onChange(serializeTokens(toggleToken(tokens, option)))}
            >
              <Figure>{isAny ? pageText(locale, 'Any', 'הכול') : option}</Figure>
            </Button>
          );
        })}
      </div>
    </div>
  );
}

// A field label with an info tooltip. Used for the pacing-strength knobs where the
// operator needs the real channel default (1.0) and a worked example spelled out, so
// a blank field and a typed value both read unambiguously (no hidden defaults).
function InfoLabel({ label, help }) {
  return (
    <span className="adv-field-label adv-field-label-info">
      {label}
      <Tooltip title={help} arrow placement="top">
        <span className="amz-stat-info" tabIndex={0} role="img" aria-label={help}>
          <Info size={11} />
        </span>
      </Tooltip>
    </span>
  );
}

// One read-only stat tile in the drawer header with a provenance tooltip.
function StatTile({ label, value, delta, tone, provenance }) {
  const shown = value === null || value === undefined || value === '' ? '-' : value;
  const isEmpty = shown === '-';
  return (
    <div className="amz-drawer-stat">
      <span className="amz-drawer-stat-label">
        {label}
        <Tooltip title={provenance} arrow placement="top">
          <span className="amz-stat-info" tabIndex={0} role="img" aria-label={provenance}>
            <Info size={11} />
          </span>
        </Tooltip>
      </span>
      <span className={`amz-drawer-stat-value ${tone || ''}${isEmpty ? ' empty' : ''}`}>
        <Figure>{shown}</Figure>
        {delta && <span className="amz-stat-delta">{delta}</span>}
      </span>
    </div>
  );
}

// The baseline (advertiser-default) rule editor: premium, allowed positions,
// allowed genres, prime-time, notes. Saves through the same PUT path the list
// used, so behaviour is unchanged - just relocated into the workspace.
function BaselineEditor({ row, locale, onSave }) {
  const [draft, setDraft] = useState(row);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraft(row);
  }, [row]);

  const dirty = isDirty(row, draft);
  const update = (field, value) => setDraft((current) => ({ ...current, [field]: value }));
  const hint = premiumHint(draft.default_premium, locale);

  async function save() {
    setSaving(true);
    await onSave(draft);
    setSaving(false);
  }

  return (
    <div className="amz-baseline">
      <div className="amz-baseline-grid">
        <div className="amz-drawer-field">
          <span className="adv-field-label">{pageText(locale, 'Premium (x rate card)', 'מקדם (× מחירון)')}</span>
          <div className="adv-premium-input">
            <TextField
              type="number"
              size="small"
              slotProps={{ htmlInput: { min: 0, step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Default premium multiplier', 'מקדם תוספת ברירת מחדל') } }}
              value={draft.default_premium ?? 1}
              onChange={(event) => update('default_premium', event.target.value === '' ? '' : Number(event.target.value))}
            />
            <Figure className={`adv-premium-hint ${hint.tone}`}>{hint.text}</Figure>
          </div>
        </div>
        <ChipField
          label={pageText(locale, 'Allowed positions', 'מיקומים מותרים')}
          presets={POSITION_PRESETS}
          value={draft.allow_positions}
          onChange={(value) => update('allow_positions', value)}
          locale={locale}
        />
        <ChipField
          label={pageText(locale, 'Allowed genres', 'ז׳אנרים מותרים')}
          presets={GENRE_PRESETS}
          value={draft.allow_genres}
          onChange={(value) => update('allow_genres', value)}
          locale={locale}
        />
        <div className="amz-drawer-field amz-drawer-prime">
          <span className="adv-field-label">{pageText(locale, 'Prime time only', 'פריים טיים בלבד')}</span>
          <Switch
            size="small"
            checked={Boolean(draft.prime_time_only)}
            onChange={(event) => update('prime_time_only', event.target.checked)}
            slotProps={{ input: { 'aria-label': pageText(locale, 'Prime time only', 'פריים טיים בלבד') } }}
          />
        </div>
        <div className="amz-drawer-field">
          <InfoLabel
            label={pageText(locale, 'Behind-pace strength', 'עוצמת השלמה כשמאחור בלוז')}
            help={pageText(
              locale,
              'How hard a behind-schedule campaign pulls breaks toward its inventory. Channel default is 1.0. Higher catches up faster (2 pulls about twice as hard as the default); 0 turns catch-up off. Leave blank to use the channel default (1.0).',
              'כמה חזק קמפיין שמאחורי הלוז מושך אליו פרסומות. ברירת המחדל של הערוץ היא 1.0. ערך גבוה יותר משלים מהר יותר (2 מושך בערך פי שניים מברירת המחדל); 0 מכבה את ההשלמה. השאר ריק כדי להשתמש בברירת המחדל של הערוץ (1.0).'
            )}
          />
          <TextField
            type="number"
            size="small"
            placeholder={pageText(locale, 'channel default (1.0)', 'ברירת מחדל של הערוץ (1.0)')}
            slotProps={{ htmlInput: { min: 0, step: 0.1, dir: 'ltr', 'aria-label': pageText(locale, 'Behind-pace pacing strength (blank uses channel default 1.0)', 'עוצמת השלמת קצב כשמאחור בלוז (ריק = ברירת מחדל של הערוץ 1.0)') } }}
            value={draft.urgency_k ?? ''}
            onChange={(event) => update('urgency_k', event.target.value)}
          />
          <span className="adv-field-hint">{pageText(locale, 'How hard behind-schedule campaigns lean toward inventory. Default 1.0. Blank uses the channel default.', 'כמה חזק קמפיינים שמאחורי הלוז נמשכים למלאי. ברירת מחדל 1.0. ריק = ברירת המחדל של הערוץ.')}</span>
        </div>
        <div className="amz-drawer-field">
          <InfoLabel
            label={pageText(locale, 'Over-delivery restraint', 'עוצמת ריסון כשמקדים את הלוז')}
            help={pageText(
              locale,
              'How hard an over-delivered campaign (ahead of its delivery pace) is steered away from inventory, so budget spreads to campaigns that still need it. Channel default is 1.0. Higher restrains harder (2 pushes about twice as hard as the default); 0 turns the over-delivery penalty off. Leave blank to use the channel default (1.0).',
              'כמה חזק קמפיין שהקדים את הלוז (מסר יותר ממה שתוכנן) מורחק מהמלאי, כדי שהתקציב יתפרס לקמפיינים שעוד זקוקים לו. ברירת המחדל של הערוץ היא 1.0. ערך גבוה יותר מרסן חזק יותר (2 דוחף בערך פי שניים מברירת המחדל); 0 מכבה את קנס ההקדמה. השאר ריק כדי להשתמש בברירת המחדל של הערוץ (1.0).'
            )}
          />
          <TextField
            type="number"
            size="small"
            placeholder={pageText(locale, 'channel default (1.0)', 'ברירת מחדל של הערוץ (1.0)')}
            slotProps={{ htmlInput: { min: 0, step: 0.1, dir: 'ltr', 'aria-label': pageText(locale, 'Over-delivery pacing restraint (blank uses channel default 1.0)', 'עוצמת ריסון בהקדמת לוז (ריק = ברירת מחדל של הערוץ 1.0)') } }}
            value={draft.ahead_k ?? ''}
            onChange={(event) => update('ahead_k', event.target.value)}
          />
          <span className="adv-field-hint">{pageText(locale, 'How hard over-delivered campaigns are steered away from inventory. Default 1.0. Blank uses the channel default.', 'כמה חזק קמפיינים שהקדימו את הלוז מורחקים מהמלאי. ברירת מחדל 1.0. ריק = ברירת המחדל של הערוץ.')}</span>
        </div>
        {/* These two knobs are the per-advertiser override of the channel pacing
            lever, and they carried no qualification at all: they read as live
            controls. The verdict is computed in lever-state.js, not asserted
            here, so it stays correct when the mechanism or the data changes. */}
        <div className="amz-drawer-field">
          {leverReasons('pacing', null, locale).slice(0, 1).map((reason) => (
            <span className="adv-field-hint" key={reason}>{reason}</span>
          ))}
        </div>
        <div className="amz-drawer-field amz-drawer-notes">
          <span className="adv-field-label">{pageText(locale, 'Notes', 'הערות')}</span>
          <TextField
            size="small"
            fullWidth
            value={draft.notes || ''}
            onChange={(event) => update('notes', event.target.value)}
            slotProps={{ htmlInput: { 'aria-label': pageText(locale, 'Notes', 'הערות') } }}
          />
        </div>
      </div>
      <div className="amz-baseline-actions">
        <Button className="run-button compact" type="button" variant="contained" disabled={!dirty || saving} onClick={save}>
          <Save size={14} />
          {saving ? pageText(locale, 'Saving...', 'שומר...') : pageText(locale, 'Save baseline', 'שמירת בסיס')}
        </Button>
        {dirty && (
          <Button className="secondary-button compact adv-revert" type="button" variant="outlined" onClick={() => setDraft(row)}>
            <RotateCcw size={14} />
            {pageText(locale, 'Revert', 'שחזור')}
          </Button>
        )}
      </div>
    </div>
  );
}

// The full per-advertiser workspace drawer. Anchored to the inline-start edge so
// it slides in correctly under RTL; embeds the at-a-glance stats, the baseline
// editor, an overlap/conflict summary, and the scoped-rules editor.
function AdvertiserDetailDrawer({
  row,
  open,
  locale,
  scopeOptions,
  onClose,
  onSaveBaseline,
  onDelete,
  onCreateCondition,
  onUpdateCondition,
  onDeleteCondition,
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  // One-shot jump command for the scoped-rules section: set from the personal
  // pricing section to open the builder on a specific rule or on the add form.
  const [condFocus, setCondFocus] = useState(null);

  useEffect(() => {
    setConfirmDelete(false);
    setCondFocus(null);
  }, [row && row.advertiser_id, open]);

  useAssistantEntity(
    'advertiser',
    open && row ? row.advertiser_id : '',
    open && row ? operatorName(row) || identityName(row) || boundName(row) || row.advertiser_id : '',
  );

  if (!row) {
    return null;
  }

  const rules = totalRules(row);
  const conflicts = conflictCount(row);
  const findings = normalizeOverlaps(row.overlaps);
  const baseline = row.baseline_premium ?? row.default_premium;
  const effective = row.avg_effective_premium;
  const anchor = locale === 'he' ? 'left' : 'right';
  // The advertiser this row prices: the operator's own label, then the daily
  // ledger's real name (joined by advertiser_id), then the rules store's own
  // name cell, which is what makes the row a named record. This mirrors
  // displayNameOf/isUnnamed in advertiser-name-helpers so the drawer never
  // disagrees with the card that opened it.
  const bound = operatorName(row) || identityName(row) || boundName(row);

  return (
    <Drawer
      anchor={anchor}
      open={open}
      onClose={onClose}
      slotProps={{ paper: {
        className: 'amz-drawer-paper',
        dir: locale === 'he' ? 'rtl' : 'ltr',
        role: 'dialog',
        'aria-modal': 'true',
        'aria-labelledby': 'advertiser-drawer-title',
      } }}
    >
      <div className="amz-drawer">
        <header className="amz-drawer-head">
          <div className="amz-drawer-title">
            <span className="amz-drawer-eyebrow">
              {bound
                ? pageText(locale, 'Pricing rule', 'כלל תמחור')
                : pageText(locale, 'Pricing rule, bound to no advertiser', 'כלל תמחור שאינו קשור לאף מפרסם')}
            </span>
            {/* A Hebrew trade name inside dir=ltr reads with its punctuation
                flipped, so the name is auto and only the raw id is ltr. */}
            <h2 id="advertiser-drawer-title"><Name>{bound || row.advertiser_id}</Name></h2>
            {bound ? <Code className="amz-drawer-rawid">{row.advertiser_id}</Code> : null}
          </div>
          <Button autoFocus type="button" className="amz-drawer-close" onClick={onClose} aria-label={pageText(locale, 'Close pricing record', 'סגירת רשומת התמחור')}>
            <X size={18} />
          </Button>
        </header>

        <div className="amz-drawer-statgrid">
          <StatTile
            label={pageText(locale, 'Scoped rules', 'כללים ממוקדים')}
            value={String(rules)}
            provenance={pageText(locale, 'Source: the conditions store (count of scoped rules)', 'מקור: מאגר הכללים (מספר הכללים הממוקדים)')}
          />
          <StatTile
            label={pageText(locale, 'Baseline premium', 'מקדם בסיס')}
            value={formatPremium(baseline)}
            delta={premiumDelta(baseline)}
            tone={Number(baseline ?? 1) > 1 ? 'teal' : Number(baseline ?? 1) < 1 ? 'amber' : ''}
            provenance={pageText(locale, 'Source: advertiser_rules.csv', 'מקור: advertiser_rules.csv')}
          />
          <StatTile
            label={pageText(locale, 'Avg effective', 'מקדם אפקטיבי')}
            value={formatPremium(effective)}
            delta={premiumDelta(effective)}
            tone={Number(effective ?? 1) > 1 ? 'teal' : Number(effective ?? 1) < 1 ? 'amber' : ''}
            provenance={pageText(locale, 'Source: rule engine (baseline times ANY-scope premium rules)', 'מקור: מנוע הכללים (הבסיס כפול כללי מקדם בהיקף ״הכול״)')}
          />
          <StatTile
            label={pageText(locale, 'Revenue', 'הכנסה')}
            value={row.revenue === null || row.revenue === undefined ? null : exactMoney(row.revenue, locale)}
            provenance={revenueProvenance(row, locale)}
          />
          <StatTile
            label={pageText(locale, 'Profitability', 'רווחיות')}
            value={null}
            provenance={revenuePendingTooltip(locale)}
          />
        </div>

        {findings.length > 0 && (
          <div className="amz-drawer-overlaps">
            {findings.map((finding, index) => (
              <div key={`${finding.kind}-${index}`} className={`adv-overlap ${overlapTone(finding.kind)}`}>
                <TriangleAlert size={14} className="adv-overlap-icon" />
                <span className="adv-overlap-text">{overlapMessage(finding)}</span>
              </div>
            ))}
          </div>
        )}

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Baseline rule', 'כלל בסיס')}</h3>
          <BaselineEditor row={row} locale={locale} onSave={onSaveBaseline} />
        </section>

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Personal pricing', 'תמחור אישי')}</h3>
          <AdvertiserPricingSummary
            advertiserId={row.advertiser_id}
            conditions={row.conditions}
            scopeOptions={scopeOptions}
            locale={locale}
            onEditRule={(ruleId) => setCondFocus({ seq: Date.now(), ruleId })}
            onAddRule={() => setCondFocus({ seq: Date.now(), ruleId: null })}
          />
        </section>

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Scoped rules', 'כללים ממוקדים')}</h3>
          <AdvertiserConditions
            advertiserId={row.advertiser_id}
            conditions={row.conditions}
            overlaps={row.overlaps}
            locale={locale}
            scopeOptions={scopeOptions}
            onCreate={onCreateCondition}
            onUpdate={onUpdateCondition}
            onDelete={onDeleteCondition}
            focusRequest={condFocus}
          />
        </section>

        <footer className="amz-drawer-foot">
          {confirmDelete ? (
            <>
              <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => onDelete(row.advertiser_id)}>
                <Trash2 size={14} />
                {pageText(locale, 'Confirm delete', 'אישור מחיקה')}
              </Button>
              <Button autoFocus className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmDelete(false)}>
                {pageText(locale, 'Cancel', 'ביטול')}
              </Button>
            </>
          ) : (
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => setConfirmDelete(true)}
              aria-label={pageText(locale, 'Delete advertiser', 'מחיקת מפרסם')}
            >
              <Trash2 size={14} />
              {pageText(locale, 'Delete advertiser', 'מחיקת מפרסם')}
            </Button>
          )}
        </footer>
      </div>
    </Drawer>
  );
}

export default AdvertiserDetailDrawer;
