import React, { useEffect, useState } from 'react';
import { Button, Drawer, Switch, TextField, Tooltip } from '@mui/material';
import { Info, RotateCcw, Save, Trash2, TriangleAlert, X } from 'lucide-react';
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
  totalRules,
} from './advertiser-stats-helpers';
import AdvertiserConditions from './AdvertiserConditions';
import { normalizeOverlaps, overlapMessage, overlapTone } from './advertisers-helpers';
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
      <span className={`amz-drawer-stat-value ${tone || ''}${isEmpty ? ' empty' : ''}`} dir="ltr">
        {shown}
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
              inputProps={{ min: 0, step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Default premium multiplier', 'מקדם תוספת ברירת מחדל') }}
              value={draft.default_premium ?? 1}
              onChange={(event) => update('default_premium', event.target.value === '' ? '' : Number(event.target.value))}
            />
            <span className={`adv-premium-hint ${hint.tone}`} dir="ltr">{hint.text}</span>
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
            inputProps={{ 'aria-label': pageText(locale, 'Prime time only', 'פריים טיים בלבד') }}
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
            inputProps={{ min: 0, step: 0.1, dir: 'ltr', 'aria-label': pageText(locale, 'Behind-pace pacing strength (blank uses channel default 1.0)', 'עוצמת השלמת קצב כשמאחור בלוז (ריק = ברירת מחדל של הערוץ 1.0)') }}
            value={draft.urgency_k ?? ''}
            onChange={(event) => update('urgency_k', event.target.value)}
          />
          <span className="adv-field-hint" dir="auto">{pageText(locale, 'How hard behind-schedule campaigns lean toward inventory. Default 1.0. Blank uses the channel default.', 'כמה חזק קמפיינים שמאחורי הלוז נמשכים למלאי. ברירת מחדל 1.0. ריק = ברירת המחדל של הערוץ.')}</span>
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
            inputProps={{ min: 0, step: 0.1, dir: 'ltr', 'aria-label': pageText(locale, 'Over-delivery pacing restraint (blank uses channel default 1.0)', 'עוצמת ריסון בהקדמת לוז (ריק = ברירת מחדל של הערוץ 1.0)') }}
            value={draft.ahead_k ?? ''}
            onChange={(event) => update('ahead_k', event.target.value)}
          />
          <span className="adv-field-hint" dir="auto">{pageText(locale, 'How hard over-delivered campaigns are steered away from inventory. Default 1.0. Blank uses the channel default.', 'כמה חזק קמפיינים שהקדימו את הלוז מורחקים מהמלאי. ברירת מחדל 1.0. ריק = ברירת המחדל של הערוץ.')}</span>
        </div>
        <div className="amz-drawer-field amz-drawer-notes">
          <span className="adv-field-label">{pageText(locale, 'Notes', 'הערות')}</span>
          <TextField
            size="small"
            fullWidth
            value={draft.notes || ''}
            onChange={(event) => update('notes', event.target.value)}
            inputProps={{ 'aria-label': pageText(locale, 'Notes', 'הערות') }}
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

  useEffect(() => {
    setConfirmDelete(false);
  }, [row && row.advertiser_id, open]);

  if (!row) {
    return null;
  }

  const rules = totalRules(row);
  const conflicts = conflictCount(row);
  const findings = normalizeOverlaps(row.overlaps);
  const baseline = row.baseline_premium ?? row.default_premium;
  const effective = row.avg_effective_premium;
  const anchor = locale === 'he' ? 'left' : 'right';

  return (
    <Drawer
      anchor={anchor}
      open={open}
      onClose={onClose}
      PaperProps={{ className: 'amz-drawer-paper', dir: locale === 'he' ? 'rtl' : 'ltr' }}
    >
      <div className="amz-drawer">
        <header className="amz-drawer-head">
          <div className="amz-drawer-title">
            <span className="amz-drawer-eyebrow">{pageText(locale, 'Management area', 'אזור ניהול')}</span>
            <h2 dir="ltr">{row.advertiser_id}</h2>
          </div>
          <button type="button" className="amz-drawer-close" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
            <X size={18} />
          </button>
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
            value={null}
            provenance={revenuePendingTooltip(locale)}
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
          />
        </section>

        <footer className="amz-drawer-foot">
          {confirmDelete ? (
            <>
              <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => onDelete(row.advertiser_id)}>
                <Trash2 size={14} />
                {pageText(locale, 'Confirm delete', 'אישור מחיקה')}
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
