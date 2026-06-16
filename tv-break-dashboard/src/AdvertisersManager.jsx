import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, Switch, TextField } from '@mui/material';
import {
  ChevronDown,
  HelpCircle,
  Plus,
  RefreshCcw,
  RotateCcw,
  Save,
  Search,
  Trash2,
  Users,
  X,
} from 'lucide-react';
import {
  EMPTY_ADVERTISER,
  GENRE_PRESETS,
  POSITION_PRESETS,
  chipOptions,
  computeSummary,
  filterAdvertisers,
  isAnySelected,
  isDirty,
  normalizeRows,
  pageText,
  parseTokens,
  premiumHint,
  serializeTokens,
  sortAdvertisers,
  suggestNextId,
  toConditionPayload,
  toPayload,
  toggleToken,
} from './advertisers-helpers';
import AdvertiserConditions from './AdvertiserConditions';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

// A keyboard-operable chip multi-select. Renders a clickable chip per option
// (ANY first, then specifics, then any unknown stored tokens) with aria-pressed.
function ChipSelect({ label, presets, value, onChange, locale }) {
  const tokens = parseTokens(value);
  const options = chipOptions(presets, tokens);
  const anyActive = isAnySelected(tokens);

  return (
    <div className="adv-chip-field">
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

// The shared smart-control cluster used by both an editable row and the add form.
function PremiumInput({ value, onChange, locale }) {
  const hint = premiumHint(value, locale);
  return (
    <div className="adv-premium-field">
      <span className="adv-field-label">{pageText(locale, 'Premium (x rate card)', 'מקדם (× מחירון)')}</span>
      <div className="adv-premium-input">
        <TextField
          type="number"
          size="small"
          inputProps={{ min: 0, step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Default premium multiplier', 'מקדם תוספת ברירת מחדל') }}
          value={value ?? 1}
          onChange={(event) => onChange(event.target.value === '' ? '' : Number(event.target.value))}
        />
        <span className={`adv-premium-hint ${hint.tone}`} dir="ltr">{hint.text}</span>
      </div>
    </div>
  );
}

function AdvertiserRow({ row, locale, onSave, onDelete, registerDraft, onCreateCondition, onUpdateCondition, onDeleteCondition }) {
  const [draft, setDraft] = useState(row);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    setDraft(row);
  }, [row]);

  const dirty = isDirty(row, draft);

  // Let the parent (Save all) reach the current dirty draft.
  useEffect(() => {
    registerDraft(row.advertiser_id, dirty ? draft : null);
    return () => registerDraft(row.advertiser_id, null);
  }, [row.advertiser_id, dirty, draft, registerDraft]);

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function handleSave() {
    setSaving(true);
    await onSave(draft);
    setSaving(false);
  }

  return (
    <div className="adv-row-group">
    <div className={`adv-row${dirty ? ' dirty' : ''}`}>
      <div className="adv-cell adv-cell-id">
        {dirty && <span className="adv-dirty-dot" title={pageText(locale, 'Unsaved changes', 'שינויים שלא נשמרו')} />}
        <span className="adv-id" dir="ltr">{row.advertiser_id}</span>
      </div>

      <div className="adv-cell">
        <PremiumInput value={draft.default_premium} onChange={(value) => update('default_premium', value)} locale={locale} />
      </div>

      <div className="adv-cell">
        <ChipSelect
          label={pageText(locale, 'Allowed positions', 'מיקומים מותרים')}
          presets={POSITION_PRESETS}
          value={draft.allow_positions}
          onChange={(value) => update('allow_positions', value)}
          locale={locale}
        />
      </div>

      <div className="adv-cell">
        <ChipSelect
          label={pageText(locale, 'Allowed genres', 'ז׳אנרים מותרים')}
          presets={GENRE_PRESETS}
          value={draft.allow_genres}
          onChange={(value) => update('allow_genres', value)}
          locale={locale}
        />
      </div>

      <div className="adv-cell adv-cell-prime">
        <span className="adv-field-label">{pageText(locale, 'Prime time only', 'פריים טיים בלבד')}</span>
        <Switch
          size="small"
          checked={Boolean(draft.prime_time_only)}
          onChange={(event) => update('prime_time_only', event.target.checked)}
          inputProps={{ 'aria-label': pageText(locale, 'Prime time only', 'פריים טיים בלבד') }}
        />
      </div>

      <div className="adv-cell">
        <span className="adv-field-label">{pageText(locale, 'Notes', 'הערות')}</span>
        <TextField
          size="small"
          fullWidth
          value={draft.notes || ''}
          onChange={(event) => update('notes', event.target.value)}
          inputProps={{ 'aria-label': pageText(locale, 'Notes', 'הערות') }}
        />
      </div>

      <div className="adv-cell adv-cell-actions">
        <Button className="secondary-button compact" type="button" variant="outlined" disabled={!dirty || saving} onClick={handleSave}>
          <Save size={14} />
          {saving ? pageText(locale, 'Saving...', 'שומר...') : pageText(locale, 'Save', 'שמירה')}
        </Button>
        {/* Delete is the second fixed anchor: it keeps its place whether or not
            the row is dirty. The optional Revert button is rendered last so it
            never pushes Save or Delete out of position. */}
        {confirmDelete ? (
          <>
            <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => onDelete(row.advertiser_id)}>
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
            aria-label={pageText(locale, 'Delete advertiser', 'מחיקת מפרסם')}
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
            onClick={() => setDraft(row)}
            aria-label={pageText(locale, 'Revert changes', 'ביטול שינויים')}
          >
            <RotateCcw size={14} />
            {pageText(locale, 'Revert', 'שחזור')}
          </Button>
        )}
      </div>
    </div>
      <AdvertiserConditions
        advertiserId={row.advertiser_id}
        conditions={row.conditions}
        overlaps={row.overlaps}
        locale={locale}
        onCreate={onCreateCondition}
        onUpdate={onUpdateCondition}
        onDelete={onDeleteCondition}
      />
    </div>
  );
}

function AddAdvertiserForm({ locale, suggestedId, existingIds, onCreate, onCancel }) {
  const [draft, setDraft] = useState({ ...EMPTY_ADVERTISER, advertiser_id: suggestedId });
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    setDraft((current) => (current.advertiser_id ? current : { ...current, advertiser_id: suggestedId }));
  }, [suggestedId]);

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  const trimmedId = draft.advertiser_id.trim();
  const duplicate = existingIds.includes(trimmedId);
  const canCreate = trimmedId.length > 0 && !duplicate && !creating;

  async function handleCreate() {
    if (!canCreate) {
      return;
    }
    setCreating(true);
    const ok = await onCreate({ ...draft, advertiser_id: trimmedId });
    setCreating(false);
    if (ok) {
      setDraft({ ...EMPTY_ADVERTISER, advertiser_id: '' });
    }
  }

  return (
    <div className="adv-add-form">
      <div className="adv-add-grid">
        <div className="adv-id-field">
          <span className="adv-field-label">{pageText(locale, 'Advertiser ID', 'מזהה מפרסם')}</span>
          <TextField
            size="small"
            value={draft.advertiser_id}
            error={duplicate}
            helperText={duplicate ? pageText(locale, 'This ID already exists', 'מזהה זה כבר קיים') : ' '}
            onChange={(event) => update('advertiser_id', event.target.value)}
            inputProps={{ dir: 'ltr', 'aria-label': pageText(locale, 'Advertiser ID', 'מזהה מפרסם') }}
          />
        </div>
        <PremiumInput value={draft.default_premium} onChange={(value) => update('default_premium', value)} locale={locale} />
        <ChipSelect
          label={pageText(locale, 'Allowed positions', 'מיקומים מותרים')}
          presets={POSITION_PRESETS}
          value={draft.allow_positions}
          onChange={(value) => update('allow_positions', value)}
          locale={locale}
        />
        <ChipSelect
          label={pageText(locale, 'Allowed genres', 'ז׳אנרים מותרים')}
          presets={GENRE_PRESETS}
          value={draft.allow_genres}
          onChange={(value) => update('allow_genres', value)}
          locale={locale}
        />
        <div className="adv-cell-prime">
          <span className="adv-field-label">{pageText(locale, 'Prime time only', 'פריים טיים בלבד')}</span>
          <Switch
            size="small"
            checked={Boolean(draft.prime_time_only)}
            onChange={(event) => update('prime_time_only', event.target.checked)}
            inputProps={{ 'aria-label': pageText(locale, 'Prime time only', 'פריים טיים בלבד') }}
          />
        </div>
        <div className="adv-notes-field">
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
      <div className="adv-add-actions">
        <Button className="run-button" type="button" variant="contained" disabled={!canCreate} onClick={handleCreate}>
          <Plus size={15} />
          {creating ? pageText(locale, 'Adding...', 'מוסיף...') : pageText(locale, 'Add advertiser', 'הוספת מפרסם')}
        </Button>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={onCancel}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </Button>
      </div>
    </div>
  );
}

const FILTERS = [
  { key: 'all', en: 'All', he: 'הכול' },
  { key: 'premium', en: 'Custom premium', he: 'מקדם מותאם' },
  { key: 'prime', en: 'Prime-only', he: 'פריים בלבד' },
  { key: 'restricted', en: 'Restricted', he: 'מוגבל' },
];

function AdvertisersManager({ copy, locale, notify }) {
  const [advertisers, setAdvertisers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortKey, setSortKey] = useState('id');
  const [showAdd, setShowAdd] = useState(false);
  const [showLegend, setShowLegend] = useState(false);
  const [dirtyDrafts, setDirtyDrafts] = useState({});
  const [savingAll, setSavingAll] = useState(false);

  const loadAdvertisers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/advertisers`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const payload = await response.json();
      setAdvertisers(normalizeRows(payload.advertisers));
      setOnline(true);
    } catch {
      setAdvertisers([]);
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAdvertisers();
  }, [loadAdvertisers]);

  const registerDraft = useCallback((id, draft) => {
    setDirtyDrafts((current) => {
      if (!draft) {
        if (!(id in current)) {
          return current;
        }
        const next = { ...current };
        delete next[id];
        return next;
      }
      return { ...current, [id]: draft };
    });
  }, []);

  const summary = useMemo(() => computeSummary(advertisers), [advertisers]);
  const existingIds = useMemo(() => advertisers.map((row) => row.advertiser_id), [advertisers]);
  const suggestedId = useMemo(() => suggestNextId(advertisers), [advertisers]);

  const visible = useMemo(
    () => sortAdvertisers(filterAdvertisers(advertisers, { search, filter }), sortKey),
    [advertisers, search, filter, sortKey],
  );

  const dirtyCount = Object.keys(dirtyDrafts).length;
  const hasActiveQuery = Boolean(search.trim()) || filter !== 'all';

  async function putAdvertiser(draft) {
    const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(draft.advertiser_id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(toPayload(draft)),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response;
  }

  async function handleSave(draft) {
    try {
      await putAdvertiser(draft);
      notify(`Advertiser ${draft.advertiser_id} saved.`, `המפרסם ${draft.advertiser_id} נשמר.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Save failed for ${draft.advertiser_id} (${error.message}).`, `השמירה נכשלה עבור ${draft.advertiser_id} (${error.message}).`);
    }
  }

  async function handleSaveAll() {
    const drafts = Object.values(dirtyDrafts);
    if (drafts.length === 0) {
      return;
    }
    setSavingAll(true);
    let saved = 0;
    const failures = [];
    for (const draft of drafts) {
      try {
        await putAdvertiser(draft);
        saved += 1;
      } catch (error) {
        failures.push(draft.advertiser_id);
      }
    }
    setSavingAll(false);
    if (failures.length === 0) {
      notify(`Saved ${saved} advertisers.`, `נשמרו ${saved} מפרסמים.`);
    } else {
      notify(
        `Saved ${saved}, ${failures.length} failed (${failures.join(', ')}).`,
        `נשמרו ${saved}, ${failures.length} נכשלו (${failures.join(', ')}).`,
      );
    }
    await loadAdvertisers();
  }

  async function handleCreate(draft) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ advertiser_id: draft.advertiser_id, ...toPayload(draft) }),
      });
      if (response.status === 409) {
        notify(`Advertiser ${draft.advertiser_id} already exists.`, `המפרסם ${draft.advertiser_id} כבר קיים.`);
        return false;
      }
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${draft.advertiser_id} created.`, `המפרסם ${draft.advertiser_id} נוצר.`);
      setShowAdd(false);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Create failed (${error.message}).`, `היצירה נכשלה (${error.message}).`);
      return false;
    }
  }

  async function handleDelete(advertiserId) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${advertiserId} deleted.`, `המפרסם ${advertiserId} נמחק.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Delete failed for ${advertiserId} (${error.message}).`, `המחיקה נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  async function handleCreateCondition(advertiserId, draft) {
    try {
      const ruleId = `rule_${Date.now().toString(36)}`;
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}/conditions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rule_id: ruleId, ...toConditionPayload(draft) }),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Scoped rule added to ${advertiserId}.`, `כלל ממוקד נוסף ל${advertiserId}.`);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Add rule failed for ${advertiserId} (${error.message}).`, `הוספת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
      return false;
    }
  }

  async function handleUpdateCondition(advertiserId, ruleId, draft) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}/conditions/${encodeURIComponent(ruleId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toConditionPayload(draft)),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Scoped rule saved for ${advertiserId}.`, `כלל ממוקד נשמר עבור ${advertiserId}.`);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Save rule failed for ${advertiserId} (${error.message}).`, `שמירת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
      return false;
    }
  }

  async function handleDeleteCondition(advertiserId, ruleId) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}/conditions/${encodeURIComponent(ruleId)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Scoped rule removed from ${advertiserId}.`, `כלל ממוקד הוסר מ${advertiserId}.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Delete rule failed for ${advertiserId} (${error.message}).`, `מחיקת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  function clearQuery() {
    setSearch('');
    setFilter('all');
  }

  const stats = [
    { key: 'total', value: summary.total, en: 'Advertisers', he: 'מפרסמים' },
    { key: 'custom', value: summary.custom, en: 'Premium adjusted', he: 'מקדם מותאם' },
    { key: 'prime', value: summary.prime, en: 'Prime-time only', he: 'פריים טיים בלבד' },
    { key: 'restricted', value: summary.restricted, en: 'Position / genre limits', he: 'מגבלות מיקום / ז׳אנר' },
  ];

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Advertisers', 'מפרסמים')}</h1>
          <p>
            {pageText(
              locale,
              'Manage advertiser rules: pricing premium, allowed break positions and programme genres, prime-time limits, and notes.',
              'ניהול תנאי מפרסמים: מקדם תמחור, מיקומי הפסקה וז׳אנרים מותרים, מגבלות פריים טיים והערות.',
            )}
          </p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadAdvertisers}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      {!loading && online && advertisers.length > 0 && (
        <div className="adv-stat-strip">
          {stats.map((stat) => (
            <div className="adv-stat-card" key={stat.key}>
              <span className="adv-stat-value" dir="ltr">{stat.value}</span>
              <span className="adv-stat-label">{pageText(locale, stat.en, stat.he)}</span>
            </div>
          ))}
        </div>
      )}

      <section className="page-panel">
        <div className="panel-head">
          <div className="adv-panel-title">
            <h2>{pageText(locale, 'Advertiser rules', 'תנאי מפרסמים')}</h2>
            <button
              type="button"
              className="adv-legend-toggle"
              aria-expanded={showLegend}
              onClick={() => setShowLegend((value) => !value)}
            >
              <HelpCircle size={13} />
              {pageText(locale, 'What do these terms mean?', 'מה המשמעות של השדות?')}
            </button>
          </div>
          <div className="adv-head-actions">
            {dirtyCount > 0 && (
              <span className="adv-unsaved-count">
                {pageText(locale, `${dirtyCount} unsaved`, `${dirtyCount} לא נשמרו`)}
              </span>
            )}
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              disabled={dirtyCount === 0 || savingAll}
              onClick={handleSaveAll}
            >
              <Save size={14} />
              {savingAll ? pageText(locale, 'Saving...', 'שומר...') : pageText(locale, 'Save all changes', 'שמירת כל השינויים')}
            </Button>
          </div>
        </div>

        {showLegend && (
          <div className="adv-legend">
            <p className="adv-legend-lead">
              {pageText(
                locale,
                'One lever changes how much money a spot earns (premium). The other three only restrict where a spot may be placed - they do not make it more profitable.',
                'שדה אחד משנה כמה כסף תשדיר מכניס (המקדם). שלושת האחרים רק מגבילים היכן מותר לשבץ תשדיר - הם לא הופכים אותו לרווחי יותר.',
              )}
            </p>
            <dl>
              <div>
                <dt>{pageText(locale, 'Premium (earns more)', 'מקדם (מכניס יותר)')}</dt>
                <dd>{pageText(locale, 'A true multiplier on revenue: 1.0 = rate card, 1.2 = this advertiser pays +20% per spot, 0.9 = -10%. This is the only field that changes real profitability.', 'מכפיל אמיתי על ההכנסה: 1.0 = מחירון, 1.2 = המפרסם משלם ‎+20%‎ לתשדיר, 0.9 = ‎-10%‎. זה השדה היחיד שמשנה רווחיות אמיתית.')}</dd>
              </div>
              <div>
                <dt>{pageText(locale, 'Positions (constraint)', 'מיקומים (אילוץ)')}</dt>
                <dd>{pageText(locale, 'Limits WHERE the spot may sit in a break. A constraint, not a profit boost: pinning to position 1 restricts placement, it does not raise the price.', 'מגביל היכן בתוך ההפסקה התשדיר יכול לשבת. אילוץ, לא תוספת רווח: נעילה למיקום 1 מגבילה שיבוץ, היא לא מעלה את המחיר.')}</dd>
              </div>
              <div>
                <dt>{pageText(locale, 'Genres (constraint)', 'ז׳אנרים (אילוץ)')}</dt>
                <dd>{pageText(locale, 'Limits which programme types may carry the spot. A placement constraint, not a profit lever. "Any" = no limit.', 'מגביל באילו סוגי תוכניות מותר לשבץ את התשדיר. אילוץ שיבוץ, לא מנוף רווח. "הכול" = ללא הגבלה.')}</dd>
              </div>
              <div>
                <dt>{pageText(locale, 'Prime-time only (constraint)', 'פריים טיים בלבד (אילוץ)')}</dt>
                <dd>{pageText(locale, 'Restricts the spot to prime-time programmes. A constraint. (Prime-time slots may already cost more in the rate card - that pricing is separate from this rule.)', 'מגביל את התשדיר לתוכניות פריים טיים. אילוץ. (משבצות פריים טיים כבר עשויות לעלות יותר במחירון - תמחור זה נפרד מהכלל הזה.)')}</dd>
              </div>
            </dl>
            <p className="adv-legend-status">
              {pageText(
                locale,
                'Status: these rules are saved to advertiser_rules.csv. The current optimizer chooses how many breaks each programme carries and prices them from the rate card (programme class x daypart x position); it does not yet read per-advertiser premium or constraints. Wiring them in is a pending engine decision.',
                'מצב: הכללים נשמרים אל advertiser_rules.csv. המנוע הנוכחי קובע כמה הפסקות כל תוכנית נושאת ומתמחר אותן מהמחירון (סוג תוכנית × חלק יום × מיקום); הוא עדיין לא קורא מקדם או אילוצים פר-מפרסם. חיבורם הוא החלטת-מנוע ממתינה.',
              )}
            </p>
          </div>
        )}

        <div className="adv-toolbar">
          <div className="adv-search">
            <Search size={15} />
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder={pageText(locale, 'Search by ID or notes', 'חיפוש לפי מזהה או הערות')}
              aria-label={pageText(locale, 'Search advertisers', 'חיפוש מפרסמים')}
            />
          </div>
          <div className="adv-filter-chips" role="group" aria-label={pageText(locale, 'Filter advertisers', 'סינון מפרסמים')}>
            {FILTERS.map((entry) => (
              <button
                key={entry.key}
                type="button"
                className={`adv-chip${filter === entry.key ? ' active' : ''}`}
                aria-pressed={filter === entry.key}
                onClick={() => setFilter(entry.key)}
              >
                {pageText(locale, entry.en, entry.he)}
              </button>
            ))}
          </div>
          <div className="adv-sort">
            <label htmlFor="adv-sort-select">{pageText(locale, 'Sort', 'מיון')}</label>
            <select id="adv-sort-select" value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
              <option value="id">{pageText(locale, 'ID (A to Z)', 'מזהה (א-ת)')}</option>
              <option value="premium-desc">{pageText(locale, 'Premium (high to low)', 'מקדם (גבוה לנמוך)')}</option>
              <option value="premium-asc">{pageText(locale, 'Premium (low to high)', 'מקדם (נמוך לגבוה)')}</option>
            </select>
          </div>
          <Button
            className="secondary-button compact"
            type="button"
            variant="outlined"
            aria-expanded={showAdd}
            onClick={() => setShowAdd((value) => !value)}
          >
            {showAdd ? <X size={14} /> : <Plus size={14} />}
            {showAdd ? pageText(locale, 'Close', 'סגירה') : pageText(locale, 'Add advertiser', 'הוספת מפרסם')}
          </Button>
        </div>

        {showAdd && online && (
          <AddAdvertiserForm
            locale={locale}
            suggestedId={suggestedId}
            existingIds={existingIds}
            onCreate={handleCreate}
            onCancel={() => setShowAdd(false)}
          />
        )}

        {loading && <div className="upload-state">{pageText(locale, 'Loading advertisers...', 'טוען מפרסמים...')}</div>}

        {!loading && !online && (
          <div className="upload-state error">
            {pageText(locale, 'The Kairos API is unavailable. Advertisers cannot be shown.', 'ה־API של Kairos לא זמין. לא ניתן להציג מפרסמים.')}
          </div>
        )}

        {!loading && online && advertisers.length === 0 && (
          <div className="upload-state">{pageText(locale, 'No advertiser rules were found.', 'לא נמצאו תנאי מפרסמים.')}</div>
        )}

        {!loading && online && advertisers.length > 0 && visible.length === 0 && (
          <div className="upload-state adv-no-match">
            <span>{pageText(locale, 'No advertisers match your search or filter.', 'אין מפרסמים שתואמים את החיפוש או הסינון.')}</span>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={clearQuery}>
              {pageText(locale, 'Clear filters', 'ניקוי סינון')}
            </Button>
          </div>
        )}

        {!loading && online && visible.length > 0 && (
          <div className="adv-table" role="table" aria-label={pageText(locale, 'Advertiser rules', 'תנאי מפרסמים')}>
            <div className="adv-row adv-head-row" role="row">
              <span className="adv-cell" role="columnheader">{pageText(locale, 'ID', 'מזהה')}</span>
              <span className="adv-cell" role="columnheader">{pageText(locale, 'Premium', 'מקדם')}</span>
              <span className="adv-cell" role="columnheader">{pageText(locale, 'Positions', 'מיקומים')}</span>
              <span className="adv-cell" role="columnheader">{pageText(locale, 'Genres', 'ז׳אנרים')}</span>
              <span className="adv-cell" role="columnheader">{pageText(locale, 'Prime', 'פריים')}</span>
              <span className="adv-cell" role="columnheader">{pageText(locale, 'Notes', 'הערות')}</span>
              <span className="adv-cell" role="columnheader">{pageText(locale, 'Actions', 'פעולות')}</span>
            </div>
            {visible.map((row) => (
              <AdvertiserRow
                key={row.advertiser_id}
                row={row}
                locale={locale}
                onSave={handleSave}
                onDelete={handleDelete}
                registerDraft={registerDraft}
                onCreateCondition={handleCreateCondition}
                onUpdateCondition={handleUpdateCondition}
                onDeleteCondition={handleDeleteCondition}
              />
            ))}
          </div>
        )}

        {!loading && online && visible.length > 0 && hasActiveQuery && (
          <div className="adv-result-note">
            {pageText(locale, `Showing ${visible.length} of ${advertisers.length}`, `מוצגים ${visible.length} מתוך ${advertisers.length}`)}
          </div>
        )}
      </section>
    </section>
  );
}

export default AdvertisersManager;
