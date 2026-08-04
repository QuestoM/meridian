import React, { useState } from 'react';
import { Button, FormControl, MenuItem, Select, TextField } from '@mui/material';
import { Save, Send, Trash2 } from 'lucide-react';
import { GroupNode, defaultGroup, serializeNode } from './constraint-predicate';
// The effect words are shared with the restriction list above this panel, so the
// two surfaces cannot say different things about the same stored value.
import { EFFECT_LIST, effectLabel } from './rules-lib';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

function t(locale, en, he) {
  return locale === 'he' ? he : en;
}

function normalizeRows(value) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.constraints)) return value.constraints;
  if (value && Array.isArray(value.rows)) return value.rows;
  return [];
}

// ---- Effect parameter fields -----------------------------------------------
function mmssToSeconds(value) {
  const [minutes, seconds] = String(value || '00:00').split(':').map((part) => Number(part));
  return (Number.isFinite(minutes) ? minutes : 0) * 60 + (Number.isFinite(seconds) ? seconds : 0);
}

function buildBody(draft, where) {
  const body = {
    scope_type: 'always',
    scope_value: '',
    channel: '',
    effect: draft.effect,
    order_index: draft.order_index === '' ? null : Number(draft.order_index),
    notes: draft.notes || '',
  };
  if (draft.effect === 'FIX_OFFSET') {
    body.offset_seconds = mmssToSeconds(draft.offset_mmss);
  } else if (draft.effect === 'OFFSET_WINDOW') {
    body.offset_min_seconds = mmssToSeconds(draft.offset_min);
    body.offset_max_seconds = mmssToSeconds(draft.offset_max);
  } else if (draft.effect === 'PIN_COUNT') {
    body.count = Number(draft.pin_count);
  } else if (draft.effect === 'DURATION_RANGE') {
    body.duration_min_seconds = Number(draft.duration_min);
    body.duration_max_seconds = Number(draft.duration_max);
  } else if (draft.effect === 'GOLD') {
    // no extra params
  } else if (draft.effect === 'FORBID') {
    // no extra params
  }
  const serializedWhere = serializeNode(where);
  if (serializedWhere.conditions && serializedWhere.conditions.length > 0) {
    body.where = serializedWhere;
  }
  return body;
}

// ---- Main ConstraintBuilder export -----------------------------------------
// onGlobalRefresh (optional) is called after a successful save or delete so the
// page-level freshness banner re-reads its verdict; both mutate a fingerprinted
// schedule input.
function ConstraintBuilder({ locale, notify, onRecompute, recomputeState, onGlobalRefresh }) {
  const he = locale === 'he';
  const [hints, setHints] = useState({ programme: [], genre: [], channels: [], available_channels: [] });
  // The value picker offers the operator's own lineup and nothing else. The
  // scope note that comes back with it says which channel was read, and when
  // none is declared it says why the lists are empty and where to declare one.
  const [scope, setScope] = useState(null);
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [items, setItems] = useState([]);
  const [available, setAvailable] = useState(true);
  const [saving, setSaving] = useState(false);

  const [draft, setDraft] = useState({
    effect: 'FIX_OFFSET',
    offset_mmss: '00:00',
    offset_min: '00:00',
    offset_max: '00:00',
    pin_count: 1,
    duration_min: 30,
    duration_max: 120,
    order_index: '',
    notes: '',
  });

  const [whereTree, setWhereTree] = useState(defaultGroup);

  React.useEffect(() => {
    let active = true;
    async function load() {
      try {
        const res = await fetch(`${API_BASE}/api/constraints/options`);
        if (res.ok && active) {
          const data = await res.json();
          setHints({
            programme: normalizeRows(data.programmes || data.programme_list),
            genre: normalizeRows(data.genres || data.genre_list),
            channels: normalizeRows(data.channels),
            available_channels: normalizeRows(data.available_channels),
            dayparts: normalizeRows(data.dayparts),
            weekdays: normalizeRows(data.weekdays),
          });
          setScope(data.scope || null);
        }
      } catch {
        // fall through to defaults
      } finally {
        if (active) setOptionsLoaded(true);
      }
      try {
        const listRes = await fetch(`${API_BASE}/api/constraints`);
        if (listRes.status === 404) {
          if (active) setAvailable(false);
          return;
        }
        if (listRes.ok && active) {
          const payload = await listRes.json();
          setItems(normalizeRows(payload));
        }
      } catch {
        // leave list empty
      }
    }
    load();
    return () => { active = false; };
  }, []);

  function updateDraft(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function saveConstraint() {
    setSaving(true);
    try {
      const body = buildBody(draft, whereTree);
      const res = await fetch(`${API_BASE}/api/constraints`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.status === 404) {
        setAvailable(false);
        notify('The constraints API is not available yet.', 'ממשק האילוצים עדיין לא זמין.');
        return;
      }
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const saved = await res.json();
      const savedId = saved.constraint_id ?? saved.id;
      setItems((current) => [...current, { ...body, id: savedId || `constraint-${current.length + 1}` }]);
      notify('Constraint saved.', 'האילוץ נשמר.');
      onGlobalRefresh?.();
    } catch (err) {
      notify(`Saving the constraint failed (${err.message}).`, `שמירת האילוץ נכשלה (${err.message}).`);
    } finally {
      setSaving(false);
    }
  }

  async function deleteConstraint(id) {
    const matchesId = (item) => (item.constraint_id ?? item.id) === id;
    try {
      const res = await fetch(`${API_BASE}/api/constraints/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (res.status === 404) {
        setItems((current) => current.filter((item) => !matchesId(item)));
        return;
      }
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setItems((current) => current.filter((item) => !matchesId(item)));
      notify('Constraint removed.', 'האילוץ הוסר.');
      onGlobalRefresh?.();
    } catch (err) {
      notify(`Removing the constraint failed (${err.message}).`, `הסרת האילוץ נכשלה (${err.message}).`);
    }
  }

  const { effect } = draft;

  return (
    <section className="settings-panel wide constraint-builder">
      <div className="settings-panel-head">
        <div>
          <h2>{t(locale, 'Constraint builder', 'בונה אילוצים')}</h2>
          <p>{t(locale, 'Set the filter conditions, then choose the effect to apply', 'הגדירו את תנאי הסינון ולאחר מכן בחרו את ההשפעה להחלה')}</p>
        </div>
      </div>

      {!available && (
        <p className="constraint-builder-warning">
          {t(locale, 'The constraints API responded with 404. Saving is disabled until it is online.', 'ממשק האילוצים החזיר 404. השמירה מושבתת עד שיהיה זמין.')}
        </p>
      )}

      <div className="cb-section-label">{t(locale, 'When (filter conditions)', 'כאשר (תנאי סינון)')}</div>
      {optionsLoaded && scope && scope.scoped && (
        <p className="cb-scope-note" dir="auto">
          {t(locale, `The suggestions cover the ${hints.programme.length} programmes on ${scope.scope_channel}, the channel this operator owns.`, `ההצעות כוללות את ${hints.programme.length} התוכניות בערוץ ${scope.scope_channel}, הערוץ שבבעלות המפעיל.`)}
        </p>
      )}
      {optionsLoaded && scope && !scope.scoped && (
        <p className="cb-scope-note empty" role="status" dir="auto">
          {t(locale, 'No channel is declared, so there are no programme suggestions. Declare the channel you own under Rules, channel and model.', 'עדיין לא הוצהר ערוץ, ולכן אין הצעות תוכניות. הצהירו על הערוץ שלכם בכללים, ערוץ ומודל.')}
        </p>
      )}
      <div className="cb-tree-root" dir={he ? 'rtl' : 'ltr'}>
        <GroupNode
          group={whereTree}
          onUpdate={setWhereTree}
          onDelete={null}
          hints={hints}
          locale={locale}
          level={0}
        />
      </div>

      <div className="cb-section-label" style={{ marginTop: 18 }}>{t(locale, 'Apply effect', 'אפקט להחלה')}</div>
      <div className="constraint-builder-form">
        <div className="constraint-field">
          <span className="adv-field-label">{t(locale, 'Effect', 'אפקט')}</span>
          <FormControl size="small">
            <Select value={effect} onChange={(e) => updateDraft('effect', e.target.value)}>
              {EFFECT_LIST.map((ef) => (
                <MenuItem key={ef.value} value={ef.value}>{t(locale, ef.label_en, ef.label_he)}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </div>

        {effect === 'FIX_OFFSET' && (
          <div className="constraint-field">
            <span className="adv-field-label">{t(locale, 'Offset (MM:SS)', 'היסט (דק:שנ)')}</span>
            <TextField size="small" value={draft.offset_mmss} onChange={(e) => updateDraft('offset_mmss', e.target.value)} inputProps={{ dir: 'ltr', placeholder: '02:30' }} />
          </div>
        )}

        {effect === 'OFFSET_WINDOW' && (
          <>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Min offset (MM:SS)', 'היסט מינ (דק:שנ)')}</span>
              <TextField size="small" value={draft.offset_min} onChange={(e) => updateDraft('offset_min', e.target.value)} inputProps={{ dir: 'ltr' }} />
            </div>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Max offset (MM:SS)', 'היסט מקס (דק:שנ)')}</span>
              <TextField size="small" value={draft.offset_max} onChange={(e) => updateDraft('offset_max', e.target.value)} inputProps={{ dir: 'ltr' }} />
            </div>
          </>
        )}

        {effect === 'PIN_COUNT' && (
          <div className="constraint-field">
            <span className="adv-field-label">{t(locale, 'Break count', 'מספר ברייקים')}</span>
            <TextField type="number" size="small" value={draft.pin_count} onChange={(e) => updateDraft('pin_count', e.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
          </div>
        )}

        {effect === 'DURATION_RANGE' && (
          <>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Min duration (s)', 'אורך מינ (שנ)')}</span>
              <TextField type="number" size="small" value={draft.duration_min} onChange={(e) => updateDraft('duration_min', e.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
            </div>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Max duration (s)', 'אורך מקס (שנ)')}</span>
              <TextField type="number" size="small" value={draft.duration_max} onChange={(e) => updateDraft('duration_max', e.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
            </div>
          </>
        )}

        <div className="constraint-field">
          <span className="adv-field-label">{t(locale, 'Order index (optional)', 'אינדקס סדר (רשות)')}</span>
          <TextField type="number" size="small" value={draft.order_index} onChange={(e) => updateDraft('order_index', e.target.value)} inputProps={{ min: 0, dir: 'ltr' }} />
        </div>

        <div className="constraint-field">
          <span className="adv-field-label">{t(locale, 'Notes (optional)', 'הערות (רשות)')}</span>
          <TextField size="small" value={draft.notes} onChange={(e) => updateDraft('notes', e.target.value)} />
        </div>
      </div>

      <div className="constraint-builder-actions">
        <Button type="button" variant="contained" className="run-button" disabled={saving || !available} onClick={saveConstraint}>
          <Save size={14} />
          {t(locale, 'Save constraint', 'שמירת אילוץ')}
        </Button>
        <Button type="button" variant="outlined" className="run-button" disabled={recomputeState === 'running'} onClick={() => onRecompute && onRecompute()}>
          <Send size={14} />
          {t(locale, 'Run the weekly plan', 'הרצת הלוח השבועי')}
        </Button>
      </div>

      <div className="constraint-list">
        <div className="panel-head">
          <h3>{t(locale, 'Existing constraints', 'אילוצים קיימים')}</h3>
          <span>{items.length}</span>
        </div>
        {items.length === 0 ? (
          <p className="constraint-list-empty">{t(locale, 'No constraints yet.', 'אין אילוצים עדיין.')}</p>
        ) : (
          <ul>
            {items.map((item, index) => {
              const itemId = item.constraint_id ?? item.id;
              return (
                <li key={itemId ?? `constraint-${index}`}>
                  <span className="constraint-chip">{effectLabel(item.effect, locale)}</span>
                  <span className="constraint-scope">{item.where ? t(locale, 'filter conditions', 'תנאי סינון') : `${item.scope_type}: ${item.scope_value || t(locale, 'any', 'הכול')}`}</span>
                  {item.notes && <span className="constraint-channel">{item.notes}</span>}
                  <Button type="button" variant="text" className="constraint-delete" onClick={() => deleteConstraint(itemId)} aria-label={t(locale, 'Delete constraint', 'מחיקת אילוץ')}>
                    <Trash2 size={14} />
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}

export default ConstraintBuilder;
