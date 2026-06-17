import React, { useState } from 'react';
import { Button, FormControl, InputLabel, MenuItem, Select, TextField } from '@mui/material';
import { Plus, PlusSquare, Save, Send, Trash2, X } from 'lucide-react';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

function t(locale, en, he) {
  return locale === 'he' ? he : en;
}

function normalizeRows(value) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.rows)) return value.rows;
  return [];
}

// ---- frozen field/operator catalog (mirrors GET /api/constraints/options shape) ----
const FIELD_DEFS = [
  { field: 'programme', label_en: 'Programme', label_he: 'תוכנית', type: 'string' },
  { field: 'genre', label_en: 'Genre', label_he: 'ז׳אנר', type: 'string' },
  { field: 'daypart', label_en: 'Daypart', label_he: 'רצועת שידור', type: 'daypart' },
  { field: 'weekday', label_en: 'Weekday', label_he: 'יום בשבוע', type: 'weekday' },
  { field: 'date', label_en: 'Date', label_he: 'תאריך', type: 'date' },
  { field: 'hour', label_en: 'Hour', label_he: 'שעה', type: 'hour' },
];

const OPERATORS_BY_TYPE = {
  string: [
    { op: 'is', label_en: 'is', label_he: 'הוא' },
    { op: 'is_not', label_en: 'is not', label_he: 'אינו' },
    { op: 'contains', label_en: 'contains', label_he: 'מכיל' },
    { op: 'not_contains', label_en: 'does not contain', label_he: 'אינו מכיל' },
    { op: 'starts_with', label_en: 'starts with', label_he: 'מתחיל ב' },
    { op: 'ends_with', label_en: 'ends with', label_he: 'מסתיים ב' },
    { op: 'regex', label_en: 'matches regex', label_he: 'תואם רגקס' },
    { op: 'in', label_en: 'is any of', label_he: 'אחד מ' },
  ],
  daypart: [
    { op: 'is', label_en: 'is', label_he: 'הוא' },
    { op: 'is_not', label_en: 'is not', label_he: 'אינו' },
    { op: 'in', label_en: 'is any of', label_he: 'אחד מ' },
  ],
  weekday: [
    { op: 'is', label_en: 'is', label_he: 'הוא' },
    { op: 'is_not', label_en: 'is not', label_he: 'אינו' },
    { op: 'in', label_en: 'is any of', label_he: 'אחד מ' },
  ],
  date: [
    { op: 'is', label_en: 'is', label_he: 'הוא' },
    { op: 'before', label_en: 'before', label_he: 'לפני' },
    { op: 'after', label_en: 'after', label_he: 'אחרי' },
    { op: 'between', label_en: 'between', label_he: 'בין' },
    { op: 'in', label_en: 'is any of', label_he: 'אחד מ' },
  ],
  hour: [
    { op: 'eq', label_en: '=', label_he: '=' },
    { op: 'lt', label_en: '<', label_he: '<' },
    { op: 'lte', label_en: '<=', label_he: '<=' },
    { op: 'gt', label_en: '>', label_he: '>' },
    { op: 'gte', label_en: '>=', label_he: '>=' },
    { op: 'between', label_en: 'between', label_he: 'בין' },
  ],
};

const DAYPART_VOCAB = ['morning', 'noon', 'evening', 'prime', 'night'];
const WEEKDAY_VOCAB = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function fieldDef(fieldName) {
  return FIELD_DEFS.find((f) => f.field === fieldName) || FIELD_DEFS[0];
}

function operatorsForField(fieldName) {
  const def = fieldDef(fieldName);
  return OPERATORS_BY_TYPE[def.type] || OPERATORS_BY_TYPE.string;
}

function defaultCondition() {
  return { _id: Math.random().toString(36).slice(2), field: 'programme', operator: 'is', value: '' };
}

function defaultGroup() {
  return { _id: Math.random().toString(36).slice(2), combinator: 'and', conditions: [defaultCondition()] };
}

// ---- Condition value input ---------------------------------------------------
function ConditionValueInput({ fieldName, operator, value, onChange, hints, locale }) {
  const def = fieldDef(fieldName);
  const he = locale === 'he';

  if (def.type === 'hour') {
    if (operator === 'between') {
      const min = typeof value === 'object' && value !== null ? (value.min ?? '') : '';
      const max = typeof value === 'object' && value !== null ? (value.max ?? '') : '';
      return (
        <div className="cb-between-pair">
          <TextField type="number" size="small" value={min} onChange={(e) => onChange({ min: Number(e.target.value), max })} inputProps={{ min: 0, max: 23, dir: 'ltr' }} placeholder="0" />
          <span className="cb-between-sep">{t(locale, 'and', 'עד')}</span>
          <TextField type="number" size="small" value={max} onChange={(e) => onChange({ min, max: Number(e.target.value) })} inputProps={{ min: 0, max: 23, dir: 'ltr' }} placeholder="23" />
        </div>
      );
    }
    return (
      <TextField type="number" size="small" value={value ?? ''} onChange={(e) => onChange(Number(e.target.value))} inputProps={{ min: 0, max: 23, dir: 'ltr' }} placeholder="0" />
    );
  }

  if (def.type === 'date') {
    if (operator === 'between') {
      const min = typeof value === 'object' && value !== null ? (value.min ?? '') : '';
      const max = typeof value === 'object' && value !== null ? (value.max ?? '') : '';
      return (
        <div className="cb-between-pair">
          <TextField type="date" size="small" value={min} onChange={(e) => onChange({ min: e.target.value, max })} InputLabelProps={{ shrink: true }} />
          <span className="cb-between-sep">{t(locale, 'and', 'עד')}</span>
          <TextField type="date" size="small" value={max} onChange={(e) => onChange({ min, max: e.target.value })} InputLabelProps={{ shrink: true }} />
        </div>
      );
    }
    if (operator === 'in') {
      const arr = Array.isArray(value) ? value : [];
      return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'yyyy-mm-dd, ...', 'yyyy-mm-dd, ...')} options={[]} locale={locale} />;
    }
    return (
      <TextField type="date" size="small" value={value ?? ''} onChange={(e) => onChange(e.target.value)} InputLabelProps={{ shrink: true }} />
    );
  }

  if (def.type === 'daypart') {
    if (operator === 'in') {
      const arr = Array.isArray(value) ? value : [];
      return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'morning, prime, ...', 'morning, prime, ...')} options={DAYPART_VOCAB} locale={locale} />;
    }
    return (
      <FormControl size="small" sx={{ minWidth: 140 }}>
        <Select value={value || ''} displayEmpty onChange={(e) => onChange(e.target.value)}>
          <MenuItem value="">{t(locale, 'Select', 'בחר')}</MenuItem>
          {DAYPART_VOCAB.map((v) => <MenuItem key={v} value={v}>{v}</MenuItem>)}
        </Select>
      </FormControl>
    );
  }

  if (def.type === 'weekday') {
    if (operator === 'in') {
      const arr = Array.isArray(value) ? value : [];
      return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'Mon, Fri, ...', 'Mon, Fri, ...')} options={WEEKDAY_VOCAB} locale={locale} />;
    }
    return (
      <FormControl size="small" sx={{ minWidth: 140 }}>
        <Select value={value || ''} displayEmpty onChange={(e) => onChange(e.target.value)}>
          <MenuItem value="">{t(locale, 'Select', 'בחר')}</MenuItem>
          {WEEKDAY_VOCAB.map((v) => <MenuItem key={v} value={v}>{v}</MenuItem>)}
        </Select>
      </FormControl>
    );
  }

  // string fields (programme / genre)
  if (operator === 'in') {
    const arr = Array.isArray(value) ? value : [];
    const optionList = hints[fieldName] || [];
    return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'Type or pick...', 'הקלד או בחר...')} options={optionList} locale={locale} />;
  }

  return (
    <TextField size="small" value={value ?? ''} onChange={(e) => onChange(e.target.value)} placeholder={t(locale, 'value', 'ערך')} inputProps={{ dir: he ? 'rtl' : 'ltr' }} />
  );
}

// ---- Chip input for "in" operators ------------------------------------------
function ChipInput({ value, onChange, placeholder, options, locale }) {
  const [text, setText] = useState('');
  const chips = Array.isArray(value) ? value : [];

  function addChip(chip) {
    const trimmed = chip.trim();
    if (trimmed && !chips.includes(trimmed)) {
      onChange([...chips, trimmed]);
    }
    setText('');
  }

  function removeChip(chip) {
    onChange(chips.filter((c) => c !== chip));
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addChip(text);
    } else if (e.key === 'Backspace' && !text && chips.length) {
      onChange(chips.slice(0, -1));
    }
  }

  return (
    <div className="cb-chip-input">
      <div className="cb-chip-list">
        {chips.map((chip) => (
          <span key={chip} className="cb-chip">
            {chip}
            <button type="button" className="cb-chip-remove" onClick={() => removeChip(chip)} aria-label={t(locale, `Remove ${chip}`, `הסר ${chip}`)}>
              <X size={10} />
            </button>
          </span>
        ))}
        <input
          className="cb-chip-text"
          value={text}
          placeholder={chips.length === 0 ? placeholder : ''}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => text.trim() && addChip(text)}
          dir="ltr"
        />
      </div>
      {options.length > 0 && (
        <div className="cb-chip-options">
          {options.filter((o) => !chips.includes(o) && (text === '' || o.toLowerCase().includes(text.toLowerCase()))).slice(0, 8).map((o) => (
            <button key={o} type="button" className="cb-chip-option" onClick={() => addChip(o)}>
              {o}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Condition row ----------------------------------------------------------
function ConditionRow({ condition, onUpdate, onDelete, hints, locale, level }) {
  const he = locale === 'he';
  const ops = operatorsForField(condition.field);

  function changeField(newField) {
    const newOps = operatorsForField(newField);
    const opStillValid = newOps.some((o) => o.op === condition.operator);
    onUpdate({ ...condition, field: newField, operator: opStillValid ? condition.operator : newOps[0].op, value: '' });
  }

  function changeOperator(newOp) {
    onUpdate({ ...condition, operator: newOp, value: '' });
  }

  return (
    <div className="cb-rule-row" style={{ marginInlineStart: `${level * 24}px` }}>
      <FormControl size="small" sx={{ minWidth: 130 }}>
        <Select value={condition.field} onChange={(e) => changeField(e.target.value)}>
          {FIELD_DEFS.map((f) => (
            <MenuItem key={f.field} value={f.field}>{t(locale, f.label_en, f.label_he)}</MenuItem>
          ))}
        </Select>
      </FormControl>
      <FormControl size="small" sx={{ minWidth: 130 }}>
        <Select value={condition.operator} onChange={(e) => changeOperator(e.target.value)}>
          {ops.map((o) => (
            <MenuItem key={o.op} value={o.op}>{t(locale, o.label_en, o.label_he)}</MenuItem>
          ))}
        </Select>
      </FormControl>
      <div className="cb-value-cell">
        <ConditionValueInput fieldName={condition.field} operator={condition.operator} value={condition.value} onChange={(v) => onUpdate({ ...condition, value: v })} hints={hints} locale={locale} />
      </div>
      <button type="button" className="cb-delete-btn" onClick={onDelete} aria-label={t(locale, 'Remove rule', 'הסר כלל')}>
        <Trash2 size={13} />
      </button>
    </div>
  );
}

// ---- Group ------------------------------------------------------------------
function GroupNode({ group, onUpdate, onDelete, hints, locale, level }) {
  const he = locale === 'he';

  function updateCondition(index, updated) {
    const next = [...group.conditions];
    next[index] = updated;
    onUpdate({ ...group, conditions: next });
  }

  function deleteCondition(index) {
    const next = group.conditions.filter((_, i) => i !== index);
    onUpdate({ ...group, conditions: next });
  }

  function addRule() {
    onUpdate({ ...group, conditions: [...group.conditions, defaultCondition()] });
  }

  function addSubGroup() {
    onUpdate({ ...group, conditions: [...group.conditions, defaultGroup()] });
  }

  function setCombinator(combinator) {
    onUpdate({ ...group, combinator });
  }

  return (
    <div className={`cb-group${level > 0 ? ' cb-group-nested' : ''}`}>
      <div className="cb-group-head">
        <div className="cb-combinator-toggle" role="group" aria-label={t(locale, 'Match condition', 'תנאי התאמה')}>
          <button
            type="button"
            className={`cb-combinator-btn${group.combinator === 'and' ? ' active' : ''}`}
            aria-pressed={group.combinator === 'and'}
            onClick={() => setCombinator('and')}
          >
            {t(locale, 'AND', 'וגם')}
          </button>
          <button
            type="button"
            className={`cb-combinator-btn${group.combinator === 'or' ? ' active' : ''}`}
            aria-pressed={group.combinator === 'or'}
            onClick={() => setCombinator('or')}
          >
            {t(locale, 'OR', 'או')}
          </button>
        </div>
        {level > 0 && onDelete && (
          <button type="button" className="cb-delete-btn" onClick={onDelete} aria-label={t(locale, 'Remove group', 'הסר קבוצה')}>
            <Trash2 size={13} />
          </button>
        )}
      </div>
      <div className="cb-group-body">
        {group.conditions.map((node, index) => {
          if (node.combinator !== undefined) {
            // nested group
            return (
              <GroupNode
                key={node._id}
                group={node}
                onUpdate={(updated) => updateCondition(index, updated)}
                onDelete={() => deleteCondition(index)}
                hints={hints}
                locale={locale}
                level={level + 1}
              />
            );
          }
          return (
            <ConditionRow
              key={node._id}
              condition={node}
              onUpdate={(updated) => updateCondition(index, updated)}
              onDelete={() => deleteCondition(index)}
              hints={hints}
              locale={locale}
              level={0}
            />
          );
        })}
      </div>
      <div className="cb-group-actions">
        <button type="button" className="cb-add-btn" onClick={addRule}>
          <Plus size={12} />
          {t(locale, 'Add rule', 'הוסף כלל')}
        </button>
        <button type="button" className="cb-add-btn" onClick={addSubGroup}>
          <PlusSquare size={12} />
          {t(locale, 'Add group', 'הוסף קבוצה')}
        </button>
      </div>
    </div>
  );
}

// ---- Serialize where tree (strip internal _id before sending) ---------------
function serializeNode(node) {
  if (node.combinator !== undefined) {
    return {
      combinator: node.combinator,
      conditions: node.conditions.map(serializeNode),
    };
  }
  return { field: node.field, operator: node.operator, value: node.value };
}

// ---- Effect parameter fields -----------------------------------------------
function mmssToSeconds(value) {
  const [minutes, seconds] = String(value || '00:00').split(':').map((part) => Number(part));
  return (Number.isFinite(minutes) ? minutes : 0) * 60 + (Number.isFinite(seconds) ? seconds : 0);
}

const EFFECT_LIST = [
  { value: 'FIX_OFFSET', label_en: 'Fix offset', label_he: 'היסט קבוע' },
  { value: 'OFFSET_WINDOW', label_en: 'Offset window', label_he: 'חלון היסט' },
  { value: 'PIN_COUNT', label_en: 'Pin count', label_he: 'מספר ברייקים קבוע' },
  { value: 'DURATION_RANGE', label_en: 'Duration range', label_he: 'טווח אורך' },
  { value: 'GOLD', label_en: 'Gold break', label_he: 'ברייק זהב' },
  { value: 'FORBID', label_en: 'Forbid', label_he: 'איסור' },
];

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
function ConstraintBuilder({ locale, notify, onRecompute, recomputeState }) {
  const he = locale === 'he';
  const [hints, setHints] = useState({ programme: [], genre: [], channels: [], available_channels: [] });
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
      setItems((current) => [...current, { ...body, id: saved.id || `constraint-${current.length + 1}` }]);
      notify('Constraint saved.', 'האילוץ נשמר.');
    } catch (err) {
      notify(`Saving the constraint failed (${err.message}).`, `שמירת האילוץ נכשלה (${err.message}).`);
    } finally {
      setSaving(false);
    }
  }

  async function deleteConstraint(id) {
    try {
      const res = await fetch(`${API_BASE}/api/constraints/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (res.status === 404) {
        setItems((current) => current.filter((item) => item.id !== id));
        return;
      }
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setItems((current) => current.filter((item) => item.id !== id));
      notify('Constraint removed.', 'האילוץ הוסר.');
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
          <p>{t(locale, 'Filter conditions (where), then choose what effect to apply', 'הגדר תנאי סינון (where) ולאחר מכן בחר אפקט להחלה')}</p>
        </div>
      </div>

      {!available && (
        <p className="constraint-builder-warning">
          {t(locale, 'The constraints API responded with 404. Saving is disabled until it is online.', 'ממשק האילוצים החזיר 404. השמירה מושבתת עד שיהיה זמין.')}
        </p>
      )}

      <div className="cb-section-label">{t(locale, 'When (filter conditions)', 'כאשר (תנאי סינון)')}</div>
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
          {t(locale, 'Save constraint', 'שמור אילוץ')}
        </Button>
        <Button type="button" variant="outlined" className="run-button" disabled={recomputeState === 'running'} onClick={() => onRecompute && onRecompute()}>
          <Send size={14} />
          {t(locale, 'Recompute weekly schedule', 'חשב מחדש את הלוח השבועי')}
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
            {items.map((item) => (
              <li key={item.id}>
                <span className="constraint-chip">{item.effect}</span>
                <span className="constraint-scope">{item.where ? t(locale, 'predicate', 'פרדיקט') : `${item.scope_type}: ${item.scope_value || t(locale, 'any', 'הכול')}`}</span>
                {item.notes && <span className="constraint-channel">{item.notes}</span>}
                <Button type="button" variant="text" className="constraint-delete" onClick={() => deleteConstraint(item.id)} aria-label={t(locale, 'Delete constraint', 'מחק אילוץ')}>
                  <Trash2 size={14} />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

export default ConstraintBuilder;
