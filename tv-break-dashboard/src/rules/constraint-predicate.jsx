import React, { useState } from 'react';
import { FormControl, MenuItem, Select, TextField } from '@mui/material';
import { Plus, PlusSquare, Trash2, X } from 'lucide-react';
import DateField from '../shell/DateField';
import { daypartLabel } from '../shell/surface-helpers';
import { InputControl, Pressable } from '../studio/dom-controls';

// The AND/OR predicate tree, split out of ConstraintBuilder so both files sit
// under the file-size law. Nothing about the grammar changed: the same frozen
// field and operator catalog, the same serialisation, the same saved rows.

function t(locale, en, he) {
  return locale === 'he' ? he : en;
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
// Israeli week order: the week starts on Sunday and ends on Saturday. The values
// stay the frozen predicate-contract tokens; only the display order is Israeli.
const WEEKDAY_VOCAB = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

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
          <TextField type="number" size="small" value={min} onChange={(e) => onChange({ min: Number(e.target.value), max })} slotProps={{ htmlInput: { min: 0, max: 23, dir: 'ltr' } }} placeholder="0" />
          <span className="cb-between-sep">{t(locale, 'and', 'עד')}</span>
          <TextField type="number" size="small" value={max} onChange={(e) => onChange({ min, max: Number(e.target.value) })} slotProps={{ htmlInput: { min: 0, max: 23, dir: 'ltr' } }} placeholder="23" />
        </div>
      );
    }
    return (
      <TextField type="number" size="small" value={value ?? ''} onChange={(e) => onChange(Number(e.target.value))} slotProps={{ htmlInput: { min: 0, max: 23, dir: 'ltr' } }} placeholder="0" />
    );
  }

  if (def.type === 'date') {
    if (operator === 'between') {
      const min = typeof value === 'object' && value !== null ? (value.min ?? '') : '';
      const max = typeof value === 'object' && value !== null ? (value.max ?? '') : '';
      return (
        <div className="cb-between-pair">
          <DateField value={min} onChange={(next) => onChange({ min: next, max })} />
          <span className="cb-between-sep">{t(locale, 'and', 'עד')}</span>
          <DateField value={max} onChange={(next) => onChange({ min, max: next })} />
        </div>
      );
    }
    if (operator === 'in') {
      const arr = Array.isArray(value) ? value : [];
      return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'yyyy-mm-dd, ...', 'yyyy-mm-dd, ...')} options={[]} locale={locale} />;
    }
    return (
      <DateField value={value ?? ''} onChange={(next) => onChange(next)} />
    );
  }

  if (def.type === 'daypart') {
    // Dayparts render as a localized option list; the stored values stay the
    // engine keys (morning, prime, ...) so the saved predicate is unchanged.
    if (operator === 'in') {
      const arr = Array.isArray(value) ? value : [];
      return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'Pick dayparts', 'בחרו רצועות שידור')} options={DAYPART_VOCAB} labelFor={(v) => daypartLabel(v, locale)} locale={locale} />;
    }
    return (
      <FormControl size="small" sx={{ minWidth: 140 }}>
        <Select value={value || ''} displayEmpty onChange={(e) => onChange(e.target.value)}>
          <MenuItem value="">{t(locale, 'Select', 'בחרו')}</MenuItem>
          {DAYPART_VOCAB.map((v) => <MenuItem key={v} value={v}>{daypartLabel(v, locale)}</MenuItem>)}
        </Select>
      </FormControl>
    );
  }

  if (def.type === 'weekday') {
    if (operator === 'in') {
      const arr = Array.isArray(value) ? value : [];
      return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'Sun, Fri, ...', 'Sun, Fri, ...')} options={WEEKDAY_VOCAB} locale={locale} />;
    }
    return (
      <FormControl size="small" sx={{ minWidth: 140 }}>
        <Select value={value || ''} displayEmpty onChange={(e) => onChange(e.target.value)}>
          <MenuItem value="">{t(locale, 'Select', 'בחרו')}</MenuItem>
          {WEEKDAY_VOCAB.map((v) => <MenuItem key={v} value={v}>{v}</MenuItem>)}
        </Select>
      </FormControl>
    );
  }

  // string fields (programme / genre)
  if (operator === 'in') {
    const arr = Array.isArray(value) ? value : [];
    const optionList = hints[fieldName] || [];
    return <ChipInput value={arr} onChange={onChange} placeholder={t(locale, 'Type or pick...', 'הקלידו או בחרו...')} options={optionList} locale={locale} />;
  }

  return (
    <TextField size="small" value={value ?? ''} onChange={(e) => onChange(e.target.value)} placeholder={t(locale, 'value', 'ערך')} slotProps={{ htmlInput: { dir: he ? 'rtl' : 'ltr' } }} />
  );
}

// ---- Chip input for "in" operators ------------------------------------------
// labelFor (optional) localizes how a stored value is DISPLAYED on chips and
// option buttons; the stored values themselves stay the raw engine keys.
function ChipInput({ value, onChange, placeholder, options, locale, labelFor }) {
  const [text, setText] = useState('');
  const chips = Array.isArray(value) ? value : [];
  const display = (chip) => (labelFor ? labelFor(chip) : chip);

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
            {display(chip)}
            <Pressable type="button" className="cb-chip-remove" onClick={() => removeChip(chip)} aria-label={t(locale, `Remove ${display(chip)}`, `הסרת ${display(chip)}`)}>
              <X size={10} />
            </Pressable>
          </span>
        ))}
        <InputControl
          className="cb-chip-text"
          value={text}
          placeholder={chips.length === 0 ? placeholder : ''}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => text.trim() && addChip(text)}
        />
      </div>
      {options.length > 0 && (
        <div className="cb-chip-options">
          {options.filter((o) => !chips.includes(o) && (text === '' || o.toLowerCase().includes(text.toLowerCase()) || String(display(o)).toLowerCase().includes(text.toLowerCase()))).slice(0, 8).map((o) => (
            <Pressable key={o} type="button" className="cb-chip-option" onClick={() => addChip(o)}>
              {display(o)}
            </Pressable>
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
    <div className="card cb-rule-row" style={{ marginInlineStart: `${level * 24}px` }}>
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
      <Pressable type="button" className="cb-delete-btn" onClick={onDelete} aria-label={t(locale, 'Remove rule', 'הסר כלל')}>
        <Trash2 size={13} />
      </Pressable>
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
    <div className={`card cb-group${level > 0 ? ' cb-group-nested' : ''}`}>
      <div className="cb-group-head">
        <div className="cb-combinator-toggle" role="group" aria-label={t(locale, 'Match condition', 'תנאי התאמה')}>
          <Pressable
            type="button"
            className={`cb-combinator-btn${group.combinator === 'and' ? ' active' : ''}`}
            aria-pressed={group.combinator === 'and'}
            onClick={() => setCombinator('and')}
          >
            {t(locale, 'AND', 'וגם')}
          </Pressable>
          <Pressable
            type="button"
            className={`cb-combinator-btn${group.combinator === 'or' ? ' active' : ''}`}
            aria-pressed={group.combinator === 'or'}
            onClick={() => setCombinator('or')}
          >
            {t(locale, 'OR', 'או')}
          </Pressable>
        </div>
        {level > 0 && onDelete && (
          <Pressable type="button" className="cb-delete-btn" onClick={onDelete} aria-label={t(locale, 'Remove group', 'הסר קבוצה')}>
            <Trash2 size={13} />
          </Pressable>
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
        <Pressable type="button" className="cb-add-btn" onClick={addRule}>
          <Plus size={12} />
          {t(locale, 'Add rule', 'הוסף כלל')}
        </Pressable>
        <Pressable type="button" className="cb-add-btn" onClick={addSubGroup}>
          <PlusSquare size={12} />
          {t(locale, 'Add group', 'הוסף קבוצה')}
        </Pressable>
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

export { FIELD_DEFS, GroupNode, defaultGroup, serializeNode };
