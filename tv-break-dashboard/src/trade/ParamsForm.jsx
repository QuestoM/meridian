import React from 'react';
import { Button } from '../studio/actions';
import { InputControl, SelectControl, TextAreaControl } from '../studio/dom-controls';
import { Plus, Trash2 } from 'lucide-react';
import { pageText } from '../shell/format';
import { PARAM_LABELS, paramLabel } from './param-labels';

// The editor for one term's parameters.
//
// WHY THE FORM IS DERIVED FROM THE EXTRACTION'S OWN SHAPE, and why that is
// schema-shaped rather than a shortcut. The strict parameter schemas live in
// kairos/trade/taxonomy_schemas.py — one per term, 812 lines of them — and the
// pipeline validates every extraction against the schema before it is ever
// stored. So the params this form receives ALREADY have the schema's shape: the
// tiers of a ladder are an array of {threshold, discount_percent}, a committed
// budget is a {amount, currency, basis} block, a tolerance is a number. Walking
// that shape produces the same fields a schema walk would, without a second copy
// of the schemas on this side of the wire to drift from the first.
//
// The one thing the shape cannot supply is a field the document never filled in,
// and that is exactly what the instance's `missing` list names. Those fields are
// added to the form as empty inputs marked required, so the gap is a place to
// type rather than a fact to work around.
//
// The server validates again on save and refuses with the field it rejected. A
// form is a convenience; the schema is the authority.

function isMoneyBlock(value) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && Object.prototype.hasOwnProperty.call(value, 'amount');
}

function fieldId(path) {
  return `param-${path.join('-').replace(/[^a-zA-Z0-9-]/g, '_')}`;
}

const BASIS_OPTIONS = ['gross', 'net_of_commission', 'net_of_discount', 'ratecard', 'unstated'];

function MoneyField({ path, value, locale, onChange }) {
  const amountId = fieldId([...path, 'amount']);
  const basisId = fieldId([...path, 'basis']);
  return (
    <fieldset className="trade-form__group">
      <legend>{paramLabel(path[path.length - 1], locale)}</legend>
      <div className="trade-form__row">
        <label htmlFor={amountId}>{pageText(locale, 'Amount', 'סכום')}</label>
        <InputControl
          id={amountId}
          type="number"
          step="0.01"
          value={value.amount ?? ''}
          onChange={(event) => onChange({ ...value, amount: event.target.value === '' ? null : Number(event.target.value) })}
        />
      </div>
      <div className="trade-form__row">
        <label htmlFor={basisId}>{pageText(locale, 'Basis the amount is stated on', 'הבסיס שעליו נקוב הסכום')}</label>
        <SelectControl
          id={basisId}
          value={value.basis || 'unstated'}
          onChange={(event) => onChange({ ...value, basis: event.target.value })}
        >
          {BASIS_OPTIONS.map((option) => (
            <option key={option} value={option}>{paramLabel(option, locale)}</option>
          ))}
        </SelectControl>
      </div>
    </fieldset>
  );
}

function ScalarField({ path, value, locale, onChange, required }) {
  const id = fieldId(path);
  const key = path[path.length - 1];
  const label = paramLabel(key, locale);
  if (typeof value === 'boolean') {
    return (
      <div className="trade-form__row trade-form__row--check">
        <InputControl
          id={id}
          type="checkbox"
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
        />
        <label htmlFor={id}>{label}</label>
      </div>
    );
  }
  if (typeof value === 'number') {
    return (
      <div className="trade-form__row">
        <label htmlFor={id}>{label}</label>
        <InputControl
          id={id}
          type="number"
          step="any"
          value={value ?? ''}
          onChange={(event) => onChange(event.target.value === '' ? null : Number(event.target.value))}
        />
      </div>
    );
  }
  const text = value === null || value === undefined ? '' : String(value);
  const long = text.length > 60 || PARAM_LABELS[key]?.long;
  return (
    <div className="trade-form__row">
      <label htmlFor={id}>
        {label}
        {required ? (
          <span className="trade-form__required">
            {pageText(locale, ' required, and the document did not supply it', ' חובה, והמסמך לא סיפק אותו')}
          </span>
        ) : null}
      </label>
      {long ? (
        <TextAreaControl id={id} rows={3} value={text} onChange={(event) => onChange(event.target.value)} />
      ) : (
        <InputControl id={id} type="text" value={text} onChange={(event) => onChange(event.target.value)} />
      )}
    </div>
  );
}

function ListField({ path, value, locale, onChange }) {
  const id = fieldId(path);
  return (
    <div className="trade-form__row">
      <label htmlFor={id}>{paramLabel(path[path.length - 1], locale)}</label>
      <InputControl
        id={id}
        type="text"
        value={value.join(', ')}
        onChange={(event) => onChange(event.target.value.split(',').map((part) => part.trim()).filter(Boolean))}
      />
      <p className="trade-form__hint">
        {pageText(locale, 'Separate each value with a comma.', 'הפרידו כל ערך בפסיק.')}
      </p>
    </div>
  );
}

function RowsField({ path, value, locale, onChange }) {
  const columns = Array.from(new Set(value.flatMap((entry) => Object.keys(entry || {}))));
  function updateCell(index, key, next) {
    onChange(value.map((entry, i) => (i === index ? { ...entry, [key]: next } : entry)));
  }
  return (
    <fieldset className="trade-form__group">
      <legend>{paramLabel(path[path.length - 1], locale)}</legend>
      {value.map((entry, index) => (
        <div className="trade-form__rowset" key={index}>
          {columns.map((key) => {
            const cell = entry ? entry[key] : undefined;
            if (isMoneyBlock(cell)) {
              return (
                <MoneyField
                  key={key}
                  path={[...path, String(index), key]}
                  value={cell}
                  locale={locale}
                  onChange={(next) => updateCell(index, key, next)}
                />
              );
            }
            return (
              <ScalarField
                key={key}
                path={[...path, String(index), key]}
                value={cell === undefined ? '' : cell}
                locale={locale}
                onChange={(next) => updateCell(index, key, next)}
              />
            );
          })}
          <Button
            type="button"
            variant="outlined"
            className="trade-secondary trade-form__remove"
            onClick={() => onChange(value.filter((_, i) => i !== index))}
          >
            <Trash2 size={14} aria-hidden="true" />
            {pageText(locale, 'Remove this row', 'הסירו את השורה')}
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outlined"
        className="trade-secondary"
        onClick={() => onChange([...value, columns.reduce((seed, key) => ({ ...seed, [key]: '' }), {})])}
      >
        <Plus size={14} aria-hidden="true" />
        {pageText(locale, 'Add a row', 'הוסיפו שורה')}
      </Button>
    </fieldset>
  );
}

export default function ParamsForm({ params, missing = [], locale, onChange }) {
  const shape = params && typeof params === 'object' ? params : {};
  // A required field the extraction never filled becomes an empty input. Without
  // this the only way to supply it would be to reject the whole term.
  const withGaps = { ...shape };
  missing.forEach((key) => {
    if (!Object.prototype.hasOwnProperty.call(withGaps, key)) withGaps[key] = '';
  });

  function setKey(key, next) {
    onChange({ ...withGaps, [key]: next });
  }

  return (
    <div className="trade-form">
      {Object.entries(withGaps).map(([key, value]) => {
        const path = [key];
        const required = missing.includes(key);
        if (Array.isArray(value)) {
          if (value.length > 0 && typeof value[0] === 'object' && value[0] !== null) {
            return <RowsField key={key} path={path} value={value} locale={locale} onChange={(next) => setKey(key, next)} />;
          }
          return <ListField key={key} path={path} value={value} locale={locale} onChange={(next) => setKey(key, next)} />;
        }
        if (isMoneyBlock(value)) {
          return <MoneyField key={key} path={path} value={value} locale={locale} onChange={(next) => setKey(key, next)} />;
        }
        if (value && typeof value === 'object') {
          return (
            <fieldset className="trade-form__group" key={key}>
              <legend>{paramLabel(key, locale)}</legend>
              {Object.entries(value).map(([inner, innerValue]) => (
                <ScalarField
                  key={inner}
                  path={[key, inner]}
                  value={innerValue}
                  locale={locale}
                  onChange={(next) => setKey(key, { ...value, [inner]: next })}
                />
              ))}
            </fieldset>
          );
        }
        return (
          <ScalarField
            key={key}
            path={path}
            value={value}
            locale={locale}
            required={required}
            onChange={(next) => setKey(key, next)}
          />
        );
      })}
    </div>
  );
}
