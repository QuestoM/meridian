import React, { useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Numeric, formatNumber } from '../shell/format';
import {
  FILTER_LABELS,
  FILTER_ORDER,
  MODEL_STATE_LABELS,
  label,
  text,
} from './sources-copy';
import { readFields, writeFields } from './sources-fields';
import FieldsMenu from './FieldsMenu';
import SourceCard from './SourceCard';
import RowsDrawer from './RowsDrawer';

// The model version the plan's numbers rest on: its name, its state, and the
// sources it was measured on. Three states and never a confident fourth. The
// remedy names its owner and carries no button, because training is the
// company team's act and an operator cannot perform it.
function ModelPanel({ model, locale }) {
  if (!model || !model.available) {
    return (
      <section className="model-strip unknown">
        <p>{text('modelUnavailable', locale)}</p>
      </section>
    );
  }
  const status = String(model.status || 'unknown');
  const tone = status === 'fresh' ? 'ok' : status === 'stale' ? 'warn' : 'muted';
  return (
    <section className={`model-strip ${tone}`}>
      <div className="model-strip-head">
        <span className="model-strip-name">
          {text('modelVersion', locale)} <Numeric>{model.version}</Numeric>
        </span>
        <span className={`source-state ${tone}`}>{label(MODEL_STATE_LABELS, status, locale)}</span>
      </div>
      <p className="model-strip-note">{locale === 'he' ? model.note_he : model.note_en}</p>
      <p className="model-strip-sources">
        <span>{text('modelMeasuredOn', locale)}</span>
        {(model.measured_on || []).map((source) => (
          <span key={source} className="model-source" dir="ltr">{source}</span>
        ))}
      </p>
      {(model.changed_sources || []).length > 0 ? (
        <p className="model-strip-sources">
          <span>{text('modelChanged', locale)}</span>
          {model.changed_sources.map((source) => (
            <span key={source} className="model-source warn" dir="ltr">{source}</span>
          ))}
        </p>
      ) : null}
    </section>
  );
}

// Every input a run reads, as one card each. The filter rail is the status
// vocabulary itself, so a state is also a place: three inputs stored and not
// read is a number you can click, and the address in the bar carries it, so
// that place can be returned to and sent to somebody else.
export function InputsView({ status, locale, canEdit, canEditReason, filter, onFilter, onOpenFile, onReload, notify }) {
  const [rowsIndex, setRowsIndex] = useState(-1);
  const [fields, setFields] = useState(readFields);

  const inputs = Array.isArray(status.inputs) ? status.inputs : [];
  const summary = status.summary || {};
  const counts = useMemo(() => {
    const tally = { all: inputs.length };
    FILTER_ORDER.forEach((key) => {
      if (key === 'all') return;
      tally[key] = Number(summary[key] ?? inputs.filter((input) => input.state === key).length) || 0;
    });
    return tally;
  }, [inputs, summary]);

  const shown = filter === 'all' ? inputs : inputs.filter((input) => input.state === filter);

  function chooseFields(next) {
    setFields(next);
    writeFields(next);
  }

  return (
    <div className="sources-view">
      <ModelPanel model={status.model} locale={locale} />

      <div className="source-filter-bar">
        <div className="source-filters" role="tablist" aria-label={text('destination', locale)}>
          {FILTER_ORDER.filter((key) => key === 'all' || counts[key] > 0).map((key) => (
            <Button
              key={key}
              type="button"
              role="tab"
              aria-selected={filter === key}
              className={filter === key ? 'source-filter active' : 'source-filter'}
              onClick={() => onFilter(key)}
            >
              {label(FILTER_LABELS, key, locale)}
              <Numeric>{formatNumber(counts[key], locale)}</Numeric>
            </Button>
          ))}
        </div>
        <FieldsMenu fields={fields} onChange={chooseFields} locale={locale} />
      </div>

      {canEdit ? null : <p className="sources-note">{canEditReason || text('readOnly', locale)}</p>}

      {shown.length === 0 ? (
        <p className="sources-note">{text('none', locale)}</p>
      ) : (
        <div className="source-grid">
          {shown.map((input, index) => (
            <SourceCard
              key={input.kind}
              input={input}
              locale={locale}
              canEdit={canEdit}
              canEditReason={canEditReason}
              fields={fields}
              onOpenRows={() => setRowsIndex(index)}
              onOpenFile={onOpenFile}
              onChanged={onReload}
              notify={notify}
            />
          ))}
        </div>
      )}

      {rowsIndex >= 0 && shown[rowsIndex] ? (
        <RowsDrawer
          input={shown[rowsIndex]}
          position={rowsIndex + 1}
          total={shown.length}
          locale={locale}
          onStep={(step) => setRowsIndex((current) => Math.min(shown.length - 1, Math.max(0, current + step)))}
          onClose={() => setRowsIndex(-1)}
        />
      ) : null}
    </div>
  );
}

export default InputsView;
