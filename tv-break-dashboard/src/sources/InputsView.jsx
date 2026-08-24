import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { useAddressParam } from '../shell/address-state';
import { ArrowRight } from 'lucide-react';
import { Numeric, formatNumber } from '../shell/format';
import { Code } from '../shell/bidi';
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
      <section className="card card-dense card-body model-strip unknown">
        <p>{text('modelUnavailable', locale)}</p>
      </section>
    );
  }
  const status = String(model.status || 'unknown');
  const tone = status === 'fresh' ? 'ok' : status === 'stale' ? 'warn' : 'muted';
  const changed = new Set(model.changed_sources || []);
  return (
    <section className={`card card-dense card-body model-strip ${tone}`}>
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
          <Code key={source} className="model-source">{source}</Code>
        ))}
      </p>
      {(model.changed_sources || []).length > 0 ? (
        <p className="model-strip-sources">
          <span>{text('modelChanged', locale)}</span>
          {model.changed_sources.map((source) => (
            <Code key={source} className="model-source warn">{source}</Code>
          ))}
        </p>
      ) : null}
      <div className="model-lineage" aria-label={locale === 'he' ? 'שרשרת מקור: קבצים, מודל ותוכנית' : 'Source chain: files, model and plan'}>
        <div className="lineage-source-stack">
          {(model.measured_on || []).map((source) => (
            <span key={source} className={changed.has(source) ? 'lineage-node source changed' : 'lineage-node source'}>
              <Code>{source}</Code>
            </span>
          ))}
        </div>
        <span className="lineage-connector" aria-hidden="true"><i /><ArrowRight size={14} /></span>
        <span className={`lineage-node model ${tone}`}>
          <small>{text('modelVersion', locale)}</small>
          <strong><Numeric>{model.version}</Numeric></strong>
        </span>
        <span className="lineage-connector" aria-hidden="true"><i /><ArrowRight size={14} /></span>
        <span className="lineage-node output">
          <small>{locale === 'he' ? 'מזין' : 'Feeds'}</small>
          <strong>{locale === 'he' ? 'תוכנית' : 'Plan'}</strong>
        </span>
      </div>
    </section>
  );
}

// Every input a run reads, as one card each. The filter rail is the status
// vocabulary itself, so a state is also a place: three inputs stored and not
// read is a number you can click, and the address in the bar carries it, so
// that place can be returned to and sent to somebody else.
export function InputsView({ status, locale, canEdit, canEditReason, filter, onFilter, onOpenFile,
                             onReload, notify, focusKind = '', focusFile = '' }) {
  const [rowsIndex, setRowsIndex] = useState(-1);
  const [fields, setFields] = useState(readFields);
  // The inspected input is an address (input in shell/nav.js), so Back steps
  // from one inspected input to the previous one, and a URL can point at one.
  const [selectedKind, setSelectedKind] = useAddressParam('input', '');
  const filterRefs = useRef([]);

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

  // A file named on another screen arrives here as a request to show it. The
  // filter is cleared when the requested card is not in the current one, because
  // landing on somebody else's filter and quietly showing a DIFFERENT card is
  // worse than landing nowhere: the reader asked for one file and would read the
  // state of another.
  useEffect(() => {
    if (!focusKind) return;
    setSelectedKind(focusKind);
    const visible = filter === 'all' || inputs.some(
      (input) => input.kind === focusKind && input.state === filter);
    if (!visible && onFilter) onFilter('all');
  }, [focusKind, filter, inputs, onFilter]);
  const visibleFilters = FILTER_ORDER.filter((key) => key === 'all' || counts[key] > 0);
  const selected = shown.find((input) => input.kind === selectedKind) || shown[0] || null;
  const selectedIndex = selected ? shown.findIndex((input) => input.kind === selected.kind) : -1;

  function chooseFields(next) {
    setFields(next);
    writeFields(next);
  }

  function chooseFilter(next) {
    setRowsIndex(-1);
    onFilter(next);
  }

  function onFilterKeyDown(event, index) {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = visibleFilters.length - 1;
    else if (event.key === 'ArrowRight') next = (index + (locale === 'he' ? -1 : 1) + visibleFilters.length) % visibleFilters.length;
    else if (event.key === 'ArrowLeft') next = (index + (locale === 'he' ? 1 : -1) + visibleFilters.length) % visibleFilters.length;
    else return;
    event.preventDefault();
    chooseFilter(visibleFilters[next]);
    filterRefs.current[next]?.focus();
  }

  return (
    <div className="sources-view">
      <ModelPanel model={status.model} locale={locale} />

      <div className="source-filter-bar">
        <div className="source-filters" role="tablist" aria-label={text('destination', locale)}>
          {visibleFilters.map((key, index) => (
            <Button
              ref={(node) => { filterRefs.current[index] = node; }}
              key={key}
              id={`source-filter-${key}`}
              type="button"
              role="tab"
              aria-selected={filter === key}
              aria-controls="source-inputs-panel"
              tabIndex={filter === key ? 0 : -1}
              className={filter === key ? 'source-filter active' : 'source-filter'}
              onClick={() => chooseFilter(key)}
              onKeyDown={(event) => onFilterKeyDown(event, index)}
            >
              {label(FILTER_LABELS, key, locale)}
              <Numeric>{formatNumber(counts[key], locale)}</Numeric>
            </Button>
          ))}
        </div>
        <FieldsMenu fields={fields} onChange={chooseFields} locale={locale} />
      </div>

      <div id="source-inputs-panel" role="tabpanel" aria-labelledby={`source-filter-${filter}`} tabIndex={0}>
        {canEdit ? null : <p className="sources-note">{canEditReason || text('readOnly', locale)}</p>}
        {focusFile && !focusKind ? (
          <p className="sources-note" role="status">
            {text('fileNotAnInput', locale)} <Code>{focusFile}</Code>
          </p>
        ) : null}

        {shown.length === 0 ? (
          <p className="sources-note">{text('none', locale)}</p>
        ) : (
          <div className="sources-control-room">
          <div className="source-grid" aria-label={text('destination', locale)}>
              {shown.map((input) => (
                <SourceCard
                  key={input.kind}
                  variant="row"
                  selected={selected && selected.kind === input.kind}
                  input={input}
                  locale={locale}
                  onSelect={() => setSelectedKind(input.kind)}
                />
              ))}
            </div>
            {selected ? (
              <aside className="card card-dense source-inspector" aria-label={locale === 'he' ? `פרטי ${selected.label_he || selected.label_en}` : `${selected.label_en} details`}>
                <SourceCard
                  key={selected.kind}
                  variant="inspector"
                  input={selected}
                  locale={locale}
                  canEdit={canEdit}
                  canEditReason={canEditReason}
                  fields={fields}
                  onOpenRows={() => setRowsIndex(selectedIndex)}
                  onOpenFile={onOpenFile}
                  onChanged={onReload}
                  notify={notify}
                />
              </aside>
            ) : null}
          </div>
        )}
      </div>

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
