import React, { useMemo, useState } from 'react';
import { Absent, Basis, Panel, RecordDrill, STATE_ORDER, Verdict } from './console-bits';
import { pick, t } from './console-words';

// The gate table. One row per gate, its state, its basis on the same row, and
// the artifact's own sentence underneath. The counts at the top are the filter,
// so a number opens the rows behind it rather than describing them.
//
// Three rules the filter keeps, each of which was broken here and each measured
// in a browser rather than read out of this file:
//
// - The subtitle counts the rows on screen, never the rows in the payload. It
//   read "13 of 13" under every chip, including the one that rendered three.
// - A chip whose count is zero opens an empty state that names the state in the
//   artifact's own words and carries the way back. It opened nothing at all: no
//   rows, no groups, no sentence, which is section 3.6's dead end exactly.
// - Every row the filter keeps is rendered under some group. The two families
//   were hard-coded, so a gate in any third family would have vanished with no
//   trace and the chip count would have disagreed with the rows on screen.

function StateLegend({ states, counts, active, onPick, locale }) {
  return (
    <div className="mc-legend" role="group" aria-label={t('gates.filter', locale)}>
      <button
        type="button"
        className={`mc-legend-item ${active === 'all' ? 'on' : ''}`}
        onClick={() => onPick('all')}
        aria-pressed={active === 'all'}
      >
        <span className="mc-legend-count">{Object.values(counts).reduce((a, b) => a + b, 0)}</span>
        <span className="mc-legend-label">{t('gates.all', locale)}</span>
      </button>
      {STATE_ORDER.map((id) => {
        const state = states.find((entry) => entry.id === id);
        if (!state) return null;
        return (
          <button
            type="button"
            key={id}
            className={`mc-legend-item mc-${id} ${active === id ? 'on' : ''}`}
            onClick={() => onPick(id)}
            aria-pressed={active === id}
            title={locale === 'en' ? state.meaning_en : state.meaning_he}
          >
            <span className="mc-legend-count">{counts[id] ?? 0}</span>
            <span className="mc-legend-label">{locale === 'en' ? state.en : state.he}</span>
            <span className="mc-legend-meaning">{locale === 'en' ? state.meaning_en : state.meaning_he}</span>
          </button>
        );
      })}
    </div>
  );
}

function GateRow({ gate, locale, blocked }) {
  const [open, setOpen] = useState(false);
  const unblock = blocked ? blocked[gate.id] : null;
  return (
    <li className={`mc-gate mc-${gate.state}`}>
      <div className="mc-gate-head">
        <span className="mc-gate-name">{locale === 'en' ? gate.label_en : gate.label_he}</span>
        <Verdict
          state={gate.state}
          labelEn={gate.state_label_en}
          labelHe={gate.state_label_he}
          locale={locale}
        />
      </div>
      <Basis basis={gate.basis} locale={locale} />
      {gate.reason ? <p className="mc-gate-reason" dir="ltr">{gate.reason}</p> : null}
      {unblock ? (
        <p className="mc-gate-unblock">
          <span className="mc-gate-unblock-label">{t('gates.unblock', locale)}</span>
          {pick(unblock, 'condition', locale)}
          {unblock.earliest && unblock.earliest.start ? (
            <span className="mc-gate-unblock-date" dir="ltr">{unblock.earliest.start}</span>
          ) : null}
        </p>
      ) : null}
      <RecordDrill record={gate.basis && gate.basis.detail} locale={locale} open={open} onToggle={() => setOpen((v) => !v)} />
    </li>
  );
}

function LayerRow({ layer, locale }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="mc-layer">
      <div className="mc-gate-head">
        <span className="mc-gate-name">{locale === 'en' ? layer.label_en : layer.label_he}</span>
        <span className={`mc-verdict mc-layer-${layer.on ? 'on' : 'off'} mc-md`}>
          {layer.on ? t('gates.layer_on', locale) : t('gates.layer_off', locale)}
        </span>
      </div>
      <p className="mc-layer-note">{pick(layer, 'note', locale)}</p>
      {layer.reason ? <p className="mc-gate-reason" dir="ltr">{layer.reason}</p> : null}
      <RecordDrill record={layer.measured} locale={locale} open={open} onToggle={() => setOpen((v) => !v)} />
    </li>
  );
}

// What a filter opens when the state it names has no members. It says which
// state is empty, in that state's own recorded words, and carries the way back
// to all of them, so a count that was pressed never opens a silent hole.
function NoGates({ state, locale, onClear }) {
  const label = state ? (locale === 'en' ? state.en : state.he) : '';
  const meaning = state ? (locale === 'en' ? state.meaning_en : state.meaning_he) : '';
  return (
    <div className="mc-gate-empty">
      <Absent
        title={state ? t('gates.none_in_state', locale) : t('gates.none_recorded', locale)}
        reason={state ? `${label}: ${meaning}` : t('provenance.no_artifacts', locale)}
        action={state ? (
          <button type="button" className="mc-link" onClick={onClear}>
            {t('gates.show_all', locale)}
          </button>
        ) : null}
      />
    </div>
  );
}

export default function GatesPanel({ payload, blocked, locale }) {
  const [filter, setFilter] = useState('all');
  const gates = payload.gates || [];
  const states = payload.states || [];
  const rows = useMemo(
    () => (filter === 'all' ? gates : gates.filter((gate) => gate.state === filter)),
    [gates, filter],
  );
  // The two families the product has, first and in that order, and then any
  // family the ledger grows later. An unknown family renders under its own key
  // rather than disappearing: a raw key on screen is a defect somebody fixes,
  // and a dropped row is a defect nobody can see.
  const groups = useMemo(() => {
    const known = ['retention', 'audience'];
    const found = Array.from(new Set(rows.map((gate) => gate.model)));
    return known.concat(found.filter((model) => !known.includes(model)))
      .map((model) => [model, rows.filter((gate) => gate.model === model)])
      .filter(([, list]) => list.length > 0);
  }, [rows]);
  return (
    <>
      <Panel
        title={t('section.gates', locale)}
        sub={`${rows.length} ${t('gates.of', locale)} ${gates.length}`}
      >
        <StateLegend
          states={states}
          counts={payload.counts || {}}
          active={filter}
          onPick={setFilter}
          locale={locale}
        />
        {groups.length === 0 ? (
          <NoGates
            state={states.find((entry) => entry.id === filter) || null}
            locale={locale}
            onClear={() => setFilter('all')}
          />
        ) : null}
        {groups.map(([model, list]) => (
          <div className="mc-gate-group" key={model}>
            <h3>{t(`gates.${model}`, locale) || model}</h3>
            <ul className="mc-gate-list">
              {list.map((gate) => (
                <GateRow gate={gate} locale={locale} blocked={blocked} key={gate.id} />
              ))}
            </ul>
          </div>
        ))}
      </Panel>
      <Panel title={t('gates.layers', locale)}>
        <ul className="mc-gate-list">
          {(payload.layers || []).map((layer) => (
            <LayerRow layer={layer} locale={locale} key={layer.id} />
          ))}
        </ul>
      </Panel>
    </>
  );
}
