import React from 'react';
import { Slider } from '@mui/material';
import { Pressable } from '../../studio/dom-controls';
import { formatPercent, pageText } from '../../shell/format';
import { Figure } from '../../shell/bidi';
import { OBJECTIVE_FOCUS, leverLabel } from './plan-week-model';

// One leg of the comparison: four objective levers and one shared licence cap.
//
// The panel used to offer the revenue weight and nothing else, and that is
// exactly the lever the engine is least sensitive to at a fixed retention floor,
// so the comparison could not separate two scenarios. Every lever the runner
// accepts is here, and the floor is first because it is the one that moves the
// plan. The hourly cap is displayed from the saved licence and cannot diverge
// between the two legs here.

function Row({ label, help, children }) {
  return (
    <div className="plan-leg-row">
      <div className="plan-leg-label">
        <span>{label}</span>
        {help ? <small>{help}</small> : null}
      </div>
      {children}
    </div>
  );
}

export function ScenarioLegForm({ leg, title, values, locale, onChange }) {
  const he = locale === 'he';
  const floorPercent = Math.round((Number(values.retention_floor) || 0) * 100);
  const caution = Math.round((Number(values.risk_lambda) || 0) * 100);
  return (
    <div className={`plan-leg plan-leg-${leg}`} aria-label={title}>
      <h3 className="plan-leg-title">{title}</h3>

      <Row
        label={leverLabel('retention_floor', locale)}
        help={pageText(locale, 'The lever that moves the plan', 'הידית שמזיזה את התוכנית')}
      >
        <div className="plan-leg-slider">
          <Slider
            size="small"
            value={floorPercent}
            min={50}
            max={99}
            step={1}
            valueLabelDisplay="auto"
            aria-label={`${title} ${leverLabel('retention_floor', locale)}`}
            onChange={(_event, value) => onChange('retention_floor', (Array.isArray(value) ? value[0] : value) / 100)}
          />
          <strong className="numeric"><Figure>{formatPercent(floorPercent, locale)}</Figure></strong>
        </div>
      </Row>

      <Row label={leverLabel('revenue_weight', locale)}>
        <div className="plan-leg-slider">
          <Slider
            size="small"
            value={Number(values.revenue_weight) || 0}
            min={0}
            max={100}
            step={5}
            valueLabelDisplay="auto"
            aria-label={`${title} ${leverLabel('revenue_weight', locale)}`}
            onChange={(_event, value) => onChange('revenue_weight', Array.isArray(value) ? value[0] : value)}
          />
          <strong className="numeric"><Figure>{Number(values.revenue_weight) || 0}</Figure></strong>
        </div>
      </Row>

      <Row
        label={leverLabel('max_breaks_per_hour', locale)}
        help={pageText(locale, 'Licence guardrail · identical in both scenarios', 'מגבלת רישיון · זהה בשני התרחישים')}
      >
        <div className="plan-leg-slider" aria-readonly="true">
          <strong className="numeric"><Figure>{Number(values.max_breaks_per_hour)}</Figure></strong>
        </div>
      </Row>

      <Row label={leverLabel('risk_lambda', locale)}>
        <div className="plan-leg-slider">
          <Slider
            size="small"
            value={caution}
            min={0}
            max={100}
            step={5}
            valueLabelDisplay="auto"
            aria-label={`${title} ${leverLabel('risk_lambda', locale)}`}
            onChange={(_event, value) => onChange('risk_lambda', (Array.isArray(value) ? value[0] : value) / 100)}
          />
          <strong className="numeric"><Figure>{caution}</Figure></strong>
        </div>
      </Row>

      <Row label={leverLabel('objective_mode', locale)}>
        <div className="plan-leg-focus">
          {OBJECTIVE_FOCUS.map((mode) => (
            <Pressable
              key={mode.key}
              type="button"
              className={`plan-leg-chip${values.objective_mode === mode.key ? ' is-active' : ''}`}
              aria-pressed={values.objective_mode === mode.key}
              onClick={() => onChange('objective_mode', mode.key)}
            >
              {he ? mode.he : mode.en}
            </Pressable>
          ))}
        </div>
      </Row>
    </div>
  );
}

export default ScenarioLegForm;
