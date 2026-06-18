import React, { useState } from 'react';
import { Button, Slider } from '@mui/material';
import { ArrowRight, GitCompare } from 'lucide-react';
import {
  API_BASE,
  finiteNumber,
  formatCurrency,
  formatNumber,
  formatPercent,
  pageText,
} from './surface-helpers';

// ScenarioCompare: a small A/B what-if. The operator picks two revenue weights
// (weight_a, weight_b) and runs both through POST /api/scenario-compare, which
// executes two genuine optimizer runs under shared guardrails. We show both
// summaries side by side plus the delta. The optimizer's score is a convex-blend
// "objective" (NOT literal revenue-minus-cost), so it is labelled "objective" and
// the API's revenue_net (always null here) is never invented.

function SummaryCard({ title, summary, accent, locale }) {
  if (!summary) {
    return null;
  }
  return (
    <div className={`scenario-card${accent ? ` ${accent}` : ''}`}>
      <div className="scenario-card-head">
        <strong>{title}</strong>
        <span className="numeric" dir="ltr">{pageText(locale, 'weight', 'משקל')} {finiteNumber(summary.revenue_weight) ?? '-'}</span>
      </div>
      <dl className="scenario-stat-list">
        <div>
          <dt>{pageText(locale, 'Projected revenue', 'הכנסה צפויה')}</dt>
          <dd className="numeric" dir="ltr">{formatCurrency(summary.projected_revenue, locale)}</dd>
        </div>
        <div>
          <dt>{pageText(locale, 'Average retention', 'שימור ממוצע')}</dt>
          <dd className="numeric" dir="ltr">{formatPercent(summary.average_retention, locale)}</dd>
        </div>
        <div>
          <dt>{pageText(locale, 'Objective (blend)', 'אובייקטיב (משוקלל)')}</dt>
          <dd className="numeric" dir="ltr">{formatNumber(summary.objective, locale)}</dd>
        </div>
        <div>
          <dt>{pageText(locale, 'Total breaks', 'סך ברייקים')}</dt>
          <dd className="numeric" dir="ltr">{formatNumber(summary.total_breaks, locale)}</dd>
        </div>
        <div>
          <dt>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</dt>
          <dd className="numeric" dir="ltr">{formatNumber(summary.total_ad_seconds, locale)}</dd>
        </div>
      </dl>
      <span className={`scenario-compliance${summary.compliant ? ' ok' : ' warn'}`}>
        {summary.compliant
          ? pageText(locale, 'Compliant with guardrails', 'עומד באילוצים')
          : pageText(locale, 'Breaches a guardrail', 'חורג מאילוץ')}
      </span>
    </div>
  );
}

function DeltaRow({ label, value, locale, formatter, suffix }) {
  const number = finiteNumber(value);
  if (number === null) {
    return (
      <div className="scenario-delta-row">
        <span>{label}</span>
        <strong>{pageText(locale, 'n/a', 'לא זמין')}</strong>
      </div>
    );
  }
  const sign = number > 0 ? '+' : '';
  const tone = number > 0 ? 'up' : number < 0 ? 'down' : '';
  return (
    <div className="scenario-delta-row">
      <span>{label}</span>
      <strong className={`numeric ${tone}`} dir="ltr">{sign}{formatter(number, locale)}{suffix || ''}</strong>
    </div>
  );
}

export default function ScenarioCompare({ locale, savedRevenueWeight = null }) {
  const he = locale === 'he';
  const baseWeight = finiteNumber(savedRevenueWeight);
  const [weightA, setWeightA] = useState(Number.isFinite(baseWeight) ? baseWeight : 40);
  const [weightB, setWeightB] = useState(Number.isFinite(baseWeight) ? Math.min(100, baseWeight + 25) : 75);
  const [state, setState] = useState({ status: 'idle', payload: null });

  async function run() {
    setState({ status: 'running', payload: null });
    try {
      const response = await fetch(`${API_BASE}/api/scenario-compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weight_a: Math.round(weightA), weight_b: Math.round(weightB) }),
      });
      if (!response.ok) throw new Error(`${response.status}`);
      const payload = await response.json();
      if (payload.available === false) {
        setState({ status: 'unavailable', payload });
      } else {
        setState({ status: 'ready', payload });
      }
    } catch {
      setState({ status: 'error', payload: null });
    }
  }

  const { status, payload } = state;

  return (
    <section className="page-panel scenario-compare">
      <div className="panel-head">
        <h2>{pageText(locale, 'Scenario A/B', 'השוואת תרחישים A/B')}</h2>
        <span>{pageText(locale, 'Two weights, one optimizer run each', 'שני משקלים, ריצת אופטימייזר לכל אחד')}</span>
      </div>

      <div className="scenario-controls" dir={he ? 'rtl' : 'ltr'}>
        <div className="scenario-weight-field">
          <label>{pageText(locale, 'Scenario A revenue weight', 'משקל הכנסה לתרחיש A')}</label>
          <div className="scenario-slider-row">
            <Slider
              size="small"
              value={weightA}
              min={0}
              max={100}
              step={5}
              valueLabelDisplay="auto"
              onChange={(_event, value) => setWeightA(Array.isArray(value) ? value[0] : value)}
            />
            <strong className="numeric" dir="ltr">{Math.round(weightA)}</strong>
          </div>
        </div>
        <div className="scenario-weight-field">
          <label>{pageText(locale, 'Scenario B revenue weight', 'משקל הכנסה לתרחיש B')}</label>
          <div className="scenario-slider-row">
            <Slider
              size="small"
              value={weightB}
              min={0}
              max={100}
              step={5}
              valueLabelDisplay="auto"
              onChange={(_event, value) => setWeightB(Array.isArray(value) ? value[0] : value)}
            />
            <strong className="numeric" dir="ltr">{Math.round(weightB)}</strong>
          </div>
        </div>
        <Button
          className="run-button"
          type="button"
          variant="contained"
          disabled={status === 'running'}
          onClick={run}
        >
          <GitCompare size={15} />
          {status === 'running'
            ? pageText(locale, 'Running...', 'מריץ...')
            : pageText(locale, 'Compare', 'השווה')}
        </Button>
      </div>

      {status === 'error' && (
        <div className="heatmap-empty">{pageText(locale, 'The comparison could not run right now.', 'ההשוואה לא הצליחה לרוץ כרגע.')}</div>
      )}
      {status === 'unavailable' && (
        <div className="heatmap-empty">{payload?.reason || pageText(locale, 'Scenario comparison is unavailable.', 'השוואת התרחישים אינה זמינה.')}</div>
      )}

      {status === 'ready' && payload && (
        <>
          <div className="scenario-grid">
            <SummaryCard title={pageText(locale, 'Scenario A', 'תרחיש A')} summary={payload.a} accent="accent-a" locale={locale} />
            <div className="scenario-arrow" aria-hidden="true"><ArrowRight size={18} /></div>
            <SummaryCard title={pageText(locale, 'Scenario B', 'תרחיש B')} summary={payload.b} accent="accent-b" locale={locale} />
          </div>

          <div className="scenario-delta">
            <h3>{pageText(locale, 'Delta (B minus A)', 'פער (B פחות A)')}</h3>
            <DeltaRow label={pageText(locale, 'Revenue', 'הכנסה')} value={payload.delta?.revenue} locale={locale} formatter={formatCurrency} />
            <DeltaRow label={pageText(locale, 'Retention', 'שימור')} value={payload.delta?.retention} locale={locale} formatter={formatNumber} suffix="pp" />
            <DeltaRow label={pageText(locale, 'Objective (blend)', 'אובייקטיב (משוקלל)')} value={payload.delta?.objective} locale={locale} formatter={formatNumber} />
            <DeltaRow label={pageText(locale, 'Breaks', 'ברייקים')} value={payload.delta?.breaks} locale={locale} formatter={formatNumber} />
            <DeltaRow label={pageText(locale, 'Ad seconds', 'שניות פרסום')} value={payload.delta?.ad_seconds} locale={locale} formatter={formatNumber} />
            <div className="scenario-delta-row muted">
              <span>{pageText(locale, 'Revenue net of retention', 'הכנסה בניכוי שימור')}</span>
              <strong>{pageText(locale, 'Not exposed', 'לא נחשף')}</strong>
            </div>
          </div>

          <p className="scenario-note">
            {pageText(
              locale,
              'Objective is the optimizer convex-blend score, not literal revenue minus retention cost. The API does not expose a revenue-net figure.',
              'האובייקטיב הוא ציון המשוקלל של האופטימייזר, לא הכנסה פחות עלות שימור ממש. ה-API אינו חושף ערך של הכנסה בניכוי שימור.',
            )}
          </p>
        </>
      )}
    </section>
  );
}
