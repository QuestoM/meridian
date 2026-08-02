import React from 'react';
import { Button, Slider } from '@mui/material';
import { RefreshCcw, SlidersHorizontal } from 'lucide-react';
import { finiteNumber } from '../shell/format';
import { NetComparisonCard } from '../today/MoneyWaterfall';

// The optimizer-balance panel of the settings surface, kept as a render
// function so the element tree is exactly what the single file produced.
export function renderObjectivePanel({
  he,
  locale,
  draft,
  revenueWeight,
  optimizerTemplates,
  applyTemplate,
  updateField,
  recomputeState,
  recomputeText,
  onRecompute,
}) {
  return (
        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{he ? 'איזון האופטימיזציה' : 'Optimizer balance'}</h2>
              <p>{he ? 'ההגדרה המרכזית שמניעה את הלוח, את ההכנסה מול השימור ואת התחזיות' : 'The central setting that drives the schedule, revenue vs retention, and forecasts'}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="optimizer-balance">
            <p className="optimizer-balance-help">
              {he
                ? 'כמה לרדוף אחרי הכנסת פרסום מול שמירה על הצופים. 0 שומר על הצפייה בלבד (כמעט בלי ברייקים), 100 ממקסם הכנסה עד גבול הרגולציה, 60 הוא איזון נוטה-להכנסה (ברירת המחדל).'
                : 'How hard to chase ad revenue versus protecting viewers. 0 protects retention only (almost no breaks), 100 maximizes revenue up to the regulatory guardrails, 60 is a revenue-leaning balance (the default).'}
            </p>
            <div className="optimizer-balance-slider">
              <span>{he ? 'צפייה' : 'Retention'}</span>
              <Slider
                value={revenueWeight}
                min={0}
                max={100}
                step={5}
                marks={[{ value: 0 }, { value: 60, label: he ? 'ברירת מחדל' : 'default' }, { value: 100 }]}
                valueLabelDisplay="on"
                onChange={(_event, value) => updateField('revenue_weight', Array.isArray(value) ? value[0] : value)}
              />
              <span>{he ? 'הכנסה' : 'Revenue'}</span>
            </div>
            <div className="optimizer-templates">
              {optimizerTemplates.map((template) => {
                const active = revenueWeight === template.values.revenue_weight && finiteNumber(draft.risk_lambda) === template.values.risk_lambda;
                return (
                  <button
                    key={template.key}
                    type="button"
                    className={`optimizer-template${active ? ' is-active' : ''}`}
                    onClick={() => applyTemplate(template.values)}
                  >
                    <strong>{template.label}</strong>
                    <small>{template.desc}</small>
                  </button>
                );
              })}
            </div>
            <div className="optimizer-objective">
              <span className="settings-field-label">{he ? 'מיקוד המנוע' : 'Engine focus'}</span>
              <div className="optimizer-objective-options">
                {[
                  { key: 'blend', label: he ? 'מאוזן, ברירת המחדל' : 'Balanced, the default',
                    desc: he ? 'המנוע מאזן בין הכנסות ברוטו לשמירה על הצופים, לפי המשקל שנקבע למעלה.' : 'The engine balances gross revenue against keeping viewers, using the weight set above.' },
                  { key: 'revenue_net', label: he ? 'ממוקד נטו' : 'Net focused',
                    desc: he ? 'המנוע מוותר על ברייקים שההכנסה שלהם נמוכה מעלות השימור שלהם: פחות ברייקים, ברוטו נמוך יותר, נטו גבוה יותר.' : 'The engine drops breaks whose revenue is below their retention cost: fewer breaks, lower gross, higher net.' },
                ].map((mode) => {
                  const active = (draft.objective_mode || 'blend') === mode.key;
                  return (
                    <button
                      key={mode.key}
                      type="button"
                      className={`optimizer-template${active ? ' is-active' : ''}`}
                      onClick={() => updateField('objective_mode', mode.key)}
                    >
                      <strong>{mode.label}</strong>
                      <small>{mode.desc}</small>
                    </button>
                  );
                })}
              </div>
              {(draft.objective_mode || 'blend') === 'revenue_net' && (
                <p className="optimizer-objective-note" role="status">
                  {he
                    ? 'שימו לב: מיקוד נטו משנה את התוכנית השמורה בהרצה הבאה, וההכנסות ברוטו בכותרת יירדו. זו בחירה מכוונת לטובת הנטו.'
                    : 'Note: net focus changes the saved plan on the next run, and the gross revenue headline will fall. It is a deliberate choice in favor of the net.'}
                </p>
              )}
              <NetComparisonCard locale={locale} refreshSignal={recomputeState || ''} currentFocus={draft.objective_mode || 'blend'} />
            </div>
            <div className="optimizer-recompute">
              <p>
                {he
                  ? 'שמרו את ההגדרות, ואז הריצו את הלוח השבועי כדי שהמסכים יראו את ההחלטה החדשה.'
                  : 'Save the settings, then run the weekly plan so the screens reflect the new decision.'}
              </p>
              <Button
                type="button"
                variant="outlined"
                className="run-button"
                disabled={recomputeState === 'running'}
                onClick={() => onRecompute && onRecompute()}
              >
                <RefreshCcw size={15} />
                {recomputeText}
              </Button>
            </div>
          </div>
        </section>
  );
}
