import React from 'react';
import { Button, Slider } from '@mui/material';
import { Check, RotateCcw, Save } from 'lucide-react';
import { finiteNumber, formatPercent, pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import {
  OBJECTIVE_FOCUS,
  OBJECTIVE_TEMPLATES,
  leverLabel,
  leverValueText,
  templateMatches,
} from './plan-week-model';

// The five fields the objective is, in the order the panel shows them.
const OBJECTIVE_FIELDS = [
  'revenue_weight',
  'min_retention_floor',
  'max_breaks_per_hour',
  'risk_lambda',
  'objective_mode',
];

// Every lever whose value would move if this draft were saved, each with the
// value on disk beside the value that would replace it.
//
// The banner used to print two of the five and neither of them as a pair, which
// is unreadable the moment a change arrives from somewhere other than the
// slider the planner just touched: adopting a compared scenario moves all five
// at once, and a planner has to be able to see what they are agreeing to.
export function objectiveChanges(draft, saved, locale) {
  if (!saved) return [];
  return OBJECTIVE_FIELDS
    .filter((field) => String(draft?.[field]) !== String(saved?.[field]))
    .map((field) => ({
      field,
      label: leverLabel(field, locale),
      was: leverValueText(field, saved[field], locale),
      next: leverValueText(field, draft[field], locale),
      // The engine focus is a name, not a figure, so it is not typeset as one.
      numeric: field !== 'objective_mode',
    }));
}

// Step one of the planner's job: what is this plan for.
//
// The four templates and the two engine focuses are the same values the settings
// surface has always applied, moved here by value because Bar 3 requires them to
// survive. What is new is that the decision now sits next to the run it drives
// and the comparison it will be judged by, instead of behind a settings page a
// programming representative has to scroll past a risk lambda to reach.

function LeverRow({ field, locale, children, help }) {
  return (
    <div className="plan-lever">
      <div className="plan-lever-head">
        <span className="plan-lever-name">{leverLabel(field, locale)}</span>
        {help ? <small className="plan-lever-help">{help}</small> : null}
      </div>
      {children}
    </div>
  );
}

export function ObjectivePanel({
  draft, saved, dirty, saveState, adopted, locale, onChange, onApplyTemplate, onSave, onRevert,
}) {
  const he = locale === 'he';
  const changes = objectiveChanges(draft, saved, locale);
  const adoptedLetter = adopted ? String(adopted).toUpperCase() : null;
  const weight = finiteNumber(draft.revenue_weight) ?? 60;
  const floor = finiteNumber(draft.min_retention_floor) ?? 0.72;
  const perHour = finiteNumber(draft.max_breaks_per_hour) ?? 4;
  const caution = Math.round((finiteNumber(draft.risk_lambda) ?? 0) * 100);
  const focus = String(draft.objective_mode || 'blend');

  return (
    <section className="plan-section" aria-labelledby="plan-objective-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-objective-title">{pageText(locale, 'What is this plan for', 'לשם מה התוכנית הזאת')}</h2>
          <p>
            {pageText(
              locale,
              'Four decisions drive every figure below. Pick a starting point, adjust, then save. Saving changes nothing on air until the plan is run.',
              'ארבע החלטות מניעות כל מספר בהמשך. בחרו נקודת פתיחה, כווננו, ואז שמרו. השמירה אינה משנה דבר בשידור עד להרצת התוכנית.',
            )}
          </p>
        </div>
        <Button
          className="run-button"
          type="button"
          variant="contained"
          disabled={!dirty || saveState === 'saving'}
          onClick={onSave}
        >
          {saveState === 'saved' ? <Check size={15} /> : <Save size={15} />}
          {saveState === 'saving'
            ? pageText(locale, 'Saving', 'שומר')
            : saveState === 'saved'
              ? pageText(locale, 'Saved', 'נשמר')
              : pageText(locale, 'Save the objective', 'שמירת המטרה')}
        </Button>
      </div>

      <div className="plan-template-row" role="group" aria-label={pageText(locale, 'Starting points', 'נקודות פתיחה')}>
        {OBJECTIVE_TEMPLATES.map((template) => {
          const active = templateMatches(template, draft);
          return (
            <button
              key={template.key}
              type="button"
              className={`plan-template${active ? ' is-active' : ''}`}
              aria-pressed={active}
              onClick={() => onApplyTemplate(template.values)}
            >
              <strong>{he ? template.he : template.en}</strong>
              <small>{he ? template.descHe : template.descEn}</small>
            </button>
          );
        })}
      </div>

      <div className="plan-lever-grid">
        <LeverRow
          field="revenue_weight"
          locale={locale}
          help={pageText(locale, '0 protects viewers only, 100 chases revenue to the guardrails', '0 מגן על הצופים בלבד, 100 רודף הכנסה עד גבול הרגולציה')}
        >
          <div className="plan-lever-slider">
            <span>{pageText(locale, 'Viewers', 'צופים')}</span>
            <Slider
              size="small"
              value={weight}
              min={0}
              max={100}
              step={5}
              valueLabelDisplay="auto"
              aria-label={leverLabel('revenue_weight', locale)}
              onChange={(_event, value) => onChange('revenue_weight', Array.isArray(value) ? value[0] : value)}
            />
            <span>{pageText(locale, 'Revenue', 'הכנסה')}</span>
            <strong className="numeric"><Figure>{weight}</Figure></strong>
          </div>
        </LeverRow>

        <LeverRow
          field="min_retention_floor"
          locale={locale}
          help={pageText(locale, 'The lever that actually moves the plan: raising it sheds the lowest-value breaks', 'הידית שבאמת מזיזה את התוכנית: העלאה שלה מוותרת על הברייקים בעלי הערך הנמוך ביותר')}
        >
          <div className="plan-lever-slider">
            <Slider
              size="small"
              value={Math.round(floor * 100)}
              min={50}
              max={99}
              step={1}
              valueLabelDisplay="auto"
              aria-label={leverLabel('min_retention_floor', locale)}
              onChange={(_event, value) => onChange('min_retention_floor', (Array.isArray(value) ? value[0] : value) / 100)}
            />
            <strong className="numeric"><Figure>{formatPercent(Math.round(floor * 100), locale)}</Figure></strong>
          </div>
        </LeverRow>

        <LeverRow field="max_breaks_per_hour" locale={locale}>
          <div className="plan-lever-slider">
            <Slider
              size="small"
              value={perHour}
              min={1}
              max={8}
              step={1}
              marks
              valueLabelDisplay="auto"
              aria-label={leverLabel('max_breaks_per_hour', locale)}
              onChange={(_event, value) => onChange('max_breaks_per_hour', Array.isArray(value) ? value[0] : value)}
            />
            <strong className="numeric"><Figure>{perHour}</Figure></strong>
          </div>
        </LeverRow>

        <LeverRow
          field="risk_lambda"
          locale={locale}
          help={pageText(locale, '0 prices the retention cost at its estimate, 100 at the worst plausible value', '0 מתמחר את עלות השימור לפי ההערכה, 100 לפי הערך הסביר הגרוע ביותר')}
        >
          <div className="plan-lever-slider">
            <Slider
              size="small"
              value={caution}
              min={0}
              max={100}
              step={5}
              valueLabelDisplay="auto"
              aria-label={leverLabel('risk_lambda', locale)}
              onChange={(_event, value) => onChange('risk_lambda', (Array.isArray(value) ? value[0] : value) / 100)}
            />
            <strong className="numeric"><Figure>{caution}</Figure></strong>
          </div>
        </LeverRow>
      </div>

      <div className="plan-focus" role="group" aria-label={leverLabel('objective_mode', locale)}>
        <span className="plan-lever-name">{leverLabel('objective_mode', locale)}</span>
        <div className="plan-focus-options">
          {OBJECTIVE_FOCUS.map((mode) => (
            <button
              key={mode.key}
              type="button"
              className={`plan-template${focus === mode.key ? ' is-active' : ''}`}
              aria-pressed={focus === mode.key}
              onClick={() => onChange('objective_mode', mode.key)}
            >
              <strong>{he ? mode.he : mode.en}</strong>
              <small>{he ? mode.descHe : mode.descEn}</small>
            </button>
          ))}
        </div>
      </div>

      {dirty && (
        <div className="plan-note plan-note-amber" role="status">
          <p>
            {pageText(
              locale,
              'These changes are not saved yet, and a saved change is not in the plan until it is run.',
              'השינויים האלה עדיין לא נשמרו, ושינוי שנשמר אינו בתוכנית עד שהיא רצה.',
            )}
          </p>
          {adoptedLetter ? (
            <p className="plan-note-detail">
              {pageText(
                locale,
                `These values are scenario ${adoptedLetter} of the comparison, exactly as it ran.`,
                `הערכים האלה הם תרחיש ${adoptedLetter} מההשוואה, בדיוק כפי שרץ.`,
              )}
            </p>
          ) : null}
          {changes.length > 0 && (
            <table className="plan-change-table">
              <thead>
                <tr>
                  <th scope="col">{pageText(locale, 'Lever', 'ידית')}</th>
                  <th scope="col">{pageText(locale, 'Saved now', 'שמור כרגע')}</th>
                  <th scope="col">{pageText(locale, 'After this change', 'אחרי השינוי')}</th>
                </tr>
              </thead>
              <tbody>
                {changes.map((change) => (
                  <tr key={change.field}>
                    <th scope="row">{change.label}</th>
                    <td className={change.numeric ? 'numeric was' : 'was'}>{change.numeric ? <Figure>{change.was}</Figure> : <Name>{change.was}</Name>}</td>
                    <td className={change.numeric ? 'numeric next' : 'next'}>{change.numeric ? <Figure>{change.next}</Figure> : <Name>{change.next}</Name>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {onRevert ? (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={onRevert}>
              <RotateCcw size={14} />
              {pageText(locale, 'Put the saved values back', 'החזרת הערכים השמורים')}
            </Button>
          ) : null}
        </div>
      )}
      {saveState === 'error' && (
        <p className="plan-note plan-note-red" role="alert">
          {pageText(locale, 'The objective could not be saved, so nothing changed.', 'לא ניתן היה לשמור את המטרה, ולכן דבר לא השתנה.')}
        </p>
      )}
    </section>
  );
}

export default ObjectivePanel;
