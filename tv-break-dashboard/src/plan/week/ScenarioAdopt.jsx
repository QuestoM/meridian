import React from 'react';
import { Button } from '@mui/material';
import { Target } from 'lucide-react';
import { pageText } from '../../shell/format';
import { ADOPT_FIELDS, leverLabel, leverValueText, objectiveFromLevers } from './plan-week-model';

// The act step 3 exists for: turn the leg that won into the plan's objective.
//
// The comparison used to end in a sentence. A planner who read that B is worth
// 1.25M more had to carry five lever values across two steps by eye, set five
// sliders, and take on trust that what they set is what was compared. The five
// values are printed here on the card that earned them and one control writes
// all five into the objective at once.
//
// What is written is the lever set the server says that leg actually ran under,
// never the form's current position, so the objective that arrives on step 1 is
// the objective the money on this card was computed on. Nothing is saved: the
// unsaved-changes banner on step 1 prints each old value beside its new one and
// the planner still saves and still runs.
export function ScenarioAdopt({ leg, summary, locale, onAdopt }) {
  const values = objectiveFromLevers(summary?.levers);
  const letter = String(leg).toUpperCase();
  if (!values) {
    return (
      <p className="plan-note plan-note-amber plan-scenario-adopt-note">
        {pageText(
          locale,
          'This run did not report the full lever set it used, so it cannot be made the objective.',
          'ההרצה הזאת לא דיווחה את מלוא הידיות שבהן השתמשה, ולכן לא ניתן להפוך אותה למטרה.',
        )}
      </p>
    );
  }
  return (
    <div className="plan-scenario-adopt">
      <Button
        className="secondary-button compact"
        type="button"
        variant="outlined"
        onClick={() => onAdopt(leg)}
      >
        <Target size={14} />
        {pageText(locale, `Use scenario ${letter} as the objective`, `קביעת תרחיש ${letter} כמטרה`)}
      </Button>
      <ul className="plan-scenario-levers">
        {ADOPT_FIELDS.map(([from, to]) => (
          <li key={to}>
            <span>{leverLabel(to, locale)}</span>
            <strong className={to === 'objective_mode' ? '' : 'numeric'} dir={to === 'objective_mode' ? 'auto' : 'ltr'}>
              {leverValueText(to, summary.levers[from], locale)}
            </strong>
          </li>
        ))}
      </ul>
      <p className="plan-scenario-adopt-foot">
        {pageText(
          locale,
          'This opens step 1 with these values against the saved ones. Nothing is saved and nothing is on air until the plan is run.',
          'זה פותח את שלב 1 עם הערכים האלה מול השמורים. דבר אינו נשמר ודבר אינו בשידור עד שהתוכנית רצה.',
        )}
      </p>
    </div>
  );
}

export default ScenarioAdopt;
