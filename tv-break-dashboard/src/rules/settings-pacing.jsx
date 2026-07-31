import React from 'react';
import { Gauge } from 'lucide-react';
import DateField from '../shell/DateField';
import { NumberControl, ToggleControl } from './SettingsControls';

// The campaign-pacing panel of the settings surface, kept as a render function
// so the element tree is exactly what the single file produced.
export function renderPacingPanel({ he, draft, updateField, updateNumber, hasCampaignFlights }) {
  return (
        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{he ? 'קצב קמפיינים' : 'Campaign pacing'}</h2>
              <p>{he ? 'מטה את השיבוץ לפי קצב הדילוור של הקמפיינים, בלי לשנות את תחזית ההכנסה' : 'Steer placement by campaign delivery pace, without changing the revenue projection'}</p>
            </div>
            <Gauge size={18} />
          </div>
          {!hasCampaignFlights && (
            <p className="settings-pacing-note">
              {he ? 'טרם הועלו יעדי דילוור לקמפיינים, ולכן הקצב אינו פעיל.' : 'No campaign flights uploaded yet, so pacing is inactive.'}
            </p>
          )}
          <div className="settings-toggle-grid">
            <ToggleControl
              label={he ? 'קצב קמפיינים' : 'Campaign pacing'}
              checked={draft.pacing_enabled ?? true}
              onChange={(value) => updateField('pacing_enabled', value)}
              helperText={he ? 'מטה את השיבוץ לעבר קמפיינים שמפגרים בקצב הדילוור והרחק מקמפיינים שדילברו יותר מדי. שיבוץ בלבד; לעולם לא משנה את תחזית ההכנסה.' : 'Steer placement toward campaigns behind delivery pace and away from over-delivered ones. Placement only; never changes the revenue projection.'}
            />
            <DateField
              label={he ? 'תאריך ייחוס לקצב' : 'Pacing reference date'}
              value={draft.pacing_reference_date}
              onChange={(value) => updateField('pacing_reference_date', value)}
              helperText={he ? 'התאריך שנחשב כהיום בעת מדידת קצב הקמפיין. אם נשאר ריק, נעשה שימוש בתאריך התחולה של הלוח.' : 'The date treated as today when measuring campaign pace. When empty, the schedule effective date is used.'}
            />
            <NumberControl
              label={he ? 'עוצמת פיגור בקצב' : 'Behind-pace strength'}
              value={draft.pacing_urgency_k ?? 1.0}
              onChange={(value) => updateNumber('pacing_urgency_k', Math.min(5, Math.max(0, Number(value))))}
              helperText={he ? 'כמה חזק קמפיין שמפגר בדילוור מושך אליו ברייקים בשיבוץ.' : 'How hard an under-delivered campaign pulls breaks toward its inventory.'}
            />
            <NumberControl
              label={he ? 'תקרת פיגור בקצב' : 'Behind-pace cap'}
              value={draft.pacing_urgency_max ?? 2.0}
              onChange={(value) => updateNumber('pacing_urgency_max', Math.min(4, Math.max(1, Number(value))))}
              helperText={he ? 'הגברת השיבוץ המרבית לקמפיין המפגר ביותר.' : 'Maximum placement boost for the most behind campaign.'}
            />
            <NumberControl
              label={he ? 'ריסון דילוור-יתר' : 'Over-delivery throttle'}
              value={draft.pacing_ahead_k ?? 1.0}
              onChange={(value) => updateNumber('pacing_ahead_k', Math.min(5, Math.max(0, Number(value))))}
              helperText={he ? 'כמה חזק קמפיין בדילוור-יתר מקבל עדיפות נמוכה בשיבוץ. אפס מבטל את קנס דילוור-היתר.' : 'How hard an over-delivered campaign is de-prioritized in placement. Zero disables the over-delivery penalty.'}
            />
            <NumberControl
              label={he ? 'רצפת דילוור-יתר' : 'Over-delivery floor'}
              value={draft.pacing_weight_floor ?? 0.5}
              onChange={(value) => updateNumber('pacing_weight_floor', Math.min(1.0, Math.max(0.25, Number(value))))}
              helperText={he ? 'המשקל הנמוך ביותר בשיבוץ שקמפיין בדילוור-יתר יכול לקבל. לעולם לא אפס, כך שאף שיבוץ אינו נחסם לחלוטין.' : 'The lowest placement weight an over-delivered campaign can receive. Never zero, so a slot is never forbidden.'}
            />
            <NumberControl
              label={he ? 'רצפת מכנה הקצב' : 'Pace denominator floor'}
              value={draft.pacing_epsilon ?? 0.05}
              onChange={(value) => updateNumber('pacing_epsilon', Math.min(0.5, Math.max(0.01, Number(value))))}
              helperText={he ? 'רצפה חישובית ששומרת על חישוב הקצב יציב ביום הראשון והאחרון של הקמפיין.' : 'A computational floor that keeps the pace calculation stable on the first and last flight day.'}
            />
          </div>
        </section>
  );
}
