import React, { useEffect, useMemo, useState } from 'react';
import { FormControl, InputLabel, MenuItem, Select, TextField } from '@mui/material';
import { Button } from '../studio/actions';
import { AlertTriangle, BookOpen, Save, SlidersHorizontal, X } from 'lucide-react';
import DateField from '../shell/DateField';
import { finiteNumber, pageText, stableSettingsKey } from '../shell/format';
import { NumberControl, ToggleControl } from './SettingsControls';
import { renderObjectivePanel } from './settings-objective';
import { renderPacingPanel } from './settings-pacing';

// The planning levers, carried across from the settings page unchanged so that
// nothing an operator could set yesterday is unreachable today. Three things
// left this panel and each has a better home: the channel declaration and the
// audience model switch are on Channel and model, and the four regulatory
// numbers are on The licence, where a change carries a date, a reason and a
// permission of its own rather than sitting beside the revenue slider.

export default function PlanningLevers({
  settings, parameters, copy, locale, saveState, onSave, onRecompute, recomputeState,
}) {
  const [draft, setDraft] = useState(settings);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  function updateField(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  function updateNumber(field, value) {
    const parsed = Number(value);
    updateField(field, Number.isFinite(parsed) ? parsed : 0);
  }

  function applyTemplate(values) {
    setDraft((current) => ({ ...current, ...values }));
  }

  const he = locale === 'he';
  const optimizerTemplates = [
    { key: 'balanced', label: he ? 'מאוזן' : 'Balanced', desc: he ? 'נוטה-להכנסה אך שומר על הצופים' : 'Revenue-leaning, viewer-protective', values: { revenue_weight: 60, risk_lambda: 0, min_retention_floor: 0.72 } },
    { key: 'revenue', label: he ? 'מקסום הכנסה' : 'Revenue priority', desc: he ? 'ממקסם הכנסה עד גבול הרגולציה' : 'Maximize revenue to the guardrails', values: { revenue_weight: 85, risk_lambda: 0, min_retention_floor: 0.70 } },
    { key: 'retention', label: he ? 'הגנת שימור' : 'Retention guardrail', desc: he ? 'פחות ברייקים, רצפת צפייה גבוהה' : 'Fewer breaks, higher floor', values: { revenue_weight: 35, risk_lambda: 0, min_retention_floor: 0.78 } },
    { key: 'conservative', label: he ? 'זהיר באי-ודאות' : 'Conservative', desc: he ? 'מדווח לפי עלות השימור הסבירה הגרועה ביותר' : 'Reports at the worst plausible retention cost', values: { revenue_weight: 60, risk_lambda: 1, min_retention_floor: 0.74 } },
  ];
  const revenueWeight = Number.isFinite(finiteNumber(draft.revenue_weight)) ? finiteNumber(draft.revenue_weight) : 60;
  const recomputeText = he ? 'בדיקת ההרצה השבועית' : 'Review weekly run';

  const protectedTypes = (draft.protected_program_types || []).join(', ');

  const flightsCount = finiteNumber(parameters?.flights_count);
  const makeGoodAvailable = parameters?.make_good?.data_available;
  const hasCampaignFlights = flightsCount !== null ? flightsCount > 0 : makeGoodAvailable !== false;
  const statusText =
    saveState === 'saved'
      ? copy.saved
      : saveState === 'saving'
        ? copy.saving
        : saveState === 'error'
          ? copy.saveFailed
          : copy.saveSettings;

  const isDirty = useMemo(() => {
    try {
      return stableSettingsKey(draft) !== stableSettingsKey(settings);
    } catch {
      return true;
    }
  }, [draft, settings]);

  const stickyStatus =
    saveState === 'saving'
      ? { text: copy.saving, tone: 'saving' }
      : saveState === 'error'
        ? { text: copy.saveFailed, tone: 'error' }
        : isDirty
          ? { text: copy.unsavedChanges, tone: 'dirty' }
          : saveState === 'saved'
            ? { text: copy.saved, tone: 'saved' }
            : { text: copy.noChanges, tone: 'clean' };

  return (
    <div className="rules-section">
      <div className="settings-grid">
        {renderObjectivePanel({
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
        })}

        <section className="card settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{copy.profile}</h2>
              <p>{draft.profile_name}</p>
            </div>
            <BookOpen size={18} />
          </div>
          <div className="settings-form-grid">
            <TextField
              label={copy.profile}
              size="small"
              value={draft.profile_name || ''}
              onChange={(event) => updateField('profile_name', event.target.value)}
            />
            <DateField
              label={copy.effectiveDate}
              value={draft.effective_date}
              onChange={(value) => updateField('effective_date', value)}
            />
            <FormControl size="small">
              <InputLabel id="settings-locale">{copy.language}</InputLabel>
              <Select
                labelId="settings-locale"
                label={copy.language}
                value={draft.locale || 'he'}
                onChange={(event) => updateField('locale', event.target.value)}
              >
                <MenuItem value="he">{copy.hebrew}</MenuItem>
                <MenuItem value="en">{copy.english}</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label={copy.source}
              size="small"
              value={draft.regulatory_source_url || ''}
              onChange={(event) => updateField('regulatory_source_url', event.target.value)}
            />
          </div>
        </section>

        <section className="card settings-panel">
          <div className="settings-panel-head">
            <div>
              <h2>{pageText(locale, 'Commercial planning controls', 'בקרות תכנון מסחריות')}</h2>
              <p>
                {pageText(
                  locale,
                  'Sales policy, not the licence. The four regulatory numbers live on The licence.',
                  'מדיניות מכירה, לא הרישיון. ארבעת מספרי הרגולציה נמצאים במדור הרישיון.',
                )}
              </p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="settings-form-stack">
            <NumberControl label={copy.retentionFloor} value={Math.round((draft.min_retention_floor || 0) * 100)} onChange={(value) => updateNumber('min_retention_floor', Number(value) / 100)} suffix="%" />
            <NumberControl
              label={copy.riskCautionSetting}
              value={Math.round((finiteNumber(draft.risk_lambda) || 0) * 100)}
              onChange={(value) => updateNumber('risk_lambda', Math.min(1, Math.max(0, Number(value) / 100)))}
              suffix="/100"
            />
            <NumberControl label={copy.dailyCap} value={draft.max_daily_ad_minutes} onChange={(value) => updateNumber('max_daily_ad_minutes', value)} suffix="min" />
          </div>
        </section>

        <section className="card settings-panel">
          <div className="settings-panel-head">
            <div>
              <h2>{pageText(locale, 'Protected content', 'תוכן מוגן')}</h2>
              <p>{pageText(locale, 'News, kids, and sensitive formats', 'חדשות, ילדים ותוכניות רגישות')}</p>
            </div>
            <AlertTriangle size={18} />
          </div>
          <div className="settings-form-stack">
            <TextField
              label={copy.protectedTypes}
              size="small"
              multiline
              minRows={3}
              value={protectedTypes}
              onChange={(event) =>
                updateField(
                  'protected_program_types',
                  event.target.value.split(',').map((item) => item.trim()).filter(Boolean),
                )
              }
            />
          </div>
        </section>

        <section className="card settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{pageText(locale, 'Commercial policy', 'מדיניות מסחרית')}</h2>
              <p>{pageText(locale, 'Sponsorships and gold breaks', 'חסויות וברייקי זהב')}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="settings-toggle-grid">
            <ToggleControl label={copy.sponsorships} checked={draft.sponsorships_enabled} onChange={(value) => updateField('sponsorships_enabled', value)} />
            <ToggleControl label={copy.gold} checked={draft.gold_breaks_enabled} onChange={(value) => updateField('gold_breaks_enabled', value)} />
            <NumberControl label={pageText(locale, 'Max gold breaks per day', 'מקסימום ברייקי זהב ביום')} value={draft.gold_breaks_max_per_day} onChange={(value) => updateNumber('gold_breaks_max_per_day', value)} suffix="/day" />
          </div>
        </section>

        {renderPacingPanel({ he, draft, updateField, updateNumber, hasCampaignFlights })}

      </div>

      <div className={`card settings-savebar tone-${stickyStatus.tone}`}>
        <span className="settings-savebar-status" aria-live="polite">
          <span className="settings-savebar-dot" aria-hidden="true" />
          {stickyStatus.text}
        </span>
        <Button
          className="secondary-button"
          type="button"
          variant="outlined"
          disabled={saveState === 'saving' || !isDirty}
          onClick={() => setDraft(settings)}
        >
          <X size={15} />
          {pageText(locale, 'Discard changes', 'ביטול שינויים')}
        </Button>
        <Button
          className="run-button"
          type="button"
          variant="contained"
          disabled={saveState === 'saving' || !isDirty}
          onClick={() => onSave(draft)}
        >
          <Save size={15} />
          {statusText}
        </Button>
      </div>
    </div>
  );
}
