import React, { useEffect, useMemo, useState } from 'react';
import { Button, FormControl, InputLabel, MenuItem, Select, TextField } from '@mui/material';
import { AlertTriangle, BookOpen, Save, ShieldCheck, SlidersHorizontal, X } from 'lucide-react';
import DateField from '../shell/DateField';
import { finiteNumber, pageText, stableSettingsKey } from '../shell/format';
import ActivityLogPanel from '../history/ActivityLogPanel';
import ConstraintBuilder from './ConstraintBuilder';
import OperatorChannelPanel from './OperatorChannelPanel';
import { NumberControl, ToggleControl } from './SettingsControls';
import { renderObjectivePanel } from './settings-objective';
import { renderPacingPanel } from './settings-pacing';

export function SettingsPanel({ settings, parameters, copy, locale, saveState, onSave, onRecompute, recomputeState, notify, onGlobalRefresh }) {
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
  // The named setups (templates) that snap the levers to a known posture. Kept
  // in sync with GET /api/settings/controls so the dashboard and the engine
  // agree on what each preset means.
  const optimizerTemplates = [
    { key: 'balanced', label: he ? 'מאוזן' : 'Balanced', desc: he ? 'נוטה-להכנסה אך שומר על הצופים' : 'Revenue-leaning, viewer-protective', values: { revenue_weight: 60, risk_lambda: 0, min_retention_floor: 0.72 } },
    { key: 'revenue', label: he ? 'מקסום הכנסה' : 'Revenue priority', desc: he ? 'ממקסם הכנסה עד גבול הרגולציה' : 'Maximize revenue to the guardrails', values: { revenue_weight: 85, risk_lambda: 0, min_retention_floor: 0.70 } },
    { key: 'retention', label: he ? 'הגנת שימור' : 'Retention guardrail', desc: he ? 'פחות ברייקים, רצפת צפייה גבוהה' : 'Fewer breaks, higher floor', values: { revenue_weight: 35, risk_lambda: 0, min_retention_floor: 0.78 } },
    { key: 'conservative', label: he ? 'זהיר באי-ודאות' : 'Conservative', desc: he ? 'מדווח לפי עלות השימור הסבירה הגרועה ביותר' : 'Reports at the worst plausible retention cost', values: { revenue_weight: 60, risk_lambda: 1, min_retention_floor: 0.74 } },
  ];
  const revenueWeight = Number.isFinite(finiteNumber(draft.revenue_weight)) ? finiteNumber(draft.revenue_weight) : 60;
  const recomputeText =
    recomputeState === 'running'
      ? (he ? 'מחשב מחדש...' : 'Recomputing...')
      : recomputeState === 'done'
        ? (he ? 'הלוח עודכן' : 'Schedule updated')
        : recomputeState === 'error'
          ? (he ? 'החישוב נכשל' : 'Recompute failed')
          : (he ? 'חישוב מחדש של הלוח השבועי' : 'Recompute weekly schedule');

  const protectedTypes = (draft.protected_program_types || []).join(', ');

  // Honest empty state for pacing: pacing can only steer placement when there
  // are real campaign FLIGHTS (delivery targets) to pace against. The
  // /api/campaigns payload is a historical-spots rollup that is always
  // non-empty, so it says nothing about flights; key on the parameters
  // payload's real flight count when the backend provides it, with the
  // make-good data_available flag as a secondary signal. When neither field
  // exists (older backend) the note stays hidden rather than guessing.
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

  // Dirty detection: compare the in-progress draft against the saved settings.
  // This drives the "unsaved changes" affordance on the sticky action bar. We
  // compare by stable JSON so field order or array identity does not matter.
  const isDirty = useMemo(() => {
    try {
      return stableSettingsKey(draft) !== stableSettingsKey(settings);
    } catch {
      return true;
    }
  }, [draft, settings]);

  // The status line for the sticky bar reflects the real save lifecycle and the
  // real draft-vs-saved comparison: saving / saved / failed come from saveState,
  // otherwise we show unsaved vs all-saved based on isDirty.
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
    <section className="settings-workspace">
      <div className="settings-hero">
        <div>
          <span className="settings-kicker">{copy.nav.Settings}</span>
          <h1>{copy.settingsTitle}</h1>
          <p>{copy.settingsIntro}</p>
        </div>
      </div>

      <OperatorChannelPanel
        settings={draft}
        parameters={parameters}
        locale={locale}
        onSave={onSave}
        saveState={saveState}
        featured
      />

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


        <section className="settings-panel wide">
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

        <section className="settings-panel">
          <div className="settings-panel-head">
            <div>
              <h2>{copy.guardrails}</h2>
              <p>{locale === 'he' ? 'בקרות תכנון מסחריות' : 'Commercial planning controls'}</p>
            </div>
            <ShieldCheck size={18} />
          </div>
          <div className="settings-form-stack">
            <NumberControl label={copy.maxAdMinutes} value={draft.max_ad_minutes_per_hour} onChange={(value) => updateNumber('max_ad_minutes_per_hour', value)} suffix="min" />
            <NumberControl label={copy.maxBreaks} value={draft.max_breaks_per_hour} onChange={(value) => updateNumber('max_breaks_per_hour', value)} suffix="/hr" />
            <NumberControl label={copy.spacing} value={draft.min_break_spacing_minutes} onChange={(value) => updateNumber('min_break_spacing_minutes', value)} suffix="min" />
            <NumberControl label={copy.retentionFloor} value={Math.round((draft.min_retention_floor || 0) * 100)} onChange={(value) => updateNumber('min_retention_floor', Number(value) / 100)} suffix="%" />
            <NumberControl
              label={copy.riskCautionSetting}
              value={Math.round((finiteNumber(draft.risk_lambda) || 0) * 100)}
              onChange={(value) => updateNumber('risk_lambda', Math.min(1, Math.max(0, Number(value) / 100)))}
              suffix="/100"
            />
          </div>
        </section>

        <section className="settings-panel">
          <div className="settings-panel-head">
            <div>
              <h2>{locale === 'he' ? 'תוכן מוגן' : 'Protected content'}</h2>
              <p>{locale === 'he' ? 'חדשות, ילדים ותוכניות רגישות' : 'News, kids, and sensitive formats'}</p>
            </div>
            <AlertTriangle size={18} />
          </div>
          <div className="settings-form-stack">
            <NumberControl label={copy.protectedMax} value={draft.protected_program_max_ad_minutes_per_hour} onChange={(value) => updateNumber('protected_program_max_ad_minutes_per_hour', value)} suffix="min" />
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

        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{locale === 'he' ? 'מדיניות מסחרית' : 'Commercial policy'}</h2>
              <p>{locale === 'he' ? 'חסויות וברייקי זהב' : 'Sponsorships and gold breaks'}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="settings-toggle-grid">
            <ToggleControl label={copy.sponsorships} checked={draft.sponsorships_enabled} onChange={(value) => updateField('sponsorships_enabled', value)} />
            <ToggleControl label={copy.gold} checked={draft.gold_breaks_enabled} onChange={(value) => updateField('gold_breaks_enabled', value)} />
            <NumberControl label={locale === 'he' ? 'מקסימום ברייקי זהב ביום' : 'Max gold breaks per day'} value={draft.gold_breaks_max_per_day} onChange={(value) => updateNumber('gold_breaks_max_per_day', value)} suffix="/day" />
            <NumberControl label={copy.dailyCap} value={draft.max_daily_ad_minutes} onChange={(value) => updateNumber('max_daily_ad_minutes', value)} suffix="min" />
          </div>
        </section>

        {renderPacingPanel({ he, draft, updateField, updateNumber, hasCampaignFlights })}


        <ConstraintBuilder
          locale={locale}
          notify={notify || (() => {})}
          onRecompute={onRecompute}
          recomputeState={recomputeState}
          onGlobalRefresh={onGlobalRefresh}
        />

        <ActivityLogPanel locale={locale} />
      </div>

      <div className={`settings-savebar tone-${stickyStatus.tone}`}>
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
    </section>
  );
}

export default SettingsPanel;
