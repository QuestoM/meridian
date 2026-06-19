import React, { useEffect, useState } from 'react';
import { Button, Switch, TextField } from '@mui/material';
import { Plus } from 'lucide-react';
import {
  EMPTY_ADVERTISER,
  GENRE_PRESETS,
  POSITION_PRESETS,
  pageText,
  parseTokens,
  premiumHint,
  serializeTokens,
  toggleToken,
} from './advertisers-helpers';

// A compact chip multi-select used by the add-advertiser form. ANY is mutually
// exclusive with specific tokens, mirroring the inline editors elsewhere.
function ChipSelect({ label, presets, value, onChange, locale }) {
  const tokens = parseTokens(value);
  const anyActive = tokens.length === 1 && tokens[0].toUpperCase() === 'ANY';
  const options = [...presets];
  tokens.forEach((token) => {
    if (token.toUpperCase() !== 'ANY' && !options.includes(token)) {
      options.push(token);
    }
  });
  return (
    <div className="adv-chip-field">
      <span className="adv-field-label">{label}</span>
      <div className="adv-chip-row" role="group" aria-label={label}>
        {options.map((option) => {
          const isAny = option.toUpperCase() === 'ANY';
          const active = isAny ? anyActive : tokens.includes(option);
          return (
            <button
              key={option}
              type="button"
              className={`adv-chip${active ? ' active' : ''}${isAny ? ' any' : ''}`}
              aria-pressed={active}
              onClick={() => onChange(serializeTokens(toggleToken(tokens, option)))}
            >
              <span dir="ltr">{isAny ? pageText(locale, 'Any', 'הכול') : option}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// The create-an-advertiser form. POSTs a new baseline advertiser rule, then the
// parent reloads the list and stats so the new card appears immediately.
function AddAdvertiserForm({ locale, suggestedId, existingIds, onCreate, onCancel }) {
  const [draft, setDraft] = useState({ ...EMPTY_ADVERTISER, advertiser_id: suggestedId });
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    setDraft((current) => (current.advertiser_id ? current : { ...current, advertiser_id: suggestedId }));
  }, [suggestedId]);

  const update = (field, value) => setDraft((current) => ({ ...current, [field]: value }));
  const trimmedId = draft.advertiser_id.trim();
  const duplicate = existingIds.includes(trimmedId);
  const canCreate = trimmedId.length > 0 && !duplicate && !creating;
  const hint = premiumHint(draft.default_premium, locale);

  async function handleCreate() {
    if (!canCreate) {
      return;
    }
    setCreating(true);
    const ok = await onCreate({ ...draft, advertiser_id: trimmedId });
    setCreating(false);
    if (ok) {
      setDraft({ ...EMPTY_ADVERTISER, advertiser_id: '' });
    }
  }

  return (
    <div className="adv-add-form">
      <div className="adv-add-grid">
        <div className="adv-id-field">
          <span className="adv-field-label">{pageText(locale, 'Advertiser ID', 'מזהה מפרסם')}</span>
          <TextField
            size="small"
            value={draft.advertiser_id}
            error={duplicate}
            helperText={duplicate ? pageText(locale, 'This ID already exists', 'מזהה זה כבר קיים') : ' '}
            onChange={(event) => update('advertiser_id', event.target.value)}
            inputProps={{ dir: 'ltr', 'aria-label': pageText(locale, 'Advertiser ID', 'מזהה מפרסם') }}
          />
        </div>
        <div className="adv-premium-field">
          <span className="adv-field-label">{pageText(locale, 'Premium (x rate card)', 'מקדם (× מחירון)')}</span>
          <div className="adv-premium-input">
            <TextField
              type="number"
              size="small"
              inputProps={{ min: 0, step: 0.05, dir: 'ltr', 'aria-label': pageText(locale, 'Default premium multiplier', 'מקדם תוספת ברירת מחדל') }}
              value={draft.default_premium ?? 1}
              onChange={(event) => update('default_premium', event.target.value === '' ? '' : Number(event.target.value))}
            />
            <span className={`adv-premium-hint ${hint.tone}`} dir="ltr">{hint.text}</span>
          </div>
        </div>
        <ChipSelect
          label={pageText(locale, 'Allowed positions', 'מיקומים מותרים')}
          presets={POSITION_PRESETS}
          value={draft.allow_positions}
          onChange={(value) => update('allow_positions', value)}
          locale={locale}
        />
        <ChipSelect
          label={pageText(locale, 'Allowed genres', 'ז׳אנרים מותרים')}
          presets={GENRE_PRESETS}
          value={draft.allow_genres}
          onChange={(value) => update('allow_genres', value)}
          locale={locale}
        />
        <div className="adv-cell-prime">
          <span className="adv-field-label">{pageText(locale, 'Prime time only', 'פריים טיים בלבד')}</span>
          <Switch
            size="small"
            checked={Boolean(draft.prime_time_only)}
            onChange={(event) => update('prime_time_only', event.target.checked)}
            inputProps={{ 'aria-label': pageText(locale, 'Prime time only', 'פריים טיים בלבד') }}
          />
        </div>
        <div className="adv-premium-field">
          <span className="adv-field-label">{pageText(locale, 'Behind-pace strength', 'עוצמת השלמה כשמאחור בלוז')}</span>
          <TextField
            type="number"
            size="small"
            placeholder={pageText(locale, 'channel default', 'ברירת מחדל של הערוץ')}
            inputProps={{ min: 0, step: 0.1, dir: 'ltr', 'aria-label': pageText(locale, 'Behind-pace pacing strength (blank uses channel default)', 'עוצמת השלמת קצב כשמאחור בלוז (ריק = ברירת מחדל של הערוץ)') }}
            value={draft.urgency_k ?? ''}
            onChange={(event) => update('urgency_k', event.target.value)}
          />
        </div>
        <div className="adv-premium-field">
          <span className="adv-field-label">{pageText(locale, 'Over-delivery restraint', 'עוצמת ריסון כשמקדים את הלוז')}</span>
          <TextField
            type="number"
            size="small"
            placeholder={pageText(locale, 'channel default', 'ברירת מחדל של הערוץ')}
            inputProps={{ min: 0, step: 0.1, dir: 'ltr', 'aria-label': pageText(locale, 'Over-delivery pacing restraint (blank uses channel default)', 'עוצמת ריסון בהקדמת לוז (ריק = ברירת מחדל של הערוץ)') }}
            value={draft.ahead_k ?? ''}
            onChange={(event) => update('ahead_k', event.target.value)}
          />
        </div>
        <div className="adv-notes-field">
          <span className="adv-field-label">{pageText(locale, 'Notes', 'הערות')}</span>
          <TextField
            size="small"
            fullWidth
            value={draft.notes || ''}
            onChange={(event) => update('notes', event.target.value)}
            inputProps={{ 'aria-label': pageText(locale, 'Notes', 'הערות') }}
          />
        </div>
      </div>
      <div className="adv-add-actions">
        <Button className="run-button" type="button" variant="contained" disabled={!canCreate} onClick={handleCreate}>
          <Plus size={15} />
          {creating ? pageText(locale, 'Adding...', 'מוסיף...') : pageText(locale, 'Add advertiser', 'הוספת מפרסם')}
        </Button>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={onCancel}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </Button>
      </div>
    </div>
  );
}

export default AddAdvertiserForm;
