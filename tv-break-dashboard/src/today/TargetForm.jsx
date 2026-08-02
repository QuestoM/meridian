import React, { useState } from 'react';
import { Button, TextField } from '@mui/material';
import { finiteNumber, formatCurrency, pageText } from '../shell/format';
import { windowLabel } from './TodayMoney';

// The control an empty target becomes.
//
// Two numbers, both supplied by the person, neither defaulted. The band is a
// field rather than a constant because the boundary between on plan, at risk
// and behind is a commercial rule, and a default here would be this product
// inventing one and then printing it as if somebody had decided it.
//
// It is inline rather than a dialog: a target belongs to the figure it
// measures, and moving it into a modal would put a wall between the two.

export function TargetForm({ today, locale, onSave, onCancel, saveState }) {
  const target = today.target || {};
  const money = today.money || {};
  const [amount, setAmount] = useState(target.amount_ils === null || target.amount_ils === undefined ? '' : String(target.amount_ils));
  const [band, setBand] = useState(target.at_risk_band_percent === null || target.at_risk_band_percent === undefined ? '' : String(target.at_risk_band_percent));
  const [note, setNote] = useState(String(target.note || ''));

  const amountValue = finiteNumber(amount);
  const bandValue = finiteNumber(band);
  const amountValid = amountValue !== null && amountValue > 0;
  const bandValid = bandValue !== null && bandValue >= 0 && bandValue <= 100;
  const busy = saveState === 'saving';

  return (
    <form
      className="today-target-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!amountValid || !bandValid || busy) return;
        onSave({ amount_ils: amountValue, at_risk_band_percent: bandValue, note });
      }}
    >
      <p className="today-note">
        {pageText(
          locale,
          `The target for ${windowLabel(money, locale)}, in shekels of expected revenue.`,
          `היעד ל${windowLabel(money, locale)}, בשקלים של הכנסה צפויה.`,
        )}
      </p>
      <div className="today-form-row">
        <TextField
          label={pageText(locale, 'Target amount in shekels', 'סכום היעד בשקלים')}
          value={amount}
          onChange={(event) => setAmount(event.target.value)}
          size="small"
          type="number"
          inputProps={{ min: 0, step: 1000, dir: 'ltr' }}
          autoFocus
          error={amount !== '' && !amountValid}
          helperText={amountValid ? formatCurrency(amountValue, locale) : pageText(locale, 'A positive number of shekels', 'מספר חיובי של שקלים')}
        />
        <TextField
          label={pageText(locale, 'At risk below the target by, in percent', 'בסיכון כשמתחת ליעד באחוזים')}
          value={band}
          onChange={(event) => setBand(event.target.value)}
          size="small"
          type="number"
          inputProps={{ min: 0, max: 100, step: 0.5, dir: 'ltr' }}
          error={band !== '' && !bandValid}
          helperText={pageText(locale, 'Below the target by more than this reads as behind', 'מתחת ליעד ביותר מזה נקרא פיגור')}
        />
      </div>
      <TextField
        label={pageText(locale, 'Note, optional', 'הערה, לא חובה')}
        value={note}
        onChange={(event) => setNote(event.target.value)}
        size="small"
        fullWidth
      />
      <div className="today-target-actions">
        <Button className="today-primary" type="submit" variant="contained" disabled={!amountValid || !bandValid || busy}>
          {busy ? pageText(locale, 'Saving', 'שומר') : pageText(locale, 'Save the target', 'שמירת היעד')}
        </Button>
        <Button className="today-secondary" type="button" onClick={onCancel} disabled={busy}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </Button>
      </div>
    </form>
  );
}

export default TargetForm;
