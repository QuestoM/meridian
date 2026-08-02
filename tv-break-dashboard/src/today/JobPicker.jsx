import React, { useState } from 'react';
import { Button } from '@mui/material';
import { pageText } from '../shell/format';
import { jobPickerRows, saveJob } from '../session.js';

// Which of these is your job.
//
// A new starter lands here because their account has no job yet, and the
// alternative design is to hope an administrator set one for them before their
// first morning. Choosing writes the field on the account and lands them, so
// the first screen a person sees is their own work.
//
// It renders for nobody whose job is set, and the company-only row is filtered
// out by the session module for a channel account, so nothing here tells that
// account the other side of the line exists.

// Where each door is in the product as it stands today. A door with no entry in
// the shell's navigation is null here and answered below rather than pointed at
// a view that does not exist: one of the two mounts its own root over the page,
// and the other is a menu rather than a page. Sending somebody to the wrong page
// is worse than naming the right place.
export const DOOR_VIEWS = {
  today: 'Overview',
  'plan.week': 'Optimizer',
  'plan.day': 'Schedule',
  'plan.tonight': 'Break Library',
  'rules.restrictions': 'Settings',
  'rules.licence': 'Overview',
  'rules.rate_card': 'Pricing',
  'clients.all': 'Advertisers',
  'clients.campaigns': 'Campaigns',
  'clients.money': 'Advertisers',
  'sources.today': 'Data',
  'account.accounts': null,
  'model.console': null,
};

const DOOR_NOTES = {
  'account.accounts': ['Opens from the account menu', 'נפתח מתפריט החשבון'],
  'model.console': ['The model console button, at the foot of the screen', 'כפתור קונסולת המודל, בתחתית המסך'],
};

// The model console is not a view of this shell. It mounts its own root over the
// page and answers to one published address, so a steward who picks that row is
// taken there rather than told a surface is being built, which it no longer is.
// The row still names the button that reopens it, because a person who lands
// somewhere they were not shown the way to cannot get back.
const CONSOLE_HASH = 'Model';

function openDoor(row, setActiveView) {
  const view = DOOR_VIEWS[row.door];
  if (view && setActiveView) {
    setActiveView(view);
    return;
  }
  // The row exists for a company account only, and the console renders for one
  // only, so this address is inert for anybody else and says nothing to them.
  if (row.door === 'model.console' && typeof window !== 'undefined') {
    window.location.hash = CONSOLE_HASH;
  }
}

export function JobPicker({ session, locale, copy, onChosen, setActiveView, notify }) {
  const [busy, setBusy] = useState('');
  const rows = jobPickerRows(session, locale);

  async function choose(row) {
    setBusy(row.id);
    const result = await saveJob(row.id);
    setBusy('');
    if (!result.ok) {
      if (notify) notify('The job could not be saved.', 'לא ניתן היה לשמור את התפקיד.');
      return;
    }
    if (onChosen) onChosen(result.session);
    openDoor(row, setActiveView);
  }

  return (
    <section className="page-panel today-job-picker" aria-label={pageText(locale, 'Which of these is your job', 'מה מבין אלה התפקיד שלכם')}>
      <div className="panel-head">
        <h2>{pageText(locale, 'Which of these is your job', 'מה מבין אלה התפקיד שלכם')}</h2>
        <span>{pageText(locale, 'It decides where you land, never what you may change', 'זה קובע לאן תגיעו, אף פעם לא מה מותר לכם לשנות')}</span>
      </div>
      <div className="today-job-rows">
        {rows.map((row) => {
          const view = DOOR_VIEWS[row.door];
          const note = DOOR_NOTES[row.door];
          const where = view ? (copy && copy.nav ? copy.nav[view] || view : view) : pageText(locale, note ? note[0] : '', note ? note[1] : '');
          return (
            <Button className="today-job-row" type="button" key={row.id} disabled={Boolean(busy)} onClick={() => choose(row)}>
              <strong>{row.label}</strong>
              <span>{row.doorLabel}</span>
              <span className="today-job-where">{where}</span>
            </Button>
          );
        })}
      </div>
      <p className="today-note">{pageText(locale, 'You can change this later from the account menu.', 'אפשר לשנות את זה אחר כך מתפריט החשבון.')}</p>
    </section>
  );
}

export default JobPicker;
