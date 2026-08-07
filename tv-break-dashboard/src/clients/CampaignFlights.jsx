import React, { useState } from 'react';
import { Figure } from '../shell/bidi';
import { Pencil, Plus, Trash2 } from 'lucide-react';
import { pageText } from '../shell/format';
import { changedFields, goalLabel, refusalText, vocabularyLabel, windowLabel } from './clients-money-helpers';
import { addFlight, removeFlight, updateFlight } from './clients-api';
import { DeliveryBasis, DeliveryCell } from './DeliveryState';
import DemoBadge from './DemoBadge';

// The flights of one campaign, as rows that can be changed in place. A flight
// is a line of the campaign, so it is added, amended and removed on the row
// itself rather than through a wizard that re-asks for the campaign.
//
// The delivered column is the campaign's own ledger, held to this flight's
// dates. It stays a state and a figure at once, never one without the other:
// what has been counted, over how many of the flight's days, marked as a floor
// while any day of the flight has no per-spot source. The instant it was counted
// at, the file it came from and the days that carry no source are printed under
// the table in both states, so the figure and its basis are on the same screen.

const EMPTY = { name: '', starts_on: '', ends_on: '', goal_kind: 'spots', goal_value: '' };

function draftOf(flight) {
  return {
    name: flight.name || '',
    starts_on: flight.starts_on || '',
    ends_on: flight.ends_on || '',
    goal_kind: flight.goal_kind || 'spots',
    goal_value: flight.goal_value === null || flight.goal_value === undefined ? '' : String(flight.goal_value),
  };
}

function FlightFields({ draft, onChange, goalKinds, goalWords, locale }) {
  return (
    <div className="clients-flight-row">
      <label className="clients-field">
        <span>{pageText(locale, 'Flight name', 'שם טיסת השידור')}</span>
        <input type="text" value={draft.name} onChange={(event) => onChange('name', event.target.value)} />
      </label>
      <label className="clients-field">
        <span>{pageText(locale, 'From', 'מתאריך')}</span>
        <input type="date" dir="ltr" value={draft.starts_on} onChange={(event) => onChange('starts_on', event.target.value)} />
      </label>
      <label className="clients-field">
        <span>{pageText(locale, 'To', 'עד תאריך')}</span>
        <input type="date" dir="ltr" value={draft.ends_on} onChange={(event) => onChange('ends_on', event.target.value)} />
      </label>
      <label className="clients-field">
        <span>{pageText(locale, 'Goal unit', 'יחידת יעד')}</span>
        <select value={draft.goal_kind} onChange={(event) => onChange('goal_kind', event.target.value)}>
          {goalKinds.map((kind) => (
            <option key={kind} value={kind}>{vocabularyLabel(goalWords, kind, locale)}</option>
          ))}
        </select>
      </label>
      <label className="clients-field">
        <span>{pageText(locale, 'Booked goal', 'יעד שהוזמן')}</span>
        <input type="number" dir="ltr" value={draft.goal_value} onChange={(event) => onChange('goal_value', event.target.value)} />
      </label>
    </div>
  );
}

export default function CampaignFlights({
  campaign,
  locale,
  goalKinds = ['spots'],
  goalWords = [],
  delivery,
  airStates = [],
  canEdit = false,
  onChanged,
  notify,
}) {
  const [editing, setEditing] = useState('');
  const [pendingRemove, setPendingRemove] = useState('');
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState({ ...EMPTY });
  const [error, setError] = useState('');

  function change(key, value) {
    setDraft((current) => ({ ...current, [key]: value }));
  }

  function fail(problem) {
    setError(refusalText(problem, locale));
  }

  async function save(flight) {
    const changes = changedFields(flight, draft);
    if (!Object.keys(changes).length) {
      setEditing('');
      return;
    }
    if (changes.goal_value !== undefined) {
      changes.goal_value = Number(changes.goal_value);
    }
    try {
      await updateFlight(campaign.campaign_id, flight.flight_id, changes);
      setEditing('');
      setError('');
      notify('Flight saved.', 'טיסת השידור נשמרה.');
      onChanged();
    } catch (problem) {
      fail(problem);
    }
  }

  async function add() {
    try {
      await addFlight(campaign.campaign_id, { ...draft, goal_value: Number(draft.goal_value) });
      setAdding(false);
      setDraft({ ...EMPTY });
      setError('');
      notify('Flight added.', 'טיסת שידור נוספה.');
      onChanged();
    } catch (problem) {
      fail(problem);
    }
  }

  async function remove(flightId) {
    try {
      await removeFlight(campaign.campaign_id, flightId);
      setPendingRemove('');
      setError('');
      notify('Flight removed.', 'טיסת השידור הוסרה.');
      onChanged();
    } catch (problem) {
      fail(problem);
    }
  }

  return (
    <div className="clients-flight-board">
      {campaign.flights.length ? (
        <table className="clients-table clients-flight-table">
          <thead>
            <tr>
              <th scope="col">{pageText(locale, 'Flight', 'טיסת שידור')}</th>
              <th scope="col">{pageText(locale, 'Window', 'חלון')}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Booked', 'הוזמן')}</th>
              <th scope="col">{pageText(locale, 'Delivered', 'סופק')}</th>
              <th scope="col">{pageText(locale, 'What to do', 'מה לעשות')}</th>
            </tr>
          </thead>
          <tbody>
            {campaign.flights.map((flight) => (
              editing === flight.flight_id ? (
                <tr key={flight.flight_id} className="clients-editing-row">
                  <td colSpan={5}>
                    <FlightFields draft={draft} onChange={change} goalKinds={goalKinds} goalWords={goalWords} locale={locale} />
                    <div className="clients-form-actions">
                      <button type="button" className="clients-primary" onClick={() => save(flight)}>
                        {pageText(locale, 'Save the flight', 'שמרו את טיסת השידור')}
                      </button>
                      <button type="button" className="clients-secondary" onClick={() => { setEditing(''); setError(''); }}>
                        {pageText(locale, 'Cancel', 'ביטול')}
                      </button>
                    </div>
                  </td>
                </tr>
              ) : (
                <tr key={flight.flight_id}>
                  <td>
                    <span className="clients-cell-name">
                      {flight.name ? <strong>{flight.name}</strong> : null}
                      <DemoBadge demo={flight.demo} locale={locale} />
                      <small className="clients-flight-id">{flight.flight_id}</small>
                    </span>
                  </td>
                  <td className="numeric"><Figure>{windowLabel(flight.starts_on, flight.ends_on, locale)}</Figure></td>
                  <td className="numeric"><Figure>{goalLabel(flight, locale, goalWords)}</Figure></td>
                  <td>
                    <DeliveryCell
                      delivery={delivery}
                      window={{ starts_on: flight.starts_on, ends_on: flight.ends_on }}
                      vocabulary={airStates}
                      locale={locale}
                    />
                  </td>
                  <td>
                    {canEdit && pendingRemove !== flight.flight_id ? (
                      <span className="clients-row-actions">
                        <button
                          type="button"
                          className="clients-inline-action"
                          onClick={() => {
                            setDraft(draftOf(flight));
                            setEditing(flight.flight_id);
                            setAdding(false);
                            setError('');
                          }}
                        >
                          <Pencil size={12} aria-hidden="true" />
                          {pageText(locale, 'Edit', 'עריכה')}
                        </button>
                        <button type="button" className="clients-inline-action" onClick={() => setPendingRemove(flight.flight_id)}>
                          <Trash2 size={12} aria-hidden="true" />
                          {pageText(locale, 'Remove', 'הסרה')}
                        </button>
                      </span>
                    ) : null}
                    {pendingRemove === flight.flight_id ? (
                      <span className="clients-confirm">
                        <small>{pageText(locale, 'The flight is deleted. The campaign stays.', 'טיסת השידור נמחקת. הקמפיין נשאר.')}</small>
                        <button type="button" className="clients-inline-action" onClick={() => remove(flight.flight_id)}>
                          {pageText(locale, 'Confirm', 'אישור')}
                        </button>
                        <button type="button" className="clients-inline-action" onClick={() => setPendingRemove('')}>
                          {pageText(locale, 'Keep it', 'השאירו')}
                        </button>
                      </span>
                    ) : null}
                  </td>
                </tr>
              )
            ))}
          </tbody>
        </table>
      ) : (
        <p className="clients-reason">
          {pageText(locale, 'No flight on this campaign yet.', 'אין טיסת שידור בקמפיין הזה עדיין.')}
        </p>
      )}

      {adding ? (
        <div className="clients-add-flight">
          <FlightFields draft={draft} onChange={change} goalKinds={goalKinds} goalWords={goalWords} locale={locale} />
          <div className="clients-form-actions">
            <button type="button" className="clients-primary" onClick={add}>
              {pageText(locale, 'Add the flight', 'הוסיפו את טיסת השידור')}
            </button>
            <button type="button" className="clients-secondary" onClick={() => { setAdding(false); setError(''); }}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </button>
          </div>
        </div>
      ) : null}

      {canEdit && !adding && !editing ? (
        <button
          type="button"
          className="clients-secondary"
          onClick={() => { setDraft({ ...EMPTY, starts_on: campaign.starts_on, ends_on: campaign.ends_on }); setAdding(true); }}
        >
          <Plus size={13} aria-hidden="true" />
          {pageText(locale, 'Add a flight', 'הוסיפו טיסת שידור')}
        </button>
      ) : null}

      {error ? <p className="clients-error" role="alert">{error}</p> : null}
      <DeliveryBasis delivery={delivery} locale={locale} />
    </div>
  );
}
