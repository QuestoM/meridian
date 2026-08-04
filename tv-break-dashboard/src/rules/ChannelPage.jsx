import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { Lock, Tv } from 'lucide-react';
import { pageText } from '../shell/format';
import { payloadCanEdit, WALLS } from '../session.js';
import {
  fetchActivation,
  fetchOperatorChannel,
  refusalSentence,
  setActivation,
  setOperatorChannel,
} from './rules-lib';

// Two switches that decide what every other number in the product means, and
// until now neither had a permission on the surface that threw it.
//
// The channel declaration has to list every channel the loaded schedule carries,
// because before the declaration nobody can know which one to hide. No figure
// travels with the list, so nothing about a rival is disclosed by it. What was
// missing is on the write: setting this to somebody else's channel does not leak
// their data, it inverts the boundary, and the product would then hide the
// operator's own channel and treat a competitor's as owned.
//
// The audience model switch decides where every forward-dated rating comes from.
// It is a configuration act, not a training one, which is why it lives here, and
// it is company staff only, which is why a channel account reads it as state.

export default function ChannelPage({ locale, session, notify, onGlobalRefresh }) {
  const [channel, setChannel] = useState(null);
  const [activation, setActivationState] = useState(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [confirming, setConfirming] = useState('');

  const load = useCallback(() => {
    fetchOperatorChannel().then(setChannel).catch((problem) => setError(problem.message));
    fetchActivation().then(setActivationState).catch(() => setActivationState(null));
  }, []);

  useEffect(() => { load(); }, [load]);

  async function declareChannel(next) {
    setPending(true);
    setConfirming('');
    try {
      const body = await setOperatorChannel(next);
      setChannel(body);
      notify?.(
        'Your channel is set. Every scoped figure now reads for it.',
        'הערוץ שלכם נקבע. כל נתון מכאן והלאה מחושב עבורו.',
      );
      onGlobalRefresh?.();
    } catch (problem) {
      notify?.(`Setting the channel failed (${problem.message}).`, `קביעת הערוץ נכשלה (${problem.message}).`);
    } finally {
      setPending(false);
    }
  }

  async function throwSwitch(next) {
    setPending(true);
    try {
      const body = await setActivation(next);
      setActivationState(body);
      notify?.(
        'The audience model switch moved. The saved plan is now out of date and needs a run.',
        'מתג מודל הקהל שונה. התוכנית השמורה אינה עדכנית ודורשת הרצה.',
      );
      onGlobalRefresh?.();
    } catch (problem) {
      notify?.(`The switch was refused (${problem.message}).`, `המתג נדחה (${problem.message}).`);
    } finally {
      setPending(false);
    }
  }

  const channelGate = payloadCanEdit(channel || {}, session, { adminOnly: true, detail: '' });
  const activationGate = payloadCanEdit(activation || {}, session, WALLS.audienceActivation);
  const options = channel?.available_channels || [];

  return (
    <div className="rules-section">
      <section className="rules-card">
        <div className="rules-card-head">
          <div>
            <h2>{pageText(locale, 'Your channel', 'הערוץ שלכם')}</h2>
            <p className="rules-card-lead">
              {pageText(
                locale,
                'The channel this operator owns. Every restriction, every price and every figure in the product is scoped to it.',
                'הערוץ שבבעלות המפעיל. כל הגבלה, כל מחיר וכל נתון במוצר מחושבים עבורו.',
              )}
            </p>
          </div>
          <Tv size={18} aria-hidden="true" />
        </div>

        {error && <p className="rules-inline-error" role="status">{error}</p>}

        <div className="rules-channel-options">
          {options.map((option) => {
            const owned = channel?.operator_channel === option;
            return (
              <button
                key={option}
                type="button"
                className={`rules-channel-option${owned ? ' owned' : ''}`}
                disabled={!channelGate.canEdit || pending || owned}
                onClick={() => setConfirming(option)}
              >
                <span dir="auto">{option}</span>
                {owned && <small>{pageText(locale, 'Yours', 'שלכם')}</small>}
              </button>
            );
          })}
        </div>

        {confirming && (
          <div className="rules-confirm-block" role="alertdialog">
            <p>
              {pageText(
                locale,
                `Declare ${confirming} as the channel this operator owns? Every scoped figure in the product changes to it.`,
                `להצהיר ש-${confirming} הוא הערוץ שבבעלות המפעיל? כל נתון מכאן והלאה יחושב עבורו.`,
              )}
            </p>
            <Button className="run-button" type="button" variant="contained" onClick={() => declareChannel(confirming)}>
              {pageText(locale, 'Declare it', 'הצהרה')}
            </Button>
            <Button className="secondary-button" type="button" variant="outlined" onClick={() => setConfirming('')}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </div>
        )}

        {!channelGate.canEdit && (
          <p className="rules-locked">
            <Lock size={13} aria-hidden="true" />
            {/* The server's refusal is authored in Hebrew, and it used to reach
                an English reader verbatim. The translation is keyed off the
                wall's own words, so a wall this page does not know still
                renders the server's sentence rather than a guess. */}
            <span dir="auto">{refusalSentence(channel?.can_edit_reason, locale) || pageText(locale, 'Only an administrator changes the channel.', 'רק מנהל המערכת משנה את הערוץ.')}</span>
          </p>
        )}
        {channel && !channel.is_declared && (
          <p className="rules-inline-error" role="status">
            {pageText(
              locale,
              'No channel is declared, so nothing is scoped and figures cover the whole loaded schedule.',
              'לא הוצהר ערוץ, ולכן שום נתון אינו מסונן והמספרים מכסים את כל הלוח שנטען.',
            )}
          </p>
        )}
      </section>

      {activation && (
        <section className="rules-card">
          <div className="rules-card-head">
            <div>
              <h2>{pageText(locale, 'The audience model', 'מודל הקהל')}</h2>
              <p className="rules-card-lead">
                {pageText(
                  locale,
                  'Where a rating for a future date comes from. This is a setting, not a training act, so it lives here.',
                  'מהיכן מגיע רייטינג לתאריך עתידי. זו הגדרה ולא פעולת אימון, ולכן היא נמצאת כאן.',
                )}
              </p>
            </div>
          </div>
          <p className="rules-activation-state">
            <span className={`rules-state-chip ${activation.state}`}>
              {activation.state === 'on'
                ? pageText(locale, 'On', 'פעיל')
                : activation.state === 'on_no_artifact'
                  ? pageText(locale, 'On, with nothing trained yet', 'פעיל, אך אין מודל מאומן')
                  : pageText(locale, 'Off', 'כבוי')}
            </span>
            <span>
              {activation.state === 'off'
                ? pageText(
                  locale,
                  'Forward-dated ratings are the historical baseline.',
                  'רייטינג לתאריכים עתידיים מגיע מהבסיס ההיסטורי.',
                )
                : activation.state === 'on_no_artifact'
                  ? pageText(
                    locale,
                    'The switch is on and nothing is trained, so the numbers are still historical.',
                    'המתג פעיל ואין מודל מאומן, ולכן המספרים עדיין היסטוריים.',
                  )
                  : pageText(
                    locale,
                    `Forward-dated ratings come from the model version of ${String(activation.computed_at || '').slice(0, 10)}.`,
                    `רייטינג לתאריכים עתידיים מגיע מגרסת המודל מ-${String(activation.computed_at || '').slice(0, 10)}.`,
                  )}
            </span>
          </p>
          <p className="rules-consequence">{locale === 'he' ? activation.consequence_he : activation.consequence_en}</p>
          {activationGate.canEdit ? (
            <Button
              className="secondary-button"
              type="button"
              variant="outlined"
              disabled={pending}
              onClick={() => throwSwitch(!activation.active)}
            >
              {activation.active
                ? pageText(locale, 'Turn it off', 'כיבוי')
                : pageText(locale, 'Turn it on', 'הפעלה')}
            </Button>
          ) : (
            <p className="rules-locked">
              <Lock size={13} aria-hidden="true" />
              <span dir="auto">{refusalSentence(activationGate.reason, locale)}</span>
            </p>
          )}
        </section>
      )}
    </div>
  );
}
