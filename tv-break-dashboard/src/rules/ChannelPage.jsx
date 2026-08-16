import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '../studio/actions';
import { Lock, Tv } from 'lucide-react';
import { pageText } from '../shell/format';
import { Name } from '../shell/bidi';
import ConsequenceDialog, { focusAfterDialogClose } from '../safety/ConsequenceDialog';
import { payloadCanEdit, WALLS } from '../session.js';
import { formatDay } from '../shell/dates';
import { Pressable } from '../studio/dom-controls';
import {
  detailWords,
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
  const channelHeadingRef = React.useRef(null);

  const load = useCallback(() => {
    fetchOperatorChannel().then(setChannel).catch((problem) => setError(detailWords(problem, locale)));
    fetchActivation().then(setActivationState).catch(() => setActivationState(null));
  }, []);

  useEffect(() => { load(); }, [load]);

  async function declareChannel(next) {
    setPending(true);
    try {
      const body = await setOperatorChannel(next);
      setChannel(body);
      notify?.(
        'Your channel is set. Every scoped figure now reads for it.',
        'הערוץ שלכם נקבע. כל נתון מכאן והלאה מחושב עבורו.',
      );
      onGlobalRefresh?.();
      return true;
    } catch (problem) {
      notify?.(`Setting the channel failed (${detailWords(problem, 'en')}).`, `קביעת הערוץ נכשלה (${detailWords(problem, 'he')}).`);
      return false;
    } finally {
      setPending(false);
    }
  }

  async function confirmChannelDeclaration() {
    if (!confirming) return;
    const changed = await declareChannel(confirming);
    if (changed) {
      setConfirming('');
      focusAfterDialogClose(channelHeadingRef);
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
      notify?.(`The switch was refused (${detailWords(problem, 'en')}).`, `המתג נדחה (${detailWords(problem, 'he')}).`);
    } finally {
      setPending(false);
    }
  }

  const channelGate = payloadCanEdit(channel || {}, session, { adminOnly: true, detail: '' });
  const activationGate = payloadCanEdit(activation || {}, session, WALLS.audienceActivation);
  const options = channel?.available_channels || [];

  return (
    <div className="rules-section">
      <section className="card rules-card">
        <div className="rules-card-head">
          <div>
            <h2 ref={channelHeadingRef} tabIndex={-1}>{pageText(locale, 'Your channel', 'הערוץ שלכם')}</h2>
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
              <Pressable
                key={option}
                type="button"
                className={`rules-channel-option${owned ? ' owned' : ''}`}
                disabled={!channelGate.canEdit || pending || owned}
                onClick={() => setConfirming(option)}
              >
                <Name>{option}</Name>
                {owned && <small>{pageText(locale, 'Yours', 'שלכם')}</small>}
              </Pressable>
            );
          })}
        </div>

        <ConsequenceDialog
          open={Boolean(confirming)}
          locale={locale}
          title={pageText(locale, 'Change the owned-channel boundary?', 'לשנות את גבול הערוץ שבבעלותכם?')}
          description={pageText(locale, 'This declaration changes the data boundary used throughout the product.', 'ההצהרה הזו משנה את גבול הנתונים שבו המוצר כולו משתמש.')}
          object={confirming ? (
            <span className="consequence-review__object">
              {pageText(locale, 'Owned channel: ', 'ערוץ בבעלות: ')}
              {channel?.operator_channel ? <Name>{channel.operator_channel}</Name> : pageText(locale, 'not declared', 'לא הוצהר')}
              {' → '}<Name>{confirming}</Name>
            </span>
          ) : ''}
          scope={pageText(locale, 'Every operator-scoped restriction, inventory figure, forecast, pricing readout and model view across this product. No source rows are deleted.', 'כל הגבלה, נתון מלאי, תחזית, תצוגת תמחור ותצוגת מודל שמסוננים לערוץ המפעיל במוצר. שורות מקור אינן נמחקות.')}
          consequence={pageText(locale, 'The current channel stops being treated as owned and the selected channel becomes the owned boundary immediately. The saved plan becomes out of date and needs a new run.', 'הערוץ הנוכחי יפסיק להיחשב בבעלות והערוץ שנבחר יהפוך מיד לגבול שבבעלותכם. התוכנית השמורה תהפוך ללא עדכנית ותדרוש הרצה חדשה.')}
          recovery={pageText(locale, 'A pre-change settings snapshot is kept on the Restore changes page.', 'תמונת מצב של ההגדרות מלפני השינוי נשמרת בעמוד שחזור שינויים.')}
          confirmLabel={pageText(locale, 'Change owned channel', 'שינוי הערוץ שבבעלות')}
          workingLabel={pageText(locale, 'Changing owned channel', 'משנה את הערוץ שבבעלות')}
          busy={pending}
          onCancel={() => setConfirming('')}
          onConfirm={confirmChannelDeclaration}
        />

        {!channelGate.canEdit && (
          <p className="rules-locked">
            <Lock size={13} aria-hidden="true" />
            {/* The server's refusal is authored in Hebrew, and it used to reach
                an English reader verbatim. The translation is keyed off the
                wall's own words, so a wall this page does not know still
                renders the server's sentence rather than a guess. */}
            <span>{refusalSentence(channel?.can_edit_reason, locale) || pageText(locale, 'Only an administrator changes the channel.', 'רק מנהל המערכת משנה את הערוץ.')}</span>
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
        <section className="card rules-card">
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
                    `Forward-dated ratings come from the model version of ${formatDay(String(activation.computed_at || '').slice(0, 10))}.`,
                    `רייטינג לתאריכים עתידיים מגיע מגרסת המודל מ-${formatDay(String(activation.computed_at || '').slice(0, 10))}.`,
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
              <span>{refusalSentence(activationGate.reason, locale)}</span>
            </p>
          )}
        </section>
      )}
    </div>
  );
}
