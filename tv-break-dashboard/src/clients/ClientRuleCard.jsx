import React, { useState } from 'react';
import { Plus, SquareArrowOutUpRight } from 'lucide-react';
import { pageText } from '../shell/format';
import {
  nextRuleId,
  parsePremium,
  premiumText,
  ruleRowFor,
  spellingRefusal,
  spellingsFor,
} from './clients-rule-helpers';

// The client's pricing rule, on the client's own record.
//
// This section used to be two empty properties whose controls sent the reader to
// a grid of forty five rows keyed ADV_01 to ADV_45, not one of which carries a
// client name, so the control that named the client's own card landed on
// strangers. Both jobs now happen here: the rule is created on the record, the
// spellings are added on the record, and the only navigation offered is to a
// card that exists and carries this client's name.
//
// What a rule is, stated because the reader is entitled to it. A row in
// `data/advertiser_rules.csv` prices an advertiser only when its name cell
// carries that advertiser's name. The forty five shipped rows carry none, so
// none of them has ever priced a spot, and the honest reading of an unbound
// client is not a blank: it is the rate card, premium 1.00.

function Spellings({ entries, locale }) {
  return (
    <ul className="clients-spellings">
      {entries.map((entry) => (
        <li key={entry.text}>
          <span dir="auto">{entry.text}</span>
          <span className={`clients-spelling-source ${entry.source}`}>
            {entry.source === 'observed'
              ? pageText(locale, 'seen in the data', 'נצפה בנתונים')
              : pageText(locale, 'typed on the rule', 'הוקלד על הכלל')}
          </span>
        </li>
      ))}
    </ul>
  );
}

// The create form. The premium is the one number the operator supplies, so it is
// asked for plainly and echoed as the percent it moves off the rate card.
function CreateRule({ client, rows, locale, busy, onCancel, onCreate }) {
  const [premium, setPremium] = useState('1.00');
  const parsed = parsePremium(premium);
  const shown = premiumText(parsed);
  const ruleId = nextRuleId(rows);

  return (
    <form
      className="clients-rule-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (parsed !== null && !busy) {
          onCreate({ advertiser_id: ruleId, name: client.advertiser, default_premium: parsed });
        }
      }}
    >
      <label htmlFor="clients-rule-premium">
        {pageText(locale, 'Premium, times the rate card', 'מקדם, כפול המחירון')}
      </label>
      <div className="clients-rule-premium">
        <input
          id="clients-rule-premium"
          type="number"
          min="0"
          step="0.05"
          dir="ltr"
          value={premium}
          onChange={(event) => setPremium(event.target.value)}
        />
        <span className="numeric" dir="ltr">{shown.delta || pageText(locale, 'rate card', 'מחיר מחירון')}</span>
      </div>
      <p className="clients-reason">
        {pageText(
          locale,
          `The rule is stored as ${ruleId} and named for this client. From then on it prices this client's spots on the daily pricing path, and the weekly plan is untouched because it does not read advertiser rules.`,
          `הכלל נשמר כ־⁦${ruleId}⁩ ועל שם הלקוח הזה. מרגע זה הוא מתמחר את התשדירים של הלקוח במסלול התמחור היומי, והתוכנית השבועית אינה משתנה משום שהיא אינה קוראת כללי מפרסם.`,
        )}
      </p>
      <div className="clients-rule-actions">
        <button type="submit" className="clients-primary compact" disabled={parsed === null || busy}>
          {busy
            ? pageText(locale, 'Creating...', 'יוצר...')
            : pageText(locale, 'Create the rule', 'יצירת הכלל')}
        </button>
        <button type="button" className="clients-inline-action" onClick={onCancel}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </button>
      </div>
    </form>
  );
}

// The add-a-spelling form. It refuses before it writes, with the reason, because
// the store refuses a spelling another row already holds and a reader should not
// meet that as an error after the fact.
function AddSpelling({ client, row, rows, locale, busy, onCancel, onAdd }) {
  const [text, setText] = useState('');
  const refusal = spellingRefusal(text, client, row, rows, locale);

  return (
    <form
      className="clients-rule-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!refusal && !busy) {
          onAdd(row, text.trim());
          setText('');
        }
      }}
    >
      <label htmlFor="clients-rule-spelling">
        {pageText(locale, 'Another spelling of this client', 'כתיב נוסף של הלקוח הזה')}
      </label>
      <input
        id="clients-rule-spelling"
        type="text"
        dir="auto"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      <p className="clients-reason">
        {pageText(
          locale,
          'A spelling listed on the rule is priced by the same rule, so a daily file that spells this client differently is still priced as this client.',
          'כתיב שרשום על הכלל מתומחר על ידי אותו כלל, כך שקובץ יומי שמאיית את הלקוח אחרת עדיין מתומחר כלקוח הזה.',
        )}
      </p>
      {text.trim() && refusal ? <p className="clients-reason">{refusal}</p> : null}
      <div className="clients-rule-actions">
        <button type="submit" className="clients-primary compact" disabled={Boolean(refusal) || busy}>
          {busy
            ? pageText(locale, 'Saving...', 'שומר...')
            : pageText(locale, 'Add the spelling', 'הוספת הכתיב')}
        </button>
        <button type="button" className="clients-inline-action" onClick={onCancel}>
          {pageText(locale, 'Cancel', 'ביטול')}
        </button>
      </div>
    </form>
  );
}

export default function ClientRuleCard({
  client,
  rows = null,
  locale = 'he',
  canEdit = true,
  busy = false,
  refusal = '',
  onCreateRule = () => {},
  onAddSpelling = () => {},
  onOpenRuleCard = () => {},
}) {
  const [open, setOpen] = useState('');
  const loaded = Array.isArray(rows);
  const row = loaded ? ruleRowFor(client, rows) : null;
  const spellings = spellingsFor(client, row);
  // The premium is the engine's own effective figure for this client, which is
  // what actually prices its spots, not the row's baseline.
  const premium = premiumText(client.effective_premium);
  const scoped = row && Array.isArray(row.conditions) ? row.conditions.length : 0;
  // Four states, and never one wearing another's copy. The last is the one that
  // must not be silent: the client read says a rule prices this client and the
  // pricing read does not hold that row, so the two disagree and say so.
  let state = 'reading';
  if (loaded && row) {
    state = 'bound';
  } else if (loaded && client.bound_to_rules_row) {
    state = 'disagreement';
  } else if (loaded) {
    state = 'unbound';
  }

  return (
    <section className="clients-rule-card">
      <h4>{pageText(locale, 'Pricing rule', 'כלל תמחור')}</h4>

      {state === 'reading' ? (
        <p className="clients-reason">
          {pageText(locale, 'Reading the pricing store...', 'קורא את מאגר התמחור...')}
        </p>
      ) : null}

      {state === 'disagreement' ? (
        <p className="clients-reason">
          {pageText(
            locale,
            'This client reads as priced by a rule, and no row in the pricing store carries its name. The two reads disagree, so neither figure is shown. Refresh, and if it persists the pricing store and the name store are out of step.',
            'הלקוח הזה נקרא כמתומחר לפי כלל, ואף שורה במאגר התמחור אינה נושאת את שמו. שתי הקריאות סותרות, ולכן אף סכום אינו מוצג. רעננו, ואם הסתירה נשארת מאגר התמחור ומאגר השמות אינם מסונכרנים.',
          )}
        </p>
      ) : null}

      {state === 'bound' ? (
        <>
          <p className="clients-rule-line">
            <span className="numeric" dir="ltr">{premium.multiplier}</span>
            {premium.delta ? <span className="numeric clients-rule-delta" dir="ltr">{premium.delta}</span> : null}
            <span className="clients-rule-id" dir="ltr">{row.advertiser_id}</span>
          </p>
          <p className="clients-reason">
            {scoped
              ? pageText(
                locale,
                `This rule prices this client's spots on the daily pricing path, and it carries ${scoped} scoped rules of its own.`,
                `הכלל הזה מתמחר את התשדירים של הלקוח במסלול התמחור היומי, ויש בו ⁦${scoped}⁩ כללים ממוקדים משלו.`,
              )
              : pageText(
                locale,
                "This rule prices this client's spots on the daily pricing path. It carries no scoped rules yet.",
                'הכלל הזה מתמחר את התשדירים של הלקוח במסלול התמחור היומי. אין בו כללים ממוקדים עדיין.',
              )}
          </p>
          <button type="button" className="clients-inline-action" onClick={() => onOpenRuleCard(row.advertiser_id)}>
            <SquareArrowOutUpRight size={12} aria-hidden="true" />
            {pageText(locale, 'Open the full rule card', 'פתחו את כרטיס הכלל המלא')}
          </button>
        </>
      ) : null}

      {state === 'unbound' ? (
        <>
          <p className="clients-rule-line">
            <span className="numeric" dir="ltr">{premium.multiplier}</span>
            <span className="clients-rule-plain">{pageText(locale, 'rate card', 'מחיר מחירון')}</span>
          </p>
          <p className="clients-reason">
            {pageText(
              locale,
              "No stored rule carries this client's name, so nothing prices its spots above the rate card.",
              'אף כלל שמור אינו נושא את שם הלקוח הזה, ולכן דבר אינו מתמחר את התשדירים שלו מעל המחירון.',
            )}
          </p>
          {canEdit && open !== 'create' ? (
            <button type="button" className="clients-inline-action" onClick={() => setOpen('create')}>
              <Plus size={12} aria-hidden="true" />
              {pageText(locale, 'Create the pricing rule for this client', 'צרו כלל תמחור ללקוח הזה')}
            </button>
          ) : null}
          {!canEdit ? <p className="clients-refusal">{refusal}</p> : null}
          {canEdit && open === 'create' ? (
            <CreateRule
              client={client}
              rows={rows}
              locale={locale}
              busy={busy}
              onCancel={() => setOpen('')}
              onCreate={(draft) => {
                setOpen('');
                onCreateRule(draft);
              }}
            />
          ) : null}
        </>
      ) : null}

      <h5>{pageText(locale, 'Known as', 'ידוע גם כ')}</h5>
      {spellings.length ? <Spellings entries={spellings} locale={locale} /> : null}
      {!spellings.length ? (
        <p className="clients-reason">
          {pageText(
            locale,
            'Only one spelling of this client has ever been seen. The observed spellings are written from the daily files themselves, so a second one appears here when a file carries it.',
            'נצפה רק כתיב אחד של הלקוח הזה. הכתיבים הנצפים נכתבים מתוך הקבצים היומיים עצמם, ולכן כתיב נוסף מופיע כאן כאשר קובץ נושא אותו.',
          )}
        </p>
      ) : null}

      {state === 'bound' && canEdit && open !== 'spelling' ? (
        <button type="button" className="clients-inline-action" onClick={() => setOpen('spelling')}>
          <Plus size={12} aria-hidden="true" />
          {pageText(locale, 'Add a spelling on the rule', 'הוסיפו כתיב על הכלל')}
        </button>
      ) : null}
      {state === 'bound' && canEdit && open === 'spelling' ? (
        <AddSpelling
          client={client}
          row={row}
          rows={rows}
          locale={locale}
          busy={busy}
          onCancel={() => setOpen('')}
          onAdd={(target, text) => {
            setOpen('');
            onAddSpelling(target, text);
          }}
        />
      ) : null}
      {state === 'unbound' ? (
        <p className="clients-reason">
          {pageText(
            locale,
            'A spelling the operator types is stored on the rule that prices it, so this client needs its pricing rule first.',
            'כתיב שהמפעיל מקליד נשמר על הכלל שמתמחר אותו, ולכן ללקוח הזה דרוש קודם כלל תמחור.',
          )}
        </p>
      ) : null}
    </section>
  );
}
