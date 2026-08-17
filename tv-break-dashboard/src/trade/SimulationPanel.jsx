import React, { useState } from 'react';
import { Card, CardBody, EmptyState, ErrorState, LoadingState, Metric, Status } from '../studio';
import { Button } from '../studio/actions';
import { InputControl } from '../studio/dom-controls';
import { Ban, Coins, Play, TriangleAlert } from 'lucide-react';
import { Code, Figure, Name, Prose } from '../shell/bidi';
import { EMPTY_VALUE, formatNumber, formatPercent, pageText } from '../shell/format';
import { termName } from './trade-terms';
import { contractMoney } from './term-language';
import {
  blockFieldKind, blockFieldLabel, BLOCK_FIELD_ORDER, moneyLineLabel, scopeResolution,
} from './trade-vocabulary';

// What this agreement WOULD do to real activity. It writes nothing.
//
// THREE THINGS THIS PANEL WILL NOT DO, each inherited from the engine and each
// the reason a commercial director can trust the figure beside it.
//
// It does not price a placement constraint. A competitive separation or a
// programme exclusion is COUNTED as placements affected, never converted into
// revenue it might have produced, because the alternative schedule was never
// run and attributing money to it would be invention.
//
// It does not hide what it could not simulate. Every term the compiler skipped,
// every guarantee in an audience the product cannot measure, every discount whose
// basis the document never stated — each appears BY NAME with the engine's own
// reason. A simulation that quietly drops half the agreement is worse than no
// simulation at all.
//
// It does not round a money figure into scale. A ladder discount of ₪3,847,200 is
// read against a signed page, and "₪3.8M" is not the amount.

function BlockDetail({ block, locale }) {
  if (!block || typeof block !== 'object') return null;
  if (block.available === false) {
    return (
      <p className="trd-sim-refused" role="note">
        <TriangleAlert size={14} aria-hidden="true" />
        <Prose as="span">{block.reason_he || pageText(locale, 'The engine refused this line.', 'המנוע סירב לשורה הזו.')}</Prose>
      </p>
    );
  }
  const fields = BLOCK_FIELD_ORDER.filter((key) => (
    block[key] !== undefined && block[key] !== null
  ));
  return (
    <>
      <dl className="trd-sim-block">
        {fields.map((key) => {
          const kind = blockFieldKind(key);
          const raw = block[key];
          return (
            <React.Fragment key={key}>
              <dt>{blockFieldLabel(key, locale)}</dt>
              <dd>
                <Figure>
                  {kind === 'money' ? contractMoney(raw, locale) : null}
                  {kind === 'percent' ? formatPercent(raw, locale) : null}
                  {kind === 'plain' ? formatNumber(raw, locale) : null}
                </Figure>
              </dd>
            </React.Fragment>
          );
        })}
      </dl>
      {block.basis_he ? <Prose className="trd-sim-basis">{block.basis_he}</Prose> : null}
      {block.merged_from && block.merged_from.length > 1 ? (
        <p className="trd-field-hint">
          {pageText(
            locale,
            'This figure combines several documents; the tiers were merged by threshold so nothing an amendment left alone was lost:',
            'הנתון הזה מאחד כמה מסמכים; המדרגות אוחדו לפי רף כך שדבר שהתיקון לא נגע בו לא אבד:',
          )}
          {' '}
          {block.merged_from.map((id) => <Code key={id} className="trd-id-chip">{id}</Code>)}
        </p>
      ) : null}
      {block.next_tier ? (
        <p className="trd-field-hint">
          {pageText(locale, 'The next tier begins at', 'המדרגה הבאה מתחילה ב')}
          {' '}
          <Figure>{contractMoney(block.next_tier.threshold, locale)}</Figure>
          {' · '}
          <Figure>{formatPercent(block.next_tier.discount_percent, locale)}</Figure>
        </p>
      ) : null}
    </>
  );
}

function MoneyLine({ line, value, locale, measured = true }) {
  if (value === null || value === undefined) return null;
  if (line.kind === 'block') {
    return (
      <Card className="trd-sim-line" dense>
        <CardBody>
          <h5>{moneyLineLabel(line.key, locale)}</h5>
          <BlockDetail block={value} locale={locale} />
        </CardBody>
      </Card>
    );
  }
  // With nothing in scope to measure, a printed ₪0 would be a result the engine
  // never produced. The tile keeps its label and says it has no value.
  if (!measured) {
    return (
      <Metric
        label={moneyLineLabel(line.key, locale)}
        value={EMPTY_VALUE}
        sub={pageText(locale, 'nothing in scope to measure', 'אין פעילות בהיקף למדידה')}
      />
    );
  }
  return (
    <Metric
      label={moneyLineLabel(line.key, locale)}
      value={contractMoney(value, locale)}
      tone={line.lead ? undefined : 'quiet'}
    />
  );
}

export default function SimulationPanel({ payload, error, locale, busy, onRun, canRun }) {
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');

  // The agreement's scope resolved to no campaign on file, so every money figure
  // is empty for want of anything to measure. Read from the scope rather than
  // inferred from a zero, because a real zero and an empty scope are different
  // facts and only one of them is a result.
  const scopeCampaigns = payload && payload.scope ? payload.scope.campaigns : null;
  const noActivity = Boolean(
    payload && payload.available && Array.isArray(scopeCampaigns) && scopeCampaigns.length === 0,
  );

  const controls = (
    <div className="trd-sim-controls">
      <label className="trd-field trd-field-inline">
        <span className="trd-field-label">{pageText(locale, 'From', 'מיום')}</span>
        <InputControl type="date" value={from} onChange={(event) => setFrom(event.target.value)} />
      </label>
      <label className="trd-field trd-field-inline">
        <span className="trd-field-label">{pageText(locale, 'Until', 'עד יום')}</span>
        <InputControl type="date" value={to} onChange={(event) => setTo(event.target.value)} />
      </label>
      <Button
        type="button"
        disabled={busy || !canRun}
        onClick={() => onRun(from || to ? { from: from || null, to: to || null } : null)}
      >
        <Play size={14} aria-hidden="true" />
        {busy
          ? pageText(locale, 'Simulating', 'מריץ סימולציה')
          : pageText(locale, 'Simulate against real activity', 'הרצת סימולציה מול פעילות אמיתית')}
      </Button>
    </div>
  );

  return (
    <section className="trd-sim" aria-label={pageText(locale, 'Simulation', 'סימולציה')}>
      <div className="trd-pane-head">
        <h4>
          <Coins size={16} aria-hidden="true" />
          {pageText(locale, 'What this agreement would do', 'מה ההסכם הזה היה עושה')}
        </h4>
      </div>
      <p className="trd-field-hint">
        {pageText(
          locale,
          'The proposal is compiled in memory and applied to activity already on file. No live store is touched and nothing is written.',
          'ההצעה מקומפלת בזיכרון ומוחלת על פעילות שכבר קיימת בקבצים. אין נגיעה במאגר פעיל ודבר אינו נכתב.',
        )}
      </p>
      {controls}

      {error ? (
        <ErrorState
          title={pageText(locale, 'The simulation could not be run', 'לא ניתן היה להריץ את הסימולציה')}
          description={error}
        />
      ) : null}

      {busy && !payload ? (
        <LoadingState
          title={pageText(locale, 'Applying the agreement to real activity', 'מחיל את ההסכם על פעילות אמיתית')}
          description={pageText(
            locale,
            'The delivery ledger and the campaign book are read for the window, then the compiled terms are applied to them.',
            'ספר האספקה וספר הקמפיינים נקראים לחלון הזמן, ואז מוחלים עליהם המונחים המקומפלים.',
          )}
        />
      ) : null}

      {payload && !payload.available ? (
        <EmptyState
          title={pageText(locale, 'There is nothing to simulate yet', 'אין עדיין מה לסמלץ')}
          description={payload.reason || ''}
        />
      ) : null}

      {payload && payload.available ? (
        <>
          {/* NOTHING MATCHED IS NOT ZERO REVENUE, and the difference is the whole
              honesty of this panel. When the agreement's own scope resolves to no
              campaign on file, every money figure below is 0 because there was
              nothing to measure — not because the deal moves no money. That is
              said before the figures, and the scope it resolved to is named so the
              reader can see what was looked for. */}
          {noActivity ? (
            <div className="trd-sim-unknown" role="note">
              <TriangleAlert size={15} aria-hidden="true" />
              <span>
                {pageText(
                  locale,
                  'No activity on file falls inside this agreement\'s scope and window, so there is nothing to price. The money figures below are empty for that reason, not because the agreement is worth nothing.',
                  'אין פעילות בקבצים שנופלת בתוך ההיקף וחלון הזמן של ההסכם הזה, ולכן אין מה לתמחר. נתוני הכסף שלמטה ריקים מהסיבה הזו, לא מפני שההסכם אינו שווה דבר.',
                )}
                {payload.scope && payload.scope.resolved ? (
                  <>
                    {' '}
                    {pageText(locale, 'It looked for', 'חיפשנו')}
                    {': '}
                    <Name>{scopeResolution(payload.scope.resolved, locale)}</Name>
                    {'.'}
                  </>
                ) : null}
              </span>
            </div>
          ) : null}

          {payload.headline_he ? (
            <Card className="trd-sim-headline">
              <CardBody><Prose>{payload.headline_he}</Prose></CardBody>
            </Card>
          ) : null}

          <div className="trd-sim-money">
            {['gross_aired', 'scheduled_ahead', 'net_after_simulated_terms'].map((key) => (
              <MoneyLine
                key={key}
                line={{ key, kind: 'money', lead: key === 'net_after_simulated_terms' }}
                value={payload.money ? payload.money[key] : null}
                locale={locale}
                measured={!noActivity}
              />
            ))}
          </div>

          {payload.money && payload.money.basis_he ? (
            <Prose className="trd-sim-basis">{payload.money.basis_he}</Prose>
          ) : null}

          {payload.money && Number(payload.money.unknown_days) > 0 ? (
            <p className="trd-sim-unknown" role="note">
              <TriangleAlert size={14} aria-hidden="true" />
              {pageText(
                locale,
                `${formatNumber(payload.money.unknown_days, locale)} campaign-days in this window have no per-spot source. They are unknown, not zero, so every figure above is a floor.`,
                `ל־${formatNumber(payload.money.unknown_days, locale)} ימי קמפיין בחלון הזה אין מקור ברמת התשדיר. הם לא ידועים, לא אפס, ולכן כל נתון שלמעלה הוא רצפה.`,
              )}
            </p>
          ) : null}

          <div className="trd-sim-blocks">
            {['discount_ladder', 'agency_commission'].map((key) => (
              <MoneyLine
                key={key}
                line={{ key, kind: 'block' }}
                value={payload.money ? payload.money[key] : null}
                locale={locale}
              />
            ))}
          </div>

          {payload.placement ? (
            <Card className="trd-sim-placement">
              <CardBody>
                <h5>{pageText(locale, 'Placement constraints', 'אילוצי שיבוץ')}</h5>
                <dl className="trd-sim-block">
                  <dt>{pageText(locale, 'Conditions it would write', 'תנאים שייכתבו')}</dt>
                  <dd><Figure>{formatNumber(payload.placement.conditions, locale)}</Figure></dd>
                  <dt>{pageText(locale, 'Frequency rules it would write', 'כללי תדירות שייכתבו')}</dt>
                  <dd><Figure>{formatNumber(payload.placement.frequency_rules, locale)}</Figure></dd>
                </dl>
                {payload.placement.note_he ? (
                  <Prose className="trd-sim-basis">{payload.placement.note_he}</Prose>
                ) : null}
              </CardBody>
            </Card>
          ) : null}

          {payload.exposure && payload.exposure.length > 0 ? (
            <Card className="trd-sim-exposure">
              <CardBody>
                <h5>{pageText(locale, 'Commitments at risk or in breach', 'התחייבויות בסיכון או בהפרה')}</h5>
                <ul>
                  {payload.exposure.map((entry) => (
                    <li key={entry.instance_id || entry.term_id}>
                      <Name>{termName(entry.term_id, locale)}</Name>
                      {entry.reason_he ? <Prose as="span">{entry.reason_he}</Prose> : null}
                    </li>
                  ))}
                </ul>
              </CardBody>
            </Card>
          ) : null}

          {payload.not_simulated && payload.not_simulated.length > 0 ? (
            <Card className="trd-sim-skipped">
              <CardBody>
                <h5>
                  <Ban size={15} aria-hidden="true" />
                  {pageText(
                    locale,
                    `${formatNumber(payload.not_simulated.length, locale)} terms could not be simulated, and here they are by name`,
                    `${formatNumber(payload.not_simulated.length, locale)} מונחים לא ניתנים לסימולציה, והרי הם בשמם`,
                  )}
                </h5>
                <ul className="trd-skipped-list">
                  {payload.not_simulated.map((entry) => (
                    <li key={entry.instance_id || entry.term_id}>
                      <Name className="trd-skipped-name">{termName(entry.term_id, locale)}</Name>
                      {entry.instance_id ? <Code className="trd-id-chip">{entry.instance_id}</Code> : null}
                      <Prose as="span" className="trd-skipped-reason">{entry.reason_he}</Prose>
                    </li>
                  ))}
                </ul>
              </CardBody>
            </Card>
          ) : null}

          {payload.scope && payload.scope.resolved ? (
            <p className="trd-field-hint">
              <Status status="info">{pageText(locale, 'Measured over', 'נמדד על')}</Status>
              <Name>{scopeResolution(payload.scope.resolved, locale)}</Name>
            </p>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
