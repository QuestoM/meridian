import React from 'react';
import { Card, CardBody, EmptyState, ErrorState, LoadingState, Status } from '../studio';
import { Activity, TriangleAlert } from 'lucide-react';
import { Code, Figure, Name, Prose } from '../shell/bidi';
import { formatDay, formatSpan } from '../shell/dates';
import { formatNumber, formatPercent, pageText } from '../shell/format';
import { termName } from './trade-terms';
import { contractMoney } from './term-language';
import {
  alarmLabel, ALARM_ORDER, alarmTone, scopeResolution, windowOf,
} from './trade-vocabulary';

// What the agreement committed the channel to, measured continuously.
//
// FOUR HONESTY RULES ARE VISIBLE ON EVERY ROW, and they are the reason this board
// is not a set of progress bars.
//
// A counted figure is a FLOOR. Days with no per-spot source are unknown, not
// zero, so a standing can only ever understate delivery. The engine says so in
// its own basis line and it is printed.
//
// UNKNOWN IS NOT COMPLIANT. A guarantee stated in an audience the product cannot
// measure reports unknown with the reason, and it is coloured as information
// rather than as health, because a green row nobody measured is the worst thing
// this board could show.
//
// A PROJECTION NAMES ITS METHOD. The committed projection is counted plus already
// booked; the pace-forward estimate rides beside it and is labelled as an
// estimate. A model's expectation is not a booking.
//
// DEFAULT BANDS ARE DISCLOSED. When the agreement stated no tolerance, the alarm
// ladder used the engine's own default bands, and the row says that out loud so
// nobody mistakes a default for a negotiated threshold.

// The unit decides how a figure reads. Money is written in full because a
// commitment is reconciled against a signed page, not skimmed for scale.
function StandingFigure({ value, unit, locale }) {
  if (value === null || value === undefined) {
    return (
      <span className="trd-unknown">
        {pageText(locale, 'not measurable', 'לא ניתן למדידה')}
      </span>
    );
  }
  if (unit === 'ILS') return <Figure>{contractMoney(value, locale)}</Figure>;
  if (unit === 'percent') return <Figure>{formatPercent(value, locale)}</Figure>;
  if (unit === 'ILS_per_point') {
    return (
      <Figure>
        {`${contractMoney(value, locale)} / ${pageText(locale, 'point', 'נקודה')}`}
      </Figure>
    );
  }
  if (unit === 'rating_points') {
    return (
      <Figure>
        {`${formatNumber(value, locale)} ${pageText(locale, 'rating points', 'נקודות רייטינג')}`}
      </Figure>
    );
  }
  return <Figure>{formatNumber(value, locale)}</Figure>;
}

function ObligationRow({ obligation, locale }) {
  const target = obligation.target || {};
  const standing = obligation.standing || {};
  const method = obligation.projection_method || null;
  const span = windowOf(obligation.window);
  return (
    <Card className="trd-obligation" data-alarm={obligation.alarm}>
      <CardBody>
        <div className="trd-ob-head">
          <div>
            <h5><Name>{termName(obligation.term_id, locale)}</Name></h5>
            <Code className="trd-ob-id">{obligation.instance_id}</Code>
          </div>
          <Status status={alarmTone(obligation.alarm)}>{alarmLabel(obligation.alarm, locale)}</Status>
        </div>

        <dl className="trd-ob-figures">
          <dt>{pageText(locale, 'Committed', 'התחייבות')}</dt>
          <dd><StandingFigure value={target.value} unit={target.unit} locale={locale} /></dd>
          <dt>{pageText(locale, 'Counted so far', 'נספר עד כה')}</dt>
          <dd><StandingFigure value={standing.counted} unit={standing.unit} locale={locale} /></dd>
          {obligation.expected_to_date !== undefined && obligation.expected_to_date !== null ? (
            <>
              <dt>{pageText(locale, 'Expected by today', 'צפוי עד היום')}</dt>
              <dd><StandingFigure value={obligation.expected_to_date} unit={target.unit} locale={locale} /></dd>
            </>
          ) : null}
          {obligation.ratio !== undefined && obligation.ratio !== null ? (
            <>
              <dt>{pageText(locale, 'Pace against expected', 'קצב מול הצפוי')}</dt>
              <dd><Figure>{formatPercent(obligation.ratio * 100, locale)}</Figure></dd>
            </>
          ) : null}
          {obligation.projection !== undefined && obligation.projection !== null ? (
            <>
              <dt>{pageText(locale, 'Projected at close', 'תחזית בסיום')}</dt>
              <dd><StandingFigure value={obligation.projection} unit={target.unit} locale={locale} /></dd>
            </>
          ) : null}
          <dt>{pageText(locale, 'Measurement window', 'חלון המדידה')}</dt>
          <dd>
            {span.openEnded ? (
              <span>{pageText(locale, 'open-ended', 'ללא מועד סיום')}</span>
            ) : (
              <Figure>{formatSpan(span.from, span.to, locale)}</Figure>
            )}
          </dd>
        </dl>

        {standing.basis ? <Prose className="trd-ob-basis">{standing.basis}</Prose> : null}
        {obligation.alarm_reason ? <Prose className="trd-ob-reason">{obligation.alarm_reason}</Prose> : null}

        {method ? (
          <div className="trd-ob-method">
            <span className="trd-card-label">{pageText(locale, 'How the projection was made', 'איך נבנתה התחזית')}</span>
            <span className="trd-ob-method-row">
              <span>{pageText(locale, 'Counted plus booked ahead', 'נספר בתוספת מתוזמן')}</span>
              <StandingFigure value={method.booked_forward} unit={target.unit} locale={locale} />
            </span>
            <span className="trd-ob-method-row">
              <span>{pageText(locale, 'At the current pace', 'לפי הקצב הנוכחי')}</span>
              <StandingFigure value={method.pace_forward} unit={target.unit} locale={locale} />
            </span>
            {method.note ? <Prose as="span" className="trd-meta-note">{method.note}</Prose> : null}
          </div>
        ) : null}

        {obligation.used_default_bands ? (
          <p className="trd-ob-default" role="note">
            <TriangleAlert size={14} aria-hidden="true" />
            {pageText(
              locale,
              'The agreement stated no tolerance, so the alarm above used the engine\'s default bands. It is not a threshold anybody negotiated.',
              'ההסכם לא קבע סטייה מותרת, ולכן ההתראה שלמעלה השתמשה ברצועות ברירת המחדל של המנוע. זה אינו רף שמישהו סיכם.',
            )}
          </p>
        ) : null}

        {obligation.resolution && obligation.resolution.resolved ? (
          <p className="trd-field-hint">
            {pageText(locale, 'Measured over', 'נמדד על')}
            {': '}
            <Name>{scopeResolution(obligation.resolution.resolved, locale)}</Name>
            {obligation.resolution.campaigns && obligation.resolution.campaigns.length > 0 ? (
              <>
                {' · '}
                <Figure>
                  {pageText(
                    locale,
                    `${formatNumber(obligation.resolution.campaigns.length, locale)} campaigns`,
                    `${formatNumber(obligation.resolution.campaigns.length, locale)} קמפיינים`,
                  )}
                </Figure>
              </>
            ) : null}
          </p>
        ) : null}
      </CardBody>
    </Card>
  );
}

export default function ObligationsBoard({ payload, error, locale, onRetry }) {
  if (error) {
    return (
      <ErrorState
        title={pageText(locale, 'The commitment standing could not be read', 'לא ניתן היה לקרוא את מצב ההתחייבויות')}
        description={error}
        action={onRetry}
      />
    );
  }
  if (payload === null || payload === undefined) {
    return (
      <LoadingState
        title={pageText(locale, 'Measuring the commitments', 'מודד את ההתחייבויות')}
        description={pageText(
          locale,
          'Each committed term is measured against the delivery ledger as it stands today.',
          'כל התחייבות נמדדת מול ספר האספקה כפי שהוא היום.',
        )}
      />
    );
  }
  if (!payload.available) {
    return (
      <EmptyState
        title={pageText(locale, 'No commitments are being measured', 'לא נמדדות התחייבויות')}
        description={payload.reason || pageText(
          locale,
          'Commitments are measured only from an approved version.',
          'התחייבויות נמדדות רק מגרסה מאושרת.',
        )}
      />
    );
  }
  const obligations = payload.obligations || [];
  if (obligations.length === 0) {
    return (
      <EmptyState
        title={pageText(locale, 'This agreement commits nothing measurable', 'ההסכם הזה אינו כולל התחייבות מדידה')}
        description={pageText(
          locale,
          'None of its approved terms is an obligation the engine tracks: no budget, no rating guarantee, no mix or continuity undertaking.',
          'אף אחד מהמונחים המאושרים שלו אינו התחייבות שהמנוע עוקב אחריה: אין תקציב, אין התחייבות רייטינג, אין תמהיל או רציפות.',
        )}
      />
    );
  }
  const counts = payload.alarm_counts || {};
  return (
    <section className="trd-obligations" aria-label={pageText(locale, 'Commitment standing', 'מצב התחייבויות')}>
      <div className="trd-pane-head">
        <h4>
          <Activity size={16} aria-hidden="true" />
          {pageText(locale, 'Commitment standing', 'מצב התחייבויות')}
        </h4>
        <span className="trd-alarm-chips">
          {ALARM_ORDER.filter((alarm) => Number(counts[alarm]) > 0).map((alarm) => (
            <Status key={alarm} status={alarmTone(alarm)}>
              {`${alarmLabel(alarm, locale)} · ${formatNumber(counts[alarm], locale)}`}
            </Status>
          ))}
        </span>
      </div>
      {payload.evaluated_at ? (
        <p className="trd-field-hint">
          {pageText(locale, 'Measured as of', 'נמדד לתאריך')}
          {': '}
          <Figure>{formatDay(payload.evaluated_at)}</Figure>
        </p>
      ) : null}
      <div className="trd-ob-grid">
        {obligations.map((obligation) => (
          <ObligationRow key={obligation.obligation_id} obligation={obligation} locale={locale} />
        ))}
      </div>
    </section>
  );
}
