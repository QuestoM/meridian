import React from 'react';
import { AlertTriangle, ChevronDown, ChevronUp, CircleCheck, CircleHelp, Plus } from 'lucide-react';
import {
  amount,
  barsFor,
  headlineLine,
  isolate,
  localized,
  otherLine,
  percent,
  pick,
  vocabularyLabel,
} from './pacing-helpers';
import PacingDays from './PacingDays';

// One campaign on the pacing board.
//
// The row is built so a reader never has to compute anything. The verdict word
// leads, the figure that produced it follows with the unit it is in, the reference
// sits on the same axis as the counted bar so the gap is visible rather than
// arithmetic, and the one thing to do about the row is a control on the row.
//
// Every figure prints the scope it was counted over on the same line: how many of
// the flight's broadcast days carry a source. A figure with no scope beside it is
// exactly the defect this board exists not to repeat, so the scope is part of the
// figure's own markup and not a tooltip.

const ICONS = {
  behind: AlertTriangle,
  at_risk: AlertTriangle,
  on_pace: CircleCheck,
  unknown: CircleHelp,
};

function Verdict({ verdict, vocabulary, locale }) {
  const Icon = ICONS[verdict] || CircleHelp;
  return (
    <span className={`pacing-verdict ${verdict}`}>
      <Icon size={13} aria-hidden="true" />
      {vocabularyLabel(vocabulary.pace_verdicts, verdict, locale)}
    </span>
  );
}

// The counted figure and the goal it is counted against, with the days it covers.
function Headline({ line, flight, locale }) {
  if (!line) return null;
  const counted = amount(line.counted.through_counted_day, line.unit, locale);
  const goal = amount(line.goal, line.unit, locale);
  const ratio = line.pace ? percent(line.pace.ratio, locale) : null;
  return (
    <div className="pacing-headline">
      <strong className="pacing-figure">
        {pick(locale, `${counted} of ${goal}`, `${isolate(counted)} מתוך ${isolate(goal)}`)}
      </strong>
      <small className="pacing-scope">
        {pick(
          locale,
          `counted over ${flight.days_counted} of ${flight.days} broadcast days`,
          `נספר על ${isolate(flight.days_counted)} מתוך ${isolate(flight.days)} ימי שידור`,
        )}
      </small>
      {ratio ? (
        <span className={`pacing-ratio ${line.pace.verdict}`} dir="ltr">{ratio}</span>
      ) : null}
    </div>
  );
}

// The counted bar with the reference marked on the same axis. The bar is
// decoration for a figure that is already printed above it, so it carries no
// number of its own and is hidden from a screen reader.
function Track({ line, locale }) {
  const bars = barsFor(line);
  if (!bars) return null;
  return (
    <div className="pacing-track" aria-hidden="true">
      <i className="pacing-booked" style={{ '--bar': bars.booked }} />
      <i className={`pacing-counted ${line.pace ? line.pace.verdict : ''}`} style={{ '--bar': bars.counted }} />
      {bars.reference === null ? null : (
        <b className="pacing-reference" style={{ '--at': bars.reference }} title={localized(line.reference, 'rule', locale)} />
      )}
    </div>
  );
}

function Sentence({ block, locale, className }) {
  const reason = localized(block, 'reason', locale);
  const path = localized(block, 'path_forward', locale);
  if (!reason && !path) return null;
  return (
    <p className={className}>
      {reason}
      {path ? ` ${path}` : ''}
    </p>
  );
}

// What the remaining days of the flight say, which fails separately from the pace.
function Forward({ line, vocabulary, locale }) {
  if (!line || !line.forward || !line.forward.state) return null;
  const remaining = amount(line.forward.remaining_to_goal, line.unit, locale);
  const missing = (line.forward.unsourced_remaining_days || []).length;
  return (
    <div className={`pacing-forward ${line.forward.state}`}>
      <span className="pacing-forward-word">
        {vocabularyLabel(vocabulary.forward_states, line.forward.state, locale)}
      </span>
      {remaining ? (
        <span className="pacing-forward-figure">
          {pick(locale, `${remaining} left to the goal`, `${isolate(remaining)} נותרו עד היעד`)}
        </span>
      ) : null}
      {missing ? (
        <span className="pacing-forward-missing">
          {pick(
            locale,
            `${missing} remaining broadcast days carry no source`,
            `${isolate(missing)} ימי שידור שנותרו בלי מקור`,
          )}
        </span>
      ) : null}
    </div>
  );
}

function Remedy({ remedy, locale, busy, onRaise, onOpenMakeGood }) {
  if (remedy.kind === 'raise') {
    const value = amount(remedy.value, remedy.unit, locale);
    return (
      <button type="button" className="pacing-remedy" disabled={busy} onClick={onRaise}>
        <Plus size={13} aria-hidden="true" />
        {pick(locale, `Raise a make-good for ${value}`, `פתחו פיצוי שידור על ${isolate(value)}`)}
      </button>
    );
  }
  if (remedy.kind === 'open') {
    return (
      <button type="button" className="pacing-remedy open" onClick={() => onOpenMakeGood(remedy.makeGoodId)}>
        {pick(locale, `Open make-good ${remedy.makeGoodId}`, `פתחו את פיצוי ${isolate(remedy.makeGoodId)}`)}
      </button>
    );
  }
  if (remedy.kind === 'book') {
    const left = amount(remedy.remaining, remedy.unit, locale);
    return (
      <span className="pacing-remedy-note">
        {pick(
          locale,
          `Book ${left} across the ${remedy.days.length} remaining days, or upload the traffic file that already holds them. Missing: ${remedy.days.join(', ')}.`,
          `הזמינו ${isolate(left)} על פני ${isolate(remedy.days.length)} הימים שנותרו, או העלו את קובץ השידור שכבר מחזיק אותם. חסרים: ${isolate(remedy.days.join(', '))}.`,
        )}
      </span>
    );
  }
  // A supply remedy is the same block the row already prints above the track,
  // where a reader meets it before anything else on the row. Printing it again in
  // the control slot said the same sentence twice, which reads as two different
  // problems rather than as one.
  return null;
}

export default function PacingRow({
  row,
  vocabulary,
  locale,
  remedy,
  expanded,
  busy,
  canEdit,
  editRefusal,
  onToggle,
  onRaise,
  onOpenMakeGood,
}) {
  const line = headlineLine(row);
  const second = otherLine(row);
  const flight = row.flight;
  const Chevron = expanded ? ChevronUp : ChevronDown;
  return (
    <article className={`pacing-row ${row.headline.verdict}`} aria-labelledby={`pacing-${row.campaign_id}`}>
      <div className="pacing-row-head">
        <Verdict verdict={row.headline.verdict} vocabulary={vocabulary} locale={locale} />
        <div className="pacing-names">
          <strong id={`pacing-${row.campaign_id}`}>{row.name || row.campaign_id}</strong>
          <small>
            {row.advertiser}
            {flight ? ` ${isolate(`${flight.starts_on} - ${flight.ends_on}`)}` : ''}
          </small>
        </div>
        {row.is_demo ? (
          <span className="pacing-demo" title={localized(row.demo, 'meaning', locale)}>
            {pick(locale, 'Demo', 'הדגמה')}
          </span>
        ) : null}
        <Headline line={line} flight={flight} locale={locale} />
      </div>

      <Track line={line} locale={locale} />

      {line && line.pace && line.pace.verdict === 'unknown' ? (
        <Sentence block={line.pace} locale={locale} className="pacing-unknown" />
      ) : null}
      {!line ? <Sentence block={row.headline} locale={locale} className="pacing-unknown" /> : null}

      <Forward line={line} vocabulary={vocabulary} locale={locale} />

      <div className="pacing-row-foot">
        {canEdit ? (
          <Remedy
            remedy={remedy}
            locale={locale}
            busy={busy}
            onRaise={onRaise}
            onOpenMakeGood={onOpenMakeGood}
          />
        ) : (
          <span className="pacing-remedy-note">{editRefusal}</span>
        )}
        {row.days.length ? (
          <button type="button" className="pacing-days-toggle" aria-expanded={expanded} onClick={onToggle}>
            <Chevron size={13} aria-hidden="true" />
            {pick(
              locale,
              `${expanded ? 'Hide' : 'Show'} the ${row.days.length} broadcast days behind this`,
              `${expanded ? 'הסתירו' : 'הציגו'} את ${isolate(row.days.length)} ימי השידור שמאחורי זה`,
            )}
          </button>
        ) : (
          <span className="pacing-remedy-note">
            {pick(
              locale,
              'The delivery ledger holds no broadcast day for this campaign at all.',
              'ספר האספקה אינו מחזיק אף יום שידור לקמפיין הזה.',
            )}
          </span>
        )}
      </div>

      {expanded ? <PacingDays row={row} second={second} locale={locale} /> : null}
    </article>
  );
}
