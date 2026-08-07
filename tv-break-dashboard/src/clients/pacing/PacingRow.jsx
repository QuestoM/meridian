import React from 'react';
import { Figure, Name } from '../../shell/bidi';
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  CircleCheck,
  CircleHelp,
  Plus,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import {
  amount,
  bare,
  barsFor,
  headlineLine,
  isolate,
  localized,
  otherLine,
  pair,
  percent,
  pick,
  vocabularyLabel,
} from './pacing-helpers';
import PacingDays from './PacingDays';

// The destination a traffic file is uploaded at, in the shell's own address form
// and under the shell's own name for it. This is a link and not a callback: the
// shell reads the hash and this piece owns no seam into its router, so the
// address contract is the thing both sides already agree on.
const UPLOAD_HASH = '#Data';

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
//
// The percentage is not the counted figure over the goal, it is the counted
// figure over the even share of the goal that the counted days were owed. Both
// operands of a number on a screen belong on the same screen: measured on the
// shipped board, the row printed 4.4 of 35 and 88 percent and left the 5.0 that
// makes 88 percent true only as the position of an unlabelled mark on the bar,
// so a reader who divided what they could see got 12.6 percent.
function Headline({ line, flight, locale }) {
  if (!line) return null;
  const figures = pair(line.counted.through_counted_day, line.goal, line.unit, locale);
  const ratio = line.pace ? percent(line.pace.ratio, locale) : null;
  const reference = line.reference ? bare(line.reference.expected_through_counted_day, line.unit, locale) : null;
  return (
    <div className="pacing-headline">
      <strong className="pacing-figure">{figures}</strong>
      <small className="pacing-scope">
        {pick(
          locale,
          `counted over ${flight.days_counted} of ${flight.days} broadcast days`,
          `נספר על ${isolate(flight.days_counted)} מתוך ${isolate(flight.days)} ימי שידור`,
        )}
      </small>
      {ratio ? (
        <Figure className={`pacing-ratio ${line.pace.verdict}`}>{ratio}</Figure>
      ) : null}
      {ratio && reference !== null ? (
        <small className="pacing-against">
          {pick(
            locale,
            `against a reference of ${reference} by that day`,
            `מול ייחוס של ${isolate(reference)} עד אותו יום`,
          )}
        </small>
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
          {pick(locale, `${remaining} left to the goal`, `${remaining} נותרו עד היעד`)}
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

// The sentence a remedy states, above the acts rather than inside them. It used
// to sit in the act group, which is how a row came to hold a sentence, a filled
// button, an outlined button and a bare link on one line at three weights.
function RemedySentence({ remedy, locale }) {
  if (remedy.kind !== 'book') return null;
  const left = amount(remedy.remaining, remedy.unit, locale);
  return (
    <p className="pacing-remedy-note">
      {pick(
        locale,
        `Book ${left} across the ${remedy.days.length} remaining broadcast days, or upload the traffic file that already holds them.`,
        `הזמינו ${left} על פני ${isolate(remedy.days.length)} ימי השידור שנותרו, או העלו את קובץ השידור שכבר מחזיק אותם.`,
      )}
    </p>
  );
}

// The days a booking remedy names, behind a disclosure and on the disclosure
// line, because a list that expands the card in place is not an act.
function RemedyDays({ remedy, locale }) {
  if (remedy.kind !== 'book' || !remedy.days.length) return null;
  return (
    <details className="pacing-remedy-days">
      <summary>
        {pick(
          locale,
          `The ${remedy.days.length} days with no source`,
          `${isolate(remedy.days.length)} הימים שאין להם מקור`,
        )}
      </summary>
      <Figure>{remedy.days.join(', ')}</Figure>
    </details>
  );
}

function Remedy({ remedy, locale, busy, onRaise, onOpenMakeGood }) {
  if (remedy.kind === 'raise') {
    const value = amount(remedy.value, remedy.unit, locale);
    return (
      <button type="button" className="pacing-remedy" disabled={busy} onClick={onRaise}>
        <Plus size={13} aria-hidden="true" />
        {pick(locale, `Raise a make-good for ${value}`, `פתחו פיצוי שידור על ${value}`)}
      </button>
    );
  }
  if (remedy.kind === 'open') {
    return (
      <button type="button" className="pacing-remedy" onClick={() => onOpenMakeGood(remedy.makeGoodId)}>
        {pick(locale, `Open make-good ${remedy.makeGoodId}`, `פתחו את פיצוי ${isolate(remedy.makeGoodId)}`)}
      </button>
    );
  }
  // The statement carries the act. A remedy that names an upload and then leaves
  // the reader to find the upload themselves is a diagnosis, not a remedy, so the
  // act this row offers is the one control that performs it.
  if (remedy.kind === 'book') {
    return (
      <a className="pacing-remedy" href={UPLOAD_HASH}>
        <Upload size={13} aria-hidden="true" />
        {pick(locale, 'Open Data to upload it', 'פתחו את נתונים כדי להעלות')}
      </a>
    );
  }
  // A supply remedy is the same block the row already prints above the track,
  // where a reader meets it before anything else on the row. Printing it again in
  // the control slot said the same sentence twice, which reads as two different
  // problems rather than as one.
  return null;
}

// The other ending. A row the board is asking a decision about is finished with
// either by acting on it or by somebody recording that the risk stands, and the
// second one is the only ending available on every such row. Once it is recorded
// the row states it, so a person scanning the board can see at a glance which
// rows have been read and which have not.
function Acceptance({ acceptance, locale, busy, onAccept, onOpenLedger }) {
  if (!acceptance || acceptance.kind === 'none') return null;
  if (acceptance.kind === 'accepted') {
    return (
      <button type="button" className="pacing-accepted" onClick={() => onOpenLedger(acceptance.makeGoodId)}>
        <ShieldCheck size={13} aria-hidden="true" />
        {pick(locale, 'Risk taken on, open the record', 'הסיכון התקבל, פתחו את הרשומה')}
      </button>
    );
  }
  return (
    <button type="button" className="pacing-accept" disabled={busy} onClick={onAccept}>
      {pick(locale, 'Take the risk on', 'קבלו את הסיכון')}
    </button>
  );
}

export default function PacingRow({
  row,
  vocabulary,
  locale,
  remedy,
  acceptance,
  demoMarking,
  drill,
  expanded,
  busy,
  canEdit,
  editRefusal,
  onToggle,
  onRaise,
  onAccept,
  onOpenMakeGood,
  onRetryDays,
}) {
  const line = headlineLine(row);
  const second = otherLine(row);
  const flight = row.flight;
  const Chevron = expanded ? ChevronUp : ChevronDown;
  return (
    <article className={`pacing-row ${row.headline.verdict}`} aria-labelledby={`pacing-${row.campaign_id}`}>
      <div className="pacing-row-head">
        <Verdict verdict={row.headline.verdict} vocabulary={vocabulary} locale={locale} />
        {/* A name is data and takes its own direction, never the surface's. The
            campaign names in this store are a period, then the advertiser, then
            the brand, and two of the three are Hebrew. Measured on the English
            pass without this, the two Hebrew segments of
            "2025-04 - עמותת מל"י - מל"י" painted in reverse order, so the screen
            named the brand before the advertiser. dir=auto takes the direction
            from the first strong character, which is the name's own. */}
        <div className="pacing-names">
          <strong id={`pacing-${row.campaign_id}`}><Name>{row.name || row.campaign_id}</Name></strong>
          <small>
            <Name>{row.advertiser}</Name>
            {flight ? ` ${isolate(`${flight.starts_on} - ${flight.ends_on}`)}` : ''}
          </small>
        </div>
        {row.is_demo ? (
          // The marking is one paragraph the seed wrote about every row it wrote,
          // so it rides the payload once. A row that carries its own keeps it, and
          // the row is never described by a sentence written about a different one.
          <span className="pacing-demo" title={localized(row.demo || demoMarking, 'meaning', locale)}>
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

      {/* Acts on one line at one height, and the disclosure that expands the card
          on its own below them. The owner reported the three weights and three
          baselines this replaces, and design-rules.md section 4 states the rule. */}
      {canEdit ? <RemedySentence remedy={remedy} locale={locale} /> : null}

      <div className="pacing-row-foot">
        {canEdit ? (
          <div className="pacing-row-acts">
            <Remedy
              remedy={remedy}
              locale={locale}
              busy={busy}
              onRaise={onRaise}
              onOpenMakeGood={onOpenMakeGood}
            />
            <Acceptance
              acceptance={acceptance}
              locale={locale}
              busy={busy}
              onAccept={onAccept}
              onOpenLedger={onOpenMakeGood}
            />
          </div>
        ) : (
          <span className="pacing-remedy-note">{editRefusal}</span>
        )}
      </div>

      <div className="pacing-row-disclosure">
        {canEdit ? <RemedyDays remedy={remedy} locale={locale} /> : null}
        {row.days_available ? (
          <button type="button" className="pacing-days-toggle" aria-expanded={expanded} onClick={onToggle}>
            <Chevron size={13} aria-hidden="true" />
            {pick(
              locale,
              `${expanded ? 'Hide' : 'Show'} the ${row.days_available} broadcast days behind this`,
              `${expanded ? 'הסתירו' : 'הציגו'} את ${isolate(row.days_available)} ימי השידור שמאחורי זה`,
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

      {expanded ? (
        <PacingDays drill={drill} second={second} locale={locale} onRetry={onRetryDays} />
      ) : null}
    </article>
  );
}
