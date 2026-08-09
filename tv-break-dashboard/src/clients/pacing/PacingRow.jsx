import React from 'react';
import { Figure, Name } from '../../shell/bidi';
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  CircleCheck,
  CircleHelp,
} from 'lucide-react';
import { formatDayList, formatSpan } from '../../shell/dates';
import {
  amount,
  bare,
  barsFor,
  headlineLine,
  isolate,
  localized,
  opensDays,
  otherLine,
  pair,
  percent,
  pick,
  vocabularyLabel,
} from './pacing-helpers';
import { Acceptance, Remedy } from './PacingActs';
import PacingDays from './PacingDays';
import PacingGoalLine from './PacingGoalLine';

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
//
// The figure itself opens the days it was summed from. Stripe's transferable
// mechanic is that an amount is the way into the rows behind it, and this board
// reached its drill only from a separately labelled button below the row while
// 4.4 of 35 rating points sat inert above it. The labelled control stays, because
// a figure that is also a control has to be discoverable by somebody who never
// hovers it, and the accessible name of the figure carries the act as well as
// the figure so the two are one control to a screen reader too.
function Headline({ line, flight, locale, days, expanded, onToggle }) {
  if (!line) return null;
  const figures = pair(line.counted.through_counted_day, line.goal, line.unit, locale);
  const ratio = line.pace ? percent(line.pace.ratio, locale) : null;
  const reference = line.reference ? bare(line.reference.expected_through_counted_day, line.unit, locale) : null;
  // The half of the counted figure whose time has not come yet. There is no
  // delivery feed, so the sharpest thing this board can say about a figure is
  // whether the traffic log's own clock has passed it, and the payload has
  // carried that split on every line all along. Measured on the shipped board
  // before this line existed: 18 of the 51 rows that carry a goal count spots
  // that have not aired, and on 7 of them nothing has aired at all, five of
  // those reading at risk. The only way to learn it was to open the drill and
  // read a state column, so the row said at risk about a campaign that had
  // aired nothing and never said so.
  const notAired = amount(line.counted.booked_not_aired, line.unit, locale);
  const opens = opensDays(figures, days, locale);
  return (
    <div className="pacing-headline">
      {days && onToggle ? (
        <button type="button" className="pacing-figure pacing-figure-open"
                aria-expanded={expanded} aria-label={opens} onClick={onToggle}>
          {figures}
        </button>
      ) : <strong className="pacing-figure">{figures}</strong>}
      <small className="pacing-scope">
        {pick(
          locale,
          `counted over ${flight.days_counted} of ${flight.days} broadcast days`,
          `נספר על ${isolate(flight.days_counted)} מתוך ${isolate(flight.days)} ימי שידור`,
        )}
      </small>
      {line.counted.booked_not_aired > 0 ? (
        <small className="pacing-not-aired">
          {pick(
            locale,
            `including ${notAired} whose time has not come yet`,
            `כולל ${notAired} שהשעה שלהן טרם הגיעה`,
          )}
        </small>
      ) : null}
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
      {/* The days are read by shell/dates.js, which collapses a consecutive run
          and prints dd/mm/yyyy. Measured on the shipped card, this line printed
          2025-04-28, 2025-04-29, 2025-04-30, 2025-05-01, 2025-05-02, 2025-05-03,
          which is one unbroken run spelled out six times in a machine format,
          and is the exact string the owner reported. */}
      <Figure>{formatDayList(remedy.days, locale)}</Figure>
    </details>
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
  onOpenCampaign,
  onRetryDays,
}) {
  const line = headlineLine(row);
  const second = otherLine(row);
  const flight = row.flight;
  const Chevron = expanded ? ChevronUp : ChevronDown;
  return (
    <article className={`card card-dense pacing-row ${row.headline.verdict}`} aria-labelledby={`pacing-${row.campaign_id}`}>
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
          {/* The name opens the campaign when the destination this panel is
              mounted in hands it a way to. It is a control only then: a name
              styled as a link that goes nowhere is worse than a name, and this
              piece owns no seam into the workspace's router. The mount in
              contracts/P11.md section 5 passes onOpenCampaign, and a mount that
              leaves it out regresses nothing: this stays the heading it was. */}
          <strong id={`pacing-${row.campaign_id}`}>
            {onOpenCampaign ? (
              <button type="button" className="pacing-name-open"
                      onClick={() => onOpenCampaign(row.campaign_id)}>
                <Name>{row.name || row.campaign_id}</Name>
              </button>
            ) : <Name>{row.name || row.campaign_id}</Name>}
          </strong>
          {/* The advertiser and the flight window are two facts, so a rule
              divides them rather than a space, which design-rules.md section 3
              asks for and which matters most in Hebrew where there are no
              capitals to find the boundary. The window is read by
              shell/dates.js: it printed the payload's raw ISO fields either
              side of a spaced hyphen, which is a machine format and a joiner a
              reader cannot tell from a list separator. */}
          <small className="pacing-name-facts">
            <Name>{row.advertiser}</Name>
            {flight ? <span>{formatSpan(flight.starts_on, flight.ends_on, locale)}</span> : null}
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
        <Headline line={line} flight={flight} locale={locale}
                  days={row.days_available} expanded={expanded} onToggle={onToggle} />
      </div>

      <Track line={line} locale={locale} />

      {/* The campaign's other goal. It is a reading and not an act, so it sits
          with the track rather than in the foot, and it is the same component
          the day drill prints under its table. */}
      <PacingGoalLine line={second} vocabulary={vocabulary} locale={locale}
                      days={row.days_available} expanded={expanded}
                      onOpen={row.days_available ? onToggle : null} />

      {line && line.pace && line.pace.verdict === 'unknown' ? (
        <Sentence block={line.pace} locale={locale} className="pacing-unknown" />
      ) : null}
      {!line ? <Sentence block={row.headline} locale={locale} className="pacing-unknown" /> : null}

      <Forward line={line} vocabulary={vocabulary} locale={locale} />

      {/* Acts on one line at one height, and the disclosure that expands the card
          on its own below them. The owner reported the three weights and three
          baselines this replaces, and design-rules.md section 4 states the rule. */}
      {/* The sentence is a diagnosis and the days behind it are a reading, so
          neither is gated on the write permission. They were both, which took
          the quantity to book and the six dates it applies to off the screen of
          every read-only account: a viewer could see that a campaign was at risk
          and not what would fix it. The permission governs the acts below, and
          those are still gated. */}
      <RemedySentence remedy={remedy} locale={locale} />

      <div className="pacing-row-foot">
        {canEdit ? (
          <div className="pacing-row-acts">
            <Remedy
              remedy={remedy}
              locale={locale}
              busy={busy}
              onRaise={onRaise}
              onOpenMakeGood={onOpenMakeGood}
              onOpenCampaign={onOpenCampaign ? () => onOpenCampaign(row.campaign_id) : null}
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
        <RemedyDays remedy={remedy} locale={locale} />
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
        <PacingDays drill={drill} line={line} second={second} vocabulary={vocabulary}
                    locale={locale} onRetry={onRetryDays} />
      ) : null}
    </article>
  );
}
