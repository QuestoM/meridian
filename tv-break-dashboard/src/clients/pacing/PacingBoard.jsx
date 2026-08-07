import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Figure, Name } from '../../shell/bidi';
import PacingRow from './PacingRow';
import { loadDays } from './pacing-api';
import { VERDICT_ORDER, acceptanceFor, instant, isolate, localized, pick, remedyFor, vocabularyLabel } from './pacing-helpers';

// The board an account manager opens in the morning.
//
// It is sorted worst first by the server, so the first row is the one to act on
// and the answer to "what needs me today" costs zero clicks. The strip above the
// list states the same counts the list is ordered by, and pressing one filters to
// it rather than sorting it, so the order never changes under the reader.
//
// The table comes first and the definition sits behind a control. Measured on the
// round that shipped it, 621 characters of basis prose sat between the strip and
// the first row and put that row at y=540 in an 851 px viewport, which is a
// definition charging rent on the thing it defines. What stays in front of the
// reader is the instant the figures were counted at and the channel they cover,
// because those two change what the numbers mean.
//
// Keyboard control is the whole list: j and k step, Enter opens the days behind a
// row, r raises the make-good a row names when it names one, and a records the
// decision to take the risk on. Nothing here needs a dialog, because none of it is
// a destructive act.

function Strip({ counts, active, vocabulary, locale, onPick }) {
  return (
    <div className="pacing-strip" role="group" aria-label={pick(locale, 'Filter by verdict', 'סינון לפי מצב')}>
      {VERDICT_ORDER.map((verdict) => (
        <button
          key={verdict}
          type="button"
          className={`pacing-chip ${verdict} ${active === verdict ? 'active' : ''}`}
          aria-pressed={active === verdict}
          onClick={() => onPick(active === verdict ? '' : verdict)}
        >
          <Figure className="pacing-chip-count">{counts[verdict] || 0}</Figure>
          {vocabularyLabel(vocabulary.pace_verdicts, verdict, locale)}
        </button>
      ))}
    </div>
  );
}

function Basis({ payload, locale }) {
  const asOf = payload.as_of || {};
  const trigger = payload.trigger || {};
  const raiseRule = payload.raise_rule || {};
  const channel = payload.scope ? payload.scope.scope_channel : '';
  return (
    <div className="pacing-basis">
      {/* The instant goes through the same reader the ledger's does. It was the
          store's raw ISO stamp here and a readable one two screens away, which is
          one product speaking two ways about one clock. */}
      <p className="pacing-basis-line">
        {pick(locale, `Counted through ${instant(asOf.instant)}.`, `נספר עד ${isolate(instant(asOf.instant))}.`)}
        {channel ? ' ' : ''}
        {channel ? pick(
          locale,
          `This board covers ${channel} and no other channel.`,
          `הלוח מכסה את ${isolate(channel)} ולא ערוץ אחר.`,
        ) : ''}
      </p>
      {/* What the counted figure is, in front of the reader rather than behind
          the disclosure. Every verdict on this board is against a planned rating
          the traffic log holds and not against a measured delivery, and a reader
          who never opened the disclosure could take an at-risk verdict for a
          delivery shortfall. The long basis stays where it was. */}
      <p className="pacing-basis-planned">{localized(payload, 'counted_is_planned', locale)}</p>
      <details className="pacing-basis-details">
        <summary>{pick(locale, 'How this is counted', 'איך זה נספר')}</summary>
        <p>{localized(payload, 'counted_basis', locale)}</p>
        <p>
          {localized(trigger, 'rule', locale)}
          {' '}
          {localized(trigger, 'not_a_commercial_term', locale)}
        </p>
        {/* When a make-good may be raised at all. It is the rule the write path
            enforces, published here so the board, the ledger and any other
            client are reading one sentence. On this data it is why no row on the
            board offers the raise. */}
        <p>{localized(raiseRule, 'rule', locale)}</p>
        {/* The delivery ledger writes these two sentences in one language only,
            so they are quoted as the source's own words rather than presented as
            this surface's copy. A reader is told where the instant and the
            figures came from either way, and a Hebrew reader now meets the
            English of them only on asking. */}
        {asOf.basis ? (
          <p className="pacing-source-quote">
            {pick(locale, 'The ledger dates itself: ', 'ספר האספקה מתארך את עצמו: ')}
            <q lang="en"><Name>{asOf.basis}</Name></q>
          </p>
        ) : null}
        {asOf.figures_basis ? (
          <p className="pacing-source-quote">
            {pick(locale, 'The ledger states its figures: ', 'ספר האספקה אומר מהם נתוניו: ')}
            <q lang="en"><Name>{asOf.figures_basis}</Name></q>
          </p>
        ) : null}
      </details>
    </div>
  );
}

export default function PacingBoard({
  payload,
  locale,
  canEdit,
  editRefusal,
  busyId,
  onRaise,
  onAccept,
  onOpenMakeGood,
  focusCampaignId = '',
  onFocused = () => {},
}) {
  const [filter, setFilter] = useState('');
  const [expanded, setExpanded] = useState('');
  const [focused, setFocused] = useState(0);
  const [drills, setDrills] = useState({});
  const listRef = useRef(null);

  const rows = (payload.rows || []).filter((row) => !filter || row.headline.verdict === filter);
  const counts = payload.counts || {};
  const vocabulary = payload.vocabulary || {};
  const countedAt = (payload.as_of || {}).instant || '';

  useEffect(() => {
    setFocused(0);
  }, [filter]);

  // A name somebody clicked in the ledger lands on its own row, not on whichever
  // row happens to be first. Measured before this: opening a ledger record for a
  // campaign sitting at index 6 of 56 returned to the board with row 0 focused,
  // unscrolled and unmarked, so a name that looks like a link went nowhere. Any
  // verdict filter is cleared first, because a row filtered out of the list
  // cannot be the row a reader was sent to.
  useEffect(() => {
    if (!focusCampaignId) return;
    const all = payload.rows || [];
    const index = all.findIndex((row) => row.campaign_id === focusCampaignId);
    if (index < 0) return;
    // Clearing the filter is its own pass. The effect above resets the focus
    // whenever the filter moves, and it is declared first, so focusing in the
    // same pass would be overwritten by it.
    if (filter) {
      setFilter('');
      return;
    }
    setFocused(index);
    onFocused();
  }, [focusCampaignId, payload.rows, filter, onFocused]);

  // A drill is a read of the same ledger the board was counted from, so a board
  // counted at a new instant invalidates every day already on screen rather than
  // leaving one row dated by an older read than the row above it.
  useEffect(() => {
    setDrills({});
    setExpanded('');
  }, [countedAt]);

  const openDays = useCallback((campaignId) => {
    setDrills((current) => ({ ...current, [campaignId]: { status: 'loading' } }));
    loadDays(campaignId)
      .then((body) => setDrills((current) => ({
        ...current,
        [campaignId]: { status: 'ready', days: body.days || [] },
      })))
      .catch(() => setDrills((current) => ({ ...current, [campaignId]: { status: 'failed' } })));
  }, []);

  const toggle = useCallback((campaignId) => {
    setExpanded((current) => (current === campaignId ? '' : campaignId));
    // A day read is kept once it lands, so opening the same row twice costs one
    // request. A failed one is retried, because a failure is not an answer.
    if (!drills[campaignId] || drills[campaignId].status === 'failed') openDays(campaignId);
  }, [drills, openDays]);

  const onKeyDown = useCallback((event) => {
    if (!rows.length) return;
    if (event.key === 'j' || event.key === 'ArrowDown') {
      event.preventDefault();
      setFocused((current) => Math.min(rows.length - 1, current + 1));
    } else if (event.key === 'k' || event.key === 'ArrowUp') {
      event.preventDefault();
      setFocused((current) => Math.max(0, current - 1));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const row = rows[focused];
      if (row) toggle(row.campaign_id);
    } else if (event.key === 'r') {
      const row = rows[focused];
      const remedy = row ? remedyFor(row, payload.make_goods) : null;
      if (row && remedy && remedy.kind === 'raise' && canEdit) {
        event.preventDefault();
        onRaise(row);
      }
    } else if (event.key === 'a') {
      const row = rows[focused];
      const acceptance = row
        ? acceptanceFor(row, payload.acceptances, payload.needs_a_decision)
        : null;
      if (row && acceptance && acceptance.kind === 'accept' && canEdit) {
        event.preventDefault();
        onAccept(row);
      }
    }
  }, [rows, focused, payload.make_goods, payload.acceptances, payload.needs_a_decision, canEdit, onRaise, onAccept, toggle]);

  useEffect(() => {
    const node = listRef.current;
    if (!node) return;
    const cards = node.querySelectorAll('.pacing-row');
    const card = cards[focused];
    if (card) card.scrollIntoView({ block: 'nearest' });
  }, [focused, rows.length]);

  // The legend names the keys that do something on the rows in front of the
  // reader, and no others. Measured on the shipped data, 0 of 56 rows reach a
  // raise, so a fixed legend advertised a key that could not fire on any row on
  // the board. A shortcut nobody can press is a claim about a capability, and
  // this piece states capability from what the rows carry rather than from what
  // the code can do in principle.
  const keys = [pick(locale, 'j and k step', 'j ו-k מדלגים')];
  if (rows.some((row) => row.days_available)) {
    keys.push(pick(locale, 'Enter opens the broadcast days', 'Enter פותח את ימי השידור'));
  }
  if (canEdit && rows.some((row) => remedyFor(row, payload.make_goods).kind === 'raise')) {
    keys.push(pick(locale, 'r raises the make-good a row names', 'r פותח את פיצוי השידור שהשורה נוקבת בו'));
  }
  if (canEdit && rows.some((row) => acceptanceFor(row, payload.acceptances, payload.needs_a_decision).kind === 'accept')) {
    keys.push(pick(locale, 'a takes the risk on', 'a מקבל את הסיכון'));
  }

  return (
    <section className="pacing-board" aria-label={pick(locale, 'Campaign pacing', 'קצב הקמפיינים')}>
      <Strip counts={counts} active={filter} vocabulary={vocabulary} locale={locale} onPick={setFilter} />

      <div className="pacing-chrome">
        <Basis payload={payload} locale={locale} />
        <p className="pacing-keys">{`${keys.join(', ')}.`}</p>
      </div>

      {rows.length === 0 ? (
        <p className="pacing-empty">
          {pick(
            locale,
            'No campaign on this channel carries that verdict.',
            'אין קמפיין בערוץ הזה שנושא את המצב הזה.',
          )}
        </p>
      ) : null}

      <div
        className="pacing-list"
        ref={listRef}
        role="list"
        tabIndex={0}
        onKeyDown={onKeyDown}
        aria-label={pick(locale, 'Campaigns, worst pacing first', 'קמפיינים, החמור בקצב ראשון')}
      >
        {rows.map((row, index) => (
          <div role="listitem" key={row.campaign_id} className={index === focused ? 'pacing-focused' : ''}>
            <PacingRow
              row={row}
              vocabulary={vocabulary}
              locale={locale}
              remedy={remedyFor(row, payload.make_goods)}
              acceptance={acceptanceFor(row, payload.acceptances, payload.needs_a_decision)}
              demoMarking={payload.demo_marking}
              drill={drills[row.campaign_id]}
              expanded={expanded === row.campaign_id}
              busy={busyId === row.campaign_id}
              canEdit={canEdit}
              editRefusal={editRefusal}
              onToggle={() => toggle(row.campaign_id)}
              onRaise={() => onRaise(row)}
              onAccept={() => onAccept(row)}
              onOpenMakeGood={onOpenMakeGood}
              onRetryDays={() => openDays(row.campaign_id)}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
