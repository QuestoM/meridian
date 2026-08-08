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
  // Both values are isolated in both languages, not only in Hebrew. One sentence
  // was applying two rules to the same two values: the Hebrew branch isolated
  // them and the English branch did not.
  //
  // Measured rather than assumed, with Range rectangles on both branches. The
  // channel name renders identically either way on this value, so nothing on
  // screen was wrong about it, and it is isolated because direction is a
  // property of the value and the next channel may not begin with a Hebrew
  // letter. The instant is the one that moves: in a Hebrew line the bare form
  // put 27/04/2025 at x=453 and the isolated form at x=401, a 52 px reorder
  // around the comma and the full stop, which is why the Hebrew branch already
  // carried one.
  const when = isolate(instant(asOf.instant));
  const named = isolate(channel);
  return (
    <div className="pacing-basis">
      {/* The instant goes through the same reader the ledger's does. It was the
          store's raw ISO stamp here and a readable one two screens away, which is
          one product speaking two ways about one clock. */}
      <p className="pacing-basis-line">
        {pick(locale, `Counted through ${when}.`, `נספר עד ${when}.`)}
        {channel ? ' ' : ''}
        {channel ? pick(
          locale,
          `This board covers ${named} and no other channel.`,
          `הלוח מכסה את ${named} ולא ערוץ אחר.`,
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
        {/* Which clock the two instants on this screen are on. The ledger's own
            counted_as_of carries no offset and a decision's raised_at carries
            UTC, so a reader meets one stamp labelled and one bare and nothing
            said the bare one was zoneless. Stripe declares the time zone of
            every report beside its range; this piece cannot, because the source
            file does not declare one, so it says that instead of implying a
            zone it does not have. */}
        <p>
          {pick(
            locale,
            'The delivery ledger declares no time zone, so a counted instant is printed as the source file records it. An instant this product recorded itself is marked UTC.',
            'ספר האספקה אינו מצהיר על אזור זמן, ולכן רגע הספירה מוצג כפי שקובץ המקור רושם אותו. רגע שהמוצר עצמו רשם מסומן UTC.',
          )}
        </p>
        {/* When a make-good may be raised at all. It is the rule the write path
            enforces, published here so the board, the ledger and any other
            client are reading one sentence. On this data it is why no row on the
            board offers the raise. */}
        <p>{localized(raiseRule, 'rule', locale)}</p>
        {/* What the ledger says its own figures are, in the language the reader
            is in. Both of the ledger's basis columns are English strings in the
            CSV, so a Hebrew operator met the sentence that decides what every
            verdict on this board means in the wrong language. The ledger's own
            module publishes the same two claims as a bilingual pair and the
            payload carries them; the column is quoted only where no pair exists,
            because a sentence this surface translated itself would be a claim
            about a store it does not own. */}
        {asOf.rating_basis_en ? (
          <>
            <p>{localized(asOf, 'rating_basis', locale)}</p>
            <p>{localized(asOf, 'spend_basis', locale)}</p>
          </>
        ) : asOf.figures_basis ? (
          <p className="pacing-source-quote">
            {pick(locale, 'The ledger states its figures: ', 'ספר האספקה אומר מהם נתוניו: ')}
            <q lang="en"><Name>{asOf.figures_basis}</Name></q>
          </p>
        ) : null}
        {/* The instant's own basis has no such pair anywhere in the product, so
            it stays the source's words, quoted and marked as the one language
            they were written in rather than paraphrased into the other. */}
        {asOf.basis ? (
          <p className="pacing-source-quote">
            {pick(locale, 'The ledger dates itself, in English only: ', 'ספר האספקה מתארך את עצמו, באנגלית בלבד: ')}
            <q lang="en"><Name>{asOf.basis}</Name></q>
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
  onOpenCampaign = null,
  focusCampaignId = '',
  onFocused = () => {},
}) {
  const [filter, setFilter] = useState('');
  const [expanded, setExpanded] = useState('');
  const [focused, setFocused] = useState(0);
  const [drills, setDrills] = useState({});
  const listRef = useRef(null);
  // Whether the last move of the mark came from a keystroke, which is the only
  // move that may take focus. Held on a ref rather than in state because it is
  // read by the effect that answers the move and is not a thing to render.
  const stepped = useRef(false);

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
        // The rules ride the same read as the days they explain, so the drill
        // never has to state a count whose cause it cannot name.
        [campaignId]: { status: 'ready', days: body.days || [], rules: body.booking_rules || {} },
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
      stepped.current = true;
      setFocused((current) => Math.min(rows.length - 1, current + 1));
    } else if (event.key === 'k' || event.key === 'ArrowUp') {
      event.preventDefault();
      stepped.current = true;
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

  // The step lands on the row rather than only colouring it.
  //
  // aria-current already said which row the keyboard was on, and a reader
  // arriving at that row heard it. What nothing said was that the keyboard had
  // moved, because focus stayed on the list while a ring moved inside it, so a
  // screen reader announced nothing between row 1 and row 56. This is the roving
  // tabindex: the focused row is the one element in the list that can take
  // focus, and a step moves focus onto it. keydown bubbles, so the list keeps
  // the handler it owns and every key still fires.
  //
  // Only a step moves focus. The same effect runs when the place marker returns
  // a reader to a row and when a filter resets the index, and taking focus on a
  // mount would steal it from wherever the reader was standing.
  useEffect(() => {
    const node = listRef.current;
    if (!node) return;
    const card = node.children[focused];
    if (!card) return;
    if (stepped.current) {
      stepped.current = false;
      card.focus({ preventScroll: true });
    }
    card.scrollIntoView({ block: 'nearest' });
  }, [focused, rows.length]);

  // The legend names the keys that do something on the rows in front of the
  // reader, and no others. Measured on the shipped data, 0 of 56 rows reach a
  // raise, so a fixed legend advertised a key that could not fire on any row on
  // the board. A shortcut nobody can press is a claim about a capability, and
  // this piece states capability from what the rows carry rather than from what
  // the code can do in principle.
  //
  // And it names what every one of those keys needs first. Measured: pressing j
  // with focus on the body did nothing at all, and only after focusing the list
  // did the marker move from row 0 to row 1. The legend read as a claim that the
  // keys worked wherever the reader was standing. The list carries the focus ring
  // this sheet already gives it, the legend now says so, and the list points at
  // the legend through aria-describedby so a reader who never sees it is told.
  const keys = [pick(locale, 'with this list focused, j and k step', 'כשהרשימה הזו במיקוד, j ו-k מדלגים')];
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
      </div>

      {/* The legend sits against the list it is about. Measured on the shipped
          board at 1512 px, it floated at the far end of the chrome row about
          440 px from the first row, opposite the basis prose, and read as an
          unattached caption. It is the same element with the same id, so the
          list still points at it. */}
      <p className="pacing-keys" id="pacing-keys">{`${keys.join(', ')}.`}</p>

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
        aria-describedby="pacing-keys"
        aria-label={pick(locale, 'Campaigns, worst pacing first', 'קמפיינים, החמור בקצב ראשון')}
      >
        {/* The keyboard mark is a fact about the list and not only a colour.
            Measured before this was written: j and k moved a 2 px ring drawn by
            pacing-row.css and the row carried no aria-current, no id and no
            tabindex, so a reader who cannot see the ring was stepping through a
            list that never said where they were. aria-current is valid on a
            listitem and changes nothing about focus, which stays on the list
            that owns the key handler. */}
        {rows.map((row, index) => (
          <div
            role="listitem"
            key={row.campaign_id}
            aria-current={index === focused ? 'true' : undefined}
            tabIndex={index === focused ? -1 : undefined}
            className={index === focused ? 'pacing-focused' : ''}
          >
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
              onOpenCampaign={onOpenCampaign}
              onRetryDays={() => openDays(row.campaign_id)}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
