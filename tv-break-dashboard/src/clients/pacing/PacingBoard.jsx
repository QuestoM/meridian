import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../../studio/actions';
import { Code, Figure, Name } from '../../shell/bidi';
import PacingRow from './PacingRow';
import { loadDays } from './pacing-api';
import { VERDICT_ORDER, acceptanceFor, headlineLine, instant, isolate, localized, pick, remedyFor, vocabularyLabel } from './pacing-helpers';

const PACING_WINDOW = 16;

// A real commitment curve, derived only from the ratios already published on
// the pacing rows. The server orders the board by operational severity, so the
// x axis keeps that order and the 100% reference stays fixed. Nothing here
// invents a goal, revenue figure or forecast.
function CommitmentCurve({ rows, locale, vocabulary }) {
  const samples = useMemo(() => rows.map((row) => {
    const line = headlineLine(row);
    const ratio = line && line.pace && Number.isFinite(Number(line.pace.ratio))
      ? Number(line.pace.ratio)
      : null;
    return {
      id: row.campaign_id,
      name: row.name || row.campaign_id,
      ratio,
      verdict: row.headline && row.headline.verdict ? row.headline.verdict : 'unknown',
    };
  }).filter((sample) => sample.ratio !== null), [rows]);

  if (!samples.length) return null;

  const width = 1000;
  const height = 164;
  const top = 16;
  const bottom = 22;
  const plot = height - top - bottom;
  const maxRatio = Math.max(1.2, ...samples.map((sample) => sample.ratio));
  const y = (ratio) => top + (1 - Math.min(maxRatio, Math.max(0, ratio)) / maxRatio) * plot;
  const x = (index) => samples.length === 1 ? width / 2 : 10 + (index / (samples.length - 1)) * (width - 20);
  const points = samples.map((sample, index) => `${x(index)},${y(sample.ratio)}`).join(' ');
  const ordered = samples.map((sample) => sample.ratio).sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  const median = ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
  const decisionCount = samples.filter((sample) => sample.verdict === 'behind' || sample.verdict === 'at_risk').length;
  const referenceY = y(1);

  return (
    <section className="card card-dense commitment-curve" aria-labelledby="commitment-curve-title">
      <header className="commitment-curve-head">
        <div>
          <Code className="commercial-module-code">PACE / COMMITMENT</Code>
          <h3 id="commitment-curve-title">{pick(locale, 'Pace against commitment', 'קצב מול התחייבות')}</h3>
          <p>{pick(locale, 'Each point is one campaign’s counted pace against its own published reference.', 'כל נקודה היא קצב שנספר לקמפיין אחד מול הייחוס שפורסם עבורו.')}</p>
        </div>
        <dl>
          <div><dt>{pick(locale, 'Known pace', 'קצב ידוע')}</dt><dd><Figure>{samples.length}</Figure></dd></div>
          <div><dt>{pick(locale, 'Need a decision', 'דורשים החלטה')}</dt><dd><Figure>{decisionCount}</Figure></dd></div>
          <div><dt>{pick(locale, 'Median pace', 'חציון קצב')}</dt><dd><Figure>{`${Math.floor(median * 1000) / 10}%`}</Figure></dd></div>
        </dl>
      </header>

      <div className="commitment-plot">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={pick(locale, 'Campaign pace ratios in the order published by the server, with the 100 percent reference marked', 'יחסי קצב הקמפיינים בסדר שפרסם השרת, עם סימון ייחוס של מאה אחוז')}>
          <rect className="commitment-band" x="0" y="0" width={width} height={height} />
          <line className="commitment-reference" x1="0" x2={width} y1={referenceY} y2={referenceY} />
          <polyline className="commitment-line" points={points} />
          {samples.map((sample, index) => (
            <circle key={sample.id} className={`commitment-point ${sample.verdict}`} cx={x(index)} cy={y(sample.ratio)} r="4">
              <title>{`${sample.name}: ${Math.floor(sample.ratio * 1000) / 10}%`}</title>
            </circle>
          ))}
        </svg>
        <span className="commitment-reference-label"><Figure>100%</Figure> {pick(locale, 'reference', 'ייחוס')}</span>
      </div>

      <div className="pacing-burn" role="img" aria-label={pick(locale, 'One segment per known campaign, coloured by its published verdict', 'מקטע אחד לכל קמפיין ידוע, בצבע של מצב הקצב שפורסם')}>
        {samples.map((sample) => (
          <i key={sample.id} className={sample.verdict} title={`${sample.name}: ${Math.floor(sample.ratio * 1000) / 10}%`} />
        ))}
      </div>
      <footer className="commitment-legend">
        {VERDICT_ORDER.map((verdict) => (
          <span key={verdict} className={verdict}>{vocabularyLabel(vocabulary.pace_verdicts, verdict, locale)}</span>
        ))}
      </footer>
    </section>
  );
}

// The server keeps the list worst-first. Filters preserve that order, and the
// keyboard operates the same progressively disclosed list.

function Strip({ counts, active, vocabulary, locale, onPick }) {
  return (
    <div className="pacing-strip" role="group" aria-label={pick(locale, 'Filter by verdict', 'סינון לפי מצב')}>
      {VERDICT_ORDER.map((verdict) => (
        <Button
          key={verdict}
          type="button"
          className={`pacing-chip ${verdict} ${active === verdict ? 'active' : ''}`}
          aria-pressed={active === verdict}
          onClick={() => onPick(active === verdict ? '' : verdict)}
        >
          <Figure className="pacing-chip-count">{counts[verdict] || 0}</Figure>
          {vocabularyLabel(vocabulary.pace_verdicts, verdict, locale)}
        </Button>
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
      <p className="pacing-basis-planned">
        {localized(trigger, 'rule', locale)}
        {' '}
        {localized(trigger, 'not_a_commercial_term', locale)}
      </p>
      <details className="pacing-basis-details">
        <summary>{pick(locale, 'How this is counted', 'איך זה נספר')}</summary>
        <p>{localized(payload, 'counted_basis', locale)}</p>
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
  const [visibleCount, setVisibleCount] = useState(PACING_WINDOW);
  const listRef = useRef(null);
  // Whether the last move of the mark came from a keystroke, which is the only
  // move that may take focus. Held on a ref rather than in state because it is
  // read by the effect that answers the move and is not a thing to render.
  const stepped = useRef(false);

  const allRows = (payload.rows || []).filter((row) => !filter || row.headline.verdict === filter);
  const rows = allRows.slice(0, visibleCount);
  const counts = payload.counts || {};
  const vocabulary = payload.vocabulary || {};
  const countedAt = (payload.as_of || {}).instant || '';

  useEffect(() => {
    setFocused(0);
    setVisibleCount(PACING_WINDOW);
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
    setVisibleCount(Math.max(PACING_WINDOW, index + 1));
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
    const row = (payload.rows || []).find((entry) => entry.campaign_id === campaignId);
    if (row && row.days_available && (!drills[campaignId] || drills[campaignId].status === 'failed')) openDays(campaignId);
  }, [payload.rows, drills, openDays]);

  const onKeyDown = useCallback((event) => {
    if (!rows.length) return;
    const fromControl = event.target.closest?.('button, a, input, select, textarea, summary');
    if (event.key === 'j' || event.key === 'ArrowDown') {
      event.preventDefault();
      stepped.current = true;
      setFocused((current) => {
        const next = Math.min(allRows.length - 1, current + 1);
        if (next >= rows.length) setVisibleCount((count) => Math.min(allRows.length, count + PACING_WINDOW));
        return next;
      });
    } else if (event.key === 'k' || event.key === 'ArrowUp') {
      event.preventDefault();
      stepped.current = true;
      setFocused((current) => Math.max(0, current - 1));
    } else if (event.key === 'Enter' && !fromControl) {
      event.preventDefault();
      const row = rows[focused];
      if (row) toggle(row.campaign_id);
    } else if (event.key === 'r' && !fromControl) {
      const row = rows[focused];
      const remedy = row ? remedyFor(row, payload.make_goods) : null;
      if (row && remedy && remedy.kind === 'raise' && canEdit) {
        event.preventDefault();
        onRaise(row);
      }
    } else if (event.key === 'a' && !fromControl) {
      const row = rows[focused];
      const acceptance = row
        ? acceptanceFor(row, payload.acceptances, payload.needs_a_decision)
        : null;
      if (row && acceptance && acceptance.kind === 'accept' && canEdit) {
        event.preventDefault();
        onAccept(row);
      }
    }
  }, [rows, allRows.length, focused, payload.make_goods, payload.acceptances, payload.needs_a_decision, canEdit, onRaise, onAccept, toggle]);

  // Keyboard stepping moves focus to the marked row; filtering and mount do not.
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

  // Advertise only shortcuts the disclosed rows can currently perform.
  const keys = [pick(locale, 'with this list focused, j and k step', 'כשהרשימה הזו במיקוד, j ו-k מדלגים')];
  if (allRows.some((row) => row.days_available)) {
    keys.push(pick(locale, 'Enter opens the broadcast days', 'Enter פותח את ימי השידור'));
  }
  if (canEdit && allRows.some((row) => remedyFor(row, payload.make_goods).kind === 'raise')) {
    keys.push(pick(locale, 'r raises the make-good a row names', 'r פותח את פיצוי השידור שהשורה נוקבת בו'));
  }
  if (canEdit && allRows.some((row) => acceptanceFor(row, payload.acceptances, payload.needs_a_decision).kind === 'accept')) {
    keys.push(pick(locale, 'a takes the risk on', 'a מקבל את הסיכון'));
  }

  return (
    <section className="pacing-board" aria-label={pick(locale, 'Campaign pacing', 'קצב הקמפיינים')}>
      <CommitmentCurve rows={payload.rows || []} locale={locale} vocabulary={vocabulary} />
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
        id="pacing-campaign-list"
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
            aria-posinset={index + 1}
            aria-setsize={allRows.length}
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
      {rows.length < allRows.length ? (
        <div className="clients-window-more" role="status">
          <span>{pick(locale, `Showing ${rows.length} of ${allRows.length} campaigns`, `מוצגים ${rows.length} מתוך ${allRows.length} קמפיינים`)}</span>
          <a href="#pacing-campaign-list" role="button" className="clients-secondary"
             onClick={(event) => { event.preventDefault(); setVisibleCount((count) => count + PACING_WINDOW); }}
             onKeyDown={(event) => { if (event.key === ' ') { event.preventDefault(); setVisibleCount((count) => count + PACING_WINDOW); } }}>
            {pick(locale, 'Show the next campaigns', 'הציגו את הקמפיינים הבאים')}
          </a>
        </div>
      ) : null}
    </section>
  );
}
