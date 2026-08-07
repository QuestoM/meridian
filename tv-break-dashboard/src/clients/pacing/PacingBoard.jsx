import React, { useCallback, useEffect, useRef, useState } from 'react';
import PacingRow from './PacingRow';
import { VERDICT_ORDER, isolate, localized, pick, remedyFor, vocabularyLabel } from './pacing-helpers';

// The board an account manager opens in the morning.
//
// It is sorted worst first by the server, so the first row is the one to act on
// and the answer to "what needs me today" costs zero clicks. The strip above the
// list states the same counts the list is ordered by, and pressing one filters to
// it rather than sorting it, so the order never changes under the reader.
//
// Keyboard control is the whole list: j and k step, Enter opens the days behind a
// row, and r raises the make-good the focused row's remedy names. Nothing here
// needs a dialog, because none of it is a destructive act.

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
          <span className="pacing-chip-count" dir="ltr">{counts[verdict] || 0}</span>
          {vocabularyLabel(vocabulary.pace_verdicts, verdict, locale)}
        </button>
      ))}
    </div>
  );
}

function Basis({ payload, locale }) {
  const asOf = payload.as_of || {};
  const trigger = payload.trigger || {};
  return (
    <div className="pacing-basis">
      <p>
        {pick(locale, `Counted through ${asOf.instant}.`, `נספר עד ${isolate(asOf.instant)}.`)}
      </p>
      {asOf.basis ? (
        // The delivery ledger writes this sentence in one language only, so it is
        // quoted as the source's own words rather than presented as this surface's
        // copy. A reader is told where the instant came from either way.
        <p className="pacing-source-quote">
          {pick(locale, 'The ledger states: ', 'ספר האספקה אומר: ')}
          <q lang="en" dir="ltr">{asOf.basis}</q>
        </p>
      ) : null}
      <p>{localized(payload, 'counted_basis', locale)}</p>
      <p>
        {localized(trigger, 'rule', locale)}
        {' '}
        {localized(trigger, 'not_a_commercial_term', locale)}
      </p>
      {payload.scope && payload.scope.scope_channel ? (
        <p className="pacing-scope-note">
          {pick(
            locale,
            `This board covers ${payload.scope.scope_channel} and no other channel.`,
            `הלוח הזה מכסה את ${isolate(payload.scope.scope_channel)} ולא ערוץ אחר.`,
          )}
        </p>
      ) : null}
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
  onOpenMakeGood,
}) {
  const [filter, setFilter] = useState('');
  const [expanded, setExpanded] = useState('');
  const [focused, setFocused] = useState(0);
  const listRef = useRef(null);

  const rows = (payload.rows || []).filter((row) => !filter || row.headline.verdict === filter);
  const counts = payload.counts || {};
  const vocabulary = payload.vocabulary || {};

  useEffect(() => {
    setFocused(0);
  }, [filter]);

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
      if (row) setExpanded((current) => (current === row.campaign_id ? '' : row.campaign_id));
    } else if (event.key === 'r') {
      const row = rows[focused];
      const remedy = row ? remedyFor(row, payload.make_goods) : null;
      if (row && remedy && remedy.kind === 'raise' && canEdit) {
        event.preventDefault();
        onRaise(row);
      }
    }
  }, [rows, focused, payload.make_goods, canEdit, onRaise]);

  useEffect(() => {
    const node = listRef.current;
    if (!node) return;
    const cards = node.querySelectorAll('.pacing-row');
    const card = cards[focused];
    if (card) card.scrollIntoView({ block: 'nearest' });
  }, [focused, rows.length]);

  return (
    <section className="pacing-board" aria-label={pick(locale, 'Campaign pacing', 'קצב הקמפיינים')}>
      <Strip counts={counts} active={filter} vocabulary={vocabulary} locale={locale} onPick={setFilter} />
      <Basis payload={payload} locale={locale} />

      <p className="pacing-keys">
        {pick(
          locale,
          'j and k step the list, Enter opens the days behind a row, r raises the make-good the row names.',
          'j ו-k מדלגים ברשימה, Enter פותח את ימי השידור שמאחורי השורה, r פותח את פיצוי השידור שהשורה נוקבת בו.',
        )}
      </p>

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
              expanded={expanded === row.campaign_id}
              busy={busyId === row.campaign_id}
              canEdit={canEdit}
              editRefusal={editRefusal}
              onToggle={() => setExpanded((current) => (current === row.campaign_id ? '' : row.campaign_id))}
              onRaise={() => onRaise(row)}
              onOpenMakeGood={onOpenMakeGood}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
