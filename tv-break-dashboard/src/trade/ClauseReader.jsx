import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { Pressable } from '../studio/dom-controls';
import { Code, Prose } from '../shell/bidi';
import { EmptyState, Status } from '../studio';
import { Check, Eye } from 'lucide-react';
import { formatNumber, pageText } from '../shell/format';
import { irrelevantClassName } from './trade-terms';

// The clauses of the document, as the pipeline segmented them, with what became
// of each one.
//
// EVERY CLAUSE IS LISTED, including the ones that produced nothing. That is the
// completeness contract: a reviewer can only certify that the whole document was
// read if the whole document is on screen. A clause classified as page furniture
// or a signature block is shown with that classification, so "irrelevant" is a
// visible decision somebody can disagree with rather than a silent omission.
//
// THE QUOTE IS MARKED INSIDE THE CLAUSE. A citation names a page and a verbatim
// quote; the quote is found in the clause's own text and marked, so a reviewer
// checking a rule sees the words that produced it rather than a page number they
// have to go hunting through. Marks never overlap: the longest quote at a given
// position wins, and the scan continues after it.

const CLAUSE_WINDOW = 20;

// Marks each quote inside the clause text without letting two marks overlap.
// Quotes are sorted longest first at each position so a citation quoting a whole
// sentence is not shredded by a shorter citation quoting three of its words.
function segmentWithQuotes(text, quotes) {
  const wanted = (quotes || [])
    .map((quote) => String(quote || ''))
    .filter((quote) => quote.length > 2 && text.includes(quote));
  if (wanted.length === 0) return [{ text, marked: false }];
  const ordered = [...wanted].sort((a, b) => b.length - a.length);
  const segments = [];
  let cursor = 0;
  while (cursor < text.length) {
    let hit = null;
    for (const quote of ordered) {
      const at = text.indexOf(quote, cursor);
      if (at === -1) continue;
      if (!hit || at < hit.at || (at === hit.at && quote.length > hit.quote.length)) {
        hit = { at, quote };
      }
    }
    if (!hit) {
      segments.push({ text: text.slice(cursor), marked: false });
      break;
    }
    if (hit.at > cursor) segments.push({ text: text.slice(cursor, hit.at), marked: false });
    segments.push({ text: hit.quote, marked: true });
    cursor = hit.at + hit.quote.length;
  }
  return segments;
}

function dispositionStatus(disposition, locale) {
  const kind = disposition ? disposition.disposition : null;
  if (kind === 'mapped') {
    const count = (disposition.instance_ids || []).length;
    return {
      tone: 'positive',
      label: pageText(
        locale,
        count === 1 ? '1 term' : `${count} terms`,
        count === 1 ? 'מונח אחד' : `${formatNumber(count, locale)} מונחים`,
      ),
    };
  }
  if (kind === 'irrelevant') {
    return {
      tone: 'neutral',
      label: irrelevantClassName(disposition.irrelevant_class, locale),
    };
  }
  if (kind === 'unmapped') {
    return {
      tone: 'warning',
      label: pageText(locale, 'commercial, no term fits', 'מסחרי, אין מונח מתאים'),
    };
  }
  return { tone: 'warning', label: pageText(locale, 'no disposition', 'ללא סיווג') };
}

export default function ClauseReader({
  clauses,
  dispositions,
  seen,
  selectedClauseId,
  highlightQuotes,
  onSelectClause,
  onMarkSeen,
  onAcknowledge,
  locale,
  canEdit,
  refusalReason,
}) {
  const [visibleCount, setVisibleCount] = useState(CLAUSE_WINDOW);
  const selectedRef = useRef(null);
  const rows = Array.isArray(clauses) ? clauses : [];
  const byClause = useMemo(() => {
    const index = {};
    (dispositions || []).forEach((entry) => { index[entry.clause_id] = entry; });
    return index;
  }, [dispositions]);

  // A clause selected from the proposal side scrolls itself into view, and the
  // window grows first if the clause sits past the current one: a citation chip
  // that jumps to nothing has not jumped.
  const selectedIndex = rows.findIndex((entry) => entry.clause_id === selectedClauseId);
  useEffect(() => {
    if (selectedIndex >= visibleCount) {
      setVisibleCount(Math.ceil((selectedIndex + 1) / CLAUSE_WINDOW) * CLAUSE_WINDOW);
    }
  }, [selectedIndex, visibleCount]);
  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [selectedClauseId, visibleCount]);

  if (rows.length === 0) {
    return (
      <EmptyState
        title={pageText(locale, 'No clauses were segmented', 'לא פולחו סעיפים')}
        description={pageText(
          locale,
          'The document has no proposal yet, or the extraction produced no clause boundaries. Neither is an empty document.',
          'למסמך אין עדיין הצעה, או שהחילוץ לא הפיק גבולות סעיפים. אף אחד מהשניים אינו מסמך ריק.',
        )}
      />
    );
  }

  const windowed = rows.slice(0, visibleCount);
  const unseenVisible = windowed
    .filter((entry) => !seen[entry.clause_id])
    .map((entry) => entry.clause_id);

  return (
    <div className="trade-clauses">
      <div className="trade-clauses__bar">
        <p className="trade-clauses__count" role="status">
          {pageText(
            locale,
            `${windowed.length} of ${rows.length} clauses, ${Object.keys(seen).length} marked read`,
            `${formatNumber(windowed.length, locale)} מתוך ${formatNumber(rows.length, locale)} סעיפים, ${formatNumber(Object.keys(seen).length, locale)} סומנו כנקראו`,
          )}
        </p>
        {canEdit && unseenVisible.length > 0 ? (
          <Button type="button" variant="outlined" className="trade-secondary" onClick={() => onMarkSeen(unseenVisible)}>
            <Eye size={14} aria-hidden="true" />
            {pageText(
              locale,
              `Mark these ${unseenVisible.length} as read`,
              `סמנו ${formatNumber(unseenVisible.length, locale)} אלה כנקראו`,
            )}
          </Button>
        ) : null}
      </div>

      <ol className="trade-clauses__list">
        {windowed.map((clause) => {
          const disposition = byClause[clause.clause_id];
          const status = dispositionStatus(disposition, locale);
          const isSelected = clause.clause_id === selectedClauseId;
          const wasSeen = Boolean(seen[clause.clause_id]);
          const quotes = isSelected ? highlightQuotes : [];
          const segments = segmentWithQuotes(String(clause.text || ''), quotes);
          return (
            <li
              key={clause.clause_id}
              className={`trade-clause${isSelected ? ' is-selected' : ''}${wasSeen ? ' is-seen' : ''}`}
              ref={isSelected ? selectedRef : null}
            >
              <Pressable
                className="trade-clause__head"
                aria-pressed={isSelected}
                aria-label={pageText(
                  locale,
                  `Clause ${clause.clause_id}, ${status.label}`,
                  `סעיף ${clause.clause_id}, ${status.label}`,
                )}
                onClick={() => onSelectClause(isSelected ? '' : clause.clause_id)}
              >
                <span className="trade-clause__id"><Code>{clause.clause_id}</Code></span>
                <Status status={status.tone} className="trade-chip">{status.label}</Status>
                {clause.is_table ? (
                  <span className="trade-clause__flag">{pageText(locale, 'table', 'טבלה')}</span>
                ) : null}
                <span className="trade-clause__page">
                  {pageText(
                    locale,
                    `page ${formatNumber((clause.pages || [])[0], locale)}`,
                    `עמוד ${formatNumber((clause.pages || [])[0], locale)}`,
                  )}
                </span>
                {wasSeen ? (
                  <span className="trade-clause__seen">
                    <Check size={13} aria-hidden="true" />
                    {pageText(locale, 'read', 'נקרא')}
                  </span>
                ) : (
                  <span className="trade-clause__unseen">{pageText(locale, 'not read yet', 'לא נקרא עדיין')}</span>
                )}
              </Pressable>

              <Prose className="trade-clause__text">
                {segments.map((segment, index) => (segment.marked
                  ? <mark key={index} className="trade-quote">{segment.text}</mark>
                  : <React.Fragment key={index}>{segment.text}</React.Fragment>))}
              </Prose>

              {disposition && disposition.disposition === 'unmapped' ? (
                <div className="trade-clause__unmapped">
                  <p>
                    {pageText(
                      locale,
                      'This clause is commercial and no term in the taxonomy represents it. It cannot be turned into a rule, and approval is blocked until somebody takes ownership of it in writing.',
                      'הסעיף מסחרי ואין בטקסונומיה מונח שמייצג אותו. אי אפשר להפוך אותו לכלל, והאישור חסום עד שמישהו ייקח עליו אחריות בכתב.',
                    )}
                  </p>
                  {disposition.reason ? <Prose className="trade-clause__reason">{disposition.reason}</Prose> : null}
                  {canEdit ? (
                    <Button type="button" variant="outlined" className="trade-secondary" onClick={() => onAcknowledge(clause.clause_id)}>
                      {pageText(locale, 'Acknowledge it with a note', 'קבלו אחריות בהערה')}
                    </Button>
                  ) : (
                    <p className="trade-refusal">{refusalReason}</p>
                  )}
                </div>
              ) : null}

              {canEdit && !wasSeen ? (
                <Button type="button" variant="outlined" className="trade-secondary trade-clause__seen-act" onClick={() => onMarkSeen([clause.clause_id])}>
                  <Check size={14} aria-hidden="true" />
                  {pageText(locale, 'I have read this clause', 'קראתי את הסעיף הזה')}
                </Button>
              ) : null}
            </li>
          );
        })}
      </ol>

      {windowed.length < rows.length ? (
        <div className="trade-clauses__more">
          <Button type="button" variant="outlined" className="trade-secondary" onClick={() => setVisibleCount((count) => count + CLAUSE_WINDOW)}>
            {pageText(
              locale,
              `Show the next clauses (${rows.length - windowed.length} left)`,
              `הציגו את הסעיפים הבאים (נותרו ${formatNumber(rows.length - windowed.length, locale)})`,
            )}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
