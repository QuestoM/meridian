import React from 'react';
import { Card, CardBody, EmptyState, Status } from '../studio';
import { Button } from '../studio/actions';
import { FilePlus2, Scale } from 'lucide-react';
import { Code, Prose } from '../shell/bidi';
import { formatNumber, pageText } from '../shell/format';
import { mechanismName, mechanismTone } from './trade-vocabulary';
import { REJECTED_GROUP } from './review-model';
import ReviewTermCard from './ReviewTermCard';

// The engine's proposal: every term it would act on, grouped by what it would DO.
//
// Split out of AgreementReviewScreen.jsx to keep that file inside the project's
// file-size law. It holds no state and performs no read; the screen keeps every
// decision and every server call, and this module renders them.
//
// TWO ORDERING DECISIONS carry the argument of the screen.
//
// Terms are grouped by MECHANISM rather than by contract family, and the group
// that will not act comes first. A reviewer reading a fifty-clause framework in a
// deal meeting is not asking "what family is this clause in"; they are asking
// "what is this going to do to my inventory and my money", and the one answer
// that costs money silently is "nothing".
//
// A clause the reading could not map to any term is a BLOCKING SECTION, not a
// footnote at the bottom. It is the only place on this surface where the machine
// admits it did not understand something, and it stays loud until a person takes
// ownership of it.

const FILTERS = [
  { key: 'all', en: 'All terms', he: 'כל המונחים' },
  { key: 'undecided', en: 'Awaiting a decision', he: 'ממתינים להחלטה' },
  { key: 'inert', en: 'Will not act', he: 'לא יפעלו' },
  { key: 'incomplete', en: 'Missing fields', he: 'שדות חסרים' },
  { key: 'conflicts', en: 'Open conflicts', he: 'סתירות פתוחות' },
];

function UnmappedClauses({ clauses, locale, canEdit, onAcknowledge }) {
  return (
    <Card className="trd-unmapped">
      <CardBody>
        <h4>
          {pageText(
            locale,
            `${formatNumber(clauses.length, locale)} clauses map to no term at all`,
            `${formatNumber(clauses.length, locale)} סעיפים אינם ממופים לשום מונח`,
          )}
        </h4>
        <p className="trd-field-hint">
          {pageText(
            locale,
            'The reading did not recognise these. Each one blocks approval until a person says what it is and takes ownership of it.',
            'הקריאה לא זיהתה אותם. כל אחד מהם חוסם את האישור עד שאדם יאמר מהו וייקח עליו אחריות.',
          )}
        </p>
        <ul className="trd-unmapped-list">
          {clauses.map((clause) => (
            <li key={clause.clause_id}>
              <Code className="trd-id-chip">{clause.clause_id}</Code>
              <Prose as="span" className="trd-unmapped-text">{clause.text}</Prose>
              {canEdit ? (
                <Button type="button" variant="outlined" onClick={() => onAcknowledge(clause)}>
                  {pageText(locale, 'Acknowledge it', 'אישור ידני')}
                </Button>
              ) : null}
            </li>
          ))}
        </ul>
      </CardBody>
    </Card>
  );
}

export default function ReviewProposalPane({
  locale, canEdit, busy, filter, counts, groups, shown, conflicts, unacknowledged,
  selectedClause, clauseTermIds, onFilter, onSelectClause, onConfirm, on,
}) {
  const openConflicts = conflicts.filter((conflict) => conflict.open).length;
  return (
    <div className="trd-proposal">
      <div className="trd-pane-head">
        <h3>{pageText(locale, 'What the engine proposes to do', 'מה שהמנוע מציע לעשות')}</h3>
        {canEdit ? (
          <Button type="button" variant="outlined" onClick={on.add}>
            <FilePlus2 size={14} aria-hidden="true" />
            {pageText(locale, 'Add a missed term', 'הוספת מונח שהוחמץ')}
          </Button>
        ) : null}
      </div>

      <div
        className="trd-filters"
        role="group"
        aria-label={pageText(locale, 'Filter the terms', 'סינון המונחים')}
      >
        {FILTERS.map((entry) => (
          <Button
            key={entry.key}
            type="button"
            className={filter === entry.key ? 'trd-filter active' : 'trd-filter'}
            aria-pressed={filter === entry.key}
            onClick={() => onFilter(entry.key)}
          >
            {pageText(locale, entry.en, entry.he)}
            <span className="trd-filter-count">{formatNumber(counts[entry.key] || 0, locale)}</span>
          </Button>
        ))}
      </div>

      {selectedClause ? (
        <p className="trd-selected-note" role="status">
          <Code>{selectedClause}</Code>
          {clauseTermIds.size > 0
            ? pageText(
              locale,
              `is the evidence for ${formatNumber(clauseTermIds.size, locale)} terms, marked below.`,
              `הוא האסמכתא ל־${formatNumber(clauseTermIds.size, locale)} מונחים, המסומנים למטה.`,
            )
            : pageText(locale, 'is evidence for no term.', 'אינו אסמכתא לאף מונח.')}
        </p>
      ) : null}

      {unacknowledged.length > 0 ? (
        <UnmappedClauses
          clauses={unacknowledged}
          locale={locale}
          canEdit={canEdit}
          onAcknowledge={on.acknowledge}
        />
      ) : null}

      {shown.length === 0 ? (
        <EmptyState
          title={pageText(locale, 'Nothing under this filter', 'אין דבר בסינון הזה')}
          description={filter === 'undecided'
            ? pageText(
              locale,
              'Every proposed term has been decided. What remains for approval, if anything, is named in the band above.',
              'כל מונח מוצע הוכרע. מה שנותר לאישור, אם נותר, מפורט ברצועה שלמעלה.',
            )
            : pageText(
              locale,
              'Choose another filter to see the rest of the agreement.',
              'בחרו סינון אחר כדי לראות את שאר ההסכם.',
            )}
          action={filter === 'all' ? null : (
            <Button type="button" variant="outlined" onClick={() => onFilter('all')}>
              {pageText(locale, 'Show every term', 'הצגת כל המונחים')}
            </Button>
          )}
        />
      ) : null}

      {groups.map((group) => (
        <section key={group.key} className="trd-group">
          <h4 className="trd-group-head">
            <Status status={group.key === REJECTED_GROUP ? 'danger' : mechanismTone(group.key)}>
              {group.key === REJECTED_GROUP
                ? pageText(locale, 'Rejected', 'נדחו')
                : mechanismName(group.key, locale)}
            </Status>
            <span className="trd-group-count">{formatNumber(group.terms.length, locale)}</span>
          </h4>
          {group.terms.map((term) => (
            <ReviewTermCard
              key={term.instance_id}
              term={term}
              locale={locale}
              canEdit={canEdit}
              busy={busy}
              highlighted={clauseTermIds.has(term.instance_id)}
              onJumpToClause={onSelectClause}
              onConfirm={onConfirm}
              onEdit={on.edit}
              onReject={on.reject}
              onResolveConflict={on.resolveConflict}
            />
          ))}
        </section>
      ))}

      {conflicts.length > 0 ? (
        <Card dense className="trd-conflict-summary">
          <CardBody>
            <Scale size={15} aria-hidden="true" />
            <p>
              {pageText(
                locale,
                `This agreement contains ${formatNumber(conflicts.length, locale)} contradictions between its own clauses; ${formatNumber(openConflicts, locale)} are still open.`,
                `ההסכם הזה מכיל ${formatNumber(conflicts.length, locale)} סתירות בין סעיפיו; ${formatNumber(openConflicts, locale)} מהן פתוחות.`,
              )}
            </p>
          </CardBody>
        </Card>
      ) : null}
    </div>
  );
}
