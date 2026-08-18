import React, { useState } from 'react';
import { Button } from '../studio/actions';
import { Code, Name, Prose } from '../shell/bidi';
import { pageText } from '../shell/format';
import { ChevronDown, ChevronUp, Plus } from 'lucide-react';
import { termName } from './trade-terms';
import './trade-interpretations.css';

// Readings that carry the shape of a term and nothing in it.
//
// The extraction proposes both kinds under one name: most proposals carry
// values a reviewer can check against the clause, and some carry only the
// suggestion that a term of some sort lives there — a discount ladder whose
// only rung is 0% at a threshold of 0, a measurement source whose every field
// came back unknown. The second kind cannot be checked, because there is
// nothing in it to check, and putting it in the list a person approves line by
// line is what made reading an agreement heavy.
//
// So they live here, folded away, and they hold nothing shut. A reader who
// opens one and recognises a real term moves it into the proposals with one
// control, and from that moment it is an ordinary proposal that blocks approval
// until it is decided — because now a person has said it is real.
//
// The server decides which list a reading belongs in (kairos.trade.standing),
// so this pane and the approval gate cannot disagree.
export default function ReviewInterpretations({ terms, locale, canEdit, busy, onPromote }) {
  const [open, setOpen] = useState(false);
  if (!terms.length) {
    return null;
  }
  return (
    <section className="trd-interpretations">
      <Button
        type="button"
        variant="text"
        className="trd-interpretations-head"
        aria-expanded={open}
        onClick={() => setOpen((was) => !was)}
      >
        {open ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
        <span>
          {pageText(
            locale,
            `${terms.length} further readings, not proposed`,
            `${terms.length} קריאות נוספות, שאינן מוצעות`,
          )}
        </span>
      </Button>
      <Prose className="trd-interpretations-why">
        {pageText(
          locale,
          'Each of these named a clause and extracted no values from it, so there is nothing in them to check against the document. They do not hold approval. Open one and add it to the proposals if the clause really carries that term.',
          'כל אחת מהן נקבה בסעיף ולא חילצה ממנו ערכים, ולכן אין בהן מה להשוות מול המסמך. הן אינן עוצרות את האישור. פתחו אחת והוסיפו אותה להצעות אם הסעיף אכן נושא את התנאי הזה.',
        )}
      </Prose>
      {open ? (
        <ul className="trd-interpretations-list">
          {terms.map((term) => (
            <li key={term.instance_id}>
              <div className="trd-interpretation-head">
                <strong><Name>{termName(term.term_id, locale)}</Name></strong>
                {term.citations.length ? (
                  <Code className="trd-interpretation-clause">{term.citations[0].clause_id}</Code>
                ) : null}
              </div>
              {term.citations.length && term.citations[0].quote ? (
                <Prose className="trd-interpretation-quote">{term.citations[0].quote}</Prose>
              ) : null}
              <Prose className="trd-interpretation-why">
                {locale === 'he' ? term.standingReason.reason_he : term.standingReason.reason_en}
              </Prose>
              {canEdit ? (
                <Button
                  type="button"
                  className="trd-interpretation-promote"
                  disabled={busy}
                  onClick={() => onPromote(term)}
                >
                  <Plus size={12} aria-hidden="true" />
                  {pageText(locale, 'Add to the proposals', 'הוסיפו להצעות')}
                </Button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
