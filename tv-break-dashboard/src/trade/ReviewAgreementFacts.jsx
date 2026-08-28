import React from 'react';
import { Name } from '../shell/bidi';
import { pageText } from '../shell/format';
import { openEndedLabel, windowOf } from './trade-vocabulary';

// The two questions a trader asks first about any agreement: until when does
// it hold, and does it bind the engine yet. Both were answerable only on the
// record screen, one navigation away from the review screen the product itself
// promotes. Measured on a first-run study: the reviewer opened the only link
// the agreement card offers, landed on the review, and found neither - so the
// validity period read as missing from the product rather than as living
// elsewhere.
//
// An agreement with no approved version is the sharper of the two: it can name
// a fourteen-million commitment and an exclusivity and still change nothing,
// and that is a headline rather than a footnote.
export default function ReviewAgreementFacts({ head, versions, locale }) {
  if (!head) return null;
  return (
    <>
      <p className="trd-review-window">
        <span>{pageText(locale, 'Effective window', 'תקופת תוקף')}: </span>
        <Name>{windowOf(head.window) || openEndedLabel(locale)}</Name>
      </p>
      {(versions || []).length === 0 ? (
        <p className="trd-review-binding">
          {pageText(
            locale,
            'No version has been approved, so this agreement changes nothing yet.',
            'לא אושרה גרסה, ולכן ההסכם הזה אינו משנה דבר בשלב הזה.',
          )}
        </p>
      ) : null}
    </>
  );
}
