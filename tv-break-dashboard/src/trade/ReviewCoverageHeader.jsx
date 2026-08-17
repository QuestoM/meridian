import React from 'react';
import { Card, CardBody, Status } from '../studio';
import { Button } from '../studio/actions';
import { Ban, Gavel, ShieldCheck } from 'lucide-react';
import { Code, Figure } from '../shell/bidi';
import { formatNumber, pageText } from '../shell/format';
import { blockerSentence } from './trade-vocabulary';

// The coverage a reviewer is held to, and the reason approval is refused.
//
// This band is persistent on purpose. The one thing a reviewer must never be
// able to lose track of, at any scroll position of a fifty-clause agreement, is
// how much of the document has actually passed before their eyes — because the
// approval that follows is the moment the channel's own machinery starts acting
// on these clauses.
//
// THE REFUSAL NAMES ITSELF. A disabled approve button with no reason is a dead
// end, so every blocker the gate reports is printed as a sentence that says what
// to do about it, with the ids it names. `aria-describedby` ties the button to
// that list, so the reason reaches a screen reader and not only the screen.

// A count, or a count against the total it has to reach. The two are different
// facts and the second is the only one that can be complete, so "complete" is
// marked only where a total exists to be reached.
function CoverageFigure({ label, value, total, done }) {
  return (
    <div className="trd-cov" data-complete={done ? 'true' : undefined}>
      <span className="trd-cov-label">{label}</span>
      <span className="trd-cov-value">
        <Figure>{total === undefined ? value : `${value} / ${total}`}</Figure>
      </span>
    </div>
  );
}

export default function ReviewCoverageHeader({
  coverage, locale, canEdit, editRefusal, approving, onApprove,
}) {
  const listId = 'trd-blockers';
  return (
    <Card className="trd-coverage">
      <CardBody>
        <div className="trd-coverage-row">
          <div className="trd-coverage-figures">
            <CoverageFigure
              label={pageText(locale, 'Clauses read', 'סעיפים שנקראו')}
              value={formatNumber(coverage.clausesSeen, locale)}
              total={formatNumber(coverage.clausesTotal, locale)}
              done={coverage.clausesSeen >= coverage.clausesTotal}
            />
            <CoverageFigure
              label={pageText(locale, 'Terms decided', 'מונחים שהוכרעו')}
              value={formatNumber(coverage.instancesDecided, locale)}
              total={formatNumber(coverage.instancesTotal, locale)}
              done={coverage.instancesDecided >= coverage.instancesTotal}
            />
            <CoverageFigure
              label={pageText(locale, 'Mapped to a term', 'מופו למונח')}
              value={formatNumber(coverage.mapped, locale)}
            />
            <CoverageFigure
              label={pageText(locale, 'Not commercial', 'לא מסחריים')}
              value={formatNumber(coverage.irrelevant, locale)}
            />
            {/* An unmapped clause is only cleared by a person taking ownership
                of it, so the figure that matters is how many of them have been
                acknowledged, not how many exist. */}
            <CoverageFigure
              label={pageText(locale, 'Mapped to nothing, acknowledged', 'לא מופו, אושרו ידנית')}
              value={formatNumber(coverage.unmappedAcknowledged, locale)}
              total={formatNumber(coverage.unmapped, locale)}
              done={coverage.unmappedAcknowledged >= coverage.unmapped}
            />
            {coverage.reviewerAdded > 0 ? (
              <CoverageFigure
                label={pageText(locale, 'Added by the reviewer', 'נוספו בסקירה')}
                value={formatNumber(coverage.reviewerAdded, locale)}
              />
            ) : null}
          </div>

          <div className="trd-coverage-act">
            <Button
              type="button"
              onClick={onApprove}
              disabled={!coverage.ready || !canEdit || approving}
              aria-describedby={coverage.ready && canEdit ? undefined : listId}
            >
              <Gavel size={14} aria-hidden="true" />
              {approving
                ? pageText(locale, 'Approving', 'מאשר')
                : pageText(locale, 'Approve the agreement', 'אישור ההסכם')}
            </Button>
          </div>
        </div>

        {coverage.ready && canEdit ? (
          <p className="trd-gate-ready" role="status">
            <Status status="positive" icon={<ShieldCheck size={14} aria-hidden="true" />}>
              {pageText(
                locale,
                'Every clause has been read and every term decided. Approving writes an immutable version and binds what the engine can bind.',
                'כל סעיף נקרא וכל מונח הוכרע. אישור יכתוב גרסה שאינה ניתנת לשינוי ויחבר את מה שהמנוע יכול לחבר.',
              )}
            </Status>
          </p>
        ) : (
          <div className="trd-gate-blocked" id={listId}>
            <p className="trd-gate-lead">
              <Ban size={14} aria-hidden="true" />
              {canEdit
                ? pageText(locale, 'Approval is refused, and this is why:', 'האישור מסורב, וזאת הסיבה:')
                : (editRefusal || pageText(locale, 'This account may not approve an agreement.', 'לחשבון הזה אין הרשאה לאשר הסכם.'))}
            </p>
            {canEdit ? (
              <ul className="trd-blocker-list">
                {coverage.blockers.map((blocker) => (
                  <li key={`${blocker.kind}-${blocker.document_id || ''}`}>
                    <Status status="warning">{blockerSentence(blocker, locale)}</Status>
                    {Array.isArray(blocker.ids) && blocker.ids.length > 0 ? (
                      <span className="trd-blocker-ids">
                        {blocker.ids.slice(0, 8).map((id) => (
                          <Code key={id} className="trd-id-chip">{id}</Code>
                        ))}
                        {blocker.ids.length < blocker.count ? (
                          <span className="trd-field-hint">
                            {pageText(
                              locale,
                              `and ${formatNumber(blocker.count - blocker.ids.length, locale)} more`,
                              `ועוד ${formatNumber(blocker.count - blocker.ids.length, locale)}`,
                            )}
                          </span>
                        ) : null}
                      </span>
                    ) : null}
                  </li>
                ))}
                {coverage.blockers.length === 0 ? (
                  <li>
                    <Status status="warning">
                      {pageText(
                        locale,
                        'The gate reports no blocker and is still not ready. That disagreement is itself the finding: do not approve until it is understood.',
                        'השער אינו מדווח על חסם ובכל זאת אינו מוכן. אי ההתאמה הזו היא עצמה הממצא: אין לאשר עד שתובן.',
                      )}
                    </Status>
                  </li>
                ) : null}
              </ul>
            ) : null}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
