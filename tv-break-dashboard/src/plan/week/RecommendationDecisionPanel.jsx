import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, ButtonBase } from '../../studio/actions';
import { ArrowUpRight, Check, Download, X } from 'lucide-react';
import { downloadJson } from '../../shell/downloads';
import {
  formatCurrency,
  formatPercent,
  pageText,
} from '../../shell/format';
import { recommendationRationale, recommendationTitle } from '../../shell/labels';
import { Card, CardBody } from '../../studio';

const PROPOSAL_COPY = {
  gold: ['Mark this as a gold break', 'סימון כברייק זהב'],
  pin: ['Pin this segment in the plan', 'נעיצת המקטע בתוכנית'],
  lower_count: ['Lower the break count in this segment', 'הפחתת מספר הברייקים במקטע'],
  force: ['Set an explicit break count', 'קביעת מספר ברייקים מפורש'],
  forbid: ['Place no break in this segment', 'מניעת ברייק במקטע'],
};

function proposalText(recommendation, locale) {
  const pair = PROPOSAL_COPY[String(recommendation?.proposed_kind || '')];
  if (!pair) return pageText(locale, 'Advisory only; no direct plan change.', 'המלצה בלבד; ללא שינוי ישיר בתוכנית.');
  return pageText(locale, pair[0], pair[1]);
}

function actionConsequence(action, recommendation, similarCount, locale) {
  if (action === 'approve') {
    if (['force', 'lower_count'].includes(String(recommendation?.proposed_kind || ''))) {
      return pageText(
        locale,
        'This opens Manual decisions with the segment prefilled. No plan change is made until you set and save the count there.',
        'הפעולה תפתח את ההחלטות הידניות כשהמקטע כבר ממולא. התוכנית לא תשתנה עד לקביעת המספר ולשמירה שם.',
      );
    }
    if (recommendation?.actionable && recommendation?.segment_id) {
      return pageText(
        locale,
        'This creates an anchored manual decision and marks the schedule stale. A new plan run is required before publication.',
        'הפעולה תיצור החלטה ידנית מעוגנת ותסמן את הלוח כלא מעודכן. תידרש הרצה חדשה לפני פרסום.',
      );
    }
    return pageText(locale, 'This records an approval in the decision log; it does not change the plan.', 'הפעולה תרשום אישור ביומן ההחלטות; היא לא תשנה את התוכנית.');
  }
  if (action === 'reject') {
    return pageText(
      locale,
      'This records the recommendation as rejected. When the recommendation is anchored, the dismissal is retained with that segment.',
      'הפעולה תרשום את ההמלצה כנדחית. כשההמלצה מעוגנת, הדחייה תישמר עם המקטע הזה.',
    );
  }
  return pageText(
    locale,
    `This records ${similarCount} recommendations of the same programme type as approved. Review the count before continuing.`,
    `הפעולה תרשום ${similarCount} המלצות מאותו סוג תוכנית כמאושרות. בדקו את הכמות לפני שממשיכים.`,
  );
}

function actionLabel(action, locale) {
  if (action === 'approve') return pageText(locale, 'Approve recommendation', 'אישור ההמלצה');
  if (action === 'reject') return pageText(locale, 'Reject recommendation', 'דחיית ההמלצה');
  return pageText(locale, 'Approve similar recommendations', 'אישור המלצות דומות');
}

export function RecommendationDecisionPanel({
  recommendations,
  selectedId,
  approved,
  rejected,
  locale,
  notify,
  onSelect,
  onApprove,
  onReject,
  onApplySimilar,
  onOpenInOverrides,
}) {
  const rows = Array.isArray(recommendations) ? recommendations : [];
  const selected = useMemo(
    () => rows.find((item) => String(item.id) === String(selectedId)) || rows[0] || null,
    [rows, selectedId],
  );
  const [pendingAction, setPendingAction] = useState('');
  const [actionState, setActionState] = useState('idle');
  const cancelRef = useRef(null);

  useEffect(() => {
    setPendingAction('');
    setActionState('idle');
  }, [selected?.id]);

  useEffect(() => {
    if (pendingAction) cancelRef.current?.focus();
  }, [pendingAction]);

  if (!selected) return null;

  const isApproved = approved?.has?.(selected.id) || approved?.has?.(String(selected.id));
  const isRejected = rejected?.has?.(selected.id) || rejected?.has?.(String(selected.id));
  const similarCount = rows.filter((item) => !selected.program_type || item.program_type === selected.program_type).length;

  async function commitAction() {
    const action = pendingAction;
    if (!action || actionState === 'running') return;
    setActionState('running');
    try {
      let recorded;
      if (action === 'approve') recorded = await onApprove?.(selected.id);
      else if (action === 'reject') recorded = await onReject?.(selected.id);
      else recorded = await onApplySimilar?.(selected.id);
      if (recorded === false) {
        setActionState('error');
        return;
      }
      setPendingAction('');
      setActionState('done');
    } catch (error) {
      setActionState('error');
      notify?.(
        `The decision could not be recorded: ${String(error?.message || error)}`,
        `לא ניתן היה לרשום את ההחלטה: ${String(error?.message || error)}`,
      );
    }
  }

  function exportDecision() {
    downloadJson('kairos-recommendation-detail.json', { recommendation: selected });
    notify?.('Recommendation detail exported as JSON.', 'פרטי ההמלצה יוצאו כ־JSON.');
  }

  return (
    <section className="card plan-section plan-recommendations" aria-labelledby="plan-recommendations-title" aria-busy={actionState === 'running'}>
      <div className="plan-section-head">
        <div>
          <h2 id="plan-recommendations-title">{pageText(locale, 'Decisions behind this plan', 'החלטות מאחורי התוכנית')}</h2>
          <p>
            {pageText(
              locale,
              'Review the evidence first. Approval and rejection are explicit recorded decisions; neither is inferred from opening a row.',
              'קודם בודקים את הראיות. אישור ודחייה הן החלטות מפורשות שנרשמות; פתיחת שורה לבדה אינה נחשבת החלטה.',
            )}
          </p>
        </div>
        <span className="plan-recommendation-count">
          {pageText(locale, `${rows.length} recommendations`, `${rows.length} המלצות`)}
        </span>
      </div>

      <div className="plan-recommendation-layout">
        <div className="plan-recommendation-list" aria-label={pageText(locale, 'Recommendations', 'המלצות')}>
          {rows.map((item) => {
            const active = String(item.id) === String(selected.id);
            const itemApproved = approved?.has?.(item.id) || approved?.has?.(String(item.id));
            const itemRejected = rejected?.has?.(item.id) || rejected?.has?.(String(item.id));
            return (
              <ButtonBase
                key={item.id || item.title}
                type="button"
                className={`plan-recommendation-row${active ? ' is-active' : ''}`}
                aria-current={active ? 'true' : undefined}
                onClick={() => onSelect?.(item.id)}
              >
                <span>
                  <strong>{recommendationTitle(item, locale)}</strong>
                  <small>{proposalText(item, locale)}</small>
                </span>
                <span className={`plan-recommendation-state${itemRejected ? ' is-rejected' : itemApproved ? ' is-approved' : ''}`}>
                  {itemRejected
                    ? pageText(locale, 'Rejected', 'נדחתה')
                    : itemApproved
                      ? pageText(locale, 'Approved', 'אושרה')
                      : pageText(locale, 'Pending', 'ממתינה')}
                </span>
              </ButtonBase>
            );
          })}
        </div>

        <Card as="article" dense className="plan-recommendation-detail" aria-label={recommendationTitle(selected, locale)}>
          <CardBody>
          <div className="plan-recommendation-detail-head">
            <div>
              <span className={`plan-recommendation-state${isRejected ? ' is-rejected' : isApproved ? ' is-approved' : ''}`}>
                {isRejected
                  ? pageText(locale, 'Rejected', 'נדחתה')
                  : isApproved
                    ? pageText(locale, 'Approved', 'אושרה')
                    : pageText(locale, 'Awaiting a decision', 'ממתינה להחלטה')}
              </span>
              <h3>{recommendationTitle(selected, locale)}</h3>
            </div>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={exportDecision}>
              <Download size={15} aria-hidden="true" />
              {pageText(locale, 'Export evidence', 'ייצוא הראיות')}
            </Button>
          </div>

          <p>{recommendationRationale(selected, locale)}</p>
          <dl className="plan-recommendation-facts">
            <div><dt>{pageText(locale, 'Proposed change', 'שינוי מוצע')}</dt><dd>{proposalText(selected, locale)}</dd></div>
            <div><dt>{pageText(locale, 'Revenue impact', 'השפעה על הכנסה')}</dt><dd>{formatCurrency(selected.impact, locale)}</dd></div>
            <div><dt>{pageText(locale, 'Audience retained', 'קהל שנשמר')}</dt><dd>{formatPercent(selected.retention, locale)}</dd></div>
            <div><dt>{pageText(locale, 'Risk', 'סיכון')}</dt><dd>{selected.risk || pageText(locale, 'Not measured', 'לא נמדד')}</dd></div>
          </dl>

          {pendingAction ? (
            <div className="plan-recommendation-review" role="group" aria-labelledby="plan-recommendation-review-title">
              <div className="plan-recommendation-review-head">
                <h4 id="plan-recommendation-review-title">{actionLabel(pendingAction, locale)}</h4>
                <Button
                  className="icon-button"
                  type="button"
                  onClick={() => setPendingAction('')}
                  aria-label={pageText(locale, 'Cancel this decision', 'ביטול ההחלטה')}
                >
                  <X size={16} aria-hidden="true" />
                </Button>
              </div>
              <p>{actionConsequence(pendingAction, selected, similarCount, locale)}</p>
              <div className="plan-recommendation-review-actions">
                <Button ref={cancelRef} className="secondary-button" type="button" variant="outlined" onClick={() => setPendingAction('')}>
                  {pageText(locale, 'Cancel', 'ביטול')}
                </Button>
                <Button className="run-button" type="button" variant="contained" disabled={actionState === 'running'} onClick={commitAction}>
                  <Check size={15} aria-hidden="true" />
                  {actionState === 'running' ? pageText(locale, 'Recording…', 'רושם…') : actionLabel(pendingAction, locale)}
                </Button>
              </div>
            </div>
          ) : (
            <div className="plan-recommendation-actions">
              <Button className="run-button" type="button" variant="contained" disabled={isApproved} onClick={() => setPendingAction('approve')}>
                {isApproved ? pageText(locale, 'Approved', 'אושרה') : pageText(locale, 'Review approval', 'בדיקת אישור')}
              </Button>
              <Button className="secondary-button" type="button" variant="outlined" disabled={isRejected} onClick={() => setPendingAction('reject')}>
                {isRejected ? pageText(locale, 'Rejected', 'נדחתה') : pageText(locale, 'Review rejection', 'בדיקת דחייה')}
              </Button>
              <Button className="secondary-button" type="button" variant="outlined" disabled={similarCount < 2} onClick={() => setPendingAction('similar')}>
                {pageText(locale, `Review ${similarCount} similar`, `בדיקת ${similarCount} דומות`)}
              </Button>
              {selected.actionable && selected.segment_id ? (
                <Button className="secondary-button" type="button" variant="outlined" onClick={() => onOpenInOverrides?.(selected)}>
                  <ArrowUpRight size={15} aria-hidden="true" />
                  {pageText(locale, 'Open in Manual decisions', 'פתיחה בהחלטות ידניות')}
                </Button>
              ) : null}
            </div>
          )}

          {actionState === 'done' ? <p className="plan-action-status" role="status">{pageText(locale, 'Decision recorded.', 'ההחלטה נרשמה.')}</p> : null}
          {actionState === 'error' ? <p className="plan-note plan-note-red" role="alert">{pageText(locale, 'The decision was not recorded.', 'ההחלטה לא נרשמה.')}</p> : null}
          </CardBody>
        </Card>
      </div>
    </section>
  );
}

export default RecommendationDecisionPanel;
