import React from 'react';
import { Scale, ShieldCheck, Stamp } from 'lucide-react';
import { Button, Card, CardBody, Status } from '../../studio';
import { Figure, Name, Prose } from '../../shell/bidi';
import { formatCurrency, formatNumber, pageText } from '../../shell/surface-helpers';

// One side of the day comparison: the headline, the deltas, the attributed
// reasons, the commitments line, the guardrail verdict and the decision
// actions. Split out of DayVersionsWorkspace to keep that file inside the
// project's file-size law; this half holds no state and makes no request;
// everything it shows arrives as a prop from the compare payload.

function signed(value, formatter) {
  if (value === null || value === undefined) return '';
  const text = formatter(Math.abs(value));
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

function AttributionCells({ attribution, locale }) {
  const cells = (attribution && attribution.cells) || [];
  if (!attribution || !attribution.available || cells.length === 0) return null;
  return (
    <div className="dvw-reasons">
      <h5>{pageText(locale, 'Where the gap comes from', 'מאיפה הפער מגיע')}</h5>
      <ul>
        {cells.slice(0, 4).map((cell, index) => (
          <li key={index}>
            <span className="dvw-reason-what">
              <Name>{cell.bucket_he || cell.bucket}</Name>
              {cell.daypart_he ? <Name>{` · ${cell.daypart_he}`}</Name> : null}
            </span>
            <Figure>{signed(cell.revenue_delta, (v) => formatCurrency(v, locale))}</Figure>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function SideCard({ side, locale, adoptedId, highest, canDecide, onDecide, deciding }) {
  const delta = side.delta || {};
  const commitments = side.commitments || {};
  const compliance = side.compliance || {};
  const isAdopted = side.status === 'adopted';
  const phrase = commitments.phrase_he
    || (commitments.available === false ? '' : '');
  return (
    <Card as="article" className="dvw-side" data-adopted={isAdopted ? 'true' : 'false'}>
      <CardBody>
        <header className="dvw-side-head">
          <h4><Name>{side.label || pageText(locale, 'The day as it stands', 'היום כפי שהוא')}</Name></h4>
          <span className="dvw-side-chips">
            {side.side_id === highest ? (
              <Status status="positive">{pageText(locale, 'Highest revenue', 'ההכנסה הגבוהה')}</Status>
            ) : null}
            {isAdopted ? (
              <Status status="positive" icon={<Stamp size={12} aria-hidden="true" />}>
                {pageText(locale, 'Adopted', 'אומצה')}
              </Status>
            ) : null}
          </span>
        </header>
        {side.author ? (
          <p className="dvw-side-author"><Name>{side.author}</Name></p>
        ) : null}
        <Prose className="dvw-side-headline">{side.headline}</Prose>
        <dl className="dvw-side-figures">
          <dt>{pageText(locale, 'Revenue vs the day', 'הכנסה מול היום')}</dt>
          <dd><Figure>{signed(delta.revenue, (v) => formatCurrency(v, locale))}</Figure></dd>
          <dt>{pageText(locale, 'Net of retention', 'בניכוי עלות נטישה')}</dt>
          <dd>
            {delta.revenue_net_of_retention === null || delta.revenue_net_of_retention === undefined
              ? <span className="dvw-quiet">{delta.net_reason || pageText(locale, 'Not computable', 'לא ניתן לחישוב')}</span>
              : <Figure>{signed(delta.revenue_net_of_retention, (v) => formatCurrency(v, locale))}</Figure>}
          </dd>
          <dt>{pageText(locale, 'Breaks', 'ברייקים')}</dt>
          <dd><Figure>{signed(delta.breaks, (v) => formatNumber(v, locale))}</Figure></dd>
          <dt>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</dt>
          <dd><Figure>{signed(delta.ad_seconds, (v) => formatNumber(v, locale))}</Figure></dd>
        </dl>
        <AttributionCells attribution={side.attribution} locale={locale} />
        <p className="dvw-side-line">
          <Scale size={14} aria-hidden="true" />
          {phrase || pageText(locale, 'Commitments: not measured', 'התחייבויות: לא נמדד')}
        </p>
        <p className="dvw-side-line">
          <ShieldCheck size={14} aria-hidden="true" />
          {compliance.compliant
            ? pageText(
              locale,
              `Passes all ${compliance.checks_run || 0} guardrail checks`,
              `עומדת בכל ${compliance.checks_run || 0} בדיקות הגבולות`,
            )
            : pageText(
              locale,
              `${(compliance.violations || []).length} guardrail violations`,
              `${(compliance.violations || []).length} חריגות מגבולות`,
            )}
        </p>
        {canDecide && side.status === 'proposed' ? (
          <div className="dvw-side-actions">
            <Button
              type="button"
              onClick={() => onDecide(side, 'adopt')}
              disabled={deciding}
            >
              {pageText(locale, 'Adopt this version', 'אימוץ הגרסה הזאת')}
            </Button>
            <Button
              type="button"
              variant="quiet"
              onClick={() => onDecide(side, 'reject')}
              disabled={deciding}
            >
              {pageText(locale, 'Reject', 'דחייה')}
            </Button>
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}
