import React, { useEffect, useState } from 'react';
import { Tv } from 'lucide-react';
import { pageText } from '../shell/format';
import { Card, CardBody } from '../studio';
import { Figure } from '../shell/bidi';
import { complianceDisclaimer, complianceUnitLabel } from '../shell/labels';
import { complianceScopeSentence, complianceViewState, fetchCompliance } from './rules-lib';
import './rules-panels.css';

// Today's landing card and the licence page one click away read the same
// verdict about the same population, so this card fetches its own scoped
// route (GET /api/compliance, the route the licence page also reads) rather
// than trusting whatever compliance payload its parent handed it. The prop
// is a fallback only, used while the card's own fetch has not yet answered
// or has failed. A fallback payload that carries no scope key is a
// market-wide figure, not the operator's, so its numbers are never printed
// as the operator's own; the card states plainly that the basis is not
// stated instead of printing a number nobody can place.

// The card the ledger falls back to when it has no basis to print. The note is
// wrapped in CardBody rather than sitting loose in the card, so it takes the
// card's inset and starts on the same line as the heading above it. Before
// that, it had no inset of any kind and printed hard against the card's border.
function BasisNote({ copy, locale, children }) {
  return (
    <Card as="div" className="analytics-panel ledger-panel">
      <div className="panel-head">
        <h2>{copy.compliance}</h2>
      </div>
      <CardBody>
        <p className="ledger-basis-missing" role="status">{children}</p>
      </CardBody>
    </Card>
  );
}

export function ComplianceLedger({ compliance, copy, locale }) {
  const [own, setOwn] = useState(null);
  const [ownFailed, setOwnFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchCompliance()
      .then((body) => {
        if (cancelled) return;
        setOwn(body);
        setOwnFailed(false);
      })
      .catch(() => { if (!cancelled) setOwnFailed(true); });
    return () => { cancelled = true; };
  }, []);

  const view = complianceViewState(own, ownFailed, compliance);

  if (view.kind === 'loading') {
    return (
      <BasisNote copy={copy} locale={locale}>
        {pageText(locale, 'Reading the compliance verdict.', 'קורא את חוות דעת התאימות.')}
      </BasisNote>
    );
  }

  if (view.kind === 'basis_missing') {
    return (
      <BasisNote copy={copy} locale={locale}>
        {pageText(
          locale,
          "The basis for this verdict is not stated, so no figure is shown as the operator's own. Open the licence page for the scoped verdict.",
          'הבסיס לחוות הדעת הזו אינו ידוע, ולכן לא מוצג נתון כשל המפעיל. פתחו את עמוד הרישיון לחוות הדעת המוגדרת.',
        )}
      </BasisNote>
    );
  }

  if (view.kind === 'no_channel') {
    return (
      <BasisNote copy={copy} locale={locale}>
        {locale === 'he' ? view.reasonHe : view.reasonEn}
      </BasisNote>
    );
  }

  const { data, scope } = view;
  const checks = data.checks || [];
  return (
    <div className="analytics-panel ledger-panel">
      <div className="panel-head">
        <h2>{copy.compliance}</h2>
        <span>{checks.length} {copy.activeRules}</span>
      </div>
      <p className="ledger-scope-line">
        <Tv size={12} aria-hidden="true" />
        <span>{complianceScopeSentence(locale, scope)}</span>
      </p>
      <div className="ledger-list">
        {checks.map((check) => {
          const formatValue = (value) => Number(value).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
          const violationCount = Number(check.violations || 0);
          const unitLabel = complianceUnitLabel(check.unit, locale);
          const isPercent = check.unit === '%';
          const observed = `${formatValue(check.observed)}${isPercent ? '%' : ''}`;
          const limit = `${formatValue(check.limit)}${isPercent ? '%' : ''}`;
          return (
            <div className="ledger-row" key={check.id}>
              <span>{locale === 'he' ? check.label_he : check.label_en}</span>
              <strong className={check.status === 'at_risk' ? 'at-risk' : ''}>
                {check.status === 'at_risk' ? copy.atRisk : copy.compliant}
              </strong>
              <small className="ledger-measure">
                <Figure className="ledger-values">{observed} / {limit}</Figure>
                {!isPercent && unitLabel && <span className="ledger-unit">{unitLabel}</span>}
                {violationCount > 0 && (
                  <span className="ledger-violations">
                    {formatValue(violationCount)} {pageText(locale, 'violations', 'חריגות')}
                  </span>
                )}
              </small>
            </div>
          );
        })}
        <p className="ledger-note">{complianceDisclaimer(data.disclaimer, locale)}</p>
      </div>
    </div>
  );
}

export default ComplianceLedger;
