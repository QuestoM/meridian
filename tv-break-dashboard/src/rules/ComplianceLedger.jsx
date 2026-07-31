import React from 'react';
import { pageText } from '../shell/format';
import { complianceDisclaimer, complianceUnitLabel } from '../shell/labels';

export function ComplianceLedger({ compliance, copy, locale }) {
  const checks = compliance?.checks || [];
  return (
    <div className="analytics-panel ledger-panel">
      <div className="panel-head">
        <h2>{copy.compliance}</h2>
        <span>{checks.length} {copy.activeRules}</span>
      </div>
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
              <small className="ledger-measure" dir={locale === 'he' ? 'rtl' : 'ltr'}>
                <span className="ledger-values" dir="ltr">{observed} / {limit}</span>
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
        <p className="ledger-note">{complianceDisclaimer(compliance?.disclaimer, locale)}</p>
      </div>
    </div>
  );
}

export default ComplianceLedger;
