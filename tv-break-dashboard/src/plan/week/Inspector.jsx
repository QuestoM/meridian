import React, { useState } from 'react';
import { Button, FormControl, IconButton, MenuItem, Select } from '@mui/material';
import { Check, Download, X } from 'lucide-react';
import {
  Numeric,
  finiteNumber,
  formatCurrency,
  formatMinutes,
  formatNumber,
  formatPercent,
  pageText,
} from '../../shell/format';
import { recommendationRationale, recommendationTitle } from '../../shell/labels';

export function SelectionGuide({ selectedProgram, onOpen, copy, locale }) {
  return (
    <aside className="inspector selection-guide" aria-label="Break detail panel closed">
      <div className="inspector-head">
        <span>{copy.selectedBreak}</span>
      </div>
      <div className="selection-guide-body">
        <strong>{selectedProgram?.title || pageText(locale, 'No break selected', 'לא נבחר ברייק')}</strong>
        <p>
          {pageText(
            locale,
            'Select a cell in the planner or reopen the details panel to review guardrails, approval state, and export options.',
            'בחרו תא במשטח התכנון או פתחו מחדש את פאנל הפרטים כדי לבדוק בקרות, סטטוס אישור ואפשרויות ייצוא.',
          )}
        </p>
        <Button className="secondary-button" type="button" variant="outlined" onClick={onOpen}>
          {pageText(locale, 'Open details', 'פתיחת פרטים')}
        </Button>
      </div>
    </aside>
  );
}

export function Inspector({ selectedProgram, recommendation, approved, rejected, retentionFloor, onApprove, onReject, onOpenInOverrides, onApplySimilar, onExport, onClose, copy, locale }) {
  const recActionable = Boolean(recommendation?.actionable && recommendation?.segment_id && recommendation?.proposed_kind);
  const approvalLabel = rejected ? pageText(locale, 'Rejected', 'נדחה') : approved ? copy.approved : copy.pending;
  const [exportScope, setExportScope] = useState('Break detail');
  const selectedBreak = selectedProgram?.selected_break;
  // Real values only: a missing duration, retention or spot count renders as a
  // dash, never a stand-in number dressed up as data.
  const durationSeconds =
    finiteNumber(selectedBreak?.duration_sec) ??
    (finiteNumber(selectedProgram?.duration_minutes) !== null ? Number(selectedProgram.duration_minutes) * 60 : null);
  const breakNumber = finiteNumber(selectedBreak?.break_num_in_program);
  const breakTotal = finiteNumber(selectedBreak?.breaks_in_program) ?? finiteNumber(selectedProgram?.break_markers);
  const breakContext = breakNumber !== null && breakTotal !== null
    ? pageText(locale, `break ${breakNumber} of ${breakTotal}`, `ברייק ${breakNumber} מתוך ${breakTotal}`)
    : breakTotal !== null
      ? pageText(locale, `${formatNumber(breakTotal, locale)} breaks`, `${formatNumber(breakTotal, locale)} ברייקים`)
      : '';
  const retentionValue = finiteNumber(selectedProgram?.retention ?? recommendation?.retention);
  const floorPercent = finiteNumber(retentionFloor) !== null ? Math.round(Number(retentionFloor) * 100) : null;
  const retentionState =
    retentionValue === null || floorPercent === null
      ? 'unknown'
      : retentionValue < floorPercent
        ? 'at_risk'
        : 'compliant';
  return (
    <aside className="inspector" aria-label="Selected break inspector">
      <div className="inspector-head">
        <span>{copy.selectedBreak}</span>
        <IconButton className="icon-button small" type="button" aria-label={pageText(locale, 'Close the break detail panel', 'סגירת פאנל פרטי הברייק')} size="small" onClick={onClose}>
          <X size={14} />
        </IconButton>
      </div>

      <div className="selected-program">
        <span className="channel-badge">{selectedProgram?.channel?.slice(0, 2) || '?'}</span>
        <div>
          <strong>{selectedProgram?.title || pageText(locale, 'No program selected', 'לא נבחרה תוכנית')}</strong>
          <small>
            {[selectedProgram?.channel, selectedProgram?.time, breakContext].filter(Boolean).join(' / ') ||
              pageText(locale, 'Select a cell in the planner', 'בחרו תא במשטח התכנון')}
          </small>
        </div>
        <span className={rejected ? 'approval rejected' : approved ? 'approval approved' : 'approval'}>{approvalLabel}</span>
      </div>

      <dl className="detail-list">
        <div><dt>{copy.detail[0]}</dt><dd>{formatCurrency(selectedProgram?.revenue, locale)}</dd></div>
        <div><dt>{copy.detail[1]}</dt><dd>{formatPercent(retentionValue, locale)}</dd></div>
        <div><dt>{copy.detail[2]}</dt><dd>{formatMinutes(durationSeconds, locale)}</dd></div>
        <div><dt>{copy.detail[3]}</dt><dd>{formatNumber(selectedBreak?.sponsorships_count, locale)}</dd></div>
      </dl>

      <div className="guardrail-block">
        <h3>{copy.guardrails}</h3>
        <div className="guardrail-row">
          <span>{pageText(locale, 'Retention floor', 'רף שימור')}</span>
          <strong className={retentionState === 'at_risk' ? 'guardrail-state at-risk' : 'guardrail-state'}>
            {retentionState === 'at_risk'
              ? copy.atRisk
              : retentionState === 'compliant'
                ? copy.compliant
                : pageText(locale, 'Not measured', 'לא נמדד')}
          </strong>
          <span className={retentionState === 'at_risk' ? 'guardrail-indicator at-risk' : 'guardrail-indicator'}>
            {retentionState === 'at_risk' ? (
              <Numeric>{`${formatNumber(retentionValue - floorPercent, locale)}pp`}</Numeric>
            ) : retentionState === 'compliant' ? (
              <Check size={14} />
            ) : (
              <Numeric>-</Numeric>
            )}
          </span>
        </div>
        {retentionState !== 'unknown' && (
          <small className="guardrail-measure">
            <Numeric>{`${formatNumber(retentionValue, locale)}% / ${formatNumber(floorPercent, locale)}%`}</Numeric>
          </small>
        )}
        <p className="guardrail-footnote">
          {pageText(
            locale,
            'Schedule-wide checks (ad minutes, spacing, protected content) live in the compliance ledger below.',
            'בדיקות לכלל הלוח (דקות פרסום, מרווחים, תוכן מוגן) מוצגות ביומן התאימות מטה.',
          )}
        </p>
      </div>

      <div className="recommendation-block">
        <h3>{copy.recommendation}</h3>
        {recommendation ? (
          <>
            <strong>{recommendationTitle(recommendation, locale)}</strong>
            <p>{recommendationRationale(recommendation, locale)}</p>
            <div className="recommendation-meta">
              <span>{copy.risk[recommendation.risk] || recommendation.risk || copy.risk.Unknown}</span>
              <span>{formatCurrency(recommendation.impact, locale)}</span>
            </div>
          </>
        ) : (
          <p>{pageText(locale, 'No recommendation for the current selection.', 'אין המלצה עבור הבחירה הנוכחית.')}</p>
        )}
      </div>

      <div className="inspector-actions">
        <Button className="primary-action" type="button" variant="contained" disabled={!recommendation} onClick={onApprove}>
          {approved ? copy.approved : copy.approve}
        </Button>
        <Button className={rejected ? 'secondary-button active' : 'secondary-button'} type="button" variant="outlined" disabled={!recommendation} onClick={onReject}>{copy.reject}</Button>
        <Button className="secondary-button" type="button" variant="outlined" disabled={!recommendation} onClick={onApplySimilar}>{copy.applySimilar}</Button>
        {recActionable && (
          <Button className="secondary-button" type="button" variant="outlined" onClick={onOpenInOverrides}>
            {pageText(locale, 'Open in overrides', 'פתיחה בעקיפות')}
          </Button>
        )}
      </div>

      <div className="export-row">
        <FormControl size="small">
          <Select aria-label={pageText(locale, 'Export scope', 'היקף הייצוא')} value={exportScope} onChange={(event) => setExportScope(event.target.value)}>
            <MenuItem value="Break detail">{copy.exportOptions[0]}</MenuItem>
            <MenuItem value="Weekly traffic plan">{copy.exportOptions[1]}</MenuItem>
            <MenuItem value="Guardrail report">{copy.exportOptions[2]}</MenuItem>
          </Select>
        </FormControl>
        <Button className="secondary-button" type="button" variant="outlined" onClick={() => onExport(exportScope)}>
          <Download size={14} />
          {copy.export}
        </Button>
      </div>
    </aside>
  );
}

export default Inspector;
