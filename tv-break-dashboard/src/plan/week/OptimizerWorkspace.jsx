import React from 'react';
import { Button, Checkbox, FormControlLabel } from '@mui/material';
import { SlidersHorizontal } from 'lucide-react';
import SummaryMetrics from '../../today/SummaryMetrics';
import ComplianceLedger from '../../rules/ComplianceLedger';
import {
  CoefficientFreshnessChip,
  OptimizationRunSummary,
  RetentionCostPanel,
} from './OptimizerRunPanels';
import GridAxisControl from './GridAxisControl';
import PlanningCanvas from './PlanningCanvas';
import TimelineView from './TimelineView';
import DaypartView from './DaypartView';
import OptimizerInventoryView from './OptimizerInventoryView';
import Inspector, { SelectionGuide } from './Inspector';
import FrontierPanel from './FrontierPanel';
import InventoryHeatmap from './InventoryHeatmap';

export function OptimizerWorkspace({
  overview,
  schedule,
  compliance,
  loading,
  activeViewMode,
  gridAxis,
  showPrograms,
  showBreaks,
  showMetrics,
  selectedProgramKey,
  selectedProgram,
  activeRec,
  approved,
  rejected,
  optimizationPlan,
  parameters,
  inspectorOpen,
  onViewChange,
  onGridAxisChange,
  onTogglePrograms,
  onToggleBreaks,
  onToggleMetrics,
  onSelectProgram,
  onCloseInspector,
  onApprove,
  onReject,
  onOpenInOverrides,
  onApplySimilar,
  onExport,
  copy,
  locale,
}) {
  const modeButtons = [
    ['grid', copy.toolbar[0]],
    ['timeline', copy.toolbar[1]],
    ['daypart', copy.toolbar[2]],
    ['inventory', copy.toolbar[3]],
  ];

  return (
    <>
      <SummaryMetrics overview={overview} copy={copy} locale={locale} />
      <OptimizationRunSummary plan={optimizationPlan} locale={locale} />
      <CoefficientFreshnessChip plan={optimizationPlan} parameters={parameters} locale={locale} />
      <RetentionCostPanel plan={optimizationPlan} parameters={parameters} copy={copy} locale={locale} />

      <div className="work-grid">
        <section className="planner-surface" aria-label={copy.canvas}>
          <div className="surface-toolbar">
            <div className="toolbar-left">
              {modeButtons.map(([mode, label]) => (
                <Button
                  key={mode}
                  className={activeViewMode === mode ? 'segmented active' : 'segmented'}
                  type="button"
                  variant="outlined"
                  aria-pressed={activeViewMode === mode}
                  onClick={() => onViewChange(mode)}
                >
                  {label}
                </Button>
              ))}
            </div>
            <div className="toolbar-right">
              {activeViewMode === 'grid' && (
                <GridAxisControl value={gridAxis} onChange={onGridAxisChange} locale={locale} />
              )}
              <FormControlLabel
                className="check-control"
                control={<Checkbox checked={showPrograms} onChange={(event) => onTogglePrograms(event.target.checked)} size="small" />}
                label={copy.toolbar[4]}
              />
              <FormControlLabel
                className="check-control"
                control={<Checkbox checked={showBreaks} onChange={(event) => onToggleBreaks(event.target.checked)} size="small" />}
                label={copy.toolbar[5]}
              />
              <Button
                className={showMetrics ? 'secondary-button compact active' : 'secondary-button compact'}
                type="button"
                variant="outlined"
                aria-pressed={showMetrics}
                onClick={onToggleMetrics}
              >
                <SlidersHorizontal size={14} />
                {copy.toolbar[6]}
              </Button>
            </div>
          </div>

          {activeViewMode === 'grid' && (
            <PlanningCanvas
              rows={schedule.rows || []}
              copy={copy}
              locale={locale}
              axis={gridAxis}
              showPrograms={showPrograms}
              showBreaks={showBreaks}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
          {activeViewMode === 'timeline' && (
            <TimelineView
              timeline={schedule.break_operations}
              rows={schedule.rows || []}
              locale={locale}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
          {activeViewMode === 'daypart' && (
            <DaypartView
              rows={schedule.rows || []}
              locale={locale}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
          {activeViewMode === 'inventory' && (
            <OptimizerInventoryView
              rows={schedule.rows || []}
              locale={locale}
              selectedProgramKey={selectedProgramKey}
              onSelectProgram={onSelectProgram}
            />
          )}
        </section>

        {inspectorOpen ? (
          <Inspector
            selectedProgram={selectedProgram}
            recommendation={activeRec}
            approved={approved.has(activeRec?.id)}
            rejected={rejected.has(activeRec?.id)}
            retentionFloor={overview.settings?.min_retention_floor}
            onApprove={onApprove}
            onReject={onReject}
            onOpenInOverrides={onOpenInOverrides}
            onApplySimilar={onApplySimilar}
            onExport={onExport}
            onClose={onCloseInspector}
            copy={copy}
            locale={locale}
          />
        ) : (
          <SelectionGuide selectedProgram={selectedProgram} onOpen={() => onSelectProgram(selectedProgram)} copy={copy} locale={locale} />
        )}
      </div>

      {showMetrics && (
        <section className="analytics-strip" aria-label="Analytics and constraint ledger">
          <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} loading={loading} operatorChannel={overview.settings?.operator_channel || ''} status={overview.frontier_status || ''} netPoint={overview.frontier_net_point || null} />
          <InventoryHeatmap copy={copy} locale={locale} />
          <ComplianceLedger compliance={compliance} copy={copy} locale={locale} />
        </section>
      )}
    </>
  );
}

export default OptimizerWorkspace;
