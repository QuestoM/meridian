import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Check, Globe, Tv } from 'lucide-react';
import {
  API_BASE,
  finiteNumber,
  formatCurrency,
  formatNumber,
  formatPercent,
  normalizeRows,
  pageText,
} from './surface-helpers';

// FrontierScopeChart: the revenue-front-vs-retention panel, upgraded with
//  (a) a SCOPE selector (whole schedule vs the operator's owned channel) wired to
//      GET /api/overview?scope=channel:<id>, with the active scope always shown as
//      a breadcrumb label;
//  (b) CLICKABLE Pareto points - selecting a point reveals its revenue_weight,
//      projected revenue and average retention, plus an "apply this weight"
//      affordance that saves it through the existing settings PUT path;
//  (c) the auto-scaled axes (paddedDomain) already used elsewhere.
//
// Honest labelling: the axes are PROJECTED revenue vs AVERAGE retention of
// whole-schedule (or scoped) Pareto scenarios. No revenue-net number is shown
// because the API does not expose one for the frontier.

const HEIGHT = 224;
const PAD_X = 46;
const PAD_Y = 30;

function paddedDomain(values, fallbackSpan, padRatio = 0.12) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return [0, fallbackSpan || 1];
  }
  const rawMin = Math.min(...finite);
  const rawMax = Math.max(...finite);
  const rawSpan = rawMax - rawMin;
  const scaleFloor = Math.max(Math.abs(rawMax), Math.abs(rawMin)) * 0.04;
  const span = Math.max(rawSpan, scaleFloor, 1e-9);
  const center = (rawMin + rawMax) / 2;
  const padding = span * padRatio;
  return [center - span / 2 - padding, center + span / 2 + padding];
}

export default function FrontierScopeChart({
  initialData,
  copy,
  locale,
  loading = false,
  operatorChannel = '',
  savedRevenueWeight = null,
  onApplyWeight,
  applyState = 'idle',
  // Real backend field overview.frontier_status (the sibling FrontierPanel already
  // consumes the same value, where 'computing' means the optimizer sweep is still
  // running on a cold cache). NOTE: the current parent call site in
  // TVBreakDashboard.jsx does not yet pass this prop, so until it adds
  // status={overview.frontier_status} the unscoped path falls back to '' and we
  // cannot distinguish computing from no-data there. We do NOT fabricate a status:
  // when it is absent the empty state stays honestly generic.
  status = '',
}) {
  const he = locale === 'he';
  const chartFrameRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(760);
  const [scope, setScope] = useState('');
  const [data, setData] = useState(initialData || []);
  const [scopeLoading, setScopeLoading] = useState(false);
  const [scopeError, setScopeError] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(null);
  // Honest frontier_status for the SCOPED view: the /api/overview payload carries
  // the same frontier_status field, so when a scope is active we read it from the
  // scoped response. When no scope is active we fall back to the parent's status
  // prop. refetchNonce lets a 'computing' cold cache trigger one re-fetch.
  const [scopeStatus, setScopeStatus] = useState('');
  const [refetchNonce, setRefetchNonce] = useState(0);

  // When no scope is active, mirror the parent-provided frontier so the panel
  // reflects the same overview payload the rest of the page consumes.
  useEffect(() => {
    if (!scope) {
      setData(initialData || []);
    }
  }, [initialData, scope]);

  // Fetch the scoped frontier only when a non-default scope is chosen. The
  // unscoped view reuses the already-loaded overview (no extra request).
  useEffect(() => {
    if (!scope) {
      setScopeError(false);
      setScopeLoading(false);
      setScopeStatus('');
      return undefined;
    }
    let active = true;
    setScopeLoading(true);
    setScopeError(false);
    fetch(`${API_BASE}/api/overview?scope=${encodeURIComponent(scope)}`)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (!active) return;
        setData(payload.frontier || []);
        setScopeStatus(payload.frontier_status || '');
        setScopeLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setScopeError(true);
        setScopeLoading(false);
      });
    return () => {
      active = false;
    };
  }, [scope, refetchNonce]);

  useEffect(() => {
    const frame = chartFrameRef.current;
    if (!frame) return undefined;
    const updateWidth = () => {
      setChartWidth(Math.max(360, Math.round(frame.getBoundingClientRect().width)));
    };
    updateWidth();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(updateWidth);
    observer.observe(frame);
    return () => observer.disconnect();
  });

  const points = useMemo(
    () =>
      normalizeRows(data)
        .map((point) => ({
          retention: finiteNumber(point.retention),
          revenue: finiteNumber(point.revenue),
          weight: finiteNumber(point.revenue_weight),
          selected: Boolean(point.selected),
        }))
        .filter((point) => point.retention !== null && point.revenue !== null),
    [data],
  );

  const savedPoint = points.find((point) => point.selected) || points[points.length - 1];
  const showSkeleton = loading || scopeLoading;
  const showEmpty = !showSkeleton && (points.length < 2 || !savedPoint);
  // When a scope is active the scoped response carries its own frontier_status;
  // otherwise we trust the parent's status prop. 'computing' means the optimizer
  // sweep is still warming the cache, which is honestly distinct from a no-data /
  // no-channel empty state. Any other (or absent) value stays generic.
  const effectiveStatus = scope ? scopeStatus : status;
  const isComputing = showEmpty && effectiveStatus === 'computing';

  // On a cold-cache 'computing' state, kick a single delayed re-fetch so the curve
  // self-heals once the sweep finishes, instead of waiting for a manual reload.
  // Only the scoped view owns its own fetch here; the unscoped view is driven by
  // the parent overview, so we bump the scope effect's nonce only when scoped.
  useEffect(() => {
    if (!isComputing || !scope) return undefined;
    const timer = setTimeout(() => setRefetchNonce((nonce) => nonce + 1), 4000);
    return () => clearTimeout(timer);
  }, [isComputing, scope]);

  const scopeOptions = useMemo(() => {
    const options = [{ value: '', labelHe: 'כל הערוצים', labelEn: 'All channels', icon: 'globe' }];
    const owned = String(operatorChannel || '').trim();
    if (owned) {
      options.push({ value: `channel:${owned}`, labelHe: owned, labelEn: owned, icon: 'tv' });
    }
    return options;
  }, [operatorChannel]);

  const width = chartWidth;
  const [retMin, retMax] = paddedDomain(points.map((point) => point.retention), 0.8);
  const [revMin, revMax] = paddedDomain(points.map((point) => point.revenue), 1);
  const xFor = (retention) =>
    PAD_X + ((retention - retMin) / Math.max(retMax - retMin, 1e-9)) * (width - PAD_X * 2);
  const yFor = (revenue) =>
    HEIGHT - PAD_Y - ((revenue - revMin) / Math.max(revMax - revMin, 1e-9)) * (HEIGHT - PAD_Y * 2);
  const path = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(point.retention).toFixed(1)} ${yFor(point.revenue).toFixed(1)}`)
    .join(' ');

  const safeSelected = selectedIndex !== null && points[selectedIndex] ? selectedIndex : null;
  const focusPoint = safeSelected !== null ? points[safeSelected] : savedPoint;
  const isSavedWeightSelected =
    focusPoint && savedRevenueWeight !== null && focusPoint.weight === Number(savedRevenueWeight);
  const canApply =
    focusPoint && focusPoint.weight !== null && !isSavedWeightSelected && typeof onApplyWeight === 'function';

  const activeScopeLabel = (() => {
    const match = scopeOptions.find((option) => option.value === scope);
    if (match) return he ? match.labelHe : match.labelEn;
    return he ? 'כל הערוצים' : 'All channels';
  })();

  return (
    <div className="analytics-panel frontier-panel frontier-scope-panel">
      <div className="panel-head">
        <h2>{copy.frontier}</h2>
        <span>{copy.frontierMode}</span>
      </div>

      <div className="frontier-scope-bar" dir={he ? 'rtl' : 'ltr'}>
        <div className="frontier-scope-control" role="group" aria-label={he ? 'היקף החזית' : 'Frontier scope'}>
          {scopeOptions.map((option) => {
            const active = option.value === scope;
            const Icon = option.icon === 'tv' ? Tv : Globe;
            return (
              <button
                key={option.value || 'all'}
                type="button"
                className={`frontier-scope-chip${active ? ' active' : ''}`}
                aria-pressed={active}
                onClick={() => {
                  setScope(option.value);
                  setSelectedIndex(null);
                }}
              >
                <Icon size={13} />
                {he ? option.labelHe : option.labelEn}
              </button>
            );
          })}
        </div>
        <span className="frontier-scope-breadcrumb">
          {he ? 'היקף נוכחי' : 'Scope'}
          <strong>{activeScopeLabel}</strong>
        </span>
      </div>

      {!operatorChannel && (
        <p className="frontier-scope-hint">
          {pageText(
            locale,
            'Set your owned channel in Settings to scope the frontier to a single channel.',
            'בחרו את הערוץ שבבעלותכם בהגדרות כדי למקד את החזית לערוץ יחיד.',
          )}
        </p>
      )}

      {showSkeleton ? (
        <div className="frontier-skeleton" aria-hidden="true" />
      ) : showEmpty ? (
        <div className="heatmap-empty">
          {scopeError
            ? pageText(locale, 'This scope could not be computed right now.', 'לא ניתן לחשב את ההיקף הזה כרגע.')
            : isComputing
              ? pageText(locale, 'The frontier is being computed, refresh in a moment.', 'החזית בחישוב, רעננו בעוד רגע.')
              : scope && !String(operatorChannel || '').trim()
                ? pageText(locale, 'No channel selected, so there is no frontier to draw.', 'לא נבחר ערוץ, ולכן אין חזית לשרטוט.')
                : pageText(locale, 'Not enough scenarios to draw a frontier yet.', 'אין מספיק תרחישים לשרטוט החזית עדיין.')}
        </div>
      ) : (
        <>
          <div ref={chartFrameRef} className="frontier-chart-frame chart-ltr" dir="ltr">
            <svg
              className="frontier-svg"
              viewBox={`0 0 ${width} ${HEIGHT}`}
              role="img"
              aria-label={pageText(locale, 'Projected revenue versus average retention frontier', 'חזית הכנסה צפויה מול שימור ממוצע')}
            >
              {[0, 1, 2, 3].map((line) => {
                const y = PAD_Y + line * ((HEIGHT - PAD_Y * 2) / 3);
                return <line key={`h-${line}`} x1={PAD_X} x2={width - PAD_X} y1={y} y2={y} />;
              })}
              {[0, 1, 2, 3, 4].map((line) => {
                const x = PAD_X + line * ((width - PAD_X * 2) / 4);
                return <line key={`v-${line}`} x1={x} x2={x} y1={PAD_Y} y2={HEIGHT - PAD_Y} />;
              })}
              <path d={path} />
              {safeSelected !== null && focusPoint && (
                <g className="frontier-hover-guides" aria-hidden="true">
                  <line x1={xFor(focusPoint.retention)} x2={xFor(focusPoint.retention)} y1={PAD_Y} y2={HEIGHT - PAD_Y} />
                  <line x1={PAD_X} x2={width - PAD_X} y1={yFor(focusPoint.revenue)} y2={yFor(focusPoint.revenue)} />
                </g>
              )}
              {points.map((point, index) => {
                const weightLabel = point.weight !== null ? `${point.weight}` : '?';
                return (
                  <circle
                    key={`${point.retention}-${point.revenue}-${index}`}
                    className={[
                      'frontier-clickable',
                      point.selected ? 'selected-point' : '',
                      safeSelected === index ? 'active-point' : '',
                    ].filter(Boolean).join(' ')}
                    cx={xFor(point.retention)}
                    cy={yFor(point.revenue)}
                    r={safeSelected === index ? 7 : point.selected ? 6 : 4}
                    tabIndex={0}
                    role="button"
                    aria-label={`${pageText(locale, 'Revenue weight', 'משקל הכנסה')} ${weightLabel}, ${formatCurrency(point.revenue, locale)}, ${formatPercent(point.retention, locale)}`}
                    onClick={() => setSelectedIndex((current) => (current === index ? null : index))}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        setSelectedIndex((current) => (current === index ? null : index));
                      }
                    }}
                  />
                );
              })}
              <text className="axis-label" x={PAD_X} y={HEIGHT - 6}>{formatPercent(retMin, locale)}</text>
              <text className="axis-label axis-label-end" x={width - PAD_X} y={HEIGHT - 6}>{formatPercent(retMax, locale)}</text>
              <text className="axis-label" x={4} y={PAD_Y + 4}>{formatCurrency(revMax, locale)}</text>
            </svg>
          </div>

          {/* The chart itself is hard-forced dir="ltr" so the plotted axes run
              low-to-high left-to-right. Keep the legend in that same chart space
              (always ltr) so the X label sits under the X axis and the Y label
              under the Y axis. An rtl legend here would mirror the labels off
              their axes even though the numbers are correct. */}
          <div className="frontier-axis-legend" dir="ltr">
            <span>{pageText(locale, 'X: average retention', 'ציר X: שימור ממוצע')}</span>
            <span>{pageText(locale, 'Y: projected revenue', 'ציר Y: הכנסה צפויה')}</span>
          </div>

          {focusPoint && (
            <div className="frontier-point-readout" dir={he ? 'rtl' : 'ltr'}>
              <div className="frontier-point-grid">
                <div>
                  <span>{pageText(locale, 'Revenue weight', 'משקל הכנסה')}</span>
                  <strong className="numeric" dir="ltr">{focusPoint.weight !== null ? formatNumber(focusPoint.weight, locale) : '-'}</strong>
                </div>
                <div>
                  <span>{pageText(locale, 'Projected revenue', 'הכנסה צפויה')}</span>
                  <strong className="numeric" dir="ltr">{formatCurrency(focusPoint.revenue, locale)}</strong>
                </div>
                <div>
                  <span>{pageText(locale, 'Average retention', 'שימור ממוצע')}</span>
                  <strong className="numeric" dir="ltr">{formatPercent(focusPoint.retention, locale)}</strong>
                </div>
              </div>
              <div className="frontier-point-action">
                {isSavedWeightSelected ? (
                  <span className="frontier-point-saved">
                    <Check size={13} />
                    {pageText(locale, 'Current saved weight', 'המשקל השמור הנוכחי')}
                  </span>
                ) : (
                  <Button
                    className="secondary-button compact"
                    type="button"
                    variant="outlined"
                    disabled={!canApply || applyState === 'saving'}
                    onClick={() => canApply && onApplyWeight(focusPoint.weight)}
                  >
                    {applyState === 'saving'
                      ? pageText(locale, 'Applying...', 'מחיל...')
                      : pageText(locale, 'Apply this weight', 'החל תרחיש')}
                  </Button>
                )}
                {safeSelected === null && (
                  <span className="frontier-point-hint">
                    {pageText(locale, 'Click a point to inspect and apply its weight.', 'לחצו על נקודה כדי לבחון ולהחיל את המשקל שלה.')}
                  </span>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
