import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../../studio/actions';
import { Check, Tv } from 'lucide-react';
import { Pressable } from '../../studio/dom-controls';
import {
  API_BASE,
  finiteNumber,
  formatCurrency,
  formatCurrencyAxis,
  formatNumber,
  formatPercent,
  normalizeRows,
  pageText,
} from '../../shell/surface-helpers';
import { Figure } from '../../shell/bidi';

// FrontierScopeChart: the revenue-vs-retention panel, upgraded with
//  (a) a SCOPE bar that DEFAULTS to the operator's owned channel whenever one is
//      configured, wired to GET /api/overview?scope=channel:<id> (warmed at
//      startup by the backend), with the active scope always shown as a
//      breadcrumb label. There is no all-channels option: an empty scope falls
//      to the engine's auto-picked representative channel-day, which can be a
//      competitor, so the unscoped path survives only as the fallback when no
//      operator channel is configured, and its breadcrumb says so honestly;
//  (b) CLICKABLE Pareto points - selecting a point reveals its retention floor,
//      projected revenue and average retention, plus an "apply this floor"
//      affordance that saves the retention floor through the settings PUT path;
//  (c) the auto-scaled axes (paddedDomain) already used elsewhere.
//
// The curve sweeps the RETENTION FLOOR at the saved revenue weight, and each
// point is a REFINED optimum. This is the genuine tradeoff: a revenue-weight
// sweep collapses onto one point under the real optimizer (the weight barely
// moves the plan once retention clears the floor), so the floor is the lever
// that actually trades revenue for retention. The axes are PROJECTED revenue vs
// AVERAGE retention of a single representative-day estimate for the selected
// scope, NOT the saved weekly total shown as the headline revenue elsewhere. The
// backend returns a `basis` disclosure object describing the scope, channel and
// method; we surface basis.disclosure as a caption, with a bilingual fallback
// when it is absent on older responses.

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
  savedRetentionFloor = null,
  onApplyFloor,
  applyState = 'idle',
  // Real backend field overview.frontier_status, passed by the parent (the
  // sibling FrontierPanel consumes the same value, where 'computing' means the
  // optimizer sweep is still running on a cold cache). We do NOT fabricate a
  // status: when it is absent the empty state stays honestly generic.
  status = '',
}) {
  const he = locale === 'he';
  const chartFrameRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(760);
  // The owned channel is the default and only channel scope; the unscoped ''
  // value survives solely as the fallback when no operator channel is set.
  const ownedChannel = String(operatorChannel || '').trim();
  const ownedScope = ownedChannel ? `channel:${ownedChannel}` : '';
  const [scope, setScope] = useState(ownedScope);
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
  // The scoped /api/overview response now carries a `basis` disclosure object
  // (scope, channel, method, disclosure text) that spells out what the frontier
  // points actually measure. Older responses omit it, so it stays null until a
  // response provides one. The unscoped view is driven by the parent's frontier
  // array, which does not include a basis, so basis is cleared there.
  const [basis, setBasis] = useState(null);

  // Keep the scope pinned to the owned channel: settings can load (or change)
  // after mount, and the owned scope must win whenever one exists.
  useEffect(() => {
    setScope(ownedScope);
    setSelectedIndex(null);
  }, [ownedScope]);

  // When no scope is active, mirror the parent-provided frontier so the panel
  // reflects the same overview payload the rest of the page consumes.
  useEffect(() => {
    if (!scope) {
      setData(initialData || []);
      setBasis(null);
    }
  }, [initialData, scope]);

  // Fetch the scoped frontier only when a non-default scope is chosen. The
  // unscoped view reuses the already-loaded overview (no extra request).
  useEffect(() => {
    if (!scope) {
      setScopeError(false);
      setScopeLoading(false);
      setScopeStatus('');
      setBasis(null);
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
        setBasis(payload.frontier_basis || null);
        setScopeLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setScopeError(true);
        setBasis(null);
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
          floor: finiteNumber(point.retention_floor),
          breaks: finiteNumber(point.num_breaks),
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

  // No all-channels chip: the owned channel is the only channel scope. Day
  // scoping does not exist on this endpoint yet; when it does its options
  // belong here beside the channel chip.
  const scopeOptions = useMemo(() => {
    if (!ownedScope) return [];
    return [{ value: ownedScope, labelHe: ownedChannel, labelEn: ownedChannel, icon: 'tv' }];
  }, [ownedScope, ownedChannel]);

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
  const isSavedFloorSelected =
    focusPoint &&
    focusPoint.floor !== null &&
    savedRetentionFloor !== null &&
    Math.abs(focusPoint.floor - Number(savedRetentionFloor)) < 1e-6;
  const canApply =
    focusPoint && focusPoint.floor !== null && !isSavedFloorSelected && typeof onApplyFloor === 'function';

  const activeScopeLabel = (() => {
    const match = scopeOptions.find((option) => option.value === scope);
    if (match) return he ? match.labelHe : match.labelEn;
    return pageText(locale, 'Auto-picked representative channel', 'ערוץ מייצג שנבחר אוטומטית');
  })();

  // Honest disclosure of what the plotted revenue actually is. Prefer the
  // backend's own basis.disclosure text when present; otherwise fall back to a
  // bilingual caption so the operator never reads the selected revenue as the
  // saved weekly plan total (a different, refined figure over all channels and
  // days) shown as the headline elsewhere.
  const basisDisclosure =
    basis && typeof basis.disclosure === 'string' && basis.disclosure.trim()
      ? basis.disclosure.trim()
      : pageText(
          locale,
          'Selected revenue here is a single representative-day estimate for the chosen scope, not the saved weekly total shown as your headline revenue across all channels and days.',
          'ההכנסה המסומנת כאן היא הערכה ליום ייצוגי יחיד עבור ההיקף הנבחר, ולא הסך השבועי השמור המוצג ככותרת ההכנסה על פני כל הערוצים והימים.',
        );

  return (
    <div className="analytics-panel frontier-panel frontier-scope-panel">
      <div className="panel-head">
        <h2>{copy.frontier}</h2>
        <span>{copy.frontierMode}</span>
      </div>

      <div className="frontier-scope-bar">
        {scopeOptions.length > 0 && (
          <div className="frontier-scope-control" role="group" aria-label={he ? 'היקף החזית' : 'Frontier scope'}>
            {scopeOptions.map((option) => {
              const active = option.value === scope;
              return (
                <Pressable
                  key={option.value}
                  type="button"
                  className={`frontier-scope-chip${active ? ' active' : ''}`}
                  aria-pressed={active}
                  onClick={() => {
                    setScope(option.value);
                    setSelectedIndex(null);
                  }}
                >
                  <Tv size={13} />
                  {he ? option.labelHe : option.labelEn}
                </Pressable>
              );
            })}
          </div>
        )}
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
                const floorLabel = point.floor !== null ? `${Math.round(point.floor * 100)}%` : '?';
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
                    aria-label={`${pageText(locale, 'Retention floor', 'רף שימור')} ${floorLabel}, ${formatCurrency(point.revenue, locale)}, ${formatPercent(point.retention, locale)}`}
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
              <text className="axis-label" x={4} y={PAD_Y + 4}>{formatCurrencyAxis(revMax, locale)}</text>
            </svg>
          </div>

          {/* The chart itself is hard-forced dir="ltr" so the plotted axes run
              low-to-high left-to-right. Keep the legend in that same chart space
              (always ltr) so the X label sits under the X axis and the Y label
              under the Y axis. An rtl legend here would mirror the labels off
              their axes even though the numbers are correct. */}
          <div className="frontier-axis-legend" dir="ltr">
            <span>{pageText(locale, 'X: average retention', 'ציר X: שימור ממוצע')}</span>
            <span>{pageText(locale, 'Y: revenue (representative day)', 'ציר Y: הכנסה (יום מייצג)')}</span>
          </div>

          <p className="frontier-scope-hint frontier-basis-note">
            {basisDisclosure}
          </p>

          {focusPoint && (
            <div className="frontier-point-readout">
              {/* Four stat tiles in one row, each with a one-line explanation
                  tooltip, replacing the old two-column grid whose empty gray
                  cells read as missing data and whose boundaries were unclear. */}
              <div className="frontier-point-tiles">
                <Tooltip title={pageText(locale, 'The constraint this point was optimized under: no break in the plan may fall below this retention floor.', 'האילוץ שתחתיו הנקודה חושבה: אף ברייק בתוכנית אינו רשאי לרדת מתחת לרף השימור הזה.')} arrow placement="bottom">
                  <div className="frontier-point-tile">
                    <span>{pageText(locale, 'Retention floor tested', 'רף שימור שנבדק')}</span>
                    <strong className="numeric"><Figure>{focusPoint.floor !== null ? `${Math.round(focusPoint.floor * 100)}%` : '-'}</Figure></strong>
                  </div>
                </Tooltip>
                <Tooltip title={pageText(locale, 'What the resulting plan actually averaged, naturally far above the floor because the floor only blocks the worst breaks.', 'הממוצע שהתוכנית שהתקבלה השיגה בפועל, גבוה בהרבה מהרף באופן טבעי כי הרף חוסם רק את הברייקים הגרועים ביותר.')} arrow placement="bottom">
                  <div className="frontier-point-tile">
                    <span>{pageText(locale, 'Retention achieved', 'שימור שהושג')}</span>
                    <strong className="numeric"><Figure>{formatPercent(focusPoint.retention, locale)}</Figure></strong>
                  </div>
                </Tooltip>
                <Tooltip title={pageText(locale, 'Projected revenue for a single representative day in this scope, not the saved weekly total.', 'הכנסה צפויה ליום מייצג יחיד בהיקף הזה, לא הסך השבועי השמור.')} arrow placement="bottom">
                  <div className="frontier-point-tile">
                    <span>{pageText(locale, 'Revenue (representative day)', 'הכנסה (יום מייצג)')}</span>
                    <strong className="numeric"><Figure>{formatCurrency(focusPoint.revenue, locale)}</Figure></strong>
                  </div>
                </Tooltip>
                <Tooltip title={pageText(locale, 'How many breaks the plan of this point airs on the representative day.', 'כמה ברייקים משדרת התוכנית של הנקודה הזו ביום המייצג.')} arrow placement="bottom">
                  <div className="frontier-point-tile">
                    <span>{pageText(locale, 'Breaks', 'ברייקים')}</span>
                    <strong className="numeric"><Figure>{focusPoint.breaks !== null ? formatNumber(focusPoint.breaks, locale) : '-'}</Figure></strong>
                  </div>
                </Tooltip>
              </div>
              <div className="frontier-point-action">
                {isSavedFloorSelected ? (
                  <span className="frontier-point-saved">
                    <Check size={13} />
                    {pageText(locale, 'Current saved floor', 'הרף השמור הנוכחי')}
                  </span>
                ) : (
                  <Button
                    className="secondary-button compact"
                    type="button"
                    variant="outlined"
                    disabled={!canApply || applyState === 'saving'}
                    onClick={() => canApply && onApplyFloor(focusPoint.floor)}
                  >
                    {applyState === 'saving'
                      ? pageText(locale, 'Applying...', 'מחיל...')
                      : pageText(locale, 'Apply this floor', 'החל רף זה')}
                  </Button>
                )}
                {safeSelected === null && (
                  <span className="frontier-point-hint">
                    {pageText(locale, 'Click a point to inspect and apply its retention floor.', 'לחצו על נקודה כדי לבחון ולהחיל את רף השימור שלה.')}
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
