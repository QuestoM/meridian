import React, { useEffect, useRef, useState } from 'react';
import {
  Numeric,
  finiteNumber,
  formatCurrency,
  formatCurrencyAxis,
  formatNumber,
  formatPercent,
  pageText,
} from '../../shell/format';
import { normalizeRows } from '../../shell/plan-model';

export function FrontierPanel({ data, copy, locale, loading = false, operatorChannel = '', status = '', netPoint = null }) {
  const chartFrameRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(760);
  const [activePointIndex, setActivePointIndex] = useState(null);
  const height = 224;
  const padX = 46;
  const padY = 30;
  const ownedChannel = String(operatorChannel || '').trim();
  // The frontier payload is an array of sweep points today; the net-focused
  // point may arrive as a net_point key on an object payload, as a sibling prop,
  // or embedded in the array under id 'net_focused'. Accept all three shapes and
  // render honestly from whichever is present, without inventing a point.
  const rawRows = Array.isArray(data) ? data : normalizeRows(data?.points);
  const netSource = (!Array.isArray(data) && data && typeof data === 'object' ? data.net_point : null) || netPoint || rawRows.find((row) => String(row?.id || '') === 'net_focused') || null;
  const points = rawRows
    .filter((row) => String(row?.id || '') !== 'net_focused')
    .map((point) => ({
      retention: finiteNumber(point.retention),
      revenue: finiteNumber(point.revenue),
      selected: Boolean(point.selected),
    }))
    .filter((point) => point.retention !== null && point.revenue !== null);
  const netFocusPoint = netSource
    ? { retention: finiteNumber(netSource.retention), revenue: finiteNumber(netSource.revenue) }
    : null;
  const hasNetPoint = Boolean(netFocusPoint && netFocusPoint.retention !== null && netFocusPoint.revenue !== null);
  // The saved settings anchor the sweep, so the point flagged selected is the
  // current plan's operating point (the sweep runs at the saved revenue weight).
  const selectedPoint = points.find((point) => point.selected) || points[points.length - 1];
  const currentPlanLabel = pageText(locale, 'Current plan', 'התוכנית הנוכחית');
  const netFocusLabel = pageText(locale, 'Net focused', 'ממוקד נטו');
  const showSkeleton = loading || points.length < 2 || !selectedPoint;
  // Honest empty state: when no channel is owned the backend returns no frontier
  // (it never forecasts an arbitrary or all-channels number). Direct the operator
  // to pick their channel instead of showing a misleading curve.
  const showPickChannel = !loading && !ownedChannel;
  // The frontier is a slow optimizer sweep computed in the background. When the
  // backend reports it is still computing and no points have arrived yet, show an
  // honest "being computed" state rather than an empty skeleton with no curve.
  const showComputing = !loading && ownedChannel && status === 'computing' && points.length < 2;
  // Subtitle: name the owned channel the curve forecasts, so the operator can see
  // at a glance the projection is scoped to their inventory only.
  const modeLabel = ownedChannel ? `${copy.frontierMode} · ${ownedChannel}` : copy.frontierMode;

  useEffect(() => {
    const frame = chartFrameRef.current;
    if (!frame) return undefined;
    const updateWidth = () => {
      setChartWidth(Math.max(360, Math.round(frame.getBoundingClientRect().width)));
    };
    updateWidth();
    if (typeof ResizeObserver === 'undefined') {
      return undefined;
    }
    const observer = new ResizeObserver(updateWidth);
    observer.observe(frame);
    return () => observer.disconnect();
  }, [showSkeleton]);

  function paddedDomain(values, fallbackSpan, padRatio = 0.12) {
    const finiteValues = values.filter((value) => Number.isFinite(value));
    if (!finiteValues.length) {
      return [0, fallbackSpan || 1];
    }
    const rawMin = Math.min(...finiteValues);
    const rawMax = Math.max(...finiteValues);
    const rawSpan = rawMax - rawMin;
    // Frame the actual data range. The floor only prevents a zero-height axis on
    // a single or flat point; it is kept tiny relative to the data so small but
    // real differences stay visible instead of being squashed into a fixed window.
    const scaleFloor = Math.max(Math.abs(rawMax), Math.abs(rawMin)) * 0.04;
    const span = Math.max(rawSpan, scaleFloor, 1e-9);
    const center = (rawMin + rawMax) / 2;
    const padding = span * padRatio;
    return [center - span / 2 - padding, center + span / 2 + padding];
  }

  const width = chartWidth;
  const domainPoints = hasNetPoint ? points.concat([netFocusPoint]) : points;
  const [retentionMin, retentionMax] = paddedDomain(domainPoints.map((point) => point.retention), 0.8);
  const [revenueMin, revenueMax] = paddedDomain(domainPoints.map((point) => point.revenue), 1);
  // Frame to the data range (auto-scale). Do not pin to 0 or a fixed window, so
  // small revenue/retention differences are visible rather than flattened.
  const minRetention = retentionMin;
  const maxRetention = retentionMax;
  const minRevenue = revenueMin;
  const maxRevenue = revenueMax;
  const xFor = (retention) =>
    padX + ((retention - minRetention) / Math.max(maxRetention - minRetention, 1e-9)) * (width - padX * 2);
  const yFor = (revenue) =>
    height - padY - ((revenue - minRevenue) / Math.max(maxRevenue - minRevenue, 1e-9)) * (height - padY * 2);
  const path = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(point.retention).toFixed(1)} ${yFor(point.revenue).toFixed(1)}`)
    .join(' ');
  const minRetentionLabel = formatPercent(minRetention, locale);
  const maxRetentionLabel = formatPercent(maxRetention, locale);
  const safeActiveIndex =
    activePointIndex !== null && points[activePointIndex] ? activePointIndex : null;
  const activePoint = safeActiveIndex !== null ? points[safeActiveIndex] : selectedPoint;
  const activeX = activePoint ? xFor(activePoint.retention) : 0;
  const activeY = activePoint ? yFor(activePoint.revenue) : 0;
  const revenueDelta = activePoint && selectedPoint ? activePoint.revenue - selectedPoint.revenue : 0;
  const retentionDelta = activePoint && selectedPoint ? activePoint.retention - selectedPoint.retention : 0;
  const tooltipClass = [
    'frontier-tooltip',
    activeX > width * 0.68 ? 'edge-right' : activeX < width * 0.32 ? 'edge-left' : '',
    activeY < 96 ? 'below' : '',
  ].filter(Boolean).join(' ');
  const hoverLabel = activePoint?.selected
    ? currentPlanLabel
    : pageText(locale, `Alternative ${safeActiveIndex + 1}`, `חלופה ${safeActiveIndex + 1}`);

  function handleChartPointerMove(event) {
    const svg = event.currentTarget.ownerSVGElement;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const cursor = point.matrixTransform(matrix.inverse());
    const nearestIndex = points.reduce((bestIndex, item, index) => {
      const bestPoint = points[bestIndex];
      const distance = Math.abs(xFor(item.retention) - cursor.x);
      const bestDistance = Math.abs(xFor(bestPoint.retention) - cursor.x);
      return distance < bestDistance ? index : bestIndex;
    }, 0);
    setActivePointIndex((current) => (current === nearestIndex ? current : nearestIndex));
  }

  return (
    <div className="analytics-panel frontier-panel">
      <div className="panel-head">
        <h2>{copy.frontier}</h2>
        <span>{modeLabel}</span>
      </div>
      {showPickChannel ? (
        <div className="frontier-empty">{copy.frontierPickChannel}</div>
      ) : showComputing ? (
        <div className="frontier-empty">{copy.frontierComputing}</div>
      ) : showSkeleton ? (
        <div className="frontier-skeleton" aria-hidden="true" />
      ) : (
        <>
          <div ref={chartFrameRef} className="frontier-chart-frame chart-ltr" dir="ltr">
            <svg
              className="frontier-svg"
              viewBox={`0 0 ${width} ${height}`}
              role="img"
              aria-label={pageText(locale, 'Revenue vs retention', 'הכנסה מול שימור')}
            >
              {[0, 1, 2, 3].map((line) => {
                const y = padY + line * ((height - padY * 2) / 3);
                return <line key={`h-${line}`} x1={padX} x2={width - padX} y1={y} y2={y} />;
              })}
              {[0, 1, 2, 3, 4].map((line) => {
                const x = padX + line * ((width - padX * 2) / 4);
                return <line key={`v-${line}`} x1={x} x2={x} y1={padY} y2={height - padY} />;
              })}
              <path d={path} />
              {safeActiveIndex !== null && activePoint && (
                <g className="frontier-hover-guides" aria-hidden="true">
                  <line x1={activeX} x2={activeX} y1={padY} y2={height - padY} />
                  <line x1={padX} x2={width - padX} y1={activeY} y2={activeY} />
                </g>
              )}
              {selectedPoint && (
                <circle
                  className="current-plan-ring"
                  cx={xFor(selectedPoint.retention)}
                  cy={yFor(selectedPoint.revenue)}
                  r={10}
                  aria-hidden="true"
                />
              )}
              {points.map((point, index) => (
                <circle
                  key={`${point.retention}-${point.revenue}-${index}`}
                  className={[
                    point.selected ? 'selected-point' : '',
                    safeActiveIndex === index ? 'active-point' : '',
                  ].filter(Boolean).join(' ')}
                  cx={xFor(point.retention)}
                  cy={yFor(point.revenue)}
                  r={safeActiveIndex === index ? 7 : point.selected ? 6 : 4}
                  tabIndex={0}
                  aria-label={`${point.selected ? `${currentPlanLabel}: ` : ''}${formatCurrency(point.revenue, locale)}, ${formatPercent(point.retention, locale)}`}
                  onFocus={() => setActivePointIndex(index)}
                  onBlur={() => setActivePointIndex(null)}
                />
              ))}
              {hasNetPoint && (
                <circle
                  className="net-focused-point"
                  cx={xFor(netFocusPoint.retention)}
                  cy={yFor(netFocusPoint.revenue)}
                  r={6}
                  tabIndex={0}
                  aria-label={`${netFocusLabel}: ${formatCurrency(netFocusPoint.revenue, locale)}, ${formatPercent(netFocusPoint.retention, locale)}`}
                />
              )}
              <rect
                className="frontier-hit-area"
                x={padX}
                y={padY}
                width={width - padX * 2}
                height={height - padY * 2}
                onPointerMove={handleChartPointerMove}
                onPointerLeave={() => setActivePointIndex(null)}
              />
              <text className="axis-label" x={padX} y={height - 6}>{minRetentionLabel}</text>
              <text className="axis-label axis-label-end" x={width - padX} y={height - 6}>{maxRetentionLabel}</text>
              <text className="axis-label" x={4} y={padY + 4}>{formatCurrencyAxis(maxRevenue, locale)}</text>
            </svg>
            {safeActiveIndex !== null && activePoint && (
              <div
                className={tooltipClass}
                dir={locale === 'he' ? 'rtl' : 'ltr'}
                style={{ left: `${(activeX / width) * 100}%`, top: `${(activeY / height) * 100}%` }}
              >
                <span>{hoverLabel}</span>
                <strong><Numeric>{formatCurrency(activePoint.revenue, locale)}</Numeric></strong>
                <small><Numeric>{formatPercent(activePoint.retention, locale)}</Numeric></small>
                <div className="frontier-tooltip-deltas">
                  <span>{pageText(locale, 'Revenue delta', 'פער הכנסה')}</span>
                  <strong><Numeric>{revenueDelta > 0 ? '+' : ''}{formatCurrency(revenueDelta, locale)}</Numeric></strong>
                  <span>{pageText(locale, 'Retention delta', 'פער שימור')}</span>
                  <strong><Numeric>{retentionDelta > 0 ? '+' : ''}{formatNumber(retentionDelta, locale)}pp</Numeric></strong>
                </div>
              </div>
            )}
          </div>
          <div className="frontier-legend" aria-hidden="true">
            {selectedPoint && (
              <span className="frontier-legend-chip current"><i />{currentPlanLabel}</span>
            )}
            {hasNetPoint && (
              <span className="frontier-legend-chip net"><i />{netFocusLabel}</span>
            )}
          </div>
          {hasNetPoint && (
            <p className="frontier-net-caption">{pageText(locale, 'Past the net focused point, toward higher gross, every additional gross shekel costs more than a shekel in retention cost.', 'מעבר לנקודה ממוקדת הנטו, לכיוון ברוטו גבוה יותר, כל שקל ברוטו נוסף עולה יותר משקל בעלות שימור.')}</p>
          )}
          <div className="frontier-readout">
            <div>
              <span>{safeActiveIndex !== null ? pageText(locale, 'Hovered revenue', 'הכנסה בחלופה') : pageText(locale, 'Current plan revenue', 'הכנסה בתוכנית הנוכחית')}</span>
              <strong><Numeric>{formatCurrency(activePoint.revenue, locale)}</Numeric></strong>
            </div>
            <div>
              <span>{safeActiveIndex !== null ? pageText(locale, 'Hovered retention', 'שימור בחלופה') : pageText(locale, 'Projected retention', 'שימור צפוי')}</span>
              <strong><Numeric>{formatPercent(activePoint.retention, locale)}</Numeric></strong>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default FrontierPanel;
