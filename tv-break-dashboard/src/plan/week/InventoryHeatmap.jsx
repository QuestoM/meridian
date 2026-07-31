import React from 'react';

export function InventoryHeatmap({ copy, locale }) {
  // No per-daypart-per-weekday revenue source is exposed by the API today, so the
  // panel renders an honest empty state rather than fabricated demo numbers. When
  // the API gains a real daypart x weekday revenue grid, render it here.
  return (
    <div className="analytics-panel heatmap-panel chart-ltr" dir={locale === 'he' ? 'rtl' : 'ltr'}>
      <div className="panel-head">
        <h2>{copy.heatmap}</h2>
        <span>{copy.opportunity}</span>
      </div>
      <div className="heatmap-empty">{copy.heatmapEmpty}</div>
    </div>
  );
}

export default InventoryHeatmap;
