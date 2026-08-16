export function timeToSeconds(time) {
  const [hour, minute] = String(time || '00:00').split(':').map((part) => Number(part));
  const safeHour = Number.isFinite(hour) ? Math.max(0, Math.min(47, hour)) : 0;
  const safeMinute = Number.isFinite(minute) ? Math.max(0, Math.min(59, minute)) : 0;
  return (safeHour * 60 + safeMinute) * 60;
}

export function normalizeEditorRows(value) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.rows)) return value.rows;
  return [];
}

export function snapSeconds(value, grid, min, max) {
  const snapped = Math.round(value / grid) * grid;
  return Math.max(min, Math.min(max, snapped));
}
