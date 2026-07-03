// Formatting helpers for the schedule editor. Kept in their own module so the
// editor stays lean and so the seconds-precise clock and human-readable offset
// live in one honest place. Every value here is derived from a real second
// count the editor already computed (programme start plus the dragged offset);
// nothing is invented.

// A break start rendered as a full HH:MM:SS clock. The wire data carries
// minute-resolution start times, so an untouched break reads HH:MM:00 (its true
// second), while a dragged break reads the exact snapped second (for example
// 12:34:30). This is the position the operator needs to read at a glance.
export function secondsToClock(seconds) {
  const max = (47 * 60 + 59) * 60 + 59;
  const safe = Math.max(0, Math.min(max, Math.round(seconds)));
  const hour = Math.floor(safe / 3600) % 24;
  const minute = Math.floor((safe % 3600) / 60);
  const second = safe % 60;
  return `${pad(hour)}:${pad(minute)}:${pad(second)}`;
}

// The offset from the programme start in a human form: 3m 20s in English,
// 3 דק׳ 20 שנ׳ in Hebrew. Minutes are dropped when zero so a short offset reads
// as 45s rather than 0m 45s.
export function humanOffset(seconds, locale) {
  const safe = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(safe / 60);
  const remainder = safe % 60;
  const he = locale === 'he';
  const minuteUnit = he ? 'דק׳' : 'm';
  const secondUnit = he ? 'שנ׳' : 's';
  const parts = [];
  if (minutes > 0) parts.push(`${minutes} ${minuteUnit}`);
  parts.push(`${remainder} ${secondUnit}`);
  return parts.join(' ');
}

// The programme window as a compact start to end range using the real clock
// strings already on the programme row. An en dash is avoided per house style;
// a plain hyphen with spaces reads cleanly in both directions.
export function windowRange(startTime, endTime) {
  const start = startTime || '';
  const end = endTime || '';
  if (!start && !end) return '';
  return `${start} - ${end}`;
}

// Localized programme class label. Mirrors the dashboard's own mapping so the
// editor names the class naturally in Hebrew; an unknown class falls back to
// the raw value rather than an invented name.
export function programClassLabel(type, locale) {
  const labels = {
    News: 'חדשות',
    Reality: 'ריאליטי',
    Drama: 'דרמה',
    Sports: 'ספורט',
    Comedy: 'קומדיה',
    Promo: 'פרומו',
    Kids: 'ילדים',
    Children: 'ילדים',
    Other: 'אחר',
    Mixed: 'מעורב',
  };
  return locale === 'he' ? labels[type] || type || '' : type || '';
}

// Localized break-position label (first, middle, last and so on). Same honest
// fallback to the raw value when a position is not in the map.
export function breakPositionLabel(position, locale) {
  const labels = {
    first: 'ראשון',
    early: 'מוקדם',
    middle: 'אמצעי',
    late: 'מאוחר',
    last: 'אחרון',
  };
  return locale === 'he' ? labels[position] || position || '' : position || '';
}

function pad(value) {
  return String(value).padStart(2, '0');
}
