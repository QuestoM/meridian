import { pageText } from './format';

export const dayKeys = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
export const daypartKeys = ['Morning', 'Daytime', 'Access', 'Primetime', 'Late night'];

export function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
}

export function programKey(channel, program) {
  return [channel, program?.day, program?.time, program?.title].map((part) => String(part || '')).join('|');
}

export function flattenScheduleRows(rows) {
  return normalizeRows(rows).flatMap((row) =>
    normalizeRows(row.programs).map((program) => ({
      ...program,
      channel: row.channel,
      key: programKey(row.channel, program),
    })),
  );
}

export function daypartForTime(time) {
  const hour = hourFromTime(time);
  if (hour >= 6 && hour < 12) return 'Morning';
  if (hour >= 12 && hour < 17) return 'Daytime';
  if (hour >= 17 && hour < 20) return 'Access';
  if (hour >= 20 && hour < 23) return 'Primetime';
  return 'Late night';
}

export function hourFromTime(time) {
  const hour = Number(String(time || '0:00').split(':')[0]);
  return Number.isFinite(hour) ? Math.max(0, Math.min(23, hour)) : 0;
}

// Derive the active planning window from the loaded schedule rather than a
// hardcoded literal. Returns a real date range when the schedule carries dates,
// otherwise a neutral label with no fabricated dates.
export function planningWeekLabel(schedule, locale) {
  const programs = normalizeRows(schedule?.break_operations?.programs);
  const dates = programs
    .map((program) => program?.date)
    .filter(Boolean)
    .sort();
  if (dates.length === 0) {
    return pageText(locale, 'Planning week', 'שבוע התכנון');
  }
  const format = (value) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleDateString(locale === 'he' ? 'he-IL' : 'en-US', { month: 'short', day: 'numeric' });
  };
  const first = dates[0];
  const last = dates[dates.length - 1];
  return first === last ? format(first) : `${format(first)} - ${format(last)}`;
}
