import { fallbackSettings } from './fallbacks';
import { finiteNumber, formatNumber, pageText } from './format';
import { dayKeys } from './plan-model';

export function impactSegmentLabel(segment, locale) {
  const labels = {
    first: pageText(locale, 'First break', 'ברייק ראשון'),
    early: pageText(locale, 'Early break', 'ברייק מוקדם'),
    middle: pageText(locale, 'Middle break', 'ברייק אמצעי'),
    last: pageText(locale, 'Last break', 'ברייק אחרון'),
    late: pageText(locale, 'Late break', 'ברייק מאוחר'),
    short: pageText(locale, 'Short', 'קצר'),
    standard: pageText(locale, 'Standard', 'סטנדרטי'),
    medium: pageText(locale, 'Medium', 'בינוני'),
    long: pageText(locale, 'Long', 'ארוך'),
    News: pageText(locale, 'News', 'חדשות'),
    Reality: pageText(locale, 'Reality', 'ריאליטי'),
    Drama: pageText(locale, 'Drama', 'דרמה'),
    Sports: pageText(locale, 'Sports', 'ספורט'),
    Comedy: pageText(locale, 'Comedy', 'קומדיה'),
    Promo: pageText(locale, 'Promo', 'פרומו'),
    Other: pageText(locale, 'Other', 'אחר'),
  };
  // Fall back to the shared genre map so classifier vocabulary stays localized
  // here too; configured class names (for example rate-card tiers) pass through.
  return labels[segment] || programTypeLabel(segment, locale) || segment;
}

export function impactSourceLabel(source, metadata, locale) {
  const measuredBreaks = finiteNumber(metadata?.total_breaks_measured);
  const suffix = measuredBreaks
    ? pageText(locale, ` · ${formatNumber(measuredBreaks, locale)} measured breaks`, ` · ${formatNumber(measuredBreaks, locale)} ברייקים נמדדו`)
    : '';
  const labels = {
    measured_detrended_pooled: pageText(locale, 'Measured retention model', 'מודל שימור מדוד'),
    measured_coefficients: pageText(locale, 'Measured retention model', 'מודל שימור מדוד'),
    legacy_csv: pageText(locale, 'Legacy impact extract', 'תוצר השפעה קודם'),
    unavailable: pageText(locale, 'Model source unavailable', 'מקור המודל לא זמין'),
  };
  return `${labels[source] || pageText(locale, 'Impact model', 'מודל השפעה')}${suffix}`;
}

export function complianceUnitLabel(unit, locale = 'en') {
  const labels = {
    en: {
      'minutes/hour': 'min/hour',
      'breaks/hour': 'breaks/hour',
      minutes: 'min',
      'minutes/day': 'min/day',
      'breaks/day': 'breaks/day',
      '%': '%',
    },
    he: {
      'minutes/hour': 'דק׳ לשעה',
      'breaks/hour': 'ברייקים לשעה',
      minutes: 'דק׳',
      'minutes/day': 'דק׳ ביום',
      'breaks/day': 'ברייקים ביום',
      '%': '%',
    },
  };
  return labels[locale === 'he' ? 'he' : 'en'][unit] || unit || '';
}

export function complianceDisclaimer(disclaimer, locale = 'en') {
  if (locale === 'he') {
    return 'בסיס הבקרה ניתן להגדרה. יש לאמת מול ייעוץ משפטי ומדיניות הערוץ לפני שימוש בפרודקשן.';
  }
  return disclaimer || fallbackSettings.notes;
}

export function daypartLabel(daypart, locale) {
  const labels = {
    Morning: 'בוקר',
    Daytime: 'יום',
    Access: 'לפני פריים',
    Primetime: 'פריים טיים',
    'Late night': 'לילה',
  };
  return locale === 'he' ? labels[daypart] || daypart : daypart;
}

export const PROGRAM_TYPE_LABELS_HE = {
  News: 'חדשות',
  Reality: 'ריאליטי',
  Drama: 'דרמה',
  Sports: 'ספורט',
  Comedy: 'קומדיה',
  Promo: 'פרומו',
  Kids: 'ילדים',
  Children: 'ילדים',
  Digital: 'דיגיטל',
  Documentary: 'דוקומנטרי',
  Lifestyle: 'לייפסטייל',
  'Morning Program': 'תוכנית בוקר',
  Music: 'מוזיקה',
  Religious: 'תוכן דתי',
  'Special Event': 'אירוע מיוחד',
  'Talk Show': 'תוכנית אירוח',
  Other: 'אחר',
  Mixed: 'מעורב',
};

export function programTypeLabel(type, locale) {
  // Covers the full classifier vocabulary observed in the live payloads, so
  // genre names never leak as raw English into the Hebrew planning surfaces.
  return locale === 'he' ? PROGRAM_TYPE_LABELS_HE[type] || type || '' : type || '';
}

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

export function breakLengthLabel(length, locale) {
  const labels = {
    short: 'קצר',
    standard: 'סטנדרטי',
    medium: 'בינוני',
    long: 'ארוך',
  };
  return locale === 'he' ? labels[length] || length || '' : length || '';
}

export function scenarioNameLabel(name, locale) {
  const labels = {
    Balanced: 'מאוזן',
    'Revenue priority': 'מקסום הכנסה',
    'Retention guardrail': 'הגנת שימור',
  };
  return locale === 'he' ? labels[name] || name || '' : name || '';
}

export function localizedModelText(text, locale) {
  if (locale !== 'he' || !text) {
    return text || '';
  }
  const translated = Object.keys(PROGRAM_TYPE_LABELS_HE)
    .sort((left, right) => right.length - left.length)
    .reduce((value, type) => value.replace(
      new RegExp(`\\b${type.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\b`, 'g'),
      programTypeLabel(type, locale),
    ), String(text));
  return translated
    .replace(/\bRevenue priority\b/g, 'מקסום הכנסה')
    .replace(/\bRetention guardrail\b/g, 'הגנת שימור')
    .replace(/\bBalanced\b/g, 'מאוזן')
    .replace(/\bmedium\b/gi, 'בינוני')
    .replace(/\bstandard\b/gi, 'סטנדרטי')
    .replace(/\bshort\b/gi, 'קצר')
    .replace(/\blong\b/gi, 'ארוך')
    .replace(/\bmiddle\b/gi, 'אמצעי')
    .replace(/\bearly\b/gi, 'מוקדם')
    .replace(/\bfirst\b/gi, 'ראשון')
    .replace(/\blast\b/gi, 'אחרון')
    .replace(/\blate\b/gi, 'מאוחר');
}

export function dayLabel(day, locale) {
  const labels = locale === 'he' ? ['ב׳', 'ג׳', 'ד׳', 'ה׳', 'ו׳', 'ש׳', 'א׳'] : dayKeys;
  const index = dayKeys.indexOf(day);
  return labels[index] || day;
}

export function gridAxisLabel(axis, locale) {
  const labels = {
    day: pageText(locale, 'Days', 'ימים'),
    daypart: pageText(locale, 'Dayparts', 'רצועות'),
    hour: pageText(locale, 'Hours', 'שעות'),
    type: pageText(locale, 'Formats', 'סוגי תוכנית'),
  };
  return labels[axis] || labels.day;
}

export function riskLabel(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return 'Unknown';
  if (score >= 68) return 'High';
  if (score >= 38) return 'Medium';
  return 'Low';
}

export function recommendationTitle(recommendation, locale) {
  if (locale !== 'he') {
    return recommendation?.title || 'Review placement';
  }
  const title = recommendation?.title_he || recommendation?.title || '';
  const fallbackTitles = {
    'Increase selected primetime break by 1 spot': 'הוספת ספוט לברייק פריים נבחר',
    'Shift a late break earlier in the hour': 'הקדמת ברייק מאוחר בתוך השעה',
    'Hold break length in news block': 'שמירת אורך הברייק במהדורת חדשות',
  };
  return fallbackTitles[title] || localizedModelText(title || 'בדיקת מיקום ברייק', locale);
}

export function recommendationRationale(recommendation, locale) {
  if (locale !== 'he') {
    return recommendation?.rationale || 'Recommendation rationale unavailable.';
  }
  const rationale = recommendation?.rationale_he || recommendation?.rationale || '';
  const fallbackRationales = {
    'Demand is concentrated in the selected slot while retention guardrail remains compliant.':
      'הביקוש מרוכז בסלוט הנבחר, ובקרת השימור עדיין תקינה.',
    'Earlier placement improves sell-through with limited churn exposure.':
      'הקדמת המיקום משפרת מכירה בלי להגדיל משמעותית את חשיפת השימור.',
    'News retention is strong, but incremental minutes are below target yield.':
      'שימור הצפייה בחדשות חזק, אך דקות נוספות אינן מגיעות לתשואת היעד.',
  };
  return localizedModelText(
    fallbackRationales[rationale] ||
      rationale ||
      'המערכת מזהה הזדמנות הכנסה, אך ההחלטה נשמרת לבקרה אנושית מול מגבלות שימור ותאימות.',
    locale,
  );
}
