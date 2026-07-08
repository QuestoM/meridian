import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CacheProvider } from '@emotion/react';
import createCache from '@emotion/cache';
import { prefixer } from 'stylis';
import rtlPlugin from '@mui/stylis-plugin-rtl';
import './coherence.css';
import DateField from './DateField';
import {
  Button,
  Checkbox,
  CssBaseline,
  FormControl,
  FormControlLabel,
  IconButton,
  InputAdornment,
  InputLabel,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Select,
  Slider,
  Switch,
  TextField,
  ThemeProvider,
  Tooltip,
  createTheme,
} from '@mui/material';
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Bell,
  Bot,
  BookOpen,
  CalendarDays,
  Check,
  ChevronDown,
  CircleDollarSign,
  ClipboardCheck,
  Clock3,
  Coins,
  Database,
  Download,
  FileBarChart,
  Gauge,
  GitCompare,
  History,
  Info,
  KeyRound,
  Languages,
  LayoutGrid,
  ListChecks,
  LogOut,
  Save,
  Play,
  Printer,
  RefreshCcw,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  Tv,
  Upload,
  Users,
  X,
} from 'lucide-react';

import UploadCenter from './UploadCenter';
import AdvertisersManager from './AdvertisersManager';
import PricingManager from './PricingManager';
import OverrideConsole from './OverrideConsole';
import ScheduleEditor, { ConstraintBuilder } from './ScheduleEditor';
import ScheduleInspector from './ScheduleInspector';
import {
  ScheduleTrackSurface,
  ProgrammeBand,
  ZoomControl,
  useScheduleZoom,
  useSegmentAnchors,
} from './schedule-track-view';
import { timeWindow, spanStyle } from './schedule-track';
import BreakChip from './BreakChip';
import FrontierScopeChart from './FrontierScopeChart';
import YieldView from './YieldView';
import { NetComparisonCard, YieldMoneyPanel } from './MoneyWaterfall';
import ScenarioCompare from './ScenarioCompare';
import GoldBreakManager from './GoldBreakManager';
import MakeGoodAlerts from './MakeGoodAlerts';
import ActivityFeed from './ActivityFeed';
import AssistantPanel from './AssistantPanel';
import VersionsPage from './VersionsPage';
import ScheduleStalenessBanner from './ScheduleStalenessBanner';
import Login, {
  ChangePasswordDialog,
  MIN_PASSWORD_LENGTH,
  createAccount,
  deleteAccount,
  fetchAccounts,
  fetchMe,
  requestLogout,
  resetAccountPassword,
  roleLabel,
} from './Login';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';
const LazyDataGrid = React.lazy(() => import('@mui/x-data-grid').then((module) => ({ default: module.DataGrid })));

const ltrCache = createCache({ key: 'mui' });
const rtlCache = createCache({
  key: 'muirtl',
  stylisPlugins: [prefixer, rtlPlugin],
});

function createKairosTheme(direction) {
  return createTheme({
    direction,
    palette: {
      mode: 'light',
      background: {
        default: '#f7f8fa',
        paper: '#ffffff',
      },
      text: {
        primary: '#111827',
        secondary: '#5b6573',
      },
      primary: {
        main: '#0d1b2a',
      },
      success: {
        main: '#0f8b7e',
      },
      warning: {
        main: '#b86e00',
      },
      divider: '#dde2e8',
    },
    shape: {
      borderRadius: 6,
    },
    typography: {
      fontFamily:
        'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      button: {
        textTransform: 'none',
        letterSpacing: 0,
        fontWeight: 620,
      },
    },
    components: {
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: {
            minHeight: 34,
            borderRadius: 6,
            fontSize: 12,
            lineHeight: 1,
            boxShadow: 'none',
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            width: 34,
            height: 34,
            borderRadius: 6,
            color: '#111827',
          },
        },
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            backgroundColor: '#ffffff',
            fontSize: 12,
          },
          input: {
            paddingTop: 8,
            paddingBottom: 8,
          },
        },
      },
      MuiInputLabel: {
        styleOverrides: {
          root: {
            fontSize: 12,
          },
        },
      },
      MuiDataGrid: {
        styleOverrides: {
          root: {
            border: 0,
            fontSize: 12,
            color: '#111827',
          },
          columnHeaders: {
            backgroundColor: '#fbfcfd',
            color: '#5b6573',
            fontSize: 11,
            fontWeight: 700,
          },
          cell: {
            borderColor: '#dde2e8',
          },
          row: {
            '&:hover': {
              backgroundColor: '#fbfcfd',
            },
          },
        },
      },
      MuiTooltip: {
        defaultProps: {
          // Hebrew tooltips read right-to-left; the popper is portaled outside
          // the rtl shell, so the bubble needs the direction set explicitly.
          slotProps: { tooltip: { dir: direction } },
        },
      },
      // Select/Menu/Popover portal their list to document.body, outside the rtl
      // shell, so without an explicit direction they open left-to-right in Hebrew.
      MuiPopover: {
        defaultProps: { slotProps: { paper: { dir: direction } } },
      },
      MuiMenu: {
        defaultProps: { slotProps: { paper: { dir: direction } } },
      },
      MuiSelect: {
        defaultProps: { MenuProps: { slotProps: { paper: { dir: direction } } } },
      },
    },
  });
}

const fallbackSettings = {
  profile_name: 'Israel commercial TV',
  locale: 'he',
  direction: 'rtl',
  chart_direction: 'ltr',
  timezone: 'Asia/Jerusalem',
  currency: 'ILS',
  effective_date: '2026-06-14',
  regulatory_source_url: 'https://www.rashut2.org.il/',
  max_ad_minutes_per_hour: 12,
  max_breaks_per_hour: 4,
  min_break_spacing_minutes: 7,
  min_retention_floor: 0.72,
  risk_lambda: 0.0,
  max_daily_ad_minutes: 160,
  protected_program_types: ['News', 'Kids', 'Children'],
  protected_program_max_ad_minutes_per_hour: 8,
  sponsorships_enabled: true,
  gold_breaks_enabled: true,
  gold_breaks_max_per_day: 3,
  require_manual_approval: true,
  notes: 'Configurable baseline. Validate with current counsel and broadcaster policy before production use.',
};

const fallbackCompliance = {
  profile: fallbackSettings.profile_name,
  effective_date: fallbackSettings.effective_date,
  source_url: fallbackSettings.regulatory_source_url,
  // API offline: there is no schedule to evaluate, so report unknown rather than
  // asserting compliance against invented observed values.
  status: 'unknown',
  disclaimer: fallbackSettings.notes,
  checks: [],
};

const copyByLocale = {
  en: {
    nav: {
      Overview: 'Overview',
      Optimizer: 'Optimizer',
      Schedule: 'Schedule',
      Inventory: 'Inventory',
      'Break Library': 'Break Library',
      Campaigns: 'Campaigns',
      Forecasts: 'Forecasts',
      Reports: 'Reports',
      Data: 'Data',
      Advertisers: 'Advertisers',
      Pricing: 'Pricing',
      Overrides: 'Overrides',
      Assistant: 'AI assistant',
      Versions: 'Versions',
      Settings: 'Settings',
    },
    workspace: 'Revenue operations',
    operatorRole: 'Revenue Ops',
    optimizer: 'Optimizer',
    scenario: 'Scenario',
    scenarios: ['Balanced', 'Revenue priority', 'Retention guardrail'],
    compare: 'Compare',
    liveApi: 'Live API',
    snapshot: 'Demo data (API offline)',
    data: 'Data',
    dataUpdated: 'Updated',
    refresh: 'Refresh',
    notifications: 'Notifications',
    runOptimization: 'Run Optimization',
    loading: 'Loading Kairos workspace',
    apiUnavailable: 'API unavailable. Showing local snapshot.',
    metrics: ['Projected revenue', 'Viewer retention D7', 'Total ad minutes', 'Risk score'],
    risk: { High: 'High', Medium: 'Medium', Low: 'Low', Unknown: 'Unknown' },
    toolbar: ['Grid View', 'Timeline', 'Daypart', 'Inventory', 'Programs', 'Breaks', 'Metrics'],
    canvas: 'Broadcast planning canvas',
    channelProgram: 'Channel / Program',
    selectedBreak: 'Selected break',
    pending: 'Pending',
    approved: 'Approved',
    detail: ['Revenue', 'Retention D7', 'Duration', 'Spots'],
    guardrails: 'Guardrails',
    recommendation: 'Recommendation',
    approve: 'Approve',
    reject: 'Reject',
    applySimilar: 'Approve similar',
    export: 'Export',
    exportOptions: ['Break detail', 'Weekly traffic plan', 'Guardrail report'],
    frontier: 'Revenue vs retention',
    frontierMode: 'Measured retention model',
    frontierPickChannel: 'Pick your channel in settings to forecast your own inventory. The frontier projects revenue for your channel only; competing programmes feed the retention model, never the revenue forecast.',
    frontierComputing: 'Computing your channel forecast. This runs a real optimisation in the background and appears here once ready; refresh in a moment.',
    heatmap: 'Daypart inventory heatmap',
    heatmapEmpty: 'No daypart heatmap data yet',
    opportunity: 'Revenue opportunity',
    compliance: 'Compliance ledger',
    activeRules: 'active rules',
    compliant: 'Compliant',
    atRisk: 'At risk',
    none: 'None',
    settingsTitle: 'Market and policy settings',
    settingsIntro: 'Controls are operational defaults, not hard-coded law. Update the profile when regulation, rate cards, or broadcaster policy changes.',
    saveSettings: 'Save changes',
    saving: 'Saving...',
    saved: 'Saved',
    saveFailed: 'Save failed',
    unsavedChanges: 'You have unsaved changes',
    noChanges: 'All changes saved',
    profile: 'Profile',
    source: 'Source',
    effectiveDate: 'Effective date',
    language: 'Language',
    hebrew: 'Hebrew',
    english: 'English',
    maxAdMinutes: 'Max ad minutes per hour',
    maxBreaks: 'Max breaks per hour',
    spacing: 'Minimum break spacing',
    retentionFloor: 'Retention floor',
    dailyCap: 'Daily ad-minute cap',
    protectedMax: 'Protected content max ad minutes',
    protectedTypes: 'Protected programme types',
    sponsorships: 'Sponsorships enabled',
    gold: 'Gold breaks enabled',
    approval: 'Manual approval required',
    riskCaution: 'Caution level',
    riskCautionHelp: 'Sets which retention cost the reported numbers carry. 0 uses the central estimate; higher values price each break at the worst plausible cost in its measured range. A reporting choice: on current data it does not change the chosen plan.',
    riskCautionSetting: 'Default caution level',
    retentionCostTitle: 'Retention cost confidence',
    retentionCostIntro: 'How trustworthy the retention cost is behind each segment in this plan.',
    retentionCostConfidence: { low: 'Low', medium: 'Medium', high: 'High' },
    retentionCostAssumption: 'assumption',
    retentionCostInterval: 'Interval',
    retentionCostBreaks: 'real breaks',
    retentionCostNoInterval: 'No interval known',
    retentionCostPoint: 'Point estimate',
    retentionCostUsed: 'Value used',
  },
  he: {
    nav: {
      Overview: 'סקירה',
      Optimizer: 'אופטימייזר',
      Schedule: 'לוח שידורים',
      Inventory: 'מלאי',
      'Break Library': 'ספריית ברייקים',
      Campaigns: 'קמפיינים',
      Forecasts: 'תחזיות',
      Reports: 'דוחות',
      Data: 'נתונים',
      Advertisers: 'מפרסמים',
      Pricing: 'תמחור',
      Overrides: 'עקיפות',
      Assistant: 'עוזר AI',
      Versions: 'ניהול גרסאות',
      Settings: 'הגדרות',
    },
    workspace: 'ניהול הכנסות מפרסום',
    operatorRole: 'Revenue Ops',
    optimizer: 'אופטימייזר',
    scenario: 'תרחיש',
    scenarios: ['מאוזן', 'מקסום הכנסה', 'הגנת שימור'],
    compare: 'השוואה',
    liveApi: 'API חי',
    snapshot: 'נתוני הדגמה (API מנותק)',
    data: 'נתונים',
    dataUpdated: 'עודכן',
    refresh: 'רענון',
    notifications: 'התראות',
    runOptimization: 'הרצת אופטימיזציה',
    loading: 'טוען סביבת Kairos',
    apiUnavailable: 'ה־API לא זמין. מוצגת תמונת מצב מקומית.',
    metrics: ['הכנסה צפויה', 'שימור צפייה D7', 'דקות פרסום', 'רמת סיכון'],
    risk: { High: 'גבוהה', Medium: 'בינונית', Low: 'נמוכה', Unknown: 'לא ידוע' },
    toolbar: ['תצוגת גריד', 'ציר זמן', 'רצועות שידור', 'מלאי', 'תוכניות', 'ברייקים', 'מדדים'],
    canvas: 'משטח תכנון שידור',
    channelProgram: 'ערוץ / תוכנית',
    selectedBreak: 'ברייק נבחר',
    pending: 'ממתין',
    approved: 'מאושר',
    detail: ['הכנסה', 'שימור D7', 'משך', 'ספוטים'],
    guardrails: 'בקרות',
    recommendation: 'המלצה',
    approve: 'אישור',
    reject: 'דחייה',
    applySimilar: 'אישור דומים',
    export: 'ייצוא',
    exportOptions: ['פרטי ברייק', 'תוכנית טראפיק שבועית', 'דוח בקרות'],
    frontier: 'הכנסה מול שימור',
    frontierMode: 'מודל שימור מדוד',
    frontierPickChannel: 'בחרו את הערוץ שלכם בהגדרות כדי לחזות את המלאי שלכם בלבד. החזית מציגה תחזית הכנסה לערוץ שלכם בלבד; תוכניות מתחרות מזינות את מודל השימור, לעולם לא את תחזית ההכנסה.',
    frontierComputing: 'מחשבים את תחזית הערוץ שלכם. זהו אופטימיזציה אמיתית שרצה ברקע ותופיע כאן ברגע שתהיה מוכנה; רעננו עוד רגע.',
    heatmap: 'מפת חום לפי רצועת שידור',
    heatmapEmpty: 'אין עדיין נתוני מפת חום לפי רצועה',
    opportunity: 'פוטנציאל הכנסה',
    compliance: 'יומן תאימות',
    activeRules: 'כללים פעילים',
    compliant: 'תקין',
    atRisk: 'דורש בדיקה',
    none: 'אין',
    settingsTitle: 'הגדרות שוק ומדיניות',
    settingsIntro: 'אלה ברירות מחדל תפעוליות, לא חוק קשיח בקוד. מעדכנים את הפרופיל כשהרגולציה, מחירונים או מדיניות הערוץ משתנים.',
    saveSettings: 'שמור שינויים',
    saving: 'שומר...',
    saved: 'נשמר',
    saveFailed: 'השמירה נכשלה',
    unsavedChanges: 'יש לך שינויים שלא נשמרו',
    noChanges: 'כל השינויים נשמרו',
    profile: 'פרופיל',
    source: 'מקור',
    effectiveDate: 'תאריך תחולה',
    language: 'שפה',
    hebrew: 'עברית',
    english: 'אנגלית',
    maxAdMinutes: 'מקסימום דקות פרסום בשעה',
    maxBreaks: 'מקסימום ברייקים בשעה',
    spacing: 'מרווח מינימלי בין ברייקים',
    retentionFloor: 'רף שימור',
    dailyCap: 'תקרת דקות פרסום יומית',
    protectedMax: 'דקות פרסום מקסימליות בתוכן מוגן',
    protectedTypes: 'סוגי תוכן מוגן',
    sponsorships: 'חסויות פעילות',
    gold: 'ברייקי זהב פעילים',
    approval: 'נדרש אישור ידני',
    riskCaution: 'רמת זהירות',
    riskCautionHelp: 'קובע איזו עלות צפייה נכנסת למספרים המדווחים. 0 משתמש באומדן המרכזי, וערכים גבוהים מתמחרים כל ברייק לפי העלות הסבירה הגרועה ביותר בטווח המדידה שלו. זו בחירת דיווח: בנתונים הנוכחיים היא אינה משנה את התוכנית שנבחרת.',
    riskCautionSetting: 'רמת זהירות כברירת מחדל',
    retentionCostTitle: 'מהימנות עלות השימור',
    retentionCostIntro: 'עד כמה אפשר לסמוך על עלות השימור שמאחורי כל סגמנט בתוכנית הזו.',
    retentionCostConfidence: { low: 'נמוכה', medium: 'בינונית', high: 'גבוהה' },
    retentionCostAssumption: 'הנחה',
    retentionCostInterval: 'טווח',
    retentionCostBreaks: 'ברייקים אמיתיים',
    retentionCostNoInterval: 'אין טווח ידוע',
    retentionCostPoint: 'אומדן נקודתי',
    retentionCostUsed: 'הערך שנעשה בו שימוש',
  },
};

const fallbackOverview = {
  brand: 'Kairos',
  workspace: 'KAI Network',
  data_freshness: new Date().toISOString(),
  // API offline: do not fabricate metrics. Null fields drive the honest empty
  // states in the consuming components rather than confident invented numbers.
  summary: {
    total_breaks: null,
    total_ad_seconds: null,
    projected_revenue: null,
    average_retention: null,
    risk_score: null,
  },
  source_counts: null,
  recommendations: [],
  frontier: [],
  settings: fallbackSettings,
  compliance: fallbackCompliance,
};

// API offline: do not fabricate a schedule. Empty rows/programs/breaks drive the
// honest empty states in the consuming components, matching fallbackOverview
// (nulled metrics) and fallbackInventory (empty rows).
const fallbackSchedule = {
  rows: [],
  break_operations: {
    programs: [],
    breaks: [],
    summary: { programs: 0, breaks: 0, ad_seconds: 0, revenue: 0 },
  },
  break_schedule: [],
};

const fallbackInventory = {
  summary: { spots: 0, revenue: 0, seconds: 0 },
  by_channel: [],
  by_hour: [],
};

const fallbackBreakLibrary = { breaks: [] };
const fallbackCampaigns = { campaigns: [] };
const fallbackForecasts = { by_day: [], scenarios: [] };
const fallbackReports = { reports: [] };
const fallbackFiles = { files: [] };
const fallbackImpact = {
  program_type_impacts: [],
  position_impacts: [],
  length_impacts: [],
  coefficient_impacts: {
    source: 'unavailable',
    metadata: {},
    program_type: [],
    position: [],
    length: [],
  },
};

const fallbackParameters = {
  settings: fallbackSettings,
  guardrails: {},
  assumptions: {},
  pricing: {},
};

const navItems = [
  ['Overview', LayoutGrid],
  ['Optimizer', Activity],
  ['Schedule', CalendarDays],
  ['Inventory', TableProperties],
  ['Break Library', ClipboardCheck],
  ['Campaigns', FileBarChart],
  ['Forecasts', Gauge],
  ['Reports', ListChecks],
  ['Data', Database],
  ['Advertisers', Users],
  ['Pricing', Coins],
  ['Overrides', SlidersHorizontal],
  ['Assistant', Bot],
  ['Versions', History],
  ['Settings', Settings],
];

const dayKeys = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const daypartKeys = ['Morning', 'Daytime', 'Access', 'Primetime', 'Late night'];

function viewFromLocation() {
  if (typeof window === 'undefined') {
    return 'Overview';
  }
  const hash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
  return navItems.some(([label]) => label === hash) ? hash : 'Overview';
}

function gridAxisFromLocation() {
  if (typeof window === 'undefined') {
    return 'day';
  }
  const axis = new URLSearchParams(window.location.search).get('axis');
  return ['day', 'daypart', 'hour', 'type'].includes(axis) ? axis : 'day';
}

// Honest empty-state sentinel: null/undefined/non-finite input renders as a
// plain hyphen, never a confident 0 that hides missing data. Callers that mean
// a real zero should pass 0 (or value || 0) to opt into the numeric path.
const EMPTY_VALUE = '-';

// Precise currency for DATA VALUES (tooltips, readout cards, deltas, stat
// figures). Compact notation with zero decimals hides material differences: it
// renders 1,571,836 and 1,100,000 both as "1M" and makes a 465,000 delta
// invisible, which is a legibility AND honesty failure. So the compact branch
// carries two fraction digits (1,571,836 -> "1.57M", 2,040,000 -> "2.04M") while
// minimumFractionDigits:0 keeps round values clean (2,000,000 -> "2M"). A 10,000
// ILS gap at the millions scale stays distinguishable (1.57M vs 1.58M). Axis
// ticks are the ONLY place compact-coarse is acceptable: use formatCurrencyAxis
// there, never this. Do not lower the fraction digits here or widen the compact
// threshold to swallow the 100K band; that reintroduces the trap.
function formatCurrency(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const compact = Math.abs(number) >= 100000;
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    notation: compact ? 'compact' : 'standard',
    maximumFractionDigits: compact ? 2 : 0,
    minimumFractionDigits: 0,
  });
  return formatter.format(number);
}

// Coarse currency for CHART AXIS TICKS ONLY, where space is tight and the label
// just conveys scale (1.6M, 465K, 12.5M). This is the deliberate compact-coarse
// exception to formatCurrency. Never use it for a data value the operator reads
// as a figure (tooltip point, readout card, delta, stat) - those must stay
// precise via formatCurrency so a 10,000 ILS difference is not rounded away.
function formatCurrencyAxis(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    notation: 'compact',
    maximumFractionDigits: 1,
    minimumFractionDigits: 0,
  });
  return formatter.format(number);
}

function formatMinutes(seconds, locale = 'en') {
  const number = finiteNumber(seconds);
  if (number === null) return EMPTY_VALUE;
  const minutes = Math.round(number / 60);
  return locale === 'he' ? `${minutes.toLocaleString('he-IL')} דק׳` : `${minutes.toLocaleString()} min`;
}

function formatNumber(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  // Full grouped digits, never compact. Compact notation on a data count (e.g.
  // "2M" for 1,571,836) hides material differences and reads as dishonest, so do
  // not add notation:'compact' or drop precision here for large counts.
  return number.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: 1,
  });
}

function formatPercent(value, locale = 'en') {
  if (finiteNumber(value) === null) return EMPTY_VALUE;
  return `${formatNumber(value, locale)}%`;
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatRetentionDelta(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) {
    return pageText(locale, 'Insufficient data', 'אין מספיק מדידות');
  }
  const points = number * 100;
  const sign = points > 0 ? '+' : '';
  return `${sign}${formatNumber(points, locale)}pp`;
}

function Numeric({ children }) {
  return (
    <span className="numeric" dir="ltr">
      {children}
    </span>
  );
}

function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

// Derive the active planning window from the loaded schedule rather than a
// hardcoded literal. Returns a real date range when the schedule carries dates,
// otherwise a neutral label with no fabricated dates.
function planningWeekLabel(schedule, locale) {
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

// stableSettingsKey produces an order-independent JSON signature for a settings
// object so the settings page can compare the in-progress draft against the
// saved settings (drives the "unsaved changes" affordance) without false
// positives from key order or fresh array identities.
function stableSettingsKey(value) {
  if (Array.isArray(value)) {
    return `[${value.map(stableSettingsKey).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stableSettingsKey(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
}

function impactSegmentLabel(segment, locale) {
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

function impactSourceLabel(source, metadata, locale) {
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

function complianceUnitLabel(unit, locale = 'en') {
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

function complianceDisclaimer(disclaimer, locale = 'en') {
  if (locale === 'he') {
    return 'בסיס הבקרה ניתן להגדרה. יש לאמת מול ייעוץ משפטי ומדיניות הערוץ לפני שימוש בפרודקשן.';
  }
  return disclaimer || fallbackSettings.notes;
}

function normalizeImpactRows(rows, segmentKey) {
  return normalizeRows(rows)
    .map((row) => {
      const coefficient =
        finiteNumber(row.average_coefficient) ??
        finiteNumber(row.average) ??
        finiteNumber(row.coefficient) ??
        finiteNumber(row.total_impact);
      return {
        segment: row.segment || row[segmentKey] || row.name || row.channel_name || '',
        coefficient,
        sampleCount: finiteNumber(row.sampleCount) ?? finiteNumber(row.sample_count) ?? finiteNumber(row.count),
        channelCount: finiteNumber(row.channelCount) ?? finiteNumber(row.channel_count),
        ciLow: finiteNumber(row.ciLow) ?? finiteNumber(row.ci_low),
        ciHigh: finiteNumber(row.ciHigh) ?? finiteNumber(row.ci_high),
      };
    })
    .filter((row) => row.segment);
}

function programKey(channel, program) {
  return [channel, program?.day, program?.time, program?.title].map((part) => String(part || '')).join('|');
}

function flattenScheduleRows(rows) {
  return normalizeRows(rows).flatMap((row) =>
    normalizeRows(row.programs).map((program) => ({
      ...program,
      channel: row.channel,
      key: programKey(row.channel, program),
    })),
  );
}

function daypartForTime(time) {
  const hour = hourFromTime(time);
  if (hour >= 6 && hour < 12) return 'Morning';
  if (hour >= 12 && hour < 17) return 'Daytime';
  if (hour >= 17 && hour < 20) return 'Access';
  if (hour >= 20 && hour < 23) return 'Primetime';
  return 'Late night';
}

function hourFromTime(time) {
  const hour = Number(String(time || '0:00').split(':')[0]);
  return Number.isFinite(hour) ? Math.max(0, Math.min(23, hour)) : 0;
}

function daypartLabel(daypart, locale) {
  const labels = {
    Morning: 'בוקר',
    Daytime: 'יום',
    Access: 'לפני פריים',
    Primetime: 'פריים טיים',
    'Late night': 'לילה',
  };
  return locale === 'he' ? labels[daypart] || daypart : daypart;
}

function programTypeLabel(type, locale) {
  // Covers the full classifier vocabulary observed in the live payloads, so
  // genre names never leak as raw English into the Hebrew planning surfaces.
  const labels = {
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
  return locale === 'he' ? labels[type] || type || '' : type || '';
}

function breakPositionLabel(position, locale) {
  const labels = {
    first: 'ראשון',
    early: 'מוקדם',
    middle: 'אמצעי',
    late: 'מאוחר',
    last: 'אחרון',
  };
  return locale === 'he' ? labels[position] || position || '' : position || '';
}

function breakLengthLabel(length, locale) {
  const labels = {
    short: 'קצר',
    standard: 'סטנדרטי',
    medium: 'בינוני',
    long: 'ארוך',
  };
  return locale === 'he' ? labels[length] || length || '' : length || '';
}

function scenarioNameLabel(name, locale) {
  const labels = {
    Balanced: 'מאוזן',
    'Revenue priority': 'מקסום הכנסה',
    'Retention guardrail': 'הגנת שימור',
  };
  return locale === 'he' ? labels[name] || name || '' : name || '';
}

function localizedModelText(text, locale) {
  if (locale !== 'he' || !text) {
    return text || '';
  }
  return String(text)
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
    .replace(/\blate\b/gi, 'מאוחר')
    .replace(/\bOther\b/g, 'אחר')
    .replace(/\bNews\b/g, 'חדשות')
    .replace(/\bReality\b/g, 'ריאליטי')
    .replace(/\bDrama\b/g, 'דרמה')
    .replace(/\bSports\b/g, 'ספורט')
    .replace(/\bComedy\b/g, 'קומדיה')
    .replace(/\bPromo\b/g, 'פרומו');
}

function dayLabel(day, locale) {
  const labels = locale === 'he' ? ['ב׳', 'ג׳', 'ד׳', 'ה׳', 'ו׳', 'ש׳', 'א׳'] : dayKeys;
  const index = dayKeys.indexOf(day);
  return labels[index] || day;
}

function gridAxisLabel(axis, locale) {
  const labels = {
    day: pageText(locale, 'Days', 'ימים'),
    daypart: pageText(locale, 'Dayparts', 'רצועות'),
    hour: pageText(locale, 'Hours', 'שעות'),
    type: pageText(locale, 'Formats', 'סוגי תוכנית'),
  };
  return labels[axis] || labels.day;
}

function buildPlannerColumns(rows, axis, locale) {
  if (axis === 'daypart') {
    return daypartKeys.map((daypart) => ({ key: daypart, label: daypartLabel(daypart, locale) }));
  }
  if (axis === 'hour') {
    const hours = Array.from(new Set(flattenScheduleRows(rows).map((program) => hourFromTime(program.time)))).sort((a, b) => a - b);
    return (hours.length ? hours : [20]).map((hour) => ({
      key: `hour-${hour}`,
      hour,
      label: `${String(hour).padStart(2, '0')}:00`,
    }));
  }
  if (axis === 'type') {
    const types = Array.from(new Set(flattenScheduleRows(rows).map((program) => program.program_type || 'Other'))).sort();
    return (types.length ? types : ['Other']).map((programType) => ({
      key: `type-${programType}`,
      programType,
      label: programTypeLabel(programType, locale),
    }));
  }
  return dayKeys.map((day) => ({ key: day, label: dayLabel(day, locale) }));
}

function programsForPlannerColumn(programs, column, axis) {
  if (axis === 'daypart') {
    return programs.filter((program) => daypartForTime(program.time) === column.key);
  }
  if (axis === 'hour') {
    return programs.filter((program) => hourFromTime(program.time) === column.hour);
  }
  if (axis === 'type') {
    return programs.filter((program) => (program.program_type || 'Other') === column.programType);
  }
  return programs.filter((program) => program.day === column.key);
}

function downloadJson(filename, payload) {
  if (typeof window === 'undefined') return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

async function downloadScheduleCsv(locale, notify) {
  if (typeof window === 'undefined') return;
  try {
    const response = await fetch(`${API_BASE}/api/export/schedule.csv`);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const disposition = response.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/i);
    const filename = match ? match[1] : 'kairos-weekly-schedule.csv';
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    if (notify) {
      notify('Schedule exported as CSV.', 'הלוח יוצא כ־CSV.');
    }
  } catch (error) {
    if (notify) {
      const status = String(error.message || '');
      if (status.startsWith('404')) {
        notify('No schedule is available to export yet.', 'אין לוח זמין לייצוא עדיין.');
      } else {
        notify(`Schedule export failed (${error.message}).`, `ייצוא הלוח נכשל (${error.message}).`);
      }
    }
  }
}

async function fetchJson(path, fallback) {
  try {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return { data: await response.json(), online: true, error: null };
  } catch (error) {
    return { data: fallback, online: false, error: error.message };
  }
}

// Backend segment overrides accept the kinds pin | force | forbid | gold. The
// recommendation payload speaks a slightly richer intent vocabulary, so map its
// proposed_kind onto the override kind the store expects: a "lower_count" intent
// resolves to a forced break count (force), everything else passes through.
function mapProposedKind(proposedKind) {
  const value = String(proposedKind || '').trim();
  if (value === 'lower_count') return 'force';
  if (value === 'gold' || value === 'pin' || value === 'forbid' || value === 'force') return value;
  return '';
}

// Posts a break decision. Returns { ok, status, error, decision }. A 404 means an
// older backend without the decision route, which is treated as ok so the annotation
// only decision log keeps working; a real error surfaces its status honestly.
async function postBreakDecision(payload) {
  try {
    const response = await fetch(`${API_BASE}/api/break-decisions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (response.status === 404) return { ok: true, status: 404, error: null, decision: null };
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body && body.detail) detail = String(body.detail);
      } catch {
        // Non-JSON error body: keep the status line as the honest message.
      }
      return { ok: false, status: response.status, error: detail, decision: null };
    }
    let decision = null;
    try {
      decision = await response.json();
    } catch {
      decision = null;
    }
    return { ok: true, status: response.status, error: null, decision };
  } catch (error) {
    // Network unreachable: the UI keeps its local decision state offline.
    return { ok: true, status: 0, error: error.message, decision: null, offline: true };
  }
}

function useKairosData(refreshKey = 0) {
  const [state, setState] = useState({
    overview: fallbackOverview,
    schedule: fallbackSchedule,
    inventory: fallbackInventory,
    breakLibrary: fallbackBreakLibrary,
    campaigns: fallbackCampaigns,
    forecasts: fallbackForecasts,
    reports: fallbackReports,
    files: fallbackFiles,
    impact: fallbackImpact,
    parameters: fallbackParameters,
    breakOperations: fallbackSchedule.break_operations,
    online: false,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let active = true;
    async function load() {
      const [
        overviewResult,
        scheduleResult,
        inventoryResult,
        breakLibraryResult,
        campaignsResult,
        forecastsResult,
        reportsResult,
        filesResult,
        impactResult,
        parametersResult,
        breakOperationsResult,
      ] = await Promise.all([
        fetchJson('/api/overview', fallbackOverview),
        fetchJson('/api/schedule', fallbackSchedule),
        fetchJson('/api/inventory', fallbackInventory),
        fetchJson('/api/break-library', fallbackBreakLibrary),
        fetchJson('/api/campaigns', fallbackCampaigns),
        fetchJson('/api/forecasts', fallbackForecasts),
        fetchJson('/api/reports', fallbackReports),
        fetchJson('/api/files', fallbackFiles),
        fetchJson('/api/impact', fallbackImpact),
        fetchJson('/api/parameters', fallbackParameters),
        fetchJson('/api/break-operations', fallbackSchedule.break_operations),
      ]);
      if (!active) return;
      const results = [
        overviewResult,
        scheduleResult,
        inventoryResult,
        breakLibraryResult,
        campaignsResult,
        forecastsResult,
        reportsResult,
        filesResult,
        impactResult,
        parametersResult,
        breakOperationsResult,
      ];
      const schedulePayload = {
        ...scheduleResult.data,
        break_operations: scheduleResult.data?.break_operations || breakOperationsResult.data,
      };
      setState({
        overview: overviewResult.data,
        schedule: schedulePayload,
        inventory: inventoryResult.data,
        breakLibrary: breakLibraryResult.data,
        campaigns: campaignsResult.data,
        forecasts: forecastsResult.data,
        reports: reportsResult.data,
        files: filesResult.data,
        impact: impactResult.data,
        parameters: parametersResult.data,
        breakOperations: breakOperationsResult.data,
        online: results.every((result) => result.online),
        loading: false,
        error: results.find((result) => result.error)?.error || null,
      });
    }
    load();
    return () => {
      active = false;
    };
  }, [refreshKey]);

  return state;
}

function TVBreakDashboard() {
  const [refreshKey, setRefreshKey] = useState(0);
  const { overview, schedule, inventory, breakLibrary, campaigns, forecasts, reports, files, impact, parameters, online, loading, error } =
    useKairosData(refreshKey);
  const [activeRecommendation, setActiveRecommendation] = useState('rec-1');
  const [approved, setApproved] = useState(new Set());
  const [rejected, setRejected] = useState(new Set());
  const [scenario, setScenario] = useState('Balanced');
  const [riskLambda, setRiskLambda] = useState(0);
  const riskLambdaTouched = useRef(false);
  const [activeView, setActiveViewState] = useState(viewFromLocation);
  const [optimizerView, setOptimizerView] = useState('grid');
  const [gridAxis, setGridAxisState] = useState(gridAxisFromLocation);
  const [showPrograms, setShowPrograms] = useState(true);
  const [showBreaks, setShowBreaks] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [settings, setSettings] = useState(overview.settings || fallbackSettings);
  const [saveState, setSaveState] = useState('idle');
  const [recomputeState, setRecomputeState] = useState('idle');
  const [applyWeightState, setApplyWeightState] = useState('idle');
  const [optimizationState, setOptimizationState] = useState('idle');
  const [optimizationPlan, setOptimizationPlan] = useState(null);
  const [actionMessage, setActionMessage] = useState('');
  const [overridePrefill, setOverridePrefill] = useState(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [recomputeProgress, setRecomputeProgress] = useState(null);
  const toastTimer = useRef(null);
  // Persistent activity feed: every notify() lands here as a dated entry (not
  // only a transient toast), so nothing scrolls away unseen. Loaded from and
  // saved to localStorage so the record survives a reload. Entries are real
  // events, never fabricated.
  const [notifications, setNotifications] = useState(() => {
    try {
      const raw = window.localStorage.getItem('kairos.activity');
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.slice(-100) : [];
    } catch {
      return [];
    }
  });
  const [feedOpen, setFeedOpen] = useState(false);
  const notifyId = useRef(0);

  // Login / session state. The wall only renders when the backend says
  // authentication is set up; an uninitialized store keeps today's open
  // single-operator flow and shows an honest "open access" chip instead.
  const [auth, setAuth] = useState({ status: 'checking', user: null });
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [accountsDialogOpen, setAccountsDialogOpen] = useState(false);

  useEffect(() => {
    let active = true;
    fetchMe().then((result) => {
      if (!active) return;
      if (result.ok && result.data && result.data.auth_disabled) {
        setAuth({ status: 'open', user: null });
      } else if (result.ok && result.data && result.data.username) {
        setAuth({ status: 'ready', user: result.data });
      } else if (result.status === 0) {
        // Server unreachable: render the app and let its offline states tell
        // the truth about connectivity; there is no session to pretend about.
        setAuth({ status: 'open', user: null });
      } else {
        setAuth({ status: 'login', user: null });
      }
    });
    return () => {
      active = false;
    };
  }, []);

  function handleLoggedIn(user) {
    setAuth({ status: 'ready', user });
    // The pre-login data fetches were rejected by the wall; refetch with the
    // session cookie in place.
    setRefreshKey((key) => key + 1);
  }

  // Session-expiry guard. Sessions live in the server process, so a server
  // restart invalidates them mid-session; without this, every panel quietly
  // renders its offline state while the operator wonders what died. Any API
  // 401 outside the auth routes flips the app to the sign-in screen instead.
  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      try {
        const url = String(args[0] || '');
        if (
          response.status === 401 &&
          url.includes('/api/') &&
          !url.includes('/api/auth/')
        ) {
          setAuth((current) => (current.status === 'login' ? current : { status: 'login', user: null }));
        }
      } catch {
        // The guard must never break a fetch.
      }
      return response;
    };
    return () => {
      window.fetch = originalFetch;
    };
  }, []);

  async function handleLogout() {
    setUserMenuAnchor(null);
    await requestLogout();
    notify('Signed out.', 'יצאת מהמערכת.');
    setAuth({ status: 'login', user: null });
  }

  useEffect(() => {
    try {
      window.localStorage.setItem('kairos.activity', JSON.stringify(notifications.slice(-100)));
    } catch {
      // localStorage may be unavailable (private mode); the in-memory feed still works.
    }
  }, [notifications]);

  // Honest progress affordance: a full-week rebuild is a synchronous call with no
  // percentage available, so we surface an elapsed-seconds timer (not a fake
  // progress bar) while an optimization or recompute is running.
  const isBusy = optimizationState === 'running' || recomputeState === 'running';
  useEffect(() => {
    if (!isBusy) {
      setElapsedSec(0);
      return undefined;
    }
    const started = Date.now();
    setElapsedSec(0);
    const id = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [isBusy]);

  function setActiveView(label) {
    setActiveViewState(label);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      url.hash = encodeURIComponent(label);
      window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    }
  }

  function setGridAxis(axis) {
    setGridAxisState(axis);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      if (axis === 'day') {
        url.searchParams.delete('axis');
      } else {
        url.searchParams.set('axis', axis);
      }
      if (!url.hash) {
        url.hash = encodeURIComponent(activeView);
      }
      window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    }
  }

  useEffect(() => {
    function handleHashChange() {
      setActiveViewState(viewFromLocation());
    }
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    const nextSettings = overview.settings || fallbackSettings;
    setSettings((current) => ({ ...current, ...nextSettings }));
  }, [overview.settings]);

  useEffect(() => {
    if (riskLambdaTouched.current) return;
    const saved = finiteNumber(settings.risk_lambda);
    const fromParameters = finiteNumber(parameters?.settings?.risk_lambda);
    const base = saved !== null ? saved : fromParameters !== null ? fromParameters : 0;
    setRiskLambda(Math.round(Math.min(1, Math.max(0, base)) * 100));
  }, [settings.risk_lambda, parameters]);

  const locale = settings.locale === 'en' ? 'en' : 'he';
  const isHebrew = locale === 'he';
  const copy = copyByLocale[locale];
  // The optimization command group (scenario, risk, run, apply, planning-week)
  // is only meaningful on the planning surfaces; hide it on Data, Pricing,
  // Advertisers, Reports and the like where it does nothing.
  const showOptimizationControls = ['Overview', 'Optimizer', 'Schedule'].includes(activeView);
  const compliance = overview.compliance || fallbackCompliance;
  const theme = useMemo(() => createKairosTheme(isHebrew ? 'rtl' : 'ltr'), [isHebrew]);
  const muiCache = isHebrew ? rtlCache : ltrCache;
  const activeNotificationCount = notifications.filter((n) => !n.dismissed).length;

  function notify(en, he) {
    setActionMessage(pageText(locale, en, he));
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setActionMessage(''), 2600);
    // Also record the event in the persistent activity feed. A stable id avoids
    // Math.random; the bilingual pair is stored so the feed renders in whatever
    // language the operator later views it in.
    notifyId.current += 1;
    const entry = { id: `n${Date.now()}-${notifyId.current}`, en, he, ts: Date.now(), dismissed: false };
    setNotifications((current) => [...current, entry].slice(-100));
  }

  useEffect(() => () => {
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
  }, []);

  const schedulePrograms = useMemo(() => flattenScheduleRows(schedule.rows || []), [schedule]);

  const selectedProgram = useMemo(() => {
    if (selectedProgramKey) {
      const selected = schedulePrograms.find((program) => program.key === selectedProgramKey);
      if (selected) return selected;
    }
    const marked = schedulePrograms.find((program) => program.selected);
    return marked || schedulePrograms[0] || null;
  }, [schedulePrograms, selectedProgramKey]);

  const activeRec =
    overview.recommendations?.find((rec) => rec.id === activeRecommendation) ||
    overview.recommendations?.[0];

  function markApprovedLocal(id) {
    setApproved((current) => new Set(current).add(id));
    setRejected((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  }

  // Send an actionable recommendation to the Overrides workspace with a prefill so
  // the operator sets the exact break count against the live segment state and the
  // projected-delta preview, instead of the model guessing a target it cannot know.
  function openRecommendationInOverrides(rec) {
    if (!rec?.segment_id) return;
    setOverridePrefill({
      segment_id: rec.segment_id,
      kind: mapProposedKind(rec.proposed_kind) || 'pin',
      anchor: rec.anchor || null,
      rec_id: rec.id || '',
    });
    setActiveView('Overrides');
  }

  async function approveRecommendation(id) {
    const rec = normalizeRows(overview.recommendations).find((item) => item.id === id) || (activeRec?.id === id ? activeRec : null);
    const kind = rec && rec.actionable ? mapProposedKind(rec.proposed_kind) : '';
    const anchor = rec?.anchor || {};

    // A forced break count needs a target the recommendation does not carry, so route
    // it to Overrides where the live segment state and preview are available rather
    // than committing a silent no-op. Everything else creates a real override inline.
    if (rec && rec.actionable && rec.segment_id && kind === 'force') {
      openRecommendationInOverrides(rec);
      notify('Set the break count in overrides, where the live segment and preview are available.',
        'קבעו את מספר הברייקים בעקיפות, שם זמינים המשבצת החיה והתצוגה המקדימה.');
      return;
    }

    if (rec && rec.actionable && rec.segment_id && kind) {
      const payload = {
        action: 'approve',
        recommendation_id: id,
        break_id: selectedProgram?.selected_break?.id,
        program_type: rec.program_type || selectedProgram?.program_type,
        scenario,
        target_id: rec.segment_id,
        kind,
        anchor_date: anchor.date,
        anchor_start: anchor.start_clock,
        anchor_title: anchor.program,
      };
      if (kind === 'gold') payload.gold = true;
      const result = await postBreakDecision(payload);
      if (result.status === 404) {
        // Older backend without the anchored decision route: keep the honest log-only
        // behavior so approvals still register on the command surface.
        markApprovedLocal(id);
        notify('Approval recorded in the decision log.', 'האישור נרשם ביומן ההחלטות.');
        return;
      }
      if (!result.ok) {
        notify(`Approval failed (${result.error}).`, `האישור נכשל (${result.error}).`);
        return;
      }
      markApprovedLocal(id);
      setRefreshKey((current) => current + 1);
      notify('Override created from this recommendation. The schedule is now marked stale; recompute when ready.',
        'נוצרה עקיפה מההמלצה הזו. לוח השידורים מסומן כעת כלא מעודכן; הריצו חישוב מחדש כשתרצו.');
      return;
    }

    // Non-actionable recommendation: annotate the decision log only, no override.
    markApprovedLocal(id);
    await postBreakDecision({
      action: 'approve',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: selectedProgram?.program_type || rec?.program_type,
      scenario,
    });
    notify('Approval recorded in the decision log.', 'האישור נרשם ביומן ההחלטות.');
  }

  function markRejectedLocal(id) {
    setRejected((current) => new Set(current).add(id));
    setApproved((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  }

  async function rejectRecommendation(id) {
    const rec = normalizeRows(overview.recommendations).find((item) => item.id === id) || (activeRec?.id === id ? activeRec : null);
    const anchor = rec?.anchor || {};
    const actionable = Boolean(rec && rec.actionable && rec.segment_id);
    const payload = {
      action: 'reject',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: rec?.program_type || selectedProgram?.program_type,
      scenario,
    };
    if (actionable) {
      // A rejection is a dismissed, anchored record. Kind is left for the backend to
      // default (forbid), since rejecting means "do not do this", not dismissing the
      // rec's specific proposed kind.
      payload.target_id = rec.segment_id;
      payload.anchor_date = anchor.date;
      payload.anchor_start = anchor.start_clock;
      payload.anchor_title = anchor.program;
    }
    const result = await postBreakDecision(payload);
    // Only an actionable rejection can create an anchored record, so only it surfaces a
    // real server error and stays unmarked on failure. A non-actionable rejection is a
    // decision-log annotation, and a 400 (no target to anchor) is expected there.
    if (actionable && !result.ok) {
      notify(`Rejection failed (${result.error}).`, `הדחייה נכשלה (${result.error}).`);
      return;
    }
    markRejectedLocal(id);
    notify('Rejection recorded in the decision log.', 'הדחייה נרשמה ביומן ההחלטות.');
  }

  function applySimilarRecommendations() {
    const targetType = activeRec?.program_type;
    const matching = normalizeRows(overview.recommendations).filter((rec) => !targetType || rec.program_type === targetType);
    setApproved((current) => {
      const next = new Set(current);
      matching.forEach((rec) => next.add(rec.id));
      return next;
    });
    setRejected((current) => {
      const next = new Set(current);
      matching.forEach((rec) => next.delete(rec.id));
      return next;
    });
    postBreakDecision({
      action: 'apply_similar',
      recommendation_id: activeRec?.id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: targetType || selectedProgram?.program_type,
      scenario,
    });
    notify('Similar recommendations recorded as approved in the decision log.', 'המלצות דומות נרשמו כמאושרות ביומן ההחלטות.');
  }

  function selectProgram(program) {
    if (!program) return;
    setSelectedProgramKey(program.key);
    setInspectorOpen(true);
    const related =
      normalizeRows(overview.recommendations).find((rec) => rec.program_type === program.program_type) ||
      normalizeRows(overview.recommendations)[0];
    if (related?.id) setActiveRecommendation(related.id);
  }

  function handleRefresh() {
    setRefreshKey((current) => current + 1);
    notify('Data refreshed from the Kairos API.', 'הנתונים רועננו מה־API של Kairos.');
  }

  function dismissNotification(id) {
    setNotifications((current) => current.map((n) => (n.id === id ? { ...n, dismissed: true } : n)));
  }
  function restoreNotification(id) {
    setNotifications((current) => current.map((n) => (n.id === id ? { ...n, dismissed: false } : n)));
  }
  function dismissAllNotifications() {
    setNotifications((current) => current.map((n) => ({ ...n, dismissed: true })));
  }
  function restoreAllNotifications() {
    setNotifications((current) => current.map((n) => ({ ...n, dismissed: false })));
  }

  function scenarioControls() {
    // The "Balanced" scenario follows the operator's saved revenue_weight so the
    // simulation opens on their real choice, not a hardcoded default.
    const savedWeight = finiteNumber(settings.revenue_weight);
    const balanced = Number.isFinite(savedWeight) ? savedWeight : 60;
    const revenueWeight = scenario === 'Revenue priority' ? 85 : scenario === 'Retention guardrail' ? 35 : balanced;
    return {
      revenue_weight: revenueWeight,
      retention_floor: settings.min_retention_floor,
      max_breaks_per_hour: settings.max_breaks_per_hour,
      risk_lambda: Math.min(1, Math.max(0, riskLambda / 100)),
    };
  }

  async function handleRunOptimization() {
    setActiveView('Optimizer');
    setOptimizerView('grid');
    setInspectorOpen(true);
    setOptimizationState('running');
    try {
      const response = await fetch(`${API_BASE}/api/optimizer-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenarioControls()),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const plan = await response.json();
      setOptimizationPlan(plan);
      notify(
        `Optimization produced ${formatNumber(plan.summary?.total_breaks || 0, locale)} compliant breaks.`,
        `האופטימיזציה יצרה ${formatNumber(plan.summary?.total_breaks || 0, locale)} ברייקים תקינים.`,
      );
    } catch {
      notify('Optimizer API is unavailable. Keeping the current working plan.', 'מנוע האופטימיזציה לא זמין. התוכנית הנוכחית נשמרת.');
    } finally {
      setOptimizationState('idle');
    }
  }

  async function persistSettings(nextSettings) {
    setSettings(nextSettings);
    setSaveState('saving');
    try {
      const response = await fetch(`${API_BASE}/api/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(nextSettings),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setSettings(await response.json());
      setSaveState('saved');
      // Bump the refresh key so dependent views refetch against the saved state
      // instead of leaving stale numbers behind a success toast.
      setRefreshKey((k) => k + 1);
      window.setTimeout(() => setSaveState('idle'), 1800);
    } catch {
      setSaveState('error');
    }
  }

  async function handleApplyFrontierFloor(floor) {
    const nextFloor = finiteNumber(floor);
    if (nextFloor === null) return;
    setApplyWeightState('saving');
    try {
      await persistSettings({ ...settings, min_retention_floor: nextFloor });
      const pct = Math.round(nextFloor * 100);
      notify(
        `Saved retention floor set to ${pct} percent.`,
        `רף השימור השמור עודכן ל־${pct} אחוז.`,
      );
    } finally {
      setApplyWeightState('idle');
    }
  }

  async function handleRecomputeSchedule(scope = null) {
    setRecomputeState('running');
    setRecomputeProgress(null);
    const finishOk = (result) => {
      setRecomputeState('done');
      // Refetch so the schedule and overview reflect the freshly computed plan.
      setRefreshKey((k) => k + 1);
      notify(
        `Weekly schedule recomputed: ${formatNumber(result.total_breaks || 0, locale)} breaks, ${formatNumber(Math.round(result.total_revenue || 0), locale)} ILS.`,
        `הלוח השבועי חושב מחדש: ${formatNumber(result.total_breaks || 0, locale)} ברייקים, ${formatNumber(Math.round(result.total_revenue || 0), locale)} ש"ח.`,
      );
      window.setTimeout(() => setRecomputeState('idle'), 2400);
    };
    const finishFail = () => {
      setRecomputeState('error');
      notify('Recompute failed. The saved schedule is unchanged.', 'החישוב מחדש נכשל. הלוח השמור לא השתנה.');
      window.setTimeout(() => setRecomputeState('idle'), 2400);
    };
    const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
    try {
      // Preferred path: the async job endpoint with honest tri-state status and
      // real per-day progress. Falls back to the synchronous endpoint when the
      // job API is absent (older backend).
      const startResponse = await fetch(`${API_BASE}/api/jobs/recompute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scope ? { scope } : {}),
      });
      if (startResponse.status === 404) {
        const response = await fetch(`${API_BASE}/api/recompute-schedule`, { method: 'POST' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        finishOk(await response.json());
        return;
      }
      if (!startResponse.ok) throw new Error(`${startResponse.status} ${startResponse.statusText}`);
      const { job_id: jobId } = await startResponse.json();
      // Poll to a terminal state; ~10 minute ceiling so a dead backend cannot
      // leave the button spinning forever.
      for (let attempt = 0; attempt < 400; attempt += 1) {
        await sleep(1500);
        const statusResponse = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        if (!statusResponse.ok) throw new Error(`${statusResponse.status} ${statusResponse.statusText}`);
        const record = await statusResponse.json();
        if (record.progress && Number.isFinite(record.progress.done) && Number.isFinite(record.progress.total)) {
          setRecomputeProgress({ done: record.progress.done, total: record.progress.total });
        }
        if (record.status === 'done') {
          finishOk(record.result || {});
          return;
        }
        if (record.status === 'failed') {
          setRecomputeState('error');
          notify(
            `Recompute failed: ${record.error || 'unknown error'}. The saved schedule is unchanged.`,
            `החישוב מחדש נכשל: ${record.error || 'שגיאה לא ידועה'}. הלוח השמור לא השתנה.`,
          );
          window.setTimeout(() => setRecomputeState('idle'), 2400);
          return;
        }
      }
      throw new Error('job polling timed out');
    } catch {
      finishFail();
    } finally {
      setRecomputeProgress(null);
    }
  }

  async function handleApplyOptimization() {
    // The optimizer preview runs on the operator's chosen levers but never saves
    // them, so the weekly schedule (saved CSV) never moves. Apply persists those
    // levers into settings, then runs the legitimate full-week recompute that
    // reads them, so the Schedule, Reports and Overview pages all catch up.
    // The preview result itself is never written to the CSV, which would corrupt
    // the rest of the week; only settings plus a full recompute are persisted.
    const controls = scenarioControls();
    // Map the scenario control fields onto their settings field names. Most match
    // by name; retention_floor lands on min_retention_floor (the setting key).
    const nextSettings = {
      ...settings,
      revenue_weight: Math.round(finiteNumber(controls.revenue_weight) ?? settings.revenue_weight),
      risk_lambda: finiteNumber(controls.risk_lambda) ?? settings.risk_lambda,
      min_retention_floor: finiteNumber(controls.retention_floor) ?? settings.min_retention_floor,
      max_breaks_per_hour: finiteNumber(controls.max_breaks_per_hour) ?? settings.max_breaks_per_hour,
    };
    await persistSettings(nextSettings);
    notify(
      'Saved these levers and rebuilding the whole weekly schedule.',
      'הלברים נשמרו והלוח השבועי כולו נבנה מחדש.',
    );
    await handleRecomputeSchedule();
  }

  function renderActiveWorkspace() {
    const common = { overview, schedule, copy, locale, compliance, loading, notify, refreshKey };

    if (activeView === 'Overview') {
      return (
        <OverviewPage
          {...common}
          files={files}
          setActiveView={setActiveView}
          operatorChannel={settings.operator_channel || ''}
          savedRetentionFloor={finiteNumber(settings.min_retention_floor)}
          onApplyFrontierFloor={handleApplyFrontierFloor}
          applyWeightState={applyWeightState}
          refreshKey={refreshKey}
        />
      );
    }

    if (activeView === 'Optimizer') {
      return (
        <OptimizerWorkspace
          {...common}
          activeViewMode={optimizerView}
          gridAxis={gridAxis}
          showPrograms={showPrograms}
          showBreaks={showBreaks}
          showMetrics={showMetrics}
          selectedProgramKey={selectedProgram?.key}
          inspectorOpen={inspectorOpen}
          selectedProgram={selectedProgram}
          activeRec={activeRec}
          approved={approved}
          rejected={rejected}
          optimizationPlan={optimizationPlan}
          parameters={parameters}
          onViewChange={(view) => setOptimizerView(view)}
          onGridAxisChange={(axis) => setGridAxis(axis)}
          onTogglePrograms={(checked) => setShowPrograms(checked)}
          onToggleBreaks={(checked) => setShowBreaks(checked)}
          onToggleMetrics={() => setShowMetrics((current) => !current)}
          onSelectProgram={selectProgram}
          onCloseInspector={() => {
            setInspectorOpen(false);
            notify('Break detail panel closed.', 'פאנל פרטי הברייק נסגר.');
          }}
          onApprove={() => activeRec && approveRecommendation(activeRec.id)}
          onReject={() => activeRec && rejectRecommendation(activeRec.id)}
          onOpenInOverrides={() => activeRec && openRecommendationInOverrides(activeRec)}
          onApplySimilar={applySimilarRecommendations}
          onExport={(exportScope) => {
            downloadJson('kairos-break-detail.json', { exportScope, selectedProgram, recommendation: activeRec, scenario });
            notify('Break detail exported as JSON.', 'פרטי הברייק יוצאו כ־JSON.');
          }}
        />
      );
    }

    if (activeView === 'Schedule') {
      return <SchedulePage {...common} onRecompute={handleRecomputeSchedule} recomputeState={recomputeState} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    if (activeView === 'Inventory') {
      return <InventoryPage inventory={inventory} overview={overview} copy={copy} locale={locale} />;
    }

    if (activeView === 'Break Library') {
      return <BreakLibraryPage breakLibrary={breakLibrary} copy={copy} locale={locale} />;
    }

    if (activeView === 'Campaigns') {
      return <CampaignsPage campaigns={campaigns} copy={copy} locale={locale} refreshKey={refreshKey} />;
    }

    if (activeView === 'Forecasts') {
      return <ForecastsPage forecasts={forecasts} overview={overview} copy={copy} locale={locale} loading={loading} />;
    }

    if (activeView === 'Reports') {
      return <ReportsPage reports={reports} files={files} copy={copy} locale={locale} />;
    }

    if (activeView === 'Data') {
      return (
        <DataPage
          files={files}
          impact={impact}
          parameters={parameters}
          overview={overview}
          copy={copy}
          locale={locale}
          notify={notify}
          onGlobalRefresh={() => setRefreshKey((k) => k + 1)}
        />
      );
    }

    if (activeView === 'Advertisers') {
      return <AdvertisersManager copy={copy} locale={locale} notify={notify} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    if (activeView === 'Pricing') {
      return <PricingManager copy={copy} locale={locale} notify={notify} onGlobalRefresh={() => setRefreshKey((k) => k + 1)} />;
    }

    if (activeView === 'Overrides') {
      return (
        <OverrideConsole
          copy={copy}
          locale={locale}
          notify={notify}
          onGlobalRefresh={() => setRefreshKey((k) => k + 1)}
          prefill={overridePrefill}
          onPrefillConsumed={() => setOverridePrefill(null)}
        />
      );
    }

    if (activeView === 'Assistant') {
      return <AssistantPanel locale={locale} notify={notify} />;
    }

    if (activeView === 'Versions') {
      return <VersionsPage locale={locale} notify={notify} />;
    }

    return (
      <SettingsPanel
        settings={settings}
        parameters={parameters}
        campaigns={campaigns}
        copy={copy}
        locale={locale}
        saveState={saveState}
        onSave={persistSettings}
        onRecompute={handleRecomputeSchedule}
        recomputeState={recomputeState}
        notify={notify}
      />
    );
  }

  // Auth gate: nothing from the workspace renders before the session check
  // resolves, so the app never flashes behind the login wall.
  if (auth.status === 'checking') {
    return (
      <CacheProvider value={muiCache}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <div className="login-screen" dir="rtl" lang="he">
            <div className="login-loading">
              <div className="login-brand-mark" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <span>רק רגע...</span>
            </div>
          </div>
        </ThemeProvider>
      </CacheProvider>
    );
  }

  if (auth.status === 'login') {
    return (
      <CacheProvider value={muiCache}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Login onLoggedIn={handleLoggedIn} />
        </ThemeProvider>
      </CacheProvider>
    );
  }

  if (auth.status === 'ready' && auth.user && auth.user.must_change_password) {
    return (
      <CacheProvider value={muiCache}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <div className="login-screen" dir="rtl" lang="he">
            <ChangePasswordDialog
              locale="he"
              forced
              onDone={(user) =>
                setAuth({ status: 'ready', user: { ...auth.user, ...user, must_change_password: false } })
              }
            />
          </div>
        </ThemeProvider>
      </CacheProvider>
    );
  }

  return (
    <CacheProvider value={muiCache}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
    <div className={`kairos-shell ${isHebrew ? 'rtl' : 'ltr'}`} dir={isHebrew ? 'rtl' : 'ltr'} lang={locale}>
      <aside className="side-rail" aria-label="Kairos navigation">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>Kairos</strong>
            <small>{copy.workspace}</small>
          </div>
        </div>

        <List component="nav" className="primary-nav" disablePadding>
          {navItems.map(([label, Icon]) => (
            <ListItemButton
              key={label}
              component="button"
              className={label === activeView ? 'nav-item active' : 'nav-item'}
              type="button"
              selected={label === activeView}
              disableRipple
              aria-current={label === activeView ? 'page' : undefined}
              onClick={() => setActiveView(label)}
            >
              <ListItemIcon className="nav-icon">
                <Icon size={16} strokeWidth={1.8} />
              </ListItemIcon>
              <ListItemText className="nav-text" disableTypography primary={<span>{copy.nav[label]}</span>} />
            </ListItemButton>
          ))}
        </List>

        {auth.status === 'ready' && auth.user ? (
          <>
            <button
              type="button"
              className="operator-card"
              onClick={(event) => setUserMenuAnchor(event.currentTarget)}
              aria-haspopup="menu"
            >
              <span className="operator-avatar">{operatorInitials(auth.user.display_name || auth.user.username)}</span>
              <div>
                <strong>{auth.user.display_name || auth.user.username}</strong>
                <small>{roleLabel(auth.user.role, locale)}</small>
              </div>
              <ChevronDown size={14} />
            </button>
            <Menu anchorEl={userMenuAnchor} open={Boolean(userMenuAnchor)} onClose={() => setUserMenuAnchor(null)}>
              <MenuItem
                onClick={() => {
                  setUserMenuAnchor(null);
                  setPasswordDialogOpen(true);
                }}
              >
                <KeyRound size={14} style={{ marginInlineEnd: 8 }} />
                {pageText(locale, 'Change password', 'החלפת סיסמה')}
              </MenuItem>
              {auth.user.role === 'admin' && (
                <MenuItem
                  onClick={() => {
                    setUserMenuAnchor(null);
                    setAccountsDialogOpen(true);
                  }}
                >
                  <Users size={14} style={{ marginInlineEnd: 8 }} />
                  {pageText(locale, 'Manage accounts', 'ניהול חשבונות')}
                </MenuItem>
              )}
              <MenuItem onClick={handleLogout}>
                <LogOut size={14} style={{ marginInlineEnd: 8 }} />
                {pageText(locale, 'Sign out', 'יציאה מהמערכת')}
              </MenuItem>
            </Menu>
          </>
        ) : (
          <div
            className="operator-card operator-open"
            title={pageText(
              locale,
              'To set up sign-in and roles, run python scripts/init_auth.py on the server.',
              'להגדרת כניסה ותפקידים יש להריץ בשרת את python scripts/init_auth.py.',
            )}
          >
            <span className="operator-avatar">?</span>
            <div>
              <strong>{pageText(locale, 'Open access', 'גישה פתוחה')}</strong>
              <small>{pageText(locale, 'Sign-in is not set up yet', 'כניסה למערכת טרם הוגדרה')}</small>
            </div>
            <Info size={14} />
          </div>
        )}
      </aside>

      {passwordDialogOpen && (
        <ChangePasswordDialog
          locale={locale}
          onClose={() => setPasswordDialogOpen(false)}
          onDone={() => {
            setPasswordDialogOpen(false);
            notify('Password updated.', 'הסיסמה עודכנה.');
          }}
        />
      )}
      {accountsDialogOpen && auth.user && auth.user.role === 'admin' && (
        <UserAdminDialog
          locale={locale}
          selfUsername={auth.user.username}
          notify={notify}
          onClose={() => setAccountsDialogOpen(false)}
        />
      )}

      <main className="workspace">
        <header className="top-bar">
          <div className="title-group">
            <span className="section-title">{copy.nav[activeView] || copy.optimizer}</span>
            {showOptimizationControls && (
              <Button
                className="date-control"
                type="button"
                variant="outlined"
                onClick={() => {
                  setActiveView('Schedule');
                  notify('Opened the schedule for the active planning week.', 'נפתח לוח השידורים לשבוע התכנון הפעיל.');
                }}
              >
                {planningWeekLabel(schedule, locale)}
                <ChevronDown size={14} />
              </Button>
            )}
          </div>

          {showOptimizationControls && (
          <div className="command-group">
            <FormControl className="scenario-select" size="small">
              <InputLabel id="scenario-label">{copy.scenario}</InputLabel>
              <Select
                labelId="scenario-label"
                value={scenario}
                label={copy.scenario}
                onChange={(event) => {
                  setScenario(event.target.value);
                  notify('Scenario selected. Run optimization to preview this planning mode.', 'התרחיש נבחר. יש להריץ אופטימיזציה כדי לצפות במצב תכנון זה.');
                }}
              >
                <MenuItem value="Balanced">{copy.scenarios[0]}</MenuItem>
                <MenuItem value="Revenue priority">{copy.scenarios[1]}</MenuItem>
                <MenuItem value="Retention guardrail">{copy.scenarios[2]}</MenuItem>
              </Select>
            </FormControl>
            <div className="risk-lambda-control">
              <div className="risk-lambda-head">
                <span className="risk-lambda-label">{copy.riskCaution}</span>
                <Tooltip title={copy.riskCautionHelp} arrow placement="bottom">
                  <Info size={13} className="risk-lambda-info" aria-label={copy.riskCautionHelp} />
                </Tooltip>
                <Numeric>{`${Math.round(Math.min(100, Math.max(0, riskLambda)))}%`}</Numeric>
              </div>
              <Slider
                size="small"
                value={riskLambda}
                min={0}
                max={100}
                step={5}
                aria-label={copy.riskCaution}
                valueLabelDisplay="off"
                onChange={(event, value) => {
                  riskLambdaTouched.current = true;
                  setRiskLambda(Array.isArray(value) ? value[0] : value);
                }}
              />
            </div>
            <Button
              className="secondary-button"
              type="button"
              variant="outlined"
              onClick={() => {
                setActiveView('Forecasts');
                notify('Opened scenario comparison.', 'נפתחה השוואת תרחישים.');
              }}
            >
              <GitCompare size={15} />
              {copy.compare}
            </Button>
          </div>
          )}

          <div className="status-group">
            <span className={online ? 'api-state online' : 'api-state offline'}>
              {online ? copy.liveApi : copy.snapshot}
            </span>
            <span className="freshness" title={locale === 'he' ? 'מועד עדכון הנתונים האחרון מה־API' : 'Time the data was last updated from the API'}>{online && overview.data_freshness ? `${copy.dataUpdated} ${new Date(overview.data_freshness).toLocaleTimeString(locale === 'he' ? 'he-IL' : [], { hour: '2-digit', minute: '2-digit' })}` : `${copy.dataUpdated} -`}</span>
            <IconButton className="icon-button" type="button" aria-label={copy.refresh} size="small" onClick={handleRefresh}>
              <RefreshCcw size={15} />
            </IconButton>
            <IconButton
              className="icon-button"
              type="button"
              aria-label={copy.notifications}
              size="small"
              onClick={() => setFeedOpen((v) => !v)}
            >
              <span className="bell-wrap">
                <Bell size={15} />
                {activeNotificationCount > 0 && (
                  <span className="bell-badge" dir="ltr">{activeNotificationCount > 9 ? '9+' : activeNotificationCount}</span>
                )}
              </span>
            </IconButton>
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => persistSettings({ ...settings, locale: locale === 'he' ? 'en' : 'he', direction: locale === 'he' ? 'ltr' : 'rtl' })}
            >
              <Languages size={14} />
              {locale === 'he' ? copy.english : copy.hebrew}
            </Button>
            {showOptimizationControls && (
              <>
                <Button className="run-button" type="button" variant="contained" disabled={optimizationState === 'running'} onClick={handleRunOptimization}>
                  {optimizationState === 'running' ? <RefreshCcw size={15} className="upload-spinner" /> : <Play size={15} fill="currentColor" />}
                  {optimizationState === 'running' ? `${pageText(locale, 'Running', 'מריץ')} ${elapsedSec}s` : copy.runOptimization}
                </Button>
                <Tooltip title={pageText(locale, 'Saves these levers and rebuilds the whole weekly schedule, not just the preview', 'שומר את הלברים האלה ובונה מחדש את כל הלוח השבועי, לא רק את התצוגה המקדימה')} arrow placement="bottom">
                  <span>
                    <Button
                      className="apply-button"
                      type="button"
                      variant="outlined"
                      disabled={optimizationState === 'running' || recomputeState === 'running'}
                      onClick={handleApplyOptimization}
                    >
                      {recomputeState === 'running' ? <RefreshCcw size={15} className="upload-spinner" /> : <CalendarDays size={15} />}
                      {recomputeState === 'running' ? `${pageText(locale, 'Applying', 'מחיל')} ${elapsedSec}s` : pageText(locale, 'Apply to weekly schedule', 'החל על לוח השבוע')}
                    </Button>
                  </span>
                </Tooltip>
              </>
            )}
          </div>
        </header>

        {isBusy && (
          <div className="rebuild-note" role="status">
            <RefreshCcw size={14} className="upload-spinner" />
            <span>{recomputeProgress ? pageText(locale, `Rebuilding the schedule: day ${recomputeProgress.done} of ${recomputeProgress.total}. Elapsed ${elapsedSec}s.`, `בונה מחדש את הלוח: יום ${recomputeProgress.done} מתוך ${recomputeProgress.total}. חלפו ${elapsedSec} שניות.`) : pageText(locale, `Rebuilding the whole weekly schedule. This can take up to a couple of minutes. Elapsed ${elapsedSec}s.`, `בונה מחדש את כל הלוח השבועי. זה יכול להימשך עד כמה דקות. חלפו ${elapsedSec} שניות.`)}</span>
          </div>
        )}

        <ScheduleStalenessBanner
          freshness={overview?.schedule_freshness}
          locale={locale}
          onRecompute={handleRecomputeSchedule}
          recomputeState={recomputeState}
        />

        {renderActiveWorkspace()}

        {actionMessage && <div className="toast">{actionMessage}</div>}
        {loading && <div className="toast">{copy.loading}</div>}
        {feedOpen && (
          <ActivityFeed
            notifications={notifications}
            locale={locale}
            onDismiss={dismissNotification}
            onRestore={restoreNotification}
            onClearAll={dismissAllNotifications}
            onRestoreAll={restoreAllNotifications}
            onClose={() => setFeedOpen(false)}
          />
        )}
        {!loading && error && <div className="toast muted">{copy.apiUnavailable}</div>}
      </main>
    </div>
      </ThemeProvider>
    </CacheProvider>
  );
}

function riskLabel(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return 'Unknown';
  if (score >= 68) return 'High';
  if (score >= 38) return 'Medium';
  return 'Low';
}

function recommendationTitle(recommendation, locale) {
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

function recommendationRationale(recommendation, locale) {
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

function Metric({ label, value, delta, icon: Icon, positive = false, tone }) {
  const hasDelta = delta !== undefined && delta !== null && delta !== '';
  return (
    <div className="metric">
      <span className={`metric-icon ${tone || ''}`}>
        <Icon size={17} strokeWidth={1.8} />
      </span>
      <span className="metric-copy">
        <span>{label}</span>
        <strong><Numeric>{value}</Numeric></strong>
      </span>
      {hasDelta ? (
        <span className={positive ? 'delta positive' : tone === 'risk' ? 'delta risk' : 'delta negative'}>
          {positive ? <ArrowUp size={12} /> : tone === 'risk' ? null : <ArrowDown size={12} />}
          <Numeric>{delta}</Numeric>
        </span>
      ) : null}
    </div>
  );
}

function SummaryMetrics({ overview, copy, locale }) {
  // A malformed-but-online response falls back to an empty summary so the
  // metrics show honest empty states, never the offline demo numbers.
  const summary = overview.summary || {};
  const riskScore = finiteNumber(summary.risk_score);
  return (
    <section className="metric-strip" aria-label="Optimization summary">
      <Metric label={copy.metrics[0]} value={formatCurrency(summary.projected_revenue, locale)} icon={CircleDollarSign} positive />
      <Metric label={copy.metrics[1]} value={formatPercent(summary.average_retention, locale)} icon={Users} />
      <Metric label={copy.metrics[2]} value={formatMinutes(summary.total_ad_seconds, locale)} icon={Clock3} positive />
      <Metric label={copy.metrics[3]} value={riskScore === null ? '-' : copy.risk[riskLabel(riskScore)]} delta={riskScore === null ? '-' : `${riskScore}/100`} icon={ShieldCheck} tone="risk" />
    </section>
  );
}

function OptimizationRunSummary({ plan, locale }) {
  if (!plan?.summary) return null;
  const summary = plan.summary;
  // The preview optimizes one channel-day (that is what keeps it responsive).
  // Name that scope, so these figures are never read as weekly totals next to
  // the whole-week metrics above.
  const scopeParts = [plan.channel, plan.day ? dayLabel(plan.day, locale) : ''].filter(Boolean);
  const scopeLabel = scopeParts.length
    ? pageText(locale, `Preview scope: ${scopeParts.join(', ')} (one channel-day, not the weekly total)`, `היקף התצוגה המקדימה: ${scopeParts.join(', ')} (יום-ערוץ אחד, לא הסך השבועי)`)
    : pageText(locale, 'Preview scope: one channel-day, not the weekly total', 'היקף התצוגה המקדימה: יום-ערוץ אחד, לא הסך השבועי');
  return (
    <section className="optimizer-run-summary">
      <p className="data-basis-note optimizer-run-scope">{scopeLabel}</p>
      <div>
        <span>{pageText(locale, 'Optimized breaks', 'ברייקים באופטימום')}</span>
        <strong><Numeric>{formatNumber(summary.total_breaks, locale)}</Numeric></strong>
      </div>
      <div>
        <span>{pageText(locale, 'Projected revenue', 'הכנסה חזויה')}</span>
        <strong><Numeric>{formatCurrency(summary.projected_revenue, locale)}</Numeric></strong>
      </div>
      <div>
        <span>{pageText(locale, 'Retention', 'שימור')}</span>
        <strong><Numeric>{formatPercent(summary.average_retention, locale)}</Numeric></strong>
      </div>
      <div>
        <span>{pageText(locale, 'Guardrail status', 'מצב בקרות')}</span>
        <strong>{summary.is_compliant ? pageText(locale, 'Compliant', 'תקין') : pageText(locale, 'Needs review', 'דורש בדיקה')}</strong>
      </div>
    </section>
  );
}

function retentionCostConfidenceWord(confidence, copy) {
  const key = String(confidence || '').toLowerCase();
  return copy.retentionCostConfidence[key] || null;
}

function RetentionCostSegment({ segment, copy, locale }) {
  const cost = segment?.retention_cost;
  if (!cost || typeof cost !== 'object') return null;

  const point = finiteNumber(cost.point);
  const used = finiteNumber(cost.used);
  const ciLow = finiteNumber(cost.ci_low);
  const ciHigh = finiteNumber(cost.ci_high);
  const count = finiteNumber(cost.n);
  const confidenceWord = retentionCostConfidenceWord(cost.confidence, copy);
  const isAssumption = count === 0 || String(cost.confidence || '').toLowerCase() === 'low';
  const hasInterval = ciLow !== null && ciHigh !== null;

  // Live-plan segments carry only segment_id; fall back to it so a confidence
  // row is never a nameless block of numbers.
  const name =
    impactSegmentLabel(segment.segment ?? segment.name ?? segment.program_type ?? '', locale) ||
    segment.label ||
    segment.segment_id ||
    '';

  return (
    <div className={isAssumption ? 'retention-cost-row assumption' : 'retention-cost-row'}>
      <div className="retention-cost-row-head">
        <strong dir="auto">{name}</strong>
        <span className={`retention-cost-confidence ${String(cost.confidence || '').toLowerCase()}`}>
          {isAssumption ? copy.retentionCostAssumption : confidenceWord || copy.retentionCostAssumption}
        </span>
      </div>
      <div className="retention-cost-row-body">
        {used !== null && (
          <span>
            {copy.retentionCostUsed}
            <Numeric>{formatNumber(used, locale)}</Numeric>
          </span>
        )}
        {point !== null && (
          <span>
            {copy.retentionCostPoint}
            <Numeric>{formatNumber(point, locale)}</Numeric>
          </span>
        )}
        <span>
          {copy.retentionCostInterval}
          {hasInterval ? (
            <Numeric>{`[${formatNumber(ciLow, locale)}, ${formatNumber(ciHigh, locale)}]`}</Numeric>
          ) : (
            <small>{copy.retentionCostNoInterval}</small>
          )}
        </span>
        {count !== null && (
          <span>
            <Numeric>{formatNumber(count, locale)}</Numeric>
            <small>{copy.retentionCostBreaks}</small>
          </span>
        )}
      </div>
    </div>
  );
}

// CoefficientFreshnessChip: an honest status chip telling the operator whether
// the measured retention coefficients still match the underlying data, or have
// gone stale. The block is read from the live optimize plan first (most current
// to the run on screen), falling back to /api/parameters. When the API returns
// no coefficient_freshness block at all, nothing is rendered (no fabricated state).
function freshnessDateLabel(value, locale) {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleDateString(locale === 'he' ? 'he-IL' : undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function CoefficientFreshnessChip({ plan, parameters, locale }) {
  const freshness = plan?.coefficient_freshness || parameters?.coefficient_freshness;
  if (!freshness || typeof freshness !== 'object') return null;

  const status = String(freshness.status || '').toLowerCase();
  if (status !== 'fresh' && status !== 'stale' && status !== 'unknown') return null;

  const computedLabel = freshnessDateLabel(freshness.computed_at, locale);
  const changedFiles = normalizeRows(freshness.changed_files).filter(
    (name) => typeof name === 'string' && name.length > 0,
  );
  const reason = typeof freshness.reason === 'string' ? freshness.reason : '';

  const label =
    status === 'fresh'
      ? pageText(locale, 'Coefficients current', 'המקדמים עדכניים')
      : status === 'stale'
        ? pageText(locale, 'Coefficients out of date', 'המקדמים אינם עדכניים')
        : pageText(locale, 'Freshness unverifiable', 'לא ניתן לאמת עדכניות');

  return (
    <section className={`coefficient-freshness ${status}`} aria-label={label}>
      <div className="coefficient-freshness-head">
        <span className="coefficient-freshness-chip">{label}</span>
        {status === 'fresh' && computedLabel && (
          <span className="coefficient-freshness-date">
            {pageText(locale, 'Measured', 'נמדד')} <Numeric>{computedLabel}</Numeric>
          </span>
        )}
      </div>
      {status === 'stale' && (
        <div className="coefficient-freshness-detail">
          {changedFiles.length > 0 && (
            <p>
              {pageText(locale, 'Changed since measurement', 'השתנו מאז המדידה')}: {changedFiles.join(', ')}
            </p>
          )}
          {reason && <p>{reason}</p>}
        </div>
      )}
      {status === 'unknown' && reason && (
        <div className="coefficient-freshness-detail">
          <p>{reason}</p>
        </div>
      )}
    </section>
  );
}

// FirstBreakNote: when the measured first-break gate is active, the optimizer
// charges each programme's FIRST break extra retention cost. This renders a short
// bilingual note with the multiplier so the operator can see the adjustment is on.
// It reads first_break_active / first_break_multiplier from the live plan first,
// then /api/parameters. When the field is false or absent (the honest default;
// the lever is off by default), nothing is rendered.
function readFirstBreak(source) {
  if (!source || typeof source !== 'object') return null;
  if (source.first_break_active === true) return source;
  const assumptions = source.assumptions;
  if (assumptions && typeof assumptions === 'object' && assumptions.first_break_active === true) {
    return assumptions;
  }
  return null;
}

function FirstBreakNote({ plan, parameters, locale }) {
  const active = readFirstBreak(plan) || readFirstBreak(parameters);
  if (!active) return null;

  const multiplier = finiteNumber(active.first_break_multiplier);
  if (multiplier === null || multiplier <= 1) return null;
  const multiplierLabel = `x${multiplier.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

  return (
    <p className="first-break-note">
      {pageText(
        locale,
        "The first break of each programme is charged extra retention cost",
        'הברייק הראשון של כל תוכנית מתומחר בעלות שימור נוספת',
      )}{' '}
      (<Numeric>{multiplierLabel}</Numeric>).
    </p>
  );
}

function RetentionCostPanel({ plan, parameters, copy, locale }) {
  const segments = normalizeRows(plan?.segments).filter(
    (segment) => segment?.retention_cost && typeof segment.retention_cost === 'object',
  );
  if (segments.length === 0) return null;

  return (
    <section className="retention-cost-panel" aria-label={copy.retentionCostTitle}>
      <div className="retention-cost-panel-head">
        <h2>{copy.retentionCostTitle}</h2>
        <p>{copy.retentionCostIntro}</p>
      </div>
      <FirstBreakNote plan={plan} parameters={parameters} locale={locale} />
      <div className="retention-cost-grid">
        {segments.map((segment, index) => (
          <RetentionCostSegment
            key={segment.id || segment.segment || segment.name || index}
            segment={segment}
            copy={copy}
            locale={locale}
          />
        ))}
      </div>
    </section>
  );
}

function OptimizerWorkspace({
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

function PageHeader({ locale, titleEn, titleHe, bodyEn, bodyHe, action }) {
  return (
    <div className="page-header">
      <div>
        <h1>{pageText(locale, titleEn, titleHe)}</h1>
        <p>{pageText(locale, bodyEn, bodyHe)}</p>
      </div>
      {action}
    </div>
  );
}

function StatusBadge({ status, locale, mode = 'inline' }) {
  const normalized = String(status || 'ready').toLowerCase();
  const labelMap = {
    ready: pageText(locale, 'Ready', 'מוכן'),
    compliant: pageText(locale, 'Compliant', 'תקין'),
    at_risk: pageText(locale, 'Needs review', 'דורש בדיקה'),
    attention: pageText(locale, 'Needs attention', 'דורש טיפול'),
    empty: pageText(locale, 'No rows yet', 'אין שורות עדיין'),
    error: pageText(locale, 'Error', 'שגיאה'),
  };
  return <span className={`status-badge ${mode} ${normalized}`}>{labelMap[normalized] || status}</span>;
}

function DataTable({ columns, rows, emptyLabel, locale = 'en' }) {
  const safeRows = normalizeRows(rows);
  const gridRows = safeRows.map((row, index) => ({
    ...row,
    id: String(row.id || row.Campaign || row.path || row.break_id || `${index}-${columns[0]?.key || 'row'}`),
  }));
  const numericKeys = new Set([
    'spots',
    'seconds',
    'revenue',
    'target_spots',
    'num_breaks',
    'total_break_time',
    'predicted_revenue',
    'predicted_retention',
    'channels',
    'breaks',
    'retention',
    'size',
    'rows',
  ]);
  const gridColumns = columns.map((column) => ({
    field: column.key,
    headerName: column.label,
    flex: column.flex || 1,
    minWidth: column.minWidth || 120,
    sortable: column.sortable !== false,
    cellClassName: column.status ? 'status-data-grid-cell' : undefined,
    renderCell: (params) => {
      const isNumeric = column.numeric || numericKeys.has(column.key);
      const value = column.render
        ? column.render(params.row, params.api.getRowIndexRelativeToVisibleRows?.(params.id) || 0)
        : params.value ?? '';
      const className = [
        'grid-cell-content',
        isNumeric ? 'numeric-cell' : '',
        column.status ? 'status-grid-content' : '',
      ].filter(Boolean).join(' ');
      return <span className={className} dir="auto">{value}</span>;
    },
    align: column.align || (column.numeric || numericKeys.has(column.key) ? 'right' : locale === 'he' ? 'right' : 'left'),
    headerAlign:
      column.headerAlign || (column.numeric || numericKeys.has(column.key) ? 'right' : locale === 'he' ? 'right' : 'left'),
  }));

  return (
    <div className="data-table-wrap mui-grid-wrap">
      <React.Suspense fallback={<div className="grid-loading">{emptyLabel}</div>}>
        <LazyDataGrid
          rows={gridRows}
          columns={gridColumns}
          density="compact"
          disableRowSelectionOnClick
          pageSizeOptions={[10, 25, 50]}
        initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
        localeText={{
          noRowsLabel: emptyLabel,
          paginationRowsPerPage: pageText(locale, 'Rows per page:', 'שורות בעמוד:'),
          paginationDisplayedRows: ({ from, to, count, estimated }) => {
            const total = count !== -1 ? formatNumber(count, locale) : pageText(locale, `more than ${to}`, `יותר מ-${to}`);
            const estimate = estimated && estimated > to ? formatNumber(estimated, locale) : total;
            return pageText(
              locale,
              `${formatNumber(from, locale)}-${formatNumber(to, locale)} of ${estimate}`,
              `${formatNumber(from, locale)}-${formatNumber(to, locale)} מתוך ${estimate}`,
            );
          },
        }}
        autoHeight
      />
      </React.Suspense>
    </div>
  );
}

function OverviewPage({ overview, compliance, files, copy, locale, setActiveView, loading, operatorChannel, savedRetentionFloor, onApplyFrontierFloor, applyWeightState, refreshKey }) {
  const sourceCounts = overview.source_counts || {};
  const recommendations = normalizeRows(overview.recommendations);
  const fileRows = normalizeRows(files.files);
  const existingFiles = fileRows.filter((file) => file.exists).length;

  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Executive operating view"
        titleHe="תמונת ניהול תפעולית"
        bodyEn="A single read on revenue, retention, compliance, and the next decisions traffic teams need to make."
        bodyHe="מבט אחד על הכנסה, שמירת צפייה, תאימות וההחלטות הבאות שצוותי הטראפיק צריכים לקבל."
        action={
          <Button className="run-button" type="button" variant="contained" onClick={() => setActiveView('Optimizer')}>
            <Activity size={15} />
            {copy.nav.Optimizer}
          </Button>
        }
      />
      <SummaryMetrics overview={overview} copy={copy} locale={locale} />
      <YieldMoneyPanel locale={locale} refreshKey={refreshKey} />
      <div className="page-grid two-one">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Priority decisions', 'החלטות בעדיפות גבוהה')}</h2>
            <span>{recommendations.length} {pageText(locale, 'actions', 'פעולות')}</span>
          </div>
          <div className="decision-list">
            {recommendations.slice(0, 5).map((item) => (
              <Button className="decision-row" type="button" key={item.id || item.title} onClick={() => setActiveView('Optimizer')}>
                <div>
                  <strong>{recommendationTitle(item, locale)}</strong>
                  <span>{programTypeLabel(item.program_type, locale) || pageText(locale, 'Mixed', 'מעורב')}</span>
                </div>
                <div>
                  <strong><Numeric>{formatCurrency(item.impact, locale)}</Numeric></strong>
                  <span><Numeric>{formatPercent(item.retention, locale)}</Numeric></span>
                </div>
              </Button>
            ))}
          </div>
        </section>
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Control room', 'חדר בקרה')}</h2>
            <span>{pageText(locale, 'Live model state', 'מצב מודל חי')}</span>
          </div>
          <div className="control-list">
            <div><span>{pageText(locale, 'Programmes', 'תוכניות')}</span><strong>{formatNumber(sourceCounts.programmes, locale)}</strong></div>
            <div><span>{pageText(locale, 'Spots', 'ספוטים')}</span><strong>{formatNumber(sourceCounts.spots, locale)}</strong></div>
            <div><span>{pageText(locale, 'Planned break rows', 'שורות תכנון ברייקים')}</span><strong>{formatNumber(sourceCounts.planned_break_rows, locale)}</strong></div>
            <div><span>{pageText(locale, 'Available source files', 'קבצי מקור זמינים')}</span><strong>{existingFiles} / {fileRows.length}</strong></div>
          </div>
        </section>
      </div>
      <div className="page-grid even">
        <ComplianceLedger compliance={compliance} copy={copy} locale={locale} />
        <FrontierScopeChart
          initialData={overview.frontier || []}
          copy={copy}
          locale={locale}
          loading={loading}
          operatorChannel={operatorChannel}
          savedRetentionFloor={savedRetentionFloor}
          onApplyFloor={onApplyFrontierFloor}
          applyState={applyWeightState}
          status={overview.frontier_status || ''}
        />
      </div>
      <YieldView locale={locale} refreshKey={refreshKey} />
    </section>
  );
}

function SchedulePage({ schedule, copy, locale, notify, onRecompute, recomputeState, refreshKey, onGlobalRefresh }) {
  const rows = normalizeRows(schedule.break_schedule);
  const [scheduleMode, setScheduleMode] = useState('grid');
  const [scheduleAxis, setScheduleAxis] = useState(gridAxisFromLocation);
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  // Zoom is shared across the timeline and editor so switching modes keeps one
  // scale (the video-editor style time scale, held in state per page visit).
  const zoom = useScheduleZoom();
  function handleSelectProgram(program) {
    setSelectedProgramKey(program.key);
  }
  return (
    <section className="page-workspace schedule-printable">
      <PageHeader
        locale={locale}
        titleEn="Schedule control"
        titleHe="בקרת לוח שידורים"
        bodyEn="Review the weekly break plan by programme type, day, length, expected revenue, and retention guardrail."
        bodyHe="בדיקת תוכנית הברייקים השבועית לפי סוג תוכנית, יום, אורך, הכנסה צפויה ושמירת צפייה."
        action={
          <div className="schedule-actions no-print">
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => downloadScheduleCsv(locale, notify)}
            >
              <Download size={14} />
              {pageText(locale, 'Download CSV', 'הורדת CSV')}
            </Button>
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => window.print()}
            >
              <Printer size={14} />
              {pageText(locale, 'Print', 'הדפסה')}
            </Button>
          </div>
        }
      />
      <section className="planner-surface compact-surface no-print">
        <div className="surface-toolbar">
          <div className="toolbar-left">
            <Button
              className={scheduleMode === 'grid' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'grid'}
              onClick={() => setScheduleMode('grid')}
            >
              {copy.toolbar[0]}
            </Button>
            <Button
              className={scheduleMode === 'daypart' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'daypart'}
              onClick={() => setScheduleMode('daypart')}
            >
              {copy.toolbar[2]}
            </Button>
            <Button
              className={scheduleMode === 'timeline' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'timeline'}
              onClick={() => setScheduleMode('timeline')}
            >
              {copy.toolbar[1]}
            </Button>
            <Button
              className={scheduleMode === 'editor' ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              aria-pressed={scheduleMode === 'editor'}
              onClick={() => setScheduleMode('editor')}
            >
              {pageText(locale, 'Editor', 'עורך')}
            </Button>
          </div>
          <div className="toolbar-right">
            {scheduleMode === 'grid' && (
              <GridAxisControl value={scheduleAxis} onChange={setScheduleAxis} locale={locale} />
            )}
            <Button
              className="secondary-button compact"
              type="button"
              variant="outlined"
              onClick={() => downloadJson('kairos-weekly-traffic-plan.json', { schedule: rows, grid: schedule.rows || [], axis: scheduleAxis })}
            >
              <Download size={14} />
              {copy.exportOptions[1]}
            </Button>
          </div>
        </div>
        {scheduleMode === 'grid' ? (
          <PlanningCanvas
            rows={schedule.rows || []}
            copy={copy}
            locale={locale}
            axis={scheduleAxis}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        ) : scheduleMode === 'timeline' ? (
          <TimelineView
            timeline={schedule.break_operations}
            rows={schedule.rows || []}
            locale={locale}
            notify={notify}
            zoom={zoom}
            onGlobalRefresh={onGlobalRefresh}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        ) : scheduleMode === 'editor' ? (
          <ScheduleEditor
            schedule={schedule}
            locale={locale}
            notify={notify}
            onRecompute={onRecompute}
            recomputeState={recomputeState}
            onGlobalRefresh={onGlobalRefresh}
            zoom={zoom}
          />
        ) : (
          <DaypartView
            rows={schedule.rows || []}
            locale={locale}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        )}
      </section>
      <section className="page-panel schedule-print-region">
        <div className="panel-head">
          <h2>{pageText(locale, 'Break plan rows', 'שורות תוכנית ברייקים')}</h2>
          <span>{rows.length} {pageText(locale, 'rows', 'שורות')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No scheduled breaks were found.', 'לא נמצאו ברייקים מתוכננים.')}
          rows={rows}
          columns={[
            { key: 'day', label: pageText(locale, 'Day', 'יום'), render: (row) => dayLabel(row.day, locale) },
            { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית'), render: (row) => programTypeLabel(row.program_type, locale) },
            { key: 'position', label: pageText(locale, 'Position', 'מיקום'), render: (row) => breakPositionLabel(row.position, locale) },
            { key: 'break_type', label: pageText(locale, 'Break type', 'סוג ברייק'), render: (row) => breakLengthLabel(row.break_type, locale) },
            { key: 'num_breaks', label: pageText(locale, 'Breaks', 'ברייקים'), render: (row) => formatNumber(row.num_breaks, locale) },
            { key: 'total_break_time', label: pageText(locale, 'Ad minutes', 'דקות פרסום'), render: (row) => formatMinutes(row.total_break_time, locale) },
            { key: 'predicted_revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.predicted_revenue, locale) },
            { key: 'predicted_retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.predicted_retention || 0) * 100, locale) },
          ]}
        />
      </section>
      <GoldBreakManager locale={locale} refreshKey={refreshKey} />
    </section>
  );
}

function InventoryPage({ inventory, overview, copy, locale }) {
  const channels = normalizeRows(inventory.by_channel);
  const hours = normalizeRows(inventory.by_hour);
  // The spots source may carry no revenue column; the API then reports
  // revenue: null with revenue_available: false. Say so once instead of
  // leaving the operator to guess why every money figure is a dash.
  const revenueAvailable = inventory.revenue_available !== false;
  const maxHourValue = Math.max(
    ...hours.map((row) => Number((revenueAvailable ? row.revenue : row.seconds) || 0)),
    1,
  );
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Inventory yield"
        titleHe="תשואת מלאי"
        bodyEn="Inspect sellable spot supply, channel mix, and hourly demand pressure before committing a plan."
        bodyHe="בדיקת היצע ספוטים, תמהיל ערוצים ולחץ ביקוש שעתי לפני אישור תוכנית."
      />
      <section className="metric-strip page-metrics">
        <Metric label={pageText(locale, 'Inventory spots', 'ספוטים במלאי')} value={formatNumber(inventory.summary?.spots, locale)} icon={TableProperties} positive />
        <Metric label={pageText(locale, 'Booked value', 'ערך מוזמן')} value={formatCurrency(inventory.summary?.revenue, locale)} icon={CircleDollarSign} positive />
        <Metric label={pageText(locale, 'Booked minutes', 'דקות מוזמנות')} value={formatMinutes(inventory.summary?.seconds, locale)} icon={Clock3} positive />
        <Metric label={copy.metrics[3]} value={finiteNumber(overview.summary?.risk_score) === null ? '-' : copy.risk[riskLabel(finiteNumber(overview.summary?.risk_score))]} delta={finiteNumber(overview.summary?.risk_score) === null ? '-' : `${finiteNumber(overview.summary?.risk_score)}/100`} icon={ShieldCheck} tone="risk" />
      </section>
      {!revenueAvailable && (
        <p className="data-basis-note">
          {pageText(
            locale,
            'The loaded spots source carries no revenue column, so money figures on this page show a dash. Upload a spots file with revenue to see booked value.',
            'למקור הספוטים שנטען אין עמודת הכנסה, ולכן ערכים כספיים בעמוד זה מוצגים כמקף. העלו קובץ ספוטים עם הכנסה כדי לראות ערך מוזמן.',
          )}
        </p>
      )}
      <div className="page-grid two-one">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Channel inventory', 'מלאי לפי ערוץ')}</h2>
            <span>{channels.length} {pageText(locale, 'channels', 'ערוצים')}</span>
          </div>
          <DataTable
            locale={locale}
            emptyLabel={pageText(locale, 'No inventory rows were found.', 'לא נמצאו שורות מלאי.')}
            rows={channels}
            columns={[
              { key: 'Channel', label: pageText(locale, 'Channel', 'ערוץ') },
              { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
              { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
              { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
              { key: 'target_spots', label: pageText(locale, 'Target', 'יעד'), render: (row) => formatNumber(row.target_spots, locale) },
            ]}
          />
        </section>
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Hourly pressure', 'לחץ לפי שעה')}</h2>
            <span>
              {revenueAvailable
                ? pageText(locale, 'Booked value', 'ערך מוזמן')
                : pageText(locale, 'Booked minutes', 'דקות מוזמנות')}
            </span>
          </div>
          <div className="bar-list chart-ltr" dir="ltr">
            {hours.slice(0, 24).map((row) => (
              <div className="bar-row" key={row.hour_of_day}>
                <span>{String(row.hour_of_day).padStart(2, '0')}:00</span>
                <i style={{ '--bar': Number((revenueAvailable ? row.revenue : row.seconds) || 0) / maxHourValue }} />
                <strong>
                  {revenueAvailable ? formatCurrency(row.revenue, locale) : formatMinutes(row.seconds, locale)}
                </strong>
              </div>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

function BreakLibraryPage({ breakLibrary, copy, locale }) {
  const rows = normalizeRows(breakLibrary.breaks);
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Break library"
        titleHe="ספריית ברייקים"
        bodyEn="A reusable working set of candidate breaks ranked by yield, retention, load, and approval status."
        bodyHe="מאגר עבודה של ברייקים מועמדים, מדורג לפי תשואה, שמירת צפייה, עומס וסטטוס אישור."
      />
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Ranked break candidates', 'ברייקים מדורגים')}</h2>
          <span>{rows.length} {pageText(locale, 'breaks', 'ברייקים')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No break candidates were found.', 'לא נמצאו ברייקים מועמדים.')}
          rows={rows}
          columns={[
            { key: 'status', label: pageText(locale, 'Status', 'סטטוס'), status: true, minWidth: 104, flex: 0.55, render: (row) => <StatusBadge status={row.status} locale={locale} mode="cell" /> },
            { key: 'channel', label: pageText(locale, 'Channel', 'ערוץ') },
            { key: 'date', label: pageText(locale, 'Airing', 'שידור'), numeric: true, render: (row) => [row.date, row.start_time].filter(Boolean).join(' ') || '-' },
            { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית'), render: (row) => programTypeLabel(row.program_type, locale) },
            { key: 'position', label: pageText(locale, 'Position', 'מיקום'), render: (row) => breakPositionLabel(row.position, locale) },
            { key: 'break_type', label: pageText(locale, 'Type', 'סוג'), render: (row) => breakLengthLabel(row.break_type, locale) },
            { key: 'total_break_time', label: pageText(locale, 'Length', 'אורך'), render: (row) => formatMinutes(row.total_break_time, locale) },
            { key: 'predicted_revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.predicted_revenue, locale) },
            { key: 'predicted_retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.predicted_retention || 0) * 100, locale) },
          ]}
        />
      </section>
    </section>
  );
}

function CampaignsPage({ campaigns, copy, locale, refreshKey }) {
  const rows = normalizeRows(campaigns.campaigns);
  const revenueAvailable = campaigns.revenue_available !== false;
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Campaign allocation"
        titleHe="הקצאת קמפיינים"
        bodyEn="Track advertiser demand, booked value, channel spread, and the campaigns that constrain optimization."
        bodyHe="מעקב אחר ביקוש מפרסמים, ערך מוזמן, פיזור ערוצים והקמפיינים שמגבילים את האופטימיזציה."
      />
      {!revenueAvailable && (
        <p className="data-basis-note">
          {pageText(
            locale,
            'The loaded spots source carries no revenue column, so campaign revenue shows a dash and campaigns are ranked by spot count.',
            'למקור הספוטים שנטען אין עמודת הכנסה, ולכן הכנסת הקמפיינים מוצגת כמקף והקמפיינים מדורגים לפי מספר ספוטים.',
          )}
        </p>
      )}
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Advertiser demand', 'ביקוש מפרסמים')}</h2>
          <span>{rows.length} {pageText(locale, 'campaigns', 'קמפיינים')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No campaign rows were found.', 'לא נמצאו שורות קמפיינים.')}
          rows={rows}
          columns={[
            { key: 'Campaign', label: pageText(locale, 'Campaign', 'קמפיין') },
            { key: 'advertiser_id', label: pageText(locale, 'Advertiser', 'מפרסם') },
            { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
            { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
            { key: 'channels', label: pageText(locale, 'Channels', 'ערוצים'), render: (row) => formatNumber(row.channels, locale) },
            { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
            { key: 'last_airing', label: pageText(locale, 'Last airing', 'שידור אחרון') },
          ]}
        />
      </section>
      <MakeGoodAlerts locale={locale} refreshKey={refreshKey} />
    </section>
  );
}

function ForecastsPage({ forecasts, overview, copy, locale, loading }) {
  // The API returns days in arbitrary (alphabetical) order; present them as a
  // week so the table reads Mon through Sun instead of a scrambled sequence.
  const days = normalizeRows(forecasts.by_day)
    .slice()
    .sort((a, b) => dayKeys.indexOf(a.day) - dayKeys.indexOf(b.day));
  const scenarios = normalizeRows(forecasts.scenarios);
  const maxRevenue = Math.max(...scenarios.map((item) => Number(item.revenue || 0)), 1);
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Forecast scenarios"
        titleHe="תרחישי תחזית"
        bodyEn="Compare revenue-forward, balanced, and retention-protected plans before committing inventory."
        bodyHe="השוואה בין תוכניות שמעדיפות הכנסה, איזון או הגנת שימור לפני נעילת המלאי."
      />
      <div className="page-grid even">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Scenario curve', 'עקומת תרחישים')}</h2>
            <span>{copy.frontierMode}</span>
          </div>
          <div className="scenario-bars chart-ltr" dir="ltr">
            {scenarios.map((item) => {
              const weight = finiteNumber(item.revenue_weight);
              const weightTitle = weight === null
                ? undefined
                : pageText(locale, `Revenue weight ${weight}`, `משקל הכנסה ${weight}`);
              return (
                <div className="scenario-row" key={item.name} title={weightTitle}>
                  <span>{scenarioNameLabel(item.name, locale)}</span>
                  <i style={{ '--bar': Number(item.revenue || 0) / maxRevenue }} />
                  <strong>{formatCurrency(item.revenue, locale)}</strong>
                  <small>{formatPercent(item.retention, locale)}</small>
                </div>
              );
            })}
          </div>
          <p className="data-basis-note">
            {pageText(
              locale,
              'Each scenario is a real optimizer run on one representative channel-day under the saved guardrails. These figures are not weekly totals; the daily forecast below sums the whole saved weekly plan.',
              'כל תרחיש הוא ריצת אופטימיזציה אמיתית על יום-ערוץ מייצג אחד תחת הבקרות השמורות. אלה אינם סכומים שבועיים; התחזית היומית מטה מסכמת את התוכנית השבועית השמורה כולה.',
            )}
          </p>
        </section>
        <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} loading={loading} operatorChannel={overview.settings?.operator_channel || ''} status={overview.frontier_status || ''} netPoint={overview.frontier_net_point || null} />
      </div>
      <ScenarioCompare locale={locale} savedRevenueWeight={finiteNumber(overview.settings?.revenue_weight)} />
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Daily forecast', 'תחזית יומית')}</h2>
          <span>{days.length} {pageText(locale, 'days', 'ימים')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No forecast rows were found.', 'לא נמצאו שורות תחזית.')}
          rows={days}
          columns={[
            { key: 'day', label: pageText(locale, 'Day', 'יום'), render: (row) => dayLabel(row.day, locale) },
            { key: 'breaks', label: pageText(locale, 'Breaks', 'ברייקים'), render: (row) => formatNumber(row.breaks, locale) },
            { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
            { key: 'retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.retention || 0) * 100, locale) },
          ]}
        />
      </section>
    </section>
  );
}

function ReportsPage({ reports, files, copy, locale }) {
  const reportRows = normalizeRows(reports.reports);
  const fileRows = normalizeRows(files.files);
  function exportReports() {
    downloadJson('kairos-report-package.json', { reports: reportRows, sources: fileRows });
  }
  // The API sends English-only titles/owners with stable report ids; localize
  // the known ids and fall back to the raw payload text for any new report.
  function reportTitle(report) {
    const titles = {
      'weekly-plan': pageText(locale, 'Weekly traffic plan', 'תוכנית טראפיק שבועית'),
      compliance: pageText(locale, 'Compliance and guardrails', 'תאימות ובקרות'),
      revenue: pageText(locale, 'Revenue forecast', 'תחזית הכנסה'),
      'data-quality': pageText(locale, 'Source file audit', 'בקרת קבצי מקור'),
    };
    return titles[report.id] || report.title;
  }
  function reportOwner(report) {
    const owners = {
      'weekly-plan': pageText(locale, 'Traffic', 'טראפיק'),
      compliance: pageText(locale, 'Legal / Ops', 'משפטי / תפעול'),
      revenue: pageText(locale, 'Revenue', 'הכנסות'),
      'data-quality': pageText(locale, 'Data', 'נתונים'),
    };
    return owners[report.id] || report.owner;
  }
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Reports and approvals"
        titleHe="דוחות ואישורים"
        bodyEn="Generate traffic, compliance, revenue, and source-audit packages for sales, operations, and legal review."
        bodyHe="הפקת חבילות טראפיק, תאימות, הכנסה ובקרת מקורות עבור מכירות, תפעול וייעוץ משפטי."
        action={
          <Button className="secondary-button" type="button" variant="outlined" onClick={exportReports}>
            <Download size={14} />
            {copy.export}
          </Button>
        }
      />
      <div className="report-grid">
        {reportRows.map((report) => (
          <article className="report-card" key={report.id}>
            <div>
              <strong>{reportTitle(report)}</strong>
              <span>{reportOwner(report)}</span>
            </div>
            <StatusBadge status={report.status} locale={locale} />
            <small>{formatNumber(report.rows, locale)} {pageText(locale, 'rows', 'שורות')}</small>
          </article>
        ))}
      </div>
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Source package', 'חבילת מקורות')}</h2>
          <span>{fileRows.filter((file) => file.exists).length} / {fileRows.length}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No report source files were found.', 'לא נמצאו קבצי מקור לדוחות.')}
          rows={fileRows}
          columns={[
            { key: 'path', label: pageText(locale, 'File', 'קובץ') },
            { key: 'exists', label: pageText(locale, 'State', 'מצב'), status: true, minWidth: 104, flex: 0.45, render: (row) => <StatusBadge status={row.exists ? 'ready' : 'error'} locale={locale} mode="cell" /> },
            { key: 'size', label: pageText(locale, 'Size', 'גודל'), render: (row) => `${formatNumber(Number(row.size || 0) / 1024, locale)} KB` },
            { key: 'modified', label: pageText(locale, 'Modified', 'עודכן'), render: (row) => (row.modified ? new Date(row.modified).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US') : '-') },
          ]}
        />
      </section>
    </section>
  );
}

function DataPage({ files, impact, parameters, overview, copy, locale, notify, onGlobalRefresh }) {
  const [dataTab, setDataTab] = useState('upload');
  return (
    <section className="page-workspace">
      <div className="surface-toolbar no-print">
        <div className="toolbar-left">
          <Button
            className={dataTab === 'upload' ? 'segmented active' : 'segmented'}
            type="button"
            variant="outlined"
            aria-pressed={dataTab === 'upload'}
            onClick={() => setDataTab('upload')}
          >
            {pageText(locale, 'Upload', 'העלאה')}
          </Button>
          <Button
            className={dataTab === 'sources' ? 'segmented active' : 'segmented'}
            type="button"
            variant="outlined"
            aria-pressed={dataTab === 'sources'}
            onClick={() => setDataTab('sources')}
          >
            {pageText(locale, 'Sources and model', 'מקורות ומודל')}
          </Button>
        </div>
      </div>
      {dataTab === 'upload' ? (
        <UploadCenter copy={copy} locale={locale} notify={notify} onGlobalRefresh={onGlobalRefresh} />
      ) : (
        <DataHubPage files={files} impact={impact} parameters={parameters} overview={overview} copy={copy} locale={locale} />
      )}
    </section>
  );
}

function DataHubPage({ files, impact, parameters, overview, copy, locale }) {
  const fileRows = normalizeRows(files.files);
  const measuredImpacts = impact.coefficient_impacts || {};
  const programImpacts = normalizeImpactRows(
    normalizeRows(measuredImpacts.program_type).length ? measuredImpacts.program_type : impact.program_type_impacts,
    'program_type',
  );
  const positionImpacts = normalizeImpactRows(
    normalizeRows(measuredImpacts.position).length ? measuredImpacts.position : impact.position_impacts,
    'position',
  );
  const lengthImpacts = normalizeImpactRows(
    normalizeRows(measuredImpacts.length).length ? measuredImpacts.length : impact.length_impacts,
    'length',
  );
  const impactSource = impactSourceLabel(measuredImpacts.source || 'legacy_csv', measuredImpacts.metadata, locale);
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Data hub"
        titleHe="מרכז נתונים"
        bodyEn="Monitor source freshness, model artifacts, and the explainability extracts that support optimization decisions."
        bodyHe="מעקב אחר רעננות מקורות, תוצרי מודל וקבצי הסבר שתומכים בהחלטות האופטימיזציה."
      />
      <section className="metric-strip page-metrics">
        <Metric label={pageText(locale, 'Programmes', 'תוכניות')} value={formatNumber(overview.source_counts?.programmes, locale)} icon={CalendarDays} positive />
        <Metric label={pageText(locale, 'Spots', 'ספוטים')} value={formatNumber(overview.source_counts?.spots, locale)} icon={TableProperties} positive />
        <Metric label={pageText(locale, 'Plan rows', 'שורות תכנון')} value={formatNumber(overview.source_counts?.planned_break_rows, locale)} icon={ClipboardCheck} positive />
        <Metric label={pageText(locale, 'Sources online', 'מקורות זמינים')} value={`${fileRows.filter((file) => file.exists).length}/${fileRows.length}`} icon={Database} positive />
      </section>
      <div className="page-grid two-one">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Source files', 'קבצי מקור')}</h2>
            <span>{pageText(locale, 'Production inputs', 'קלטי פרודקשן')}</span>
          </div>
          <DataTable
            locale={locale}
            emptyLabel={pageText(locale, 'No source files were found.', 'לא נמצאו קבצי מקור.')}
            rows={fileRows}
            columns={[
              { key: 'path', label: pageText(locale, 'Path', 'נתיב') },
              { key: 'exists', label: pageText(locale, 'State', 'מצב'), status: true, minWidth: 104, flex: 0.45, render: (row) => <StatusBadge status={row.exists ? 'ready' : 'error'} locale={locale} mode="cell" /> },
              { key: 'size', label: pageText(locale, 'Size', 'גודל'), render: (row) => `${formatNumber(Number(row.size || 0) / 1024, locale)} KB` },
            ]}
          />
        </section>
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Model explainability', 'הסבריות מודל')}</h2>
            <span>{impactSource}</span>
          </div>
          <div className="impact-stack">
            <ImpactPreview title={pageText(locale, 'Programme type impact', 'השפעת סוג תוכנית')} rows={programImpacts} locale={locale} />
            <ImpactPreview title={pageText(locale, 'Position impact', 'השפעת מיקום')} rows={positionImpacts} locale={locale} />
            <ImpactPreview title={pageText(locale, 'Length impact', 'השפעת אורך')} rows={lengthImpacts} locale={locale} />
            <DriftMonitorCard drift={impact.drift} locale={locale} />
            {typeof measuredImpacts.pooling_note === 'string' && measuredImpacts.pooling_note.trim() && (
              <p className="data-basis-note">
                {pageText(locale, 'Model reliability note:', 'הערת מהימנות מהמודל:')}{' '}
                <span dir="ltr">{measuredImpacts.pooling_note}</span>
              </p>
            )}
          </div>
        </section>
      </div>
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Optimizer parameters', 'פרמטרי אופטימיזציה')}</h2>
          <span>{pageText(locale, 'Guardrails, assumptions, pricing', 'בקרות, הנחות ותמחור')}</span>
        </div>
        <ParameterLedger parameters={parameters} locale={locale} />
      </section>
    </section>
  );
}

function ImpactPreview({ title, rows, locale }) {
  const first = normalizeImpactRows(rows, 'segment').slice(0, 4);
  const maxMagnitude = Math.max(...first.map((row) => Math.abs(row.coefficient || 0)), 0.01);
  return (
    <div className="impact-preview">
      <header>
        <strong>{title}</strong>
        <small>{pageText(locale, 'Retention delta per break', 'שינוי שימור לכל ברייק')}</small>
      </header>
      {first.length === 0 ? (
        <span>{pageText(locale, 'No extract', 'אין קובץ')}</span>
      ) : (
        first.map((row, index) => {
          const magnitude = row.coefficient === null ? 0 : Math.abs(row.coefficient);
          const sample = row.sampleCount ? `n=${formatNumber(row.sampleCount, locale)}` : pageText(locale, 'sample pending', 'מדגם לא זמין');
          const range = row.ciLow !== null && row.ciHigh !== null
            ? `${formatRetentionDelta(row.ciLow, locale)} / ${formatRetentionDelta(row.ciHigh, locale)}`
            : sample;
          const coefficientLabel = formatRetentionDelta(row.coefficient, locale);
          return (
            <div className="impact-row" key={`${title}-${row.segment}-${index}`}>
              <span className="impact-label">{impactSegmentLabel(row.segment, locale)}</span>
              <span className="impact-meter" aria-hidden="true">
                <i style={{ '--impact-width': `${Math.max(8, (magnitude / maxMagnitude) * 100)}%` }} />
              </span>
              <strong>{row.coefficient === null ? coefficientLabel : <Numeric>{coefficientLabel}</Numeric>}</strong>
              <small className={row.ciLow !== null && row.ciHigh !== null ? 'numeric' : undefined}>{range}</small>
            </div>
          );
        })
      )}
    </div>
  );
}

// Log-effect values are close enough to fractional level changes at the drift
// monitor's magnitudes, so value * 100 is shown as a signed percent-like figure.
function formatDriftPercent(value, locale) {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const points = number * 100;
  const sign = points > 0 ? '+' : '';
  return `${sign}${points.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 })}%`;
}

// Audience level stability: surfaces the weekly level-drift measurement the
// coefficient rebuild stores in the artifact metadata and /api/impact echoes
// as `drift`. Renders the measured block or the honest absent reason; when the
// backend sends no verdict, none is invented here.
function DriftMonitorCard({ drift, locale }) {
  const block = drift && typeof drift === 'object' ? drift : null;
  const title = pageText(locale, 'Audience level stability', 'יציבות רמת הצפייה');
  if (!block || block.status !== 'measured') {
    const reason = typeof block?.reason === 'string' && block.reason.trim() ? block.reason : null;
    return (
      <div className="impact-preview drift-card">
        <header>
          <strong>{title}</strong>
          <small>{pageText(locale, 'Weekly monitor', 'ניטור שבועי')}</small>
        </header>
        <p className="drift-note">{pageText(locale, 'No level-drift measurement is available for the current coefficients.', 'מדידת סחיפת הרמה אינה זמינה עבור המקדמים הנוכחיים.')}</p>
        {reason ? <p className="drift-reason" dir="ltr">{reason}</p> : null}
      </div>
    );
  }
  const driftLabel = formatDriftPercent(block.drift_per_week, locale);
  const seNumber = finiteNumber(block.drift_se);
  const seLabel = seNumber === null ? null : `± ${(seNumber * 100).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { maximumFractionDigits: 2 })}%`;
  const bindingState = block.binding === true ? 'binding' : block.binding === false ? 'stable' : 'unknown';
  const chipLabel = bindingState === 'binding' ? pageText(locale, 'Needs attention', 'דורש תשומת לב') : bindingState === 'stable' ? pageText(locale, 'Stable', 'יציב') : pageText(locale, 'Not determined', 'לא נקבע');
  const weeks = normalizeRows(block.weekly_levels);
  const means = weeks.map((week) => finiteNumber(week.mean_log_effect)).filter((value) => value !== null);
  const minMean = means.length ? Math.min(...means) : 0;
  const meanSpan = means.length ? Math.max(...means) - minMean : 0;
  return (
    <div className="impact-preview drift-card">
      <header>
        <strong>{title}</strong>
        <small><Numeric>{formatNumber(block.n_weeks, locale)}</Numeric> {pageText(locale, 'weeks', 'שבועות')}, <Numeric>{formatNumber(block.n_breaks, locale)}</Numeric> {pageText(locale, 'breaks', 'ברייקים')}</small>
      </header>
      <div className="drift-headline">
        <div className="drift-stat">
          <strong><Numeric>{seLabel ? `${driftLabel} ${seLabel}` : driftLabel}</Numeric></strong>
          <small>{pageText(locale, 'Drift per week', 'סחיפה לשבוע')}</small>
        </div>
        <span className={`drift-chip ${bindingState}`} title={typeof block.criterion === 'string' ? block.criterion : undefined}>{chipLabel}</span>
      </div>
      {weeks.length > 0 ? (
        <div className="drift-week-block">
          <small className="drift-strip-caption">{pageText(locale, 'Weekly mean level', 'רמה שבועית ממוצעת')}</small>
          <div className="drift-week-strip">
            {weeks.map((week, index) => {
              const mean = finiteNumber(week.mean_log_effect);
              const ratio = mean === null || meanSpan <= 0 ? 1 : (mean - minMean) / meanSpan;
              return (
                <div className="drift-week" key={`drift-week-${week.week ?? index}`}>
                  <small>{pageText(locale, `Week ${week.week ?? index + 1}`, `שבוע ${week.week ?? index + 1}`)}</small>
                  <span className="drift-week-bar" aria-hidden="true"><i style={{ '--drift-week-width': `${Math.round(12 + ratio * 88)}%` }} /></span>
                  <strong><Numeric>{formatDriftPercent(mean, locale)}</Numeric></strong>
                  <small><Numeric>{`n=${formatNumber(week.n, locale)}`}</Numeric></small>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
      <p className="drift-note">{pageText(locale, "The plan's coefficients assume a steady audience level. A drift above the threshold means the weekly level moves more than the measurement's own precision, so recompute the coefficients when new data lands.", 'מקדמי התוכנית מניחים רמת צפייה יציבה. סחיפה מעל הסף פירושה שהרמה השבועית זזה יותר מדיוק המדידה עצמה, ולכן מומלץ לחשב את המקדמים מחדש כשנקלטים נתונים חדשים.')}</p>
    </div>
  );
}

function ParameterLedger({ parameters, locale }) {
  const settings = parameters?.settings || fallbackSettings;
  const guardrails = parameters?.guardrails || {};
  const assumptions = parameters?.assumptions || {};
  const pricing = parameters?.pricing || {};
  const retentionAssumption = finiteNumber(assumptions.retention_impact_per_break);
  const basePrice = finiteNumber(pricing.base_price_per_second_per_tvr_point);
  const rows = [
    {
      label: pageText(locale, 'Ad minutes per hour', 'דקות פרסום לשעה'),
      value: `${formatNumber(settings.max_ad_minutes_per_hour, locale)} ${pageText(locale, 'min', 'דק׳')}`,
      detail: pageText(locale, 'Regulatory ceiling', 'תקרת רגולציה'),
    },
    {
      label: pageText(locale, 'Breaks per hour', 'ברייקים לשעה'),
      value: formatNumber(settings.max_breaks_per_hour, locale),
      detail: pageText(locale, 'Operational guardrail', 'בקרה תפעולית'),
    },
    {
      label: pageText(locale, 'Minimum spacing', 'מרווח מינימלי'),
      value: `${formatNumber(settings.min_break_spacing_minutes, locale)} ${pageText(locale, 'min', 'דק׳')}`,
      detail: pageText(locale, 'Between break starts', 'בין תחילות ברייקים'),
    },
    {
      label: pageText(locale, 'Retention floor', 'רף שימור'),
      value: formatPercent(Number(settings.min_retention_floor || 0) * 100, locale),
      detail: guardrails.min_retention_floor ? pageText(locale, 'Engine guardrail', 'בקרת מנוע') : pageText(locale, 'Saved setting', 'הגדרה שמורה'),
    },
    {
      label: pageText(locale, 'Retention assumption', 'הנחת שימור'),
      value: retentionAssumption === null ? '-' : formatRetentionDelta(retentionAssumption, locale),
      detail: pageText(locale, 'Fallback when a cell is unseen', 'fallback לסגמנט שלא נמדד'),
    },
    {
      label: pageText(locale, 'Base price', 'מחיר בסיס'),
      value: basePrice === null ? '-' : formatCurrency(basePrice, locale),
      detail: pageText(locale, 'Per TVR-second', 'ל-TVR שנייה'),
    },
  ];
  const premiumRows = Object.entries(pricing.program_type_premiums || {})
    .slice(0, 6)
    .map(([name, value]) => ({ name, value: finiteNumber(value) }));
  return (
    <div className="parameter-ledger">
      <div className="parameter-grid">
        {rows.map((row) => (
          <div className="parameter-row" key={row.label}>
            <span>{row.label}</span>
            <strong><Numeric>{row.value}</Numeric></strong>
            <small>{row.detail}</small>
          </div>
        ))}
      </div>
      <div className="premium-list">
        <strong>{pageText(locale, 'Programme pricing premiums', 'פרמיות תמחור לפי סוג תוכנית')}</strong>
        {premiumRows.length === 0 ? (
          <span>{pageText(locale, 'No pricing model loaded', 'מודל תמחור לא נטען')}</span>
        ) : (
          premiumRows.map((row) => (
            <span key={row.name}>
              <b>{row.name}</b>
              <Numeric>{row.value === null ? '-' : `${formatNumber(row.value, locale)}x`}</Numeric>
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function SelectionGuide({ selectedProgram, onOpen, copy, locale }) {
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
            'בחר תא במשטח התכנון או פתח מחדש את פאנל הפרטים כדי לבדוק בקרות, סטטוס אישור ואפשרויות ייצוא.',
          )}
        </p>
        <Button className="secondary-button" type="button" variant="outlined" onClick={onOpen}>
          {pageText(locale, 'Open details', 'פתח פרטים')}
        </Button>
      </div>
    </aside>
  );
}

function GridAxisControl({ value, onChange, locale }) {
  const options = ['day', 'daypart', 'hour', 'type'];
  return (
    <div className="axis-control" aria-label={pageText(locale, 'Grid split', 'חלוקת גריד')}>
      {options.map((axis) => (
        <Button
          key={axis}
          className={value === axis ? 'axis-segment active' : 'axis-segment'}
          type="button"
          variant="outlined"
          aria-pressed={value === axis}
          onClick={() => onChange(axis)}
        >
          {gridAxisLabel(axis, locale)}
        </Button>
      ))}
    </div>
  );
}

function timeToMinutes(time) {
  const [hour, minute] = String(time || '00:00').split(':').map((part) => Number(part));
  const safeHour = Number.isFinite(hour) ? Math.max(0, Math.min(47, hour)) : 0;
  const safeMinute = Number.isFinite(minute) ? Math.max(0, Math.min(59, minute)) : 0;
  return safeHour * 60 + safeMinute;
}

function minutesToTime(minutes) {
  const safe = Math.max(0, Math.min(47 * 60 + 59, Math.round(minutes)));
  const hour = Math.floor(safe / 60) % 24;
  const minute = safe % 60;
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
}

function buildTimelineFallback(rows) {
  const programs = flattenScheduleRows(rows).slice(0, 24).map((program, index) => {
    const duration = Number(program.duration_minutes || 30);
    const start = timeToMinutes(program.time);
    return {
      id: `fallback-program-${index}`,
      key: program.key,
      lane: `${program.channel} / ${program.day}`,
      channel: program.channel,
      title: program.title,
      program_type: program.program_type || 'Other',
      day: program.day,
      start_time: minutesToTime(start),
      end_time: minutesToTime(start + duration),
      duration_minutes: duration,
      revenue: Number(program.revenue || 0),
      retention: Number(program.retention || 0),
      break_markers: Number(program.break_markers || 0),
    };
  });
  const breaks = programs.flatMap((program) => {
    const count = Math.max(0, Math.min(5, Number(program.break_markers || 0)));
    const duration = 120;
    const start = timeToMinutes(program.start_time);
    const programDuration = Number(program.duration_minutes || 30);
    return Array.from({ length: count }).map((_, index) => {
      const breakStart = start + ((programDuration * 60) / (count + 1) / 60) * (index + 1);
      return {
        id: `${program.key}-fallback-break-${index + 1}`,
        program_key: program.key,
        program_title: program.title,
        lane: program.lane,
        channel: program.channel,
        day: program.day,
        program_type: program.program_type,
        break_num_in_program: index + 1,
        breaks_in_program: count,
        start_time: minutesToTime(breakStart),
        end_time: minutesToTime(breakStart + duration / 60),
        duration_sec: duration,
        sponsorships_count: 0,
        is_gold: false,
        source: 'Model',
        revenue_calculated: Number(program.revenue || 0) / Math.max(count, 1),
        retention: program.retention,
        status: Number(program.retention || 0) < 72 ? 'at_risk' : 'ready',
      };
    });
  });
  return {
    programs,
    breaks,
    summary: {
      programs: programs.length,
      breaks: breaks.length,
      ad_seconds: breaks.reduce((sum, item) => sum + Number(item.duration_sec || 0), 0),
      revenue: breaks.reduce((sum, item) => sum + Number(item.revenue_calculated || 0), 0),
    },
  };
}

function normalizedTimeline(timeline, rows) {
  const fallback = buildTimelineFallback(rows);
  const programs = normalizeRows(timeline?.programs).length ? normalizeRows(timeline.programs) : fallback.programs;
  const breaks = normalizeRows(timeline?.breaks).length ? normalizeRows(timeline.breaks) : fallback.breaks;
  const summary = timeline?.summary || fallback.summary;
  return { programs, breaks, summary };
}

function TimelineView({ timeline, rows, locale, notify, zoom, onGlobalRefresh, selectedProgramKey, onSelectProgram }) {
  const { programs, breaks, summary } = normalizedTimeline(timeline, rows);
  const lanes = Array.from(new Set([...programs.map((item) => item.lane), ...breaks.map((item) => item.lane)].filter(Boolean)));
  const allMinutes = [
    ...programs.flatMap((item) => [timeToMinutes(item.start_time), timeToMinutes(item.end_time)]),
    ...breaks.flatMap((item) => [timeToMinutes(item.start_time), timeToMinutes(item.end_time)]),
  ].filter((value) => Number.isFinite(value));
  // Shared time axis and zoom, so the timeline and the editor line up on the
  // same hour window and pixel mapping at whatever scale is set.
  const axis = timeWindow(allMinutes.length ? allMinutes : [20 * 60, 23 * 60]);
  const localZoom = useScheduleZoom();
  const { pxPerMin, setZoom, zoomBy } = zoom || localZoom;
  const positionStyle = (startTime, endTime) => spanStyle(axis, pxPerMin, timeToMinutes(startTime), timeToMinutes(endTime));

  // Owned-channel segment anchors, resolved through the shared hook so a click
  // on a programme band opens the same inspector the editor uses. A programme
  // that is not on the owned channel has no editable segment; we say so plainly.
  const { resolve } = useSegmentAnchors();
  const [inspect, setInspect] = useState(null);
  const openInspector = (program) => {
    if (!program) return;
    const hit = resolve(program.channel, program.date, program.start_time);
    if (hit) {
      setInspect(hit);
    } else if (notify) {
      notify(
        'This programme is not on your owned channel, so it has no editable segment.',
        'התוכנית אינה בערוץ שבבעלותכם, ולכן אין לה מקטע לעריכה.',
      );
    }
  };

  return (
    <div className="timeline-view">
      <div className="timeline-topbar no-print">
        <div className="timeline-summary" dir={locale === 'he' ? 'rtl' : 'ltr'}>
          <div>
            <strong>{formatNumber(summary.programs, locale)}</strong>
            <span>{pageText(locale, 'programs on timeline', 'תוכניות בציר')}</span>
          </div>
          <div>
            <strong>{formatNumber(summary.breaks, locale)}</strong>
            <span>{pageText(locale, 'planned breaks', 'ברייקים מתוכננים')}</span>
          </div>
          <div>
            <strong><Numeric>{formatMinutes(summary.ad_seconds, locale)}</Numeric></strong>
            <span>{pageText(locale, 'commercial time', 'זמן פרסום')}</span>
          </div>
          <div>
            <strong><Numeric>{formatCurrency(summary.revenue, locale)}</Numeric></strong>
            <span>{pageText(locale, 'modelled revenue', 'הכנסה מחושבת')}</span>
          </div>
        </div>
        <ZoomControl pxPerMin={pxPerMin} onZoom={setZoom} onStep={zoomBy} locale={locale} />
      </div>

      <ScheduleTrackSurface axis={axis} pxPerMin={pxPerMin} onZoom={setZoom} locale={locale}>
        {({ width, minWidth, ticks }) => lanes.map((lane) => {
          const lanePrograms = programs.filter((item) => item.lane === lane);
          const laneBreaks = breaks.filter((item) => item.lane === lane);
          const laneRevenue = laneBreaks.reduce((sum, item) => sum + Number(item.revenue_calculated || 0), 0);
          return (
            <div className="timeline-row" key={lane} style={{ minWidth }}>
              <div className="timeline-lane" dir={locale === 'he' ? 'rtl' : 'ltr'}>
                <strong>{lane}</strong>
                <span>{laneBreaks.length} {pageText(locale, 'breaks', 'ברייקים')} / <Numeric>{formatCurrency(laneRevenue, locale)}</Numeric></span>
              </div>
              <div className="timeline-track" style={{ width }}>
                {ticks.filter((tick) => tick.major).map((tick) => (
                  <i key={`${lane}-${tick.minute}`} style={{ left: `${tick.left}px` }} />
                ))}
                {lanePrograms.map((program) => (
                  <ProgrammeBand
                    key={program.key || `${program.title}-${program.start_time}`}
                    title={program.title}
                    classLabel={programTypeLabel(program.program_type, locale)}
                    windowText={`${program.start_time} - ${program.end_time}`}
                    style={positionStyle(program.start_time, program.end_time)}
                    clickable
                    onOpen={() => openInspector(program)}
                  />
                ))}
                {laneBreaks.map((breakItem) => {
                  const selected = selectedProgramKey === breakItem.program_key;
                  const selectedProgram = {
                    key: breakItem.program_key,
                    title: breakItem.program_title,
                    channel: breakItem.channel,
                    day: breakItem.day,
                    time: breakItem.start_time,
                    duration_minutes: Math.round(Number(breakItem.duration_sec || 0) / 60),
                    revenue: breakItem.revenue_calculated,
                    retention: breakItem.retention,
                    break_markers: breakItem.breaks_in_program,
                    program_type: breakItem.program_type,
                    selected_break: breakItem,
                  };
                  const className = [
                    'break-chip',
                    'break-chip-readonly',
                    'timeline-break',
                    selected ? 'selected' : '',
                    breakItem.status === 'at_risk' ? 'risk' : '',
                    breakItem.is_gold ? 'gold' : '',
                  ].filter(Boolean).join(' ');
                  // Anchor the chip at its start time and let the shared chip
                  // width govern legibility. A break is a fixed 120s span, so
                  // scaling the width to that duration collapses it to a few
                  // pixels at low zoom; keeping only the left keeps every chip
                  // as readable as the editor's.
                  const { left } = positionStyle(breakItem.start_time, breakItem.end_time);
                  return (
                    <Button
                      className={className}
                      key={breakItem.id}
                      type="button"
                      variant="contained"
                      disableRipple
                      style={{ left }}
                      title={`${breakItem.program_title} / ${breakItem.start_time}-${breakItem.end_time}`}
                      aria-pressed={selected}
                      onClick={() => onSelectProgram(selectedProgram)}
                    >
                      <BreakChip
                        clock={breakItem.start_time}
                        detail={`${breakItem.break_num_in_program}/${breakItem.breaks_in_program}`}
                        gold={Boolean(breakItem.is_gold)}
                        goldLabel={pageText(locale, 'gold', 'זהב')}
                      />
                    </Button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </ScheduleTrackSurface>

      {inspect && (
        <ScheduleInspector
          segmentId={inspect.segmentId}
          channel={inspect.channel}
          day={inspect.day}
          locale={locale}
          notify={notify}
          onClose={() => setInspect(null)}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
    </div>
  );
}

function DaypartView({ rows, locale, selectedProgramKey, onSelectProgram }) {
  const programs = flattenScheduleRows(rows);
  const groups = daypartKeys.map((daypart) => ({
    daypart,
    items: programs.filter((program) => daypartForTime(program.time) === daypart),
  }));
  const populatedGroups = groups.filter((group) => group.items.length > 0);
  const emptyGroups = groups.filter((group) => group.items.length === 0);
  return (
    <div className="daypart-view">
      {populatedGroups.map(({ daypart, items }) => {
        const revenue = items.reduce((sum, program) => sum + Number(program.revenue || 0), 0);
        const avgRetention = items.length
          ? items.reduce((sum, program) => sum + Number(program.retention || 0), 0) / items.length
          : 0;
        return (
          <section className="daypart-card" key={daypart}>
            <div className="daypart-card-head">
              <div>
                <strong>{daypartLabel(daypart, locale)}</strong>
                <span>{items.length} {pageText(locale, 'programs', 'תוכניות')}</span>
              </div>
              <div>
                <strong><Numeric>{formatCurrency(revenue, locale)}</Numeric></strong>
                <span><Numeric>{formatPercent(avgRetention, locale)}</Numeric></span>
              </div>
            </div>
            <div className="daypart-programs">
              {items.slice(0, 7).map((program) => (
                <Button
                  key={program.key}
                  className={program.key === selectedProgramKey ? 'daypart-program active' : 'daypart-program'}
                  type="button"
                  variant="text"
                  onClick={() => onSelectProgram(program)}
                >
                  <span>{program.title}</span>
                  <small>{program.channel} / {program.day} / {program.time}</small>
                  <strong><Numeric>{formatCurrency(program.revenue, locale)}</Numeric></strong>
                </Button>
              ))}
            </div>
          </section>
        );
      })}
      {emptyGroups.length > 0 && (
        <section className="daypart-empty-summary">
          <strong>{pageText(locale, 'No planned inventory', 'אין מלאי מתוכנן')}</strong>
          <span>
            {emptyGroups.map((group) => daypartLabel(group.daypart, locale)).join(' / ')}
          </span>
        </section>
      )}
    </div>
  );
}

function OptimizerInventoryView({ rows, locale, selectedProgramKey, onSelectProgram }) {
  const channelRows = normalizeRows(rows).map((row) => {
    const programs = normalizeRows(row.programs).map((program) => ({
      ...program,
      channel: row.channel,
      key: programKey(row.channel, program),
    }));
    const revenue = programs.reduce((sum, program) => sum + Number(program.revenue || 0), 0);
    const breaks = programs.reduce((sum, program) => sum + Number(program.break_markers || 0), 0);
    const retention = programs.length
      ? programs.reduce((sum, program) => sum + Number(program.retention || 0), 0) / programs.length
      : 0;
    return { channel: row.channel, programs, revenue, breaks, retention };
  });
  const maxRevenue = Math.max(...channelRows.map((row) => row.revenue), 1);

  return (
    <div className="optimizer-inventory-view">
      {channelRows.map((row) => (
        <section className="inventory-channel-card" key={row.channel}>
          <div className="inventory-channel-head">
            <div>
              <strong>{row.channel}</strong>
              <span>{row.programs.length} {pageText(locale, 'programs', 'תוכניות')} / {formatNumber(row.breaks, locale)} {pageText(locale, 'breaks', 'ברייקים')}</span>
            </div>
            <strong><Numeric>{formatCurrency(row.revenue, locale)}</Numeric></strong>
          </div>
          <i className="inventory-pressure" style={{ '--bar': row.revenue / maxRevenue }} />
          <div className="inventory-channel-meta">
            <span>{pageText(locale, 'Avg retention', 'שימור ממוצע')}</span>
            <strong><Numeric>{formatPercent(row.retention, locale)}</Numeric></strong>
          </div>
          <div className="inventory-program-list">
            {row.programs.slice(0, 4).map((program) => (
              <Button
                key={program.key}
                className={program.key === selectedProgramKey ? 'inventory-program active' : 'inventory-program'}
                type="button"
                variant="text"
                onClick={() => onSelectProgram(program)}
              >
                <span>{program.title}</span>
                <small>{program.day} / {program.time}</small>
              </Button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function PlanningCanvas({ rows, copy, locale, axis = 'day', showPrograms = true, showBreaks = true, selectedProgramKey, onSelectProgram }) {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const columns = buildPlannerColumns(rows, axis, locale);
  const cellMinWidth = axis === 'hour' ? 112 : 136;
  const gridTemplateColumns = `142px repeat(${columns.length}, minmax(${cellMinWidth}px, 1fr))`;
  const minWidth = 142 + columns.length * cellMinWidth;
  const dayLabels = locale === 'he' ? ['ב׳', 'ג׳', 'ד׳', 'ה׳', 'ו׳', 'ש׳', 'א׳'] : days;
  return (
    <div className="planning-canvas">
      <div className="canvas-header" style={{ gridTemplateColumns, minWidth }}>
        <span>{copy.channelProgram} / {gridAxisLabel(axis, locale)}</span>
        {columns.map((column) => (
          <span key={column.key}>{column.label}</span>
        ))}
      </div>
      {rows.map((row, rowIndex) => {
        const programs = Array.isArray(row.programs) ? row.programs : [];
        const channelName = String(row.channel || 'Channel');
        return (
        <div className="channel-row" key={channelName || `channel-${rowIndex}`} style={{ gridTemplateColumns, minWidth }}>
          <div className="channel-name">
            <span>{channelName.replace('ערוץ', 'K')}</span>
            <small>{programTypeLabel(programs[0]?.program_type, locale) || pageText(locale, 'Mixed', 'מעורב')}</small>
          </div>
          {columns.map((column) => {
            const cellPrograms = programsForPlannerColumn(programs, column, axis);
            const program = cellPrograms.find((item) => item.selected) || cellPrograms[0];
            const programWithChannel = program
              ? { channel: channelName, ...program, key: programKey(channelName, program) }
              : null;
            const totalRevenue = cellPrograms.reduce((sum, item) => sum + Number(item.revenue || 0), 0);
            const averageRetention = cellPrograms.length
              ? cellPrograms.reduce((sum, item) => sum + Number(item.retention || 0), 0) / cellPrograms.length
              : 0;
            const markerCount = cellPrograms.reduce((sum, item) => sum + Number(item.break_markers || 0), 0);
            const timeRange = cellPrograms.length
              ? `${cellPrograms[0].time} - ${cellPrograms[cellPrograms.length - 1].time}`
              : '';
            const selectedInCell = selectedProgramKey
              ? cellPrograms.some((item) => programKey(channelName, item) === selectedProgramKey)
              : cellPrograms.some((item) => item.selected);
            return (
              <ProgramCell
                key={`${channelName}-${column.key}`}
                program={programWithChannel}
                locale={locale}
                selected={selectedInCell}
                programCount={cellPrograms.length}
                totalRevenue={totalRevenue}
                averageRetention={averageRetention}
                markerCount={markerCount}
                timeRange={timeRange}
                showPrograms={showPrograms}
                showBreaks={showBreaks}
                onSelect={onSelectProgram}
              />
            );
          })}
        </div>
        );
      })}
    </div>
  );
}

function ProgramCell({
  program,
  locale,
  selected = false,
  programCount = 1,
  totalRevenue,
  averageRetention,
  markerCount,
  timeRange,
  showPrograms = true,
  showBreaks = true,
  onSelect,
}) {
  if (!program) return <div className="program-cell empty" />;
  // Marker dots mirror the planned break count; zero breaks shows zero dots
  // (the fixed-height strip keeps the cell layout stable) instead of a
  // fabricated minimum of one.
  const markers = Array.from({
    length: Math.max(0, Math.min(10, Number(markerCount ?? program.break_markers ?? 0) || 0)),
  });
  const revenue = totalRevenue ?? program.revenue;
  const retention = averageRetention ?? program.retention;
  const meta = programCount > 1
    ? `${formatNumber(programCount, locale)} ${pageText(locale, 'programs', 'תוכניות')} / ${timeRange}`
    : `${program.time} / ${formatMinutes(Number(program.duration_minutes || 0) * 60, locale)}`;
  return (
    <Button
      className={selected ? 'program-cell selected' : 'program-cell'}
      type="button"
      variant="text"
      disableRipple
      aria-pressed={selected}
      title={`${program.title} / ${program.channel} / ${program.day} ${program.time}`}
      onClick={() => onSelect?.(program)}
    >
      {showPrograms ? (
        <span className="program-title">{program.title}</span>
      ) : (
        <span className="program-title muted-title">{programTypeLabel(program.program_type, locale) || pageText(locale, 'Program hidden', 'תוכנית מוסתרת')}</span>
      )}
      <span className="program-meta">{meta}</span>
      {showBreaks && (
        <span className="break-markers">
          {markers.map((_, index) => (
            <i key={index} className={index % 3 === 0 ? 'marker revenue' : 'marker'} />
          ))}
        </span>
      )}
      <span className="cell-metrics">
        <span><Numeric>{formatCurrency(revenue, locale)}</Numeric></span>
        <span><Numeric>{formatPercent(retention, locale)}</Numeric></span>
      </span>
    </Button>
  );
}

function Inspector({ selectedProgram, recommendation, approved, rejected, retentionFloor, onApprove, onReject, onOpenInOverrides, onApplySimilar, onExport, onClose, copy, locale }) {
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
        <IconButton className="icon-button small" type="button" aria-label={pageText(locale, 'Close inspector', 'סגירת המפקח')} size="small" onClick={onClose}>
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

function FrontierPanel({ data, copy, locale, loading = false, operatorChannel = '', status = '', netPoint = null }) {
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

function InventoryHeatmap({ copy, locale }) {
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

function ComplianceLedger({ compliance, copy, locale }) {
  const checks = compliance?.checks || [];
  return (
    <div className="analytics-panel ledger-panel">
      <div className="panel-head">
        <h2>{copy.compliance}</h2>
        <span>{checks.length} {copy.activeRules}</span>
      </div>
      <div className="ledger-list">
        {checks.map((check) => {
          const formatValue = (value) => Number(value).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
          const violationCount = Number(check.violations || 0);
          const unitLabel = complianceUnitLabel(check.unit, locale);
          const isPercent = check.unit === '%';
          const observed = `${formatValue(check.observed)}${isPercent ? '%' : ''}`;
          const limit = `${formatValue(check.limit)}${isPercent ? '%' : ''}`;
          return (
            <div className="ledger-row" key={check.id}>
              <span>{locale === 'he' ? check.label_he : check.label_en}</span>
              <strong className={check.status === 'at_risk' ? 'at-risk' : ''}>
                {check.status === 'at_risk' ? copy.atRisk : copy.compliant}
              </strong>
              <small className="ledger-measure" dir={locale === 'he' ? 'rtl' : 'ltr'}>
                <span className="ledger-values" dir="ltr">{observed} / {limit}</span>
                {!isPercent && unitLabel && <span className="ledger-unit">{unitLabel}</span>}
                {violationCount > 0 && (
                  <span className="ledger-violations">
                    {formatValue(violationCount)} {pageText(locale, 'violations', 'חריגות')}
                  </span>
                )}
              </small>
            </div>
          );
        })}
        <p className="ledger-note">{complianceDisclaimer(compliance?.disclaimer, locale)}</p>
      </div>
    </div>
  );
}

// OperatorChannelPanel: shows available_channels from /api/parameters and lets
// the operator choose which channel they own. The selection is persisted via
// the same PUT /api/settings path as all other settings.
function OperatorChannelPanel({ settings, parameters, locale, onSave, saveState, featured }) {
  const he = locale === 'he';
  const availableChannels = normalizeRows(
    parameters?.available_channels || parameters?.settings?.available_channels,
  );
  const currentChannel = settings?.operator_channel || '';

  function handleChange(channel) {
    onSave({ ...settings, operator_channel: channel });
  }

  return (
    <section className={`settings-panel wide${featured ? ' settings-panel-featured' : ''}`}>
      <div className="settings-panel-head">
        <div>
          {featured && (
            <span className="settings-channel-kicker">{he ? 'נקודת הפתיחה' : 'Start here'}</span>
          )}
          <h2>{he ? 'הערוץ שלך' : 'Your channel'}</h2>
          <p>{he ? 'הערוץ שבבעלות האופרטור. האילוצים שלך חלים על ערוץ זה, והוא משער את ההכנסה מול שמירת הצופים.' : 'The channel this operator owns. Your constraints apply to this channel, and it is the gateway to the revenue versus retention view.'}</p>
        </div>
        <Tv size={18} />
      </div>
      <label htmlFor="operator-channel-select" style={{ display: 'block', marginBottom: 6, fontSize: 12, fontWeight: 600, color: 'var(--muted)' }}>
        {he ? 'ערוץ' : 'Channel'}
      </label>
      <FormControl size="small" sx={{ minWidth: 220 }}>
        <Select
          id="operator-channel-select"
          value={currentChannel}
          displayEmpty
          onChange={(e) => handleChange(e.target.value)}
          renderValue={(selected) => selected || (he ? 'לא נבחר' : 'Not set')}
        >
          <MenuItem value="">{he ? 'לא נבחר' : 'Not set'}</MenuItem>
          {availableChannels.map((ch) => {
            const val = typeof ch === 'string' ? ch : ch.key || ch.value || ch.name || String(ch);
            return <MenuItem key={val} value={val}>{val}</MenuItem>;
          })}
        </Select>
      </FormControl>
      {currentChannel && (
        <p className="cb-operator-channel-note">
          {he ? `האילוצים החדשים יחולו על ערוץ "${currentChannel}".` : `New constraints will be scoped to channel "${currentChannel}".`}
        </p>
      )}
      {!currentChannel && (
        <p className="cb-operator-channel-warning">
          {he ? 'אזהרה: הערוץ אינו מוגדר. מסנן הערוץ המתחרה אינו פעיל - האילוצים חלים על כל הערוצים עד שתבחר ערוץ.' : 'Warning: no channel is set. The competitor-channel boundary filter is inactive - constraints match all channels until you pick your channel.'}
        </p>
      )}
    </section>
  );
}

// Compact humanized label for an activity entry. Known actions get a plain
// language name; everything else falls back to a method+path code chip.
function activityActionLabel(entry, he) {
  const event = entry.event || '';
  if (event === 'login') return he ? 'כניסה למערכת' : 'Signed in';
  if (event === 'login_failed') return he ? 'ניסיון כניסה שנכשל' : 'Failed sign-in attempt';
  if (event === 'logout') return he ? 'יציאה מהמערכת' : 'Signed out';
  const method = entry.method || '';
  const path = entry.path || '';
  if (method === 'PUT' && path === '/api/settings') return he ? 'עדכון הגדרות' : 'Settings update';
  if (method === 'POST' && /^\/api\/assistant\/proposals\/[^/]+\/apply$/.test(path)) return he ? 'אישור הצעות העוזר' : 'Assistant proposal approval';
  if (method === 'POST' && (path === '/api/recompute-schedule' || path === '/api/jobs/recompute')) return he ? 'חישוב מחדש' : 'Recompute';
  return null;
}

function activityTimeLabel(ts, he) {
  if (!ts) return '';
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(he ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

// The settings-page activity log: who changed what and when, served by
// GET /api/activity-log. The API decides visibility (admin sees everyone, any
// other role only itself, dev mode without login sees everything), so this
// panel renders the scope it was given and says so honestly instead of
// pretending to filter anything client-side.
function ActivityLogPanel({ locale }) {
  const he = locale === 'he';
  const [log, setLog] = useState({ status: 'loading', entries: [], scope: 'all' });
  const [userFilter, setUserFilter] = useState('');
  const [knownUsers, setKnownUsers] = useState([]);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLog((current) => ({ ...current, status: 'loading' }));
    (async () => {
      try {
        const filter = userFilter ? `&user=${encodeURIComponent(userFilter)}` : '';
        const response = await fetch(`${API_BASE}/api/activity-log?limit=100${filter}`, { credentials: 'include' });
        if (!response.ok) throw new Error(`${response.status}`);
        const data = await response.json();
        if (cancelled) return;
        const entries = Array.isArray(data.entries) ? data.entries : [];
        setLog({ status: 'ready', entries, scope: data.scope === 'self' ? 'self' : 'all' });
        if (!userFilter) {
          setKnownUsers((current) => {
            const merged = new Set(current);
            entries.forEach((entry) => {
              if (entry.user) merged.add(entry.user);
            });
            return Array.from(merged).sort();
          });
        }
      } catch {
        if (!cancelled) setLog({ status: 'error', entries: [], scope: 'all' });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [userFilter, reloadKey]);

  const showUserColumn = log.scope === 'all';
  const filterLabel = he ? 'סינון לפי משתמש' : 'Filter by user';
  return (
    <section className="settings-panel wide">
      <div className="settings-panel-head">
        <div>
          <h2>{he ? 'יומן פעילות' : 'Activity log'}</h2>
          <p>{he ? 'מי שינה מה ומתי, כולל פעולות שבוצעו דרך עוזר ה־AI' : 'Who changed what and when, including actions made through the AI assistant'}</p>
        </div>
        <Activity size={18} />
      </div>
      <div className="alog-toolbar">
        <div className="alog-controls">
          {showUserColumn && knownUsers.length > 0 && (
            <FormControl size="small" className="alog-filter">
              <InputLabel id="alog-user-filter">{filterLabel}</InputLabel>
              <Select
                labelId="alog-user-filter"
                label={filterLabel}
                value={userFilter}
                onChange={(event) => setUserFilter(event.target.value)}
              >
                <MenuItem value="">{he ? 'כל המשתמשים' : 'All users'}</MenuItem>
                {knownUsers.map((name) => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {log.scope === 'self' && (
            <span className="alog-self-note">{he ? 'מוצגת הפעילות שלך בלבד' : 'Showing your own activity only'}</span>
          )}
        </div>
        <Button
          type="button"
          variant="outlined"
          className="run-button"
          disabled={log.status === 'loading'}
          onClick={() => setReloadKey((key) => key + 1)}
        >
          <RefreshCcw size={15} />
          {he ? 'רענון' : 'Refresh'}
        </Button>
      </div>
      {log.status === 'loading' && <p className="alog-note">{he ? 'רק רגע...' : 'Loading...'}</p>}
      {log.status === 'error' && (
        <p className="alog-note alog-error" role="alert">{he ? 'טעינת יומן הפעילות נכשלה. אפשר לנסות לרענן.' : 'Could not load the activity log. Try refreshing.'}</p>
      )}
      {log.status === 'ready' && log.entries.length === 0 && (
        <p className="alog-note">
          {userFilter
            ? (he ? 'אין רשומות למשתמש שנבחר.' : 'No entries for the selected user.')
            : (he ? 'אין עדיין רשומות ביומן. פעולות שינוי יופיעו כאן.' : 'No activity recorded yet. Changes will appear here.')}
        </p>
      )}
      {log.status === 'ready' && log.entries.length > 0 && (
        <div className="alog-table-wrap">
          <table className="alog-table">
            <thead>
              <tr>
                <th>{he ? 'זמן' : 'Time'}</th>
                {showUserColumn && <th>{he ? 'משתמש' : 'User'}</th>}
                <th>{he ? 'פעולה' : 'Action'}</th>
                <th>{he ? 'סטטוס' : 'Status'}</th>
              </tr>
            </thead>
            <tbody>
              {log.entries.map((entry, index) => {
                const label = activityActionLabel(entry, he);
                const status = Number(entry.status);
                const hasStatus = Number.isFinite(status) && status > 0;
                return (
                  <tr key={`${entry.ts || 'entry'}-${index}`}>
                    <td><span className="alog-time" dir="ltr">{activityTimeLabel(entry.ts, he)}</span></td>
                    {showUserColumn && <td><span className="alog-user">{entry.user || ''}</span></td>}
                    <td>
                      {label ? <span>{label}</span> : <code className="alog-code" dir="ltr">{`${entry.method || ''} ${entry.path || ''}`.trim()}</code>}
                      {entry.via === 'assistant' && <span className="alog-via">{he ? 'עוזר AI' : 'AI assistant'}</span>}
                    </td>
                    <td>{hasStatus ? <span className={`alog-status${status >= 400 ? ' warn' : ''}`} dir="ltr">{status}</span> : null}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function SettingsPanel({ settings, parameters, campaigns, copy, locale, saveState, onSave, onRecompute, recomputeState, notify }) {
  const [draft, setDraft] = useState(settings);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  function updateField(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  function updateNumber(field, value) {
    const parsed = Number(value);
    updateField(field, Number.isFinite(parsed) ? parsed : 0);
  }

  function applyTemplate(values) {
    setDraft((current) => ({ ...current, ...values }));
  }

  const he = locale === 'he';
  // The named setups (templates) that snap the levers to a known posture. Kept
  // in sync with GET /api/settings/controls so the dashboard and the engine
  // agree on what each preset means.
  const optimizerTemplates = [
    { key: 'balanced', label: he ? 'מאוזן' : 'Balanced', desc: he ? 'נוטה-להכנסה אך שומר על הצופים' : 'Revenue-leaning, viewer-protective', values: { revenue_weight: 60, risk_lambda: 0, min_retention_floor: 0.72 } },
    { key: 'revenue', label: he ? 'עדיפות להכנסה' : 'Revenue priority', desc: he ? 'ממקסם הכנסה עד גבול הרגולציה' : 'Maximize revenue to the guardrails', values: { revenue_weight: 85, risk_lambda: 0, min_retention_floor: 0.70 } },
    { key: 'retention', label: he ? 'שמירה על צפייה' : 'Retention guardrail', desc: he ? 'פחות הפסקות, רצפת צפייה גבוהה' : 'Fewer breaks, higher floor', values: { revenue_weight: 35, risk_lambda: 0, min_retention_floor: 0.78 } },
    { key: 'conservative', label: he ? 'זהיר באי-ודאות' : 'Conservative', desc: he ? 'מדווח לפי עלות הצפייה הסבירה הגרועה ביותר' : 'Reports at the worst plausible retention cost', values: { revenue_weight: 60, risk_lambda: 1, min_retention_floor: 0.74 } },
  ];
  const revenueWeight = Number.isFinite(finiteNumber(draft.revenue_weight)) ? finiteNumber(draft.revenue_weight) : 60;
  const recomputeText =
    recomputeState === 'running'
      ? (he ? 'מחשב מחדש...' : 'Recomputing...')
      : recomputeState === 'done'
        ? (he ? 'הלוח עודכן' : 'Schedule updated')
        : recomputeState === 'error'
          ? (he ? 'החישוב נכשל' : 'Recompute failed')
          : (he ? 'חשב מחדש את הלוח השבועי' : 'Recompute weekly schedule');

  const protectedTypes = (draft.protected_program_types || []).join(', ');

  // Honest empty state for pacing: pacing can only steer placement when there
  // are real campaign flights to pace against. We read the live campaigns
  // payload (the same one the Campaigns page uses) rather than fabricating any
  // count, and treat an empty list as "no flights uploaded yet".
  const campaignFlights = normalizeRows(campaigns?.campaigns);
  const hasCampaignFlights = campaignFlights.length > 0;
  const statusText =
    saveState === 'saved'
      ? copy.saved
      : saveState === 'saving'
        ? copy.saving
        : saveState === 'error'
          ? copy.saveFailed
          : copy.saveSettings;

  // Dirty detection: compare the in-progress draft against the saved settings.
  // This drives the "unsaved changes" affordance on the sticky action bar. We
  // compare by stable JSON so field order or array identity does not matter.
  const isDirty = useMemo(() => {
    try {
      return stableSettingsKey(draft) !== stableSettingsKey(settings);
    } catch {
      return true;
    }
  }, [draft, settings]);

  // The status line for the sticky bar reflects the real save lifecycle and the
  // real draft-vs-saved comparison: saving / saved / failed come from saveState,
  // otherwise we show unsaved vs all-saved based on isDirty.
  const stickyStatus =
    saveState === 'saving'
      ? { text: copy.saving, tone: 'saving' }
      : saveState === 'error'
        ? { text: copy.saveFailed, tone: 'error' }
        : isDirty
          ? { text: copy.unsavedChanges, tone: 'dirty' }
          : saveState === 'saved'
            ? { text: copy.saved, tone: 'saved' }
            : { text: copy.noChanges, tone: 'clean' };

  return (
    <section className="settings-workspace">
      <div className="settings-hero">
        <div>
          <span className="settings-kicker">{copy.nav.Settings}</span>
          <h1>{copy.settingsTitle}</h1>
          <p>{copy.settingsIntro}</p>
        </div>
      </div>

      <OperatorChannelPanel
        settings={draft}
        parameters={parameters}
        locale={locale}
        onSave={onSave}
        saveState={saveState}
        featured
      />

      <div className="settings-grid">
        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{he ? 'איזון האופטימיזציה' : 'Optimizer balance'}</h2>
              <p>{he ? 'הלֶבֶר המרכזי שמניע את הלוח, ההכנסה מול השימור והתחזיות' : 'The central lever that drives the schedule, revenue vs retention, and forecasts'}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="optimizer-balance">
            <p className="optimizer-balance-help">
              {he
                ? 'כמה לרדוף אחרי הכנסת פרסום מול שמירה על הצופים. 0 שומר על הצפייה בלבד (כמעט בלי הפסקות), 100 ממקסם הכנסה עד גבול הרגולציה, 60 הוא איזון נוטה-להכנסה (ברירת המחדל).'
                : 'How hard to chase ad revenue versus protecting viewers. 0 protects retention only (almost no breaks), 100 maximizes revenue up to the regulatory guardrails, 60 is a revenue-leaning balance (the default).'}
            </p>
            <div className="optimizer-balance-slider">
              <span>{he ? 'צפייה' : 'Retention'}</span>
              <Slider
                value={revenueWeight}
                min={0}
                max={100}
                step={5}
                marks={[{ value: 0 }, { value: 60, label: he ? 'דיפולט' : 'default' }, { value: 100 }]}
                valueLabelDisplay="on"
                onChange={(_event, value) => updateField('revenue_weight', Array.isArray(value) ? value[0] : value)}
              />
              <span>{he ? 'הכנסה' : 'Revenue'}</span>
            </div>
            <div className="optimizer-templates">
              {optimizerTemplates.map((template) => {
                const active = revenueWeight === template.values.revenue_weight && finiteNumber(draft.risk_lambda) === template.values.risk_lambda;
                return (
                  <button
                    key={template.key}
                    type="button"
                    className={`optimizer-template${active ? ' is-active' : ''}`}
                    onClick={() => applyTemplate(template.values)}
                  >
                    <strong>{template.label}</strong>
                    <small>{template.desc}</small>
                  </button>
                );
              })}
            </div>
            <div className="optimizer-objective">
              <span className="settings-field-label">{he ? 'מיקוד המנוע' : 'Engine focus'}</span>
              <div className="optimizer-objective-options">
                {[
                  { key: 'blend', label: he ? 'מאוזן, ברירת המחדל' : 'Balanced, the default',
                    desc: he ? 'המנוע מאזן בין הכנסות ברוטו לשמירה על הצופים, לפי המשקל שנקבע למעלה.' : 'The engine balances gross revenue against keeping viewers, using the weight set above.' },
                  { key: 'revenue_net', label: he ? 'ממוקד נטו' : 'Net focused',
                    desc: he ? 'המנוע מוותר על ברייקים שההכנסה שלהם נמוכה מעלות השימור שלהם: פחות ברייקים, ברוטו נמוך יותר, נטו גבוה יותר.' : 'The engine drops breaks whose revenue is below their retention cost: fewer breaks, lower gross, higher net.' },
                ].map((mode) => {
                  const active = (draft.objective_mode || 'blend') === mode.key;
                  return (
                    <button
                      key={mode.key}
                      type="button"
                      className={`optimizer-template${active ? ' is-active' : ''}`}
                      onClick={() => updateField('objective_mode', mode.key)}
                    >
                      <strong>{mode.label}</strong>
                      <small>{mode.desc}</small>
                    </button>
                  );
                })}
              </div>
              {(draft.objective_mode || 'blend') === 'revenue_net' && (
                <p className="optimizer-objective-note" role="status">
                  {he
                    ? 'שימו לב: מיקוד נטו משנה את התוכנית השמורה בעת חישוב מחדש, וההכנסות ברוטו בכותרת יירדו. זו בחירה מכוונת לטובת הנטו.'
                    : 'Note: net focus changes the saved plan on recompute, and the gross revenue headline will fall. It is a deliberate choice in favor of the net.'}
                </p>
              )}
              <NetComparisonCard locale={locale} refreshSignal={recomputeState || ''} currentFocus={draft.objective_mode || 'blend'} />
            </div>
            <div className="optimizer-recompute">
              <p>
                {he
                  ? 'שמור את ההגדרות, ואז חשב מחדש את הלוח השבועי כדי שהמסכים יראו את ההחלטה החדשה.'
                  : 'Save the settings, then recompute the weekly schedule so the screens reflect the new decision.'}
              </p>
              <Button
                type="button"
                variant="outlined"
                className="run-button"
                disabled={recomputeState === 'running'}
                onClick={() => onRecompute && onRecompute()}
              >
                <RefreshCcw size={15} />
                {recomputeText}
              </Button>
            </div>
          </div>
        </section>

        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{copy.profile}</h2>
              <p>{draft.profile_name}</p>
            </div>
            <BookOpen size={18} />
          </div>
          <div className="settings-form-grid">
            <TextField
              label={copy.profile}
              size="small"
              value={draft.profile_name || ''}
              onChange={(event) => updateField('profile_name', event.target.value)}
            />
            <DateField
              label={copy.effectiveDate}
              value={draft.effective_date}
              onChange={(value) => updateField('effective_date', value)}
            />
            <FormControl size="small">
              <InputLabel id="settings-locale">{copy.language}</InputLabel>
              <Select
                labelId="settings-locale"
                label={copy.language}
                value={draft.locale || 'he'}
                onChange={(event) => updateField('locale', event.target.value)}
              >
                <MenuItem value="he">{copy.hebrew}</MenuItem>
                <MenuItem value="en">{copy.english}</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label={copy.source}
              size="small"
              value={draft.regulatory_source_url || ''}
              onChange={(event) => updateField('regulatory_source_url', event.target.value)}
            />
          </div>
        </section>

        <section className="settings-panel">
          <div className="settings-panel-head">
            <div>
              <h2>{copy.guardrails}</h2>
              <p>{locale === 'he' ? 'בקרות תכנון מסחריות' : 'Commercial planning controls'}</p>
            </div>
            <ShieldCheck size={18} />
          </div>
          <div className="settings-form-stack">
            <NumberControl label={copy.maxAdMinutes} value={draft.max_ad_minutes_per_hour} onChange={(value) => updateNumber('max_ad_minutes_per_hour', value)} suffix="min" />
            <NumberControl label={copy.maxBreaks} value={draft.max_breaks_per_hour} onChange={(value) => updateNumber('max_breaks_per_hour', value)} suffix="/hr" />
            <NumberControl label={copy.spacing} value={draft.min_break_spacing_minutes} onChange={(value) => updateNumber('min_break_spacing_minutes', value)} suffix="min" />
            <NumberControl label={copy.retentionFloor} value={Math.round((draft.min_retention_floor || 0) * 100)} onChange={(value) => updateNumber('min_retention_floor', Number(value) / 100)} suffix="%" />
            <NumberControl
              label={copy.riskCautionSetting}
              value={Math.round((finiteNumber(draft.risk_lambda) || 0) * 100)}
              onChange={(value) => updateNumber('risk_lambda', Math.min(1, Math.max(0, Number(value) / 100)))}
              suffix="/100"
            />
          </div>
        </section>

        <section className="settings-panel">
          <div className="settings-panel-head">
            <div>
              <h2>{locale === 'he' ? 'תוכן מוגן' : 'Protected content'}</h2>
              <p>{locale === 'he' ? 'חדשות, ילדים ותוכניות רגישות' : 'News, kids, and sensitive formats'}</p>
            </div>
            <AlertTriangle size={18} />
          </div>
          <div className="settings-form-stack">
            <NumberControl label={copy.protectedMax} value={draft.protected_program_max_ad_minutes_per_hour} onChange={(value) => updateNumber('protected_program_max_ad_minutes_per_hour', value)} suffix="min" />
            <TextField
              label={copy.protectedTypes}
              size="small"
              multiline
              minRows={3}
              value={protectedTypes}
              onChange={(event) =>
                updateField(
                  'protected_program_types',
                  event.target.value.split(',').map((item) => item.trim()).filter(Boolean),
                )
              }
            />
          </div>
        </section>

        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{locale === 'he' ? 'מדיניות מסחרית' : 'Commercial policy'}</h2>
              <p>{locale === 'he' ? 'חסויות וברייקי זהב' : 'Sponsorships and gold breaks'}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="settings-toggle-grid">
            <ToggleControl label={copy.sponsorships} checked={draft.sponsorships_enabled} onChange={(value) => updateField('sponsorships_enabled', value)} />
            <ToggleControl label={copy.gold} checked={draft.gold_breaks_enabled} onChange={(value) => updateField('gold_breaks_enabled', value)} />
            <NumberControl label={locale === 'he' ? 'מקסימום ברייקי זהב ביום' : 'Max gold breaks per day'} value={draft.gold_breaks_max_per_day} onChange={(value) => updateNumber('gold_breaks_max_per_day', value)} suffix="/day" />
            <NumberControl label={copy.dailyCap} value={draft.max_daily_ad_minutes} onChange={(value) => updateNumber('max_daily_ad_minutes', value)} suffix="min" />
          </div>
        </section>

        <section className="settings-panel wide">
          <div className="settings-panel-head">
            <div>
              <h2>{he ? 'קצב קמפיינים' : 'Campaign pacing'}</h2>
              <p>{he ? 'מטה את השיבוץ לפי קצב הדילוור של הקמפיינים, בלי לשנות את תחזית ההכנסה' : 'Steer placement by campaign delivery pace, without changing the revenue projection'}</p>
            </div>
            <Gauge size={18} />
          </div>
          {!hasCampaignFlights && (
            <p className="settings-pacing-note">
              {he ? 'טרם הועלו קמפיינים, ולכן הקצב אינו פעיל.' : 'No campaign flights uploaded yet, so pacing is inactive.'}
            </p>
          )}
          <div className="settings-toggle-grid">
            <ToggleControl
              label={he ? 'קצב קמפיינים' : 'Campaign pacing'}
              checked={draft.pacing_enabled ?? true}
              onChange={(value) => updateField('pacing_enabled', value)}
              helperText={he ? 'מטה את השיבוץ לעבר קמפיינים שמפגרים בקצב הדילוור והרחק מקמפיינים שדילברו יותר מדי. שיבוץ בלבד; לעולם לא משנה את תחזית ההכנסה.' : 'Steer placement toward campaigns behind delivery pace and away from over-delivered ones. Placement only; never changes the revenue projection.'}
            />
            <DateField
              label={he ? 'תאריך ייחוס לקצב' : 'Pacing reference date'}
              value={draft.pacing_reference_date}
              onChange={(value) => updateField('pacing_reference_date', value)}
              helperText={he ? 'התאריך שנחשב כהיום בעת מדידת קצב הקמפיין. ריק משתמש בתאריך התוקף של הלוח.' : 'The date treated as today when measuring campaign pace. Empty uses the schedule effective date.'}
            />
            <NumberControl
              label={he ? 'עוצמת פיגור בקצב' : 'Behind-pace strength'}
              value={draft.pacing_urgency_k ?? 1.0}
              onChange={(value) => updateNumber('pacing_urgency_k', Math.min(5, Math.max(0, Number(value))))}
              helperText={he ? 'כמה חזק קמפיין בתת-דילוור מושך פרסומות למלאי שלו.' : 'How hard an under-delivered campaign pulls breaks toward its inventory.'}
            />
            <NumberControl
              label={he ? 'תקרת פיגור בקצב' : 'Behind-pace cap'}
              value={draft.pacing_urgency_max ?? 2.0}
              onChange={(value) => updateNumber('pacing_urgency_max', Math.min(4, Math.max(1, Number(value))))}
              helperText={he ? 'הגברת השיבוץ המרבית לקמפיין המפגר ביותר.' : 'Maximum placement boost for the most behind campaign.'}
            />
            <NumberControl
              label={he ? 'ריסון דילוור-יתר' : 'Over-delivery throttle'}
              value={draft.pacing_ahead_k ?? 1.0}
              onChange={(value) => updateNumber('pacing_ahead_k', Math.min(5, Math.max(0, Number(value))))}
              helperText={he ? 'כמה חזק קמפיין בדילוור-יתר מקבל עדיפות נמוכה בשיבוץ. אפס מבטל את קנס דילוור-היתר.' : 'How hard an over-delivered campaign is de-prioritized in placement. Zero disables the over-delivery penalty.'}
            />
            <NumberControl
              label={he ? 'רצפת דילוור-יתר' : 'Over-delivery floor'}
              value={draft.pacing_weight_floor ?? 0.5}
              onChange={(value) => updateNumber('pacing_weight_floor', Math.min(1.0, Math.max(0.25, Number(value))))}
              helperText={he ? 'המשקל הנמוך ביותר בשיבוץ שקמפיין בדילוור-יתר יכול לקבל. לעולם לא אפס, כך שפרסומת לעולם אינה נחסמת.' : 'The lowest placement weight an over-delivered campaign can receive. Never zero, so a slot is never forbidden.'}
            />
            <NumberControl
              label={he ? 'רצפת מכנה הקצב' : 'Pace denominator floor'}
              value={draft.pacing_epsilon ?? 0.05}
              onChange={(value) => updateNumber('pacing_epsilon', Math.min(0.5, Math.max(0.01, Number(value))))}
              helperText={he ? 'רצפה נומרית כדי שהדחיפות תישאר סופית ביום הראשון והאחרון של הקמפיין.' : 'Numerical floor so urgency stays finite on the first and last flight day.'}
            />
          </div>
        </section>

        <ConstraintBuilder
          locale={locale}
          notify={notify || (() => {})}
          onRecompute={onRecompute}
          recomputeState={recomputeState}
        />

        <ActivityLogPanel locale={locale} />
      </div>

      <div className={`settings-savebar tone-${stickyStatus.tone}`}>
        <span className="settings-savebar-status" aria-live="polite">
          <span className="settings-savebar-dot" aria-hidden="true" />
          {stickyStatus.text}
        </span>
        <Button
          className="run-button"
          type="button"
          variant="contained"
          disabled={saveState === 'saving' || !isDirty}
          onClick={() => onSave(draft)}
        >
          <Save size={15} />
          {statusText}
        </Button>
      </div>
    </section>
  );
}

// The unit (min, /day, %, ...) sits INSIDE the field as an end adornment, so every
// settings field is one full-width frame at the same width whether it carries a unit
// or not. Native number spinners are hidden in CSS (they only showed on hover and
// looked out of place); the value stays fully typeable.
function NumberControl({ label, value, onChange, suffix, helperText }) {
  return (
    <TextField
      className="settings-number"
      label={label}
      type="number"
      size="small"
      fullWidth
      value={value ?? 0}
      onChange={(event) => onChange(event.target.value)}
      helperText={helperText}
      slotProps={suffix ? {
        input: { endAdornment: <InputAdornment position="end">{suffix}</InputAdornment> },
      } : undefined}
    />
  );
}

function ToggleControl({ label, checked, onChange, helperText }) {
  return (
    <div className="toggle-field">
      <div className="toggle-control">
        <span>{label}</span>
        <Switch size="small" checked={Boolean(checked)} onChange={(event) => onChange(event.target.checked)} />
      </div>
      {helperText ? <p className="settings-field-help">{helperText}</p> : null}
    </div>
  );
}

function operatorInitials(name) {
  const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return '?';
  const first = parts[0][0] || '';
  const second = parts.length > 1 ? parts[parts.length - 1][0] || '' : '';
  return (first + second).toUpperCase() || '?';
}

// Admin-only account management over /api/auth/users*: list, create, delete
// and reset passwords. Every failure surfaces honestly; nothing is optimistic.
function UserAdminDialog({ locale, selfUsername, notify, onClose }) {
  const t = (en, he) => pageText(locale, en, he);
  const [accounts, setAccounts] = useState([]);
  const [loadState, setLoadState] = useState('loading');
  const [reloadKey, setReloadKey] = useState(0);
  const [form, setForm] = useState({ username: '', display_name: '', role: 'viewer', password: '' });
  const [formError, setFormError] = useState('');
  const [busy, setBusy] = useState(false);
  const [resetFor, setResetFor] = useState('');
  const [resetValue, setResetValue] = useState('');
  const [rowError, setRowError] = useState('');
  const [confirmDelete, setConfirmDelete] = useState('');

  useEffect(() => {
    let active = true;
    setLoadState('loading');
    fetchAccounts().then((result) => {
      if (!active) return;
      if (result.ok && result.data && Array.isArray(result.data.users)) {
        setAccounts(result.data.users);
        setLoadState('ready');
      } else {
        setLoadState('error');
      }
    });
    return () => {
      active = false;
    };
  }, [reloadKey]);

  const adminTotal = accounts.filter((account) => account.role === 'admin').length;

  function describeFailure(result) {
    if (result.status === 0) return t('No connection to the server.', 'אין חיבור לשרת.');
    if (result.status === 409) return t('That username is already taken.', 'שם המשתמש הזה כבר תפוס.');
    if (result.status === 422) {
      return t(
        `The password needs at least ${MIN_PASSWORD_LENGTH} characters.`,
        `הסיסמה צריכה להכיל לפחות ${MIN_PASSWORD_LENGTH} תווים.`,
      );
    }
    const detail = result.data && result.data.detail ? String(result.data.detail) : '';
    if (detail.toLowerCase().includes('last admin')) {
      return t('The last admin account cannot be deleted.', 'אי אפשר למחוק את חשבון הניהול האחרון.');
    }
    if (detail.toLowerCase().includes('signed in with')) {
      return t('You cannot delete the account you are signed in with.', 'אי אפשר למחוק את החשבון שאיתו נכנסת למערכת.');
    }
    if (detail) return detail;
    return t(`The request failed (status ${result.status}).`, `הפעולה נכשלה (סטטוס ${result.status}).`);
  }

  async function submitCreate(event) {
    event.preventDefault();
    if (busy) return;
    if (form.password.length < MIN_PASSWORD_LENGTH) {
      setFormError(t(
        `The temporary password needs at least ${MIN_PASSWORD_LENGTH} characters.`,
        `הסיסמה הזמנית צריכה להכיל לפחות ${MIN_PASSWORD_LENGTH} תווים.`,
      ));
      return;
    }
    setBusy(true);
    setFormError('');
    const result = await createAccount({
      username: form.username.trim().toLowerCase(),
      password: form.password,
      role: form.role,
      display_name: form.display_name.trim(),
      must_change_password: true,
    });
    setBusy(false);
    if (result.ok && result.data) {
      setForm({ username: '', display_name: '', role: 'viewer', password: '' });
      setReloadKey((key) => key + 1);
      notify('Account created.', 'החשבון נוצר.');
    } else {
      setFormError(describeFailure(result));
    }
  }

  async function submitReset(username) {
    if (busy) return;
    setBusy(true);
    setRowError('');
    const result = await resetAccountPassword(username, resetValue);
    setBusy(false);
    if (result.ok) {
      setResetFor('');
      setResetValue('');
      setReloadKey((key) => key + 1);
      notify('Temporary password set.', 'נקבעה סיסמה זמנית חדשה.');
    } else {
      setRowError(describeFailure(result));
    }
  }

  async function submitDelete(username) {
    if (busy) return;
    if (confirmDelete !== username) {
      setConfirmDelete(username);
      return;
    }
    setBusy(true);
    setRowError('');
    const result = await deleteAccount(username);
    setBusy(false);
    setConfirmDelete('');
    if (result.ok) {
      setReloadKey((key) => key + 1);
      notify('Account deleted.', 'החשבון נמחק.');
    } else {
      setRowError(describeFailure(result));
    }
  }

  return (
    <div className="auth-overlay" dir={locale === 'he' ? 'rtl' : 'ltr'} role="dialog" aria-modal="true">
      <div className="auth-dialog auth-dialog-wide">
        <button type="button" className="auth-close" onClick={onClose} aria-label={t('Close', 'סגירה')}>
          ×
        </button>
        <h2>{t('Manage accounts', 'ניהול חשבונות')}</h2>
        <p className="auth-hint">
          {t(
            'Each teammate signs in with a personal account; the role decides what the account can change.',
            'לכל אחד ואחת בצוות חשבון אישי; התפקיד קובע אילו פעולות פתוחות בחשבון.',
          )}
        </p>
        {loadState === 'loading' && <p className="auth-empty">{t('Loading accounts...', 'רק רגע...')}</p>}
        {loadState === 'error' && (
          <div>
            <p className="auth-error">{t('Could not load the account list.', 'טעינת רשימת החשבונות נכשלה.')}</p>
            <div className="auth-actions">
              <button type="button" className="auth-secondary" onClick={() => setReloadKey((key) => key + 1)}>
                {t('Try again', 'לנסות שוב')}
              </button>
            </div>
          </div>
        )}
        {loadState === 'ready' && (
          <table className="auth-table">
            <thead>
              <tr>
                <th>{t('Name', 'שם')}</th>
                <th>{t('Display name', 'שם תצוגה')}</th>
                <th>{t('Role', 'תפקיד')}</th>
                <th aria-label={t('Actions', 'פעולות')} />
              </tr>
            </thead>
            <tbody>
              {accounts.map((account) => {
                const isSelf = account.username === selfUsername;
                const lastAdmin = account.role === 'admin' && adminTotal <= 1;
                return (
                  <React.Fragment key={account.username}>
                    <tr>
                      <td className="auth-mono">{account.username}</td>
                      <td>{account.display_name}</td>
                      <td>
                        {roleLabel(account.role, locale)}
                        {account.must_change_password && (
                          <span className="auth-flag">{t('Temporary password', 'סיסמה זמנית')}</span>
                        )}
                      </td>
                      <td>
                        <div className="auth-row-actions">
                          <button
                            type="button"
                            className="auth-mini"
                            disabled={busy}
                            onClick={() => {
                              setResetFor(resetFor === account.username ? '' : account.username);
                              setResetValue('');
                              setRowError('');
                            }}
                          >
                            {t('Reset password', 'איפוס סיסמה')}
                          </button>
                          <button
                            type="button"
                            className={`auth-mini auth-danger${confirmDelete === account.username ? ' auth-confirming' : ''}`}
                            disabled={busy || isSelf || lastAdmin}
                            title={
                              isSelf
                                ? t(
                                    'You cannot delete the account you are signed in with.',
                                    'אי אפשר למחוק את החשבון שאיתו נכנסת למערכת.',
                                  )
                                : lastAdmin
                                  ? t('The last admin account cannot be deleted.', 'אי אפשר למחוק את חשבון הניהול האחרון.')
                                  : undefined
                            }
                            onClick={() => submitDelete(account.username)}
                          >
                            {confirmDelete === account.username ? t('Confirm delete', 'לאשר מחיקה') : t('Delete', 'מחיקה')}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {resetFor === account.username && (
                      <tr className="auth-reset-row">
                        <td colSpan={4}>
                          <div className="auth-inline-form">
                            <input
                              type="password"
                              dir="ltr"
                              autoComplete="new-password"
                              placeholder={t('New temporary password', 'סיסמה זמנית חדשה')}
                              value={resetValue}
                              onChange={(event) => setResetValue(event.target.value)}
                            />
                            <button
                              type="button"
                              className="auth-mini"
                              disabled={busy || resetValue.length < MIN_PASSWORD_LENGTH}
                              onClick={() => submitReset(account.username)}
                            >
                              {t('Set password', 'קביעת הסיסמה')}
                            </button>
                            <span className="auth-hint">
                              {t('A change is required at the next sign-in.', 'בכניסה הבאה תידרש החלפת סיסמה.')}
                            </span>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
        {loadState === 'ready' && accounts.length === 0 && (
          <p className="auth-empty">{t('No accounts yet.', 'אין עדיין חשבונות.')}</p>
        )}
        {loadState === 'ready' && rowError && (
          <p className="auth-error" role="alert">
            {rowError}
          </p>
        )}
        {loadState === 'ready' && (
          <form onSubmit={submitCreate}>
            <h3>{t('New account', 'חשבון חדש')}</h3>
            <div className="auth-create-grid">
              <label className="auth-field">
                <span>{t('Username', 'שם משתמש')}</span>
                <input
                  dir="ltr"
                  autoComplete="off"
                  value={form.username}
                  onChange={(event) => setForm({ ...form, username: event.target.value })}
                />
              </label>
              <label className="auth-field">
                <span>{t('Display name', 'שם תצוגה')}</span>
                <input
                  value={form.display_name}
                  onChange={(event) => setForm({ ...form, display_name: event.target.value })}
                />
              </label>
              <label className="auth-field">
                <span>{t('Role', 'תפקיד')}</span>
                <select value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value })}>
                  <option value="viewer">{roleLabel('viewer', locale)}</option>
                  <option value="operator">{roleLabel('operator', locale)}</option>
                  <option value="admin">{roleLabel('admin', locale)}</option>
                </select>
              </label>
              <label className="auth-field">
                <span>{t('Temporary password', 'סיסמה זמנית')}</span>
                <input
                  type="password"
                  dir="ltr"
                  autoComplete="new-password"
                  value={form.password}
                  onChange={(event) => setForm({ ...form, password: event.target.value })}
                />
              </label>
            </div>
            <p className="auth-hint">
              {t(
                'At least 10 characters; a password change is required at the first sign-in. The viewer role reads only, operator edits and runs, admin also manages accounts.',
                'לפחות 10 תווים; בכניסה הראשונה תידרש החלפת סיסמה. תפקיד צפייה מאפשר קריאה בלבד, תפעול מאפשר עריכה והרצה, וניהול מוסיף ניהול חשבונות.',
              )}
            </p>
            {formError && (
              <p className="auth-error" role="alert">
                {formError}
              </p>
            )}
            <div className="auth-actions">
              <button
                type="submit"
                className="auth-primary"
                disabled={busy || form.username.trim() === '' || form.password === ''}
              >
                {t('Create account', 'יצירת חשבון')}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

export default TVBreakDashboard;
