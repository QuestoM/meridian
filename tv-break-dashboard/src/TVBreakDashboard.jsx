import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CacheProvider } from '@emotion/react';
import createCache from '@emotion/cache';
import { prefixer } from 'stylis';
import rtlPlugin from '@mui/stylis-plugin-rtl';
import {
  Button,
  Checkbox,
  CssBaseline,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
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
  Info,
  Languages,
  LayoutGrid,
  ListChecks,
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
import ScheduleEditor, { ConstraintBuilder } from './ScheduleEditor';
import FrontierScopeChart from './FrontierScopeChart';
import YieldView from './YieldView';
import ScenarioCompare from './ScenarioCompare';
import GoldBreakManager from './GoldBreakManager';
import MakeGoodAlerts from './MakeGoodAlerts';

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
      Settings: 'Settings',
    },
    workspace: 'Revenue operations',
    operatorRole: 'Revenue Ops',
    optimizer: 'Optimizer',
    dateRange: 'May 19 - May 25, 2025',
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
    risk: { High: 'High', Medium: 'Medium', Low: 'Low' },
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
    applySimilar: 'Apply Similar',
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
    riskCautionHelp: 'How cautious to be when the retention estimate is uncertain. 0 uses the central estimate; higher values treat an uncertain break at its worst plausible cost, for a more conservative plan.',
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
      Settings: 'הגדרות',
    },
    workspace: 'ניהול הכנסות מפרסום',
    operatorRole: 'Revenue Ops',
    optimizer: 'אופטימייזר',
    dateRange: '19 במאי - 25 במאי 2025',
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
    risk: { High: 'גבוהה', Medium: 'בינונית', Low: 'נמוכה' },
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
    applySimilar: 'החלה דומה',
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
    riskCautionHelp: 'כמה להיזהר כשאומדן השימור אינו ודאי. 0 = שימוש באומדן המרכזי; ככל שעולה, ברייק עם חוסר ודאות מתומחר לפי העלות הגרועה הסבירה שלו, כלומר תוכנית שמרנית יותר.',
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

const fallbackSchedule = {
  rows: [
    {
      channel: 'KAI 1',
      programs: [
        { title: 'The Voice', program_type: 'Reality', day: 'Mon', time: '20:00', duration_minutes: 120, revenue: 382000, retention: 74.1, break_markers: 5, selected: false },
        { title: 'NCIS', program_type: 'Drama', day: 'Tue', time: '20:00', duration_minutes: 60, revenue: 246000, retention: 73.6, break_markers: 4, selected: false },
        { title: "Grey's Anatomy", program_type: 'Drama', day: 'Thu', time: '20:00', duration_minutes: 60, revenue: 456000, retention: 72.3, break_markers: 6, selected: true },
        { title: 'Movie: Top Gun', program_type: 'Sports', day: 'Sat', time: '20:00', duration_minutes: 180, revenue: 512000, retention: 71.6, break_markers: 8, selected: false },
      ],
    },
    {
      channel: 'KAI 2',
      programs: [
        { title: 'The Big Bang Theory', program_type: 'Comedy', day: 'Mon', time: '20:00', duration_minutes: 30, revenue: 186000, retention: 77.2, break_markers: 2, selected: false },
        { title: 'Chicago P.D.', program_type: 'Drama', day: 'Wed', time: '20:00', duration_minutes: 60, revenue: 212000, retention: 75.6, break_markers: 3, selected: false },
        { title: 'NHL Playoffs', program_type: 'Sports', day: 'Fri', time: '20:00', duration_minutes: 150, revenue: 410000, retention: 74.6, break_markers: 7, selected: false },
        { title: 'The Simpsons', program_type: 'Comedy', day: 'Sun', time: '20:00', duration_minutes: 30, revenue: 156000, retention: 76.9, break_markers: 2, selected: false },
      ],
    },
    {
      channel: 'KAI News',
      programs: [
        { title: 'Kai News 8PM', program_type: 'News', day: 'Mon', time: '20:00', duration_minutes: 30, revenue: 98000, retention: 81.3, break_markers: 2, selected: false },
        { title: 'Kai News 8PM', program_type: 'News', day: 'Tue', time: '20:00', duration_minutes: 30, revenue: 98000, retention: 81.6, break_markers: 2, selected: false },
        { title: 'Kai News 8PM', program_type: 'News', day: 'Thu', time: '20:00', duration_minutes: 30, revenue: 98000, retention: 81.0, break_markers: 2, selected: false },
        { title: 'Kai News 8PM', program_type: 'News', day: 'Sun', time: '20:00', duration_minutes: 30, revenue: 98000, retention: 81.2, break_markers: 2, selected: false },
      ],
    },
  ],
  break_operations: {
    programs: [
      { id: 'fallback-1', key: 'KAI 1-fallback-1-2000', lane: 'KAI 1 / Mon', channel: 'KAI 1', title: 'The Voice', program_type: 'Reality', day: 'Mon', date: '2025-05-19', start_time: '20:00', end_time: '22:00', duration_minutes: 120, revenue: 382000, retention: 74.1, break_markers: 5 },
      { id: 'fallback-2', key: 'KAI 2-fallback-2-2000', lane: 'KAI 2 / Mon', channel: 'KAI 2', title: 'The Big Bang Theory', program_type: 'Comedy', day: 'Mon', date: '2025-05-19', start_time: '20:00', end_time: '20:30', duration_minutes: 30, revenue: 186000, retention: 77.2, break_markers: 2 },
      { id: 'fallback-3', key: 'KAI News-fallback-3-2000', lane: 'KAI News / Mon', channel: 'KAI News', title: 'Kai News 8PM', program_type: 'News', day: 'Mon', date: '2025-05-19', start_time: '20:00', end_time: '20:30', duration_minutes: 30, revenue: 98000, retention: 81.3, break_markers: 2 },
    ],
    breaks: [
      { id: 'fallback-1-br-1', program_key: 'KAI 1-fallback-1-2000', program_title: 'The Voice', lane: 'KAI 1 / Mon', channel: 'KAI 1', day: 'Mon', program_type: 'Reality', break_num_in_program: 1, breaks_in_program: 5, start_time: '20:20', end_time: '20:22', duration_sec: 120, sponsorships_count: 0, is_gold: false, source: 'Model', revenue_calculated: 76400, retention: 74.1, status: 'ready' },
      { id: 'fallback-1-br-2', program_key: 'KAI 1-fallback-1-2000', program_title: 'The Voice', lane: 'KAI 1 / Mon', channel: 'KAI 1', day: 'Mon', program_type: 'Reality', break_num_in_program: 2, breaks_in_program: 5, start_time: '20:40', end_time: '20:42', duration_sec: 120, sponsorships_count: 0, is_gold: false, source: 'Model', revenue_calculated: 76400, retention: 74.1, status: 'ready' },
      { id: 'fallback-2-br-1', program_key: 'KAI 2-fallback-2-2000', program_title: 'The Big Bang Theory', lane: 'KAI 2 / Mon', channel: 'KAI 2', day: 'Mon', program_type: 'Comedy', break_num_in_program: 1, breaks_in_program: 2, start_time: '20:10', end_time: '20:12', duration_sec: 120, sponsorships_count: 0, is_gold: false, source: 'Model', revenue_calculated: 93000, retention: 77.2, status: 'ready' },
      { id: 'fallback-3-br-1', program_key: 'KAI News-fallback-3-2000', program_title: 'Kai News 8PM', lane: 'KAI News / Mon', channel: 'KAI News', day: 'Mon', program_type: 'News', break_num_in_program: 1, breaks_in_program: 2, start_time: '20:12', end_time: '20:14', duration_sec: 120, sponsorships_count: 0, is_gold: false, source: 'Model', revenue_calculated: 49000, retention: 81.3, status: 'ready' },
    ],
    summary: { programs: 3, breaks: 4, ad_seconds: 480, revenue: 294800 },
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

function formatCurrency(value, locale = 'en') {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const magnitude = Math.abs(number);
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: magnitude >= 100000 ? 0 : 1,
    notation: magnitude >= 100000 ? 'compact' : 'standard',
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
  return labels[segment] || segment;
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

async function postBreakDecision(payload) {
  try {
    await fetch(`${API_BASE}/api/break-decisions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch {
    // The UI keeps the local decision state when the API is offline.
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
  const [approved, setApproved] = useState(new Set(['rec-1']));
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
  const toastTimer = useRef(null);

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
  const compliance = overview.compliance || fallbackCompliance;
  const theme = useMemo(() => createKairosTheme(isHebrew ? 'rtl' : 'ltr'), [isHebrew]);
  const muiCache = isHebrew ? rtlCache : ltrCache;

  function notify(en, he) {
    setActionMessage(pageText(locale, en, he));
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setActionMessage(''), 2600);
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

  function toggleApproval(id) {
    setApproved((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setRejected((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
    postBreakDecision({
      action: 'approve',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: selectedProgram?.program_type || activeRec?.program_type,
      scenario,
    });
    notify('Decision state updated.', 'סטטוס ההחלטה עודכן.');
  }

  function rejectRecommendation(id) {
    setRejected((current) => new Set(current).add(id));
    setApproved((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
    postBreakDecision({
      action: 'reject',
      recommendation_id: id,
      break_id: selectedProgram?.selected_break?.id,
      program_type: selectedProgram?.program_type || activeRec?.program_type,
      scenario,
    });
    notify('Recommendation rejected for this plan.', 'ההמלצה נדחתה עבור התוכנית הזו.');
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
    notify('Similar recommendations were marked approved.', 'המלצות דומות סומנו כמאושרות.');
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

  function handleNotifications() {
    const checks = overview?.compliance?.checks || [];
    if (checks.length === 0) {
      notify('No compliance checks available to evaluate.', 'אין בדיקות תאימות זמינות להערכה.');
      return;
    }
    const atRisk = checks.filter((check) => check.status === 'at_risk').length;
    if (atRisk === 0) {
      notify('No open operational alerts. All compliance checks pass.', 'אין התראות תפעוליות פתוחות. כל בדיקות התאימות תקינות.');
      return;
    }
    notify(`${atRisk} compliance check(s) need review.`, `${atRisk} בדיקות תאימות דורשות בדיקה.`);
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

  async function handleApplyFrontierWeight(weight) {
    const nextWeight = finiteNumber(weight);
    if (nextWeight === null) return;
    setApplyWeightState('saving');
    try {
      await persistSettings({ ...settings, revenue_weight: Math.round(nextWeight) });
      notify(
        `Saved revenue weight set to ${Math.round(nextWeight)}.`,
        `משקל ההכנסה השמור עודכן ל־${Math.round(nextWeight)}.`,
      );
    } finally {
      setApplyWeightState('idle');
    }
  }

  async function handleRecomputeSchedule() {
    setRecomputeState('running');
    try {
      const response = await fetch(`${API_BASE}/api/recompute-schedule`, { method: 'POST' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const result = await response.json();
      setRecomputeState('done');
      // Refetch so the schedule and overview reflect the freshly computed plan.
      setRefreshKey((k) => k + 1);
      notify(
        `Weekly schedule recomputed: ${formatNumber(result.total_breaks || 0, locale)} breaks, ${formatNumber(Math.round(result.total_revenue || 0), locale)} ILS.`,
        `הלוח השבועי חושב מחדש: ${formatNumber(result.total_breaks || 0, locale)} ברייקים, ${formatNumber(Math.round(result.total_revenue || 0), locale)} ש"ח.`,
      );
      window.setTimeout(() => setRecomputeState('idle'), 2400);
    } catch {
      setRecomputeState('error');
      notify('Recompute failed. The saved schedule is unchanged.', 'החישוב מחדש נכשל. הלוח השמור לא השתנה.');
      window.setTimeout(() => setRecomputeState('idle'), 2400);
    }
  }

  function renderActiveWorkspace() {
    const common = { overview, schedule, copy, locale, compliance, loading, notify };

    if (activeView === 'Overview') {
      return (
        <OverviewPage
          {...common}
          files={files}
          setActiveView={setActiveView}
          operatorChannel={settings.operator_channel || ''}
          savedRevenueWeight={finiteNumber(settings.revenue_weight)}
          onApplyFrontierWeight={handleApplyFrontierWeight}
          applyWeightState={applyWeightState}
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
          onApprove={() => activeRec && toggleApproval(activeRec.id)}
          onReject={() => activeRec && rejectRecommendation(activeRec.id)}
          onApplySimilar={applySimilarRecommendations}
          onExport={(exportScope) => {
            downloadJson('kairos-break-detail.json', { exportScope, selectedProgram, recommendation: activeRec, scenario });
            notify('Break detail exported as JSON.', 'פרטי הברייק יוצאו כ־JSON.');
          }}
        />
      );
    }

    if (activeView === 'Schedule') {
      return <SchedulePage {...common} onRecompute={handleRecomputeSchedule} recomputeState={recomputeState} />;
    }

    if (activeView === 'Inventory') {
      return <InventoryPage inventory={inventory} overview={overview} copy={copy} locale={locale} />;
    }

    if (activeView === 'Break Library') {
      return <BreakLibraryPage breakLibrary={breakLibrary} copy={copy} locale={locale} />;
    }

    if (activeView === 'Campaigns') {
      return <CampaignsPage campaigns={campaigns} copy={copy} locale={locale} />;
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
        />
      );
    }

    if (activeView === 'Advertisers') {
      return <AdvertisersManager copy={copy} locale={locale} notify={notify} />;
    }

    if (activeView === 'Pricing') {
      return <PricingManager copy={copy} locale={locale} notify={notify} />;
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

        <div className="operator-card">
          <span className="operator-avatar">AK</span>
          <div>
            <strong>Alex Kim</strong>
            <small>{copy.operatorRole}</small>
          </div>
          <ChevronDown size={14} />
        </div>
      </aside>

      <main className="workspace">
        <header className="top-bar">
          <div className="title-group">
            <span className="section-title">{copy.nav[activeView] || copy.optimizer}</span>
            <Button
              className="date-control"
              type="button"
              variant="outlined"
              onClick={() => {
                setActiveView('Schedule');
                notify('Opened the schedule for the active planning week.', 'נפתח לוח השידורים לשבוע התכנון הפעיל.');
              }}
            >
              {copy.dateRange}
              <ChevronDown size={14} />
            </Button>
          </div>

          <div className="command-group">
            <FormControl className="scenario-select" size="small">
              <InputLabel id="scenario-label">{copy.scenario}</InputLabel>
              <Select
                labelId="scenario-label"
                value={scenario}
                label={copy.scenario}
                onChange={(event) => {
                  setScenario(event.target.value);
                  notify('Scenario switched. Metrics are shown for the selected planning mode.', 'התרחיש הוחלף. המדדים מוצגים לפי מצב התכנון שנבחר.');
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

          <div className="status-group">
            <span className={online ? 'api-state online' : 'api-state offline'}>
              {online ? copy.liveApi : copy.snapshot}
            </span>
            <span className="freshness" title={locale === 'he' ? 'מועד עדכון הנתונים האחרון מה־API' : 'Time the data was last updated from the API'}>{online && overview.data_freshness ? `${copy.dataUpdated} ${new Date(overview.data_freshness).toLocaleTimeString(locale === 'he' ? 'he-IL' : [], { hour: '2-digit', minute: '2-digit' })}` : `${copy.dataUpdated} —`}</span>
            <IconButton className="icon-button" type="button" aria-label={copy.refresh} size="small" onClick={handleRefresh}>
              <RefreshCcw size={15} />
            </IconButton>
            <IconButton
              className="icon-button"
              type="button"
              aria-label={copy.notifications}
              size="small"
              onClick={handleNotifications}
            >
              <Bell size={15} />
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
            <Button className="run-button" type="button" variant="contained" disabled={optimizationState === 'running'} onClick={handleRunOptimization}>
              <Play size={15} fill="currentColor" />
              {optimizationState === 'running' ? pageText(locale, 'Running', 'מריץ') : copy.runOptimization}
            </Button>
          </div>
        </header>

        {renderActiveWorkspace()}

        {actionMessage && <div className="toast">{actionMessage}</div>}
        {loading && <div className="toast">{copy.loading}</div>}
        {!loading && error && <div className="toast muted">{copy.apiUnavailable}</div>}
      </main>
    </div>
      </ThemeProvider>
    </CacheProvider>
  );
}

function riskLabel(score) {
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
  return (
    <section className="optimizer-run-summary">
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

  const name = impactSegmentLabel(segment.segment ?? segment.name ?? segment.program_type ?? '', locale) || segment.label || '';

  return (
    <div className={isAssumption ? 'retention-cost-row assumption' : 'retention-cost-row'}>
      <div className="retention-cost-row-head">
        <strong>{name}</strong>
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
            onApprove={onApprove}
            onReject={onReject}
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
          <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} loading={loading} operatorChannel={overview.settings?.operator_channel || ''} status={overview.frontier_status || ''} />
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

function OverviewPage({ overview, compliance, files, copy, locale, setActiveView, loading, operatorChannel, savedRevenueWeight, onApplyFrontierWeight, applyWeightState }) {
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
          savedRevenueWeight={savedRevenueWeight}
          onApplyWeight={onApplyFrontierWeight}
          applyState={applyWeightState}
        />
      </div>
      <YieldView locale={locale} />
    </section>
  );
}

function SchedulePage({ schedule, copy, locale, notify, onRecompute, recomputeState }) {
  const rows = normalizeRows(schedule.break_schedule);
  const [scheduleMode, setScheduleMode] = useState('grid');
  const [scheduleAxis, setScheduleAxis] = useState(gridAxisFromLocation);
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
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
      <GoldBreakManager locale={locale} />
    </section>
  );
}

function InventoryPage({ inventory, overview, copy, locale }) {
  const channels = normalizeRows(inventory.by_channel);
  const hours = normalizeRows(inventory.by_hour);
  const maxRevenue = Math.max(...hours.map((row) => Number(row.revenue || 0)), 1);
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
        <Metric label={pageText(locale, 'Inventory spots', 'ספוטים במלאי')} value={formatNumber(inventory.summary?.spots, locale)} delta={pageText(locale, 'source', 'מקור')} icon={TableProperties} positive />
        <Metric label={pageText(locale, 'Booked value', 'ערך מוזמן')} value={formatCurrency(inventory.summary?.revenue, locale)} icon={CircleDollarSign} positive />
        <Metric label={pageText(locale, 'Booked minutes', 'דקות מוזמנות')} value={formatMinutes(inventory.summary?.seconds, locale)} delta={copy.nav.Schedule} icon={Clock3} />
        <Metric label={copy.metrics[3]} value={copy.risk[riskLabel(overview.summary?.risk_score)]} delta={`${overview.summary?.risk_score || 0}/100`} icon={ShieldCheck} tone="risk" />
      </section>
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
            <span>{pageText(locale, 'Booked value', 'ערך מוזמן')}</span>
          </div>
          <div className="bar-list chart-ltr" dir="ltr">
            {hours.slice(0, 24).map((row) => (
              <div className="bar-row" key={row.hour_of_day}>
                <span>{String(row.hour_of_day).padStart(2, '0')}:00</span>
                <i style={{ '--bar': Number(row.revenue || 0) / maxRevenue }} />
                <strong>{formatCurrency(row.revenue, locale)}</strong>
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

function CampaignsPage({ campaigns, copy, locale }) {
  const rows = normalizeRows(campaigns.campaigns);
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Campaign allocation"
        titleHe="הקצאת קמפיינים"
        bodyEn="Track advertiser demand, booked value, channel spread, and the campaigns that constrain optimization."
        bodyHe="מעקב אחר ביקוש מפרסמים, ערך מוזמן, פיזור ערוצים והקמפיינים שמגבילים את האופטימיזציה."
      />
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
      <MakeGoodAlerts locale={locale} />
    </section>
  );
}

function ForecastsPage({ forecasts, overview, copy, locale, loading }) {
  const days = normalizeRows(forecasts.by_day);
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
            {scenarios.map((item) => (
              <div className="scenario-row" key={item.name}>
                <span>{scenarioNameLabel(item.name, locale)}</span>
                <i style={{ '--bar': Number(item.revenue || 0) / maxRevenue }} />
                <strong>{formatCurrency(item.revenue, locale)}</strong>
                <small>{formatPercent(item.retention, locale)}</small>
              </div>
            ))}
          </div>
        </section>
        <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} loading={loading} operatorChannel={overview.settings?.operator_channel || ''} status={overview.frontier_status || ''} />
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
              <strong>{report.title}</strong>
              <span>{report.owner}</span>
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

function DataPage({ files, impact, parameters, overview, copy, locale, notify }) {
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
        <UploadCenter copy={copy} locale={locale} notify={notify} />
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
        <Metric label={pageText(locale, 'Programmes', 'תוכניות')} value={formatNumber(overview.source_counts?.programmes, locale)} delta={copy.nav.Schedule} icon={CalendarDays} positive />
        <Metric label={pageText(locale, 'Spots', 'ספוטים')} value={formatNumber(overview.source_counts?.spots, locale)} delta={copy.nav.Inventory} icon={TableProperties} positive />
        <Metric label={pageText(locale, 'Plan rows', 'שורות תכנון')} value={formatNumber(overview.source_counts?.planned_break_rows, locale)} delta={copy.nav['Break Library']} icon={ClipboardCheck} />
        <Metric label={pageText(locale, 'Sources online', 'מקורות זמינים')} value={`${fileRows.filter((file) => file.exists).length}/${fileRows.length}`} delta={copy.data} icon={Database} positive />
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

function TimelineView({ timeline, rows, locale, selectedProgramKey, onSelectProgram }) {
  const { programs, breaks, summary } = normalizedTimeline(timeline, rows);
  const lanes = Array.from(new Set([...programs.map((item) => item.lane), ...breaks.map((item) => item.lane)].filter(Boolean)));
  const allTimes = [
    ...programs.flatMap((item) => [timeToMinutes(item.start_time), timeToMinutes(item.end_time)]),
    ...breaks.flatMap((item) => [timeToMinutes(item.start_time), timeToMinutes(item.end_time)]),
  ].filter((value) => Number.isFinite(value));
  const startHour = Math.max(0, Math.floor((Math.min(...allTimes, 20 * 60) - 30) / 60));
  const endHour = Math.min(24, Math.max(startHour + 4, Math.ceil((Math.max(...allTimes, 23 * 60) + 30) / 60)));
  const totalMinutes = Math.max(60, (endHour - startHour) * 60);
  const hours = Array.from({ length: endHour - startHour + 1 }, (_, index) => startHour + index);
  const minWidth = 164 + Math.max(680, totalMinutes * 3.8);
  const positionStyle = (startTime, endTime) => {
    const start = timeToMinutes(startTime);
    const end = Math.max(start + 5, timeToMinutes(endTime));
    const left = ((start - startHour * 60) / totalMinutes) * 100;
    const width = ((end - start) / totalMinutes) * 100;
    return {
      left: `${Math.max(0, Math.min(99, left))}%`,
      width: `${Math.max(1.2, Math.min(100 - Math.max(0, left), width))}%`,
    };
  };

  return (
    <div className="timeline-view">
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

      <div className="timeline-scroll chart-ltr" dir="ltr">
        <div className="timeline-ruler" style={{ minWidth }}>
          <span />
          <div className="timeline-hours">
            {hours.map((hour) => (
              <span key={hour} style={{ left: `${((hour - startHour) / Math.max(1, endHour - startHour)) * 100}%` }}>
                {String(hour % 24).padStart(2, '0')}:00
              </span>
            ))}
          </div>
        </div>
        {lanes.map((lane) => {
          const lanePrograms = programs.filter((item) => item.lane === lane);
          const laneBreaks = breaks.filter((item) => item.lane === lane);
          const laneRevenue = laneBreaks.reduce((sum, item) => sum + Number(item.revenue_calculated || 0), 0);
          return (
            <div className="timeline-row" key={lane} style={{ minWidth }}>
              <div className="timeline-lane" dir={locale === 'he' ? 'rtl' : 'ltr'}>
                <strong>{lane}</strong>
                <span>{laneBreaks.length} {pageText(locale, 'breaks', 'ברייקים')} / <Numeric>{formatCurrency(laneRevenue, locale)}</Numeric></span>
              </div>
              <div className="timeline-track">
                {hours.map((hour) => (
                  <i key={`${lane}-${hour}`} style={{ left: `${((hour - startHour) / Math.max(1, endHour - startHour)) * 100}%` }} />
                ))}
                {lanePrograms.map((program) => (
                  <div
                    className="timeline-program-band"
                    key={program.key || `${program.title}-${program.start_time}`}
                    style={positionStyle(program.start_time, program.end_time)}
                    title={`${program.title} / ${program.start_time}-${program.end_time}`}
                  >
                    <span>{program.title}</span>
                  </div>
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
                  return (
                    <Button
                      className={selected ? 'timeline-break selected' : `timeline-break ${breakItem.status === 'at_risk' ? 'risk' : ''}`}
                      key={breakItem.id}
                      type="button"
                      variant="contained"
                      disableRipple
                      style={positionStyle(breakItem.start_time, breakItem.end_time)}
                      title={`${breakItem.program_title} / ${breakItem.start_time}-${breakItem.end_time}`}
                      aria-pressed={selected}
                      onClick={() => onSelectProgram(selectedProgram)}
                    >
                      <span>{breakItem.start_time}</span>
                      <strong>{breakItem.break_num_in_program}/{breakItem.breaks_in_program}</strong>
                      {breakItem.is_gold && <em>Gold</em>}
                    </Button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
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
  const markers = Array.from({ length: Math.max(1, Math.min(10, markerCount || program.break_markers || 1)) });
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

function Inspector({ selectedProgram, recommendation, approved, rejected, onApprove, onReject, onApplySimilar, onExport, onClose, copy, locale }) {
  const approvalLabel = rejected ? pageText(locale, 'Rejected', 'נדחה') : approved ? copy.approved : copy.pending;
  const [exportScope, setExportScope] = useState('Break detail');
  const selectedBreak = selectedProgram?.selected_break;
  const durationSeconds = Number(selectedBreak?.duration_sec || selectedProgram?.duration_minutes * 60 || 120);
  const breakNumber = selectedBreak?.break_num_in_program || 1;
  const breakTotal = selectedBreak?.breaks_in_program || selectedProgram?.break_markers || 1;
  const retentionValue = Number(selectedProgram?.retention ?? recommendation?.retention ?? 0);
  const retentionAtRisk = retentionValue > 0 && retentionValue < 72;
  return (
    <aside className="inspector" aria-label="Selected break inspector">
      <div className="inspector-head">
        <span>{copy.selectedBreak}</span>
        <IconButton className="icon-button small" type="button" aria-label="Close inspector" size="small" onClick={onClose}>
          <X size={14} />
        </IconButton>
      </div>

      <div className="selected-program">
        <span className="channel-badge">{selectedProgram?.channel?.slice(0, 2) || 'K1'}</span>
        <div>
          <strong>{selectedProgram?.title || 'Selected program'}</strong>
          <small>
            {selectedProgram?.channel || 'KAI 1'} / {selectedProgram?.time || '20:00'} /{' '}
            {locale === 'he' ? `ברייק ${breakNumber} מתוך ${breakTotal}` : `break ${breakNumber} of ${breakTotal}`}
          </small>
        </div>
        <span className={rejected ? 'approval rejected' : approved ? 'approval approved' : 'approval'}>{approvalLabel}</span>
      </div>

      <dl className="detail-list">
        <div><dt>{copy.detail[0]}</dt><dd>{formatCurrency(selectedProgram?.revenue, locale)}</dd></div>
        <div><dt>{copy.detail[1]}</dt><dd>{formatPercent(retentionValue || 72.3, locale)}</dd></div>
        <div><dt>{copy.detail[2]}</dt><dd>{formatMinutes(durationSeconds, locale)}</dd></div>
        <div><dt>{copy.detail[3]}</dt><dd>{formatNumber(selectedBreak?.sponsorships_count ?? selectedProgram?.break_markers ?? 0, locale)}</dd></div>
      </dl>

      <div className="guardrail-block">
        <h3>{copy.guardrails}</h3>
        {[
          locale === 'he' ? 'דקות פרסום בשעה' : 'Max ads per hour',
          locale === 'he' ? 'אורך ברייק מינימלי' : 'Minimum break length',
          locale === 'he' ? 'הגנת תוכנית' : 'Program protection',
          locale === 'he' ? 'רף שימור' : 'Retention floor',
        ].map((item, index) => {
          const isAtRisk = index === 3 && retentionAtRisk;
          return (
            <div className="guardrail-row" key={item}>
              <span>{item}</span>
              <strong className={isAtRisk ? 'guardrail-state at-risk' : 'guardrail-state'}>{isAtRisk ? copy.atRisk : copy.compliant}</strong>
              <span className={isAtRisk ? 'guardrail-indicator at-risk' : 'guardrail-indicator'}>
                {isAtRisk ? <Numeric>{`${formatNumber(retentionValue - 72, locale)}pp`}</Numeric> : <Check size={14} />}
              </span>
            </div>
          );
        })}
      </div>

      <div className="recommendation-block">
        <h3>{copy.recommendation}</h3>
        <strong>{recommendationTitle(recommendation, locale)}</strong>
        <p>{recommendationRationale(recommendation, locale)}</p>
        <div className="recommendation-meta">
          <span>{copy.risk[recommendation?.risk || 'Medium'] || recommendation?.risk}</span>
          <span>{formatCurrency(recommendation?.impact || 0, locale)}</span>
        </div>
      </div>

      <div className="inspector-actions">
        <Button className="primary-action" type="button" variant="contained" onClick={onApprove}>
          {approved ? copy.approved : copy.approve}
        </Button>
        <Button className={rejected ? 'secondary-button active' : 'secondary-button'} type="button" variant="outlined" onClick={onReject}>{copy.reject}</Button>
        <Button className="secondary-button" type="button" variant="outlined" onClick={onApplySimilar}>{copy.applySimilar}</Button>
      </div>

      <div className="export-row">
        <FormControl size="small">
          <Select aria-label="Export scope" value={exportScope} onChange={(event) => setExportScope(event.target.value)}>
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

function FrontierPanel({ data, copy, locale, loading = false, operatorChannel = '', status = '' }) {
  const chartFrameRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(760);
  const [activePointIndex, setActivePointIndex] = useState(null);
  const height = 224;
  const padX = 46;
  const padY = 30;
  const ownedChannel = String(operatorChannel || '').trim();
  const points = normalizeRows(data)
    .map((point) => ({
      retention: finiteNumber(point.retention),
      revenue: finiteNumber(point.revenue),
      selected: Boolean(point.selected),
    }))
    .filter((point) => point.retention !== null && point.revenue !== null);
  const selectedPoint = points.find((point) => point.selected) || points[points.length - 1];
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
  const [retentionMin, retentionMax] = paddedDomain(points.map((point) => point.retention), 0.8);
  const [revenueMin, revenueMax] = paddedDomain(points.map((point) => point.revenue), 1);
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
    ? pageText(locale, 'Selected plan', 'תוכנית נבחרת')
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
                  aria-label={`${formatCurrency(point.revenue, locale)}, ${formatPercent(point.retention, locale)}`}
                  onFocus={() => setActivePointIndex(index)}
                  onBlur={() => setActivePointIndex(null)}
                />
              ))}
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
              <text className="axis-label" x={4} y={padY + 4}>{formatCurrency(maxRevenue, locale)}</text>
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
          <div className="frontier-readout">
            <div>
              <span>{safeActiveIndex !== null ? pageText(locale, 'Hovered revenue', 'הכנסה בחלופה') : pageText(locale, 'Selected revenue', 'הכנסה בתוכנית')}</span>
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
    { key: 'conservative', label: he ? 'זהיר באי-ודאות' : 'Conservative', desc: he ? 'מתמחר את התרחיש הגרוע ביותר' : 'Prices the worst-case cost', values: { revenue_weight: 60, risk_lambda: 1, min_retention_floor: 0.74 } },
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
            <TextField
              label={copy.effectiveDate}
              type="date"
              size="small"
              value={draft.effective_date || ''}
              onChange={(event) => updateField('effective_date', event.target.value)}
              InputLabelProps={{ shrink: true }}
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
            />
            <TextField
              label={he ? 'תאריך ייחוס לקצב' : 'Pacing reference date'}
              type="date"
              size="small"
              value={draft.pacing_reference_date ?? ''}
              onChange={(event) => updateField('pacing_reference_date', event.target.value)}
              InputLabelProps={{ shrink: true }}
            />
            <NumberControl
              label={he ? 'עוצמת פיגור בקצב' : 'Behind-pace strength'}
              value={draft.pacing_urgency_k ?? 1.0}
              onChange={(value) => updateNumber('pacing_urgency_k', Math.min(5, Math.max(0, Number(value))))}
            />
            <NumberControl
              label={he ? 'תקרת פיגור בקצב' : 'Behind-pace cap'}
              value={draft.pacing_urgency_max ?? 2.0}
              onChange={(value) => updateNumber('pacing_urgency_max', Math.min(4, Math.max(1, Number(value))))}
            />
            <NumberControl
              label={he ? 'ריסון דילוור-יתר' : 'Over-delivery throttle'}
              value={draft.pacing_ahead_k ?? 1.0}
              onChange={(value) => updateNumber('pacing_ahead_k', Math.min(5, Math.max(0, Number(value))))}
            />
            <NumberControl
              label={he ? 'רצפת דילוור-יתר' : 'Over-delivery floor'}
              value={draft.pacing_weight_floor ?? 0.5}
              onChange={(value) => updateNumber('pacing_weight_floor', Math.min(1.0, Math.max(0.25, Number(value))))}
            />
            <NumberControl
              label={he ? 'רצפת מכנה הקצב' : 'Pace denominator floor'}
              value={draft.pacing_epsilon ?? 0.05}
              onChange={(value) => updateNumber('pacing_epsilon', Math.min(0.5, Math.max(0.01, Number(value))))}
            />
          </div>
          <div className="settings-pacing-help">
            <p>{he ? 'מטה את השיבוץ לעבר קמפיינים שמפגרים בקצב הדילוור והרחק מקמפיינים שדילברו יותר מדי. שיבוץ בלבד; לעולם לא משנה את תחזית ההכנסה.' : 'Steer placement toward campaigns behind delivery pace and away from over-delivered ones. Placement only; never changes the revenue projection.'}</p>
            <p>{he ? 'התאריך שנחשב כהיום בעת מדידת קצב הקמפיין. ריק משתמש בתאריך התוקף של הלוח.' : 'The date treated as today when measuring campaign pace. Empty uses the schedule effective date.'}</p>
            <p>{he ? 'כמה חזק קמפיין בתת-דילוור מושך פרסומות למלאי שלו.' : 'How hard an under-delivered campaign pulls breaks toward its inventory.'}</p>
            <p>{he ? 'הגברת השיבוץ המרבית לקמפיין המפגר ביותר.' : 'Maximum placement boost for the most behind campaign.'}</p>
            <p>{he ? 'כמה חזק קמפיין בדילוור-יתר מקבל עדיפות נמוכה בשיבוץ. אפס מבטל את קנס דילוור-היתר.' : 'How hard an over-delivered campaign is de-prioritized in placement. Zero disables the over-delivery penalty.'}</p>
            <p>{he ? 'המשקל הנמוך ביותר בשיבוץ שקמפיין בדילוור-יתר יכול לקבל. לעולם לא אפס, כך שפרסומת לעולם אינה נחסמת.' : 'The lowest placement weight an over-delivered campaign can receive. Never zero, so a slot is never forbidden.'}</p>
            <p>{he ? 'רצפה נומרית כדי שהדחיפות תישאר סופית ביום הראשון והאחרון של הקמפיין.' : 'Numerical floor so urgency stays finite on the first and last flight day.'}</p>
          </div>
        </section>

        <ConstraintBuilder
          locale={locale}
          notify={notify || (() => {})}
          onRecompute={onRecompute}
          recomputeState={recomputeState}
        />
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

function NumberControl({ label, value, onChange, suffix }) {
  return (
    <div className="number-control">
      <div>
        <TextField
          label={label}
          type="number"
          size="small"
          value={value ?? 0}
          onChange={(event) => onChange(event.target.value)}
        />
        <small>{suffix}</small>
      </div>
    </div>
  );
}

function ToggleControl({ label, checked, onChange }) {
  return (
    <div className="toggle-control">
      <span>{label}</span>
      <Switch size="small" checked={Boolean(checked)} onChange={(event) => onChange(event.target.checked)} />
    </div>
  );
}

export default TVBreakDashboard;
