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
  Switch,
  TextField,
  ThemeProvider,
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
  Database,
  Download,
  FileBarChart,
  Gauge,
  GitCompare,
  Languages,
  LayoutGrid,
  ListChecks,
  Save,
  Play,
  RefreshCcw,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  Tv,
  Users,
  X,
} from 'lucide-react';

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
  status: 'compliant',
  disclaimer: fallbackSettings.notes,
  checks: [
    { id: 'hourly_ad_load', label_en: 'Ad minutes per broadcast hour', label_he: 'דקות פרסום לשעת שידור', status: 'compliant', observed: 7.2, limit: 12, unit: 'minutes/hour' },
    { id: 'break_density', label_en: 'Breaks per hour', label_he: 'מספר ברייקים בשעה', status: 'compliant', observed: 3, limit: 4, unit: 'breaks/hour' },
    { id: 'retention_floor', label_en: 'Viewer retention floor', label_he: 'רף שימור צפייה', status: 'compliant', observed: 74.2, limit: 72, unit: '%' },
    { id: 'protected_programs', label_en: 'Protected programme ad load', label_he: 'עומס פרסום בתוכן מוגן', status: 'compliant', observed: 5.1, limit: 8, unit: 'minutes/hour' },
  ],
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
      'Data Hub': 'Data Hub',
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
    snapshot: 'Snapshot',
    data: 'Data',
    refresh: 'Refresh',
    notifications: 'Notifications',
    runOptimization: 'Run Optimization',
    loading: 'Loading Kairos workspace',
    apiUnavailable: 'API unavailable. Showing local snapshot.',
    metrics: ['Projected revenue', 'Viewer retention D7', 'Total ad minutes', 'Risk score'],
    risk: { High: 'High', Medium: 'Medium', Low: 'Low' },
    toolbar: ['Grid View', 'Daypart', 'Inventory', 'Programs', 'Breaks', 'Metrics'],
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
    frontier: 'Revenue vs retention frontier',
    frontierMode: 'D7 model',
    heatmap: 'Daypart inventory heatmap',
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
      'Data Hub': 'מרכז נתונים',
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
    snapshot: 'Snapshot',
    data: 'נתונים',
    refresh: 'רענון',
    notifications: 'התראות',
    runOptimization: 'הרצת אופטימיזציה',
    loading: 'טוען סביבת Kairos',
    apiUnavailable: 'ה־API לא זמין. מוצגת תמונת מצב מקומית.',
    metrics: ['הכנסה צפויה', 'שימור צפייה D7', 'דקות פרסום', 'רמת סיכון'],
    risk: { High: 'גבוהה', Medium: 'בינונית', Low: 'נמוכה' },
    toolbar: ['תצוגת גריד', 'רצועות שידור', 'מלאי', 'תוכניות', 'ברייקים', 'מדדים'],
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
    frontier: 'חזית הכנסה מול שימור',
    frontierMode: 'מודל D7',
    heatmap: 'מפת חום לפי רצועת שידור',
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
  },
};

const fallbackOverview = {
  brand: 'Kairos',
  workspace: 'KAI Network',
  data_freshness: new Date().toISOString(),
  summary: {
    total_breaks: 89,
    total_ad_seconds: 8010,
    projected_revenue: 429100,
    average_retention: 74.2,
    risk_score: 52,
  },
  source_counts: {
    programmes: 1200,
    spots: 5400,
    planned_break_rows: 18,
  },
  recommendations: [
    {
      id: 'rec-1',
      title: 'Increase selected primetime break by 1 spot',
      program_type: 'Reality',
      impact: 18000,
      retention: 72.3,
      risk: 'Medium',
      rationale: 'Demand is concentrated in the selected slot while retention guardrail remains compliant.',
    },
    {
      id: 'rec-2',
      title: 'Shift a late break earlier in the hour',
      program_type: 'Drama',
      impact: 9200,
      retention: 73.1,
      risk: 'Low',
      rationale: 'Earlier placement improves sell-through with limited churn exposure.',
    },
    {
      id: 'rec-3',
      title: 'Hold break length in news block',
      program_type: 'News',
      impact: 0,
      retention: 80.8,
      risk: 'Low',
      rationale: 'News retention is strong, but incremental minutes are below target yield.',
    },
  ],
  frontier: [
    { retention: 67.4, revenue: 488000, selected: false },
    { retention: 69.0, revenue: 470000, selected: false },
    { retention: 71.1, revenue: 451000, selected: false },
    { retention: 74.2, revenue: 429100, selected: true },
    { retention: 75.6, revenue: 401000, selected: false },
    { retention: 77.2, revenue: 372000, selected: false },
  ],
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
  ['Data Hub', Database],
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
  return ['day', 'daypart', 'hour'].includes(axis) ? axis : 'day';
}

function formatCurrency(value, locale = 'en') {
  const number = Number(value || 0);
  const formatter = new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: 'ILS',
    maximumFractionDigits: number >= 100000 ? 0 : 1,
    notation: number >= 100000 ? 'compact' : 'standard',
  });
  return formatter.format(number);
}

function formatMinutes(seconds, locale = 'en') {
  const minutes = Math.round(Number(seconds || 0) / 60);
  return locale === 'he' ? `${minutes.toLocaleString('he-IL')} דק׳` : `${minutes.toLocaleString()} min`;
}

function formatNumber(value, locale = 'en') {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: 1,
  });
}

function formatPercent(value, locale = 'en') {
  return `${formatNumber(value, locale)}%`;
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

function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
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
  return dayKeys.map((day) => ({ key: day, label: dayLabel(day, locale) }));
}

function programsForPlannerColumn(programs, column, axis) {
  if (axis === 'daypart') {
    return programs.filter((program) => daypartForTime(program.time) === column.key);
  }
  if (axis === 'hour') {
    return programs.filter((program) => hourFromTime(program.time) === column.hour);
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
      ];
      setState({
        overview: overviewResult.data,
        schedule: scheduleResult.data,
        inventory: inventoryResult.data,
        breakLibrary: breakLibraryResult.data,
        campaigns: campaignsResult.data,
        forecasts: forecastsResult.data,
        reports: reportsResult.data,
        files: filesResult.data,
        impact: impactResult.data,
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
  const { overview, schedule, inventory, breakLibrary, campaigns, forecasts, reports, files, impact, online, loading, error } =
    useKairosData(refreshKey);
  const [activeRecommendation, setActiveRecommendation] = useState('rec-1');
  const [approved, setApproved] = useState(new Set(['rec-1']));
  const [rejected, setRejected] = useState(new Set());
  const [scenario, setScenario] = useState('Balanced');
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
    notify('Decision state updated.', 'סטטוס ההחלטה עודכן.');
  }

  function rejectRecommendation(id) {
    setRejected((current) => new Set(current).add(id));
    setApproved((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
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

  function handleRunOptimization() {
    setActiveView('Optimizer');
    setOptimizerView('grid');
    setInspectorOpen(true);
    notify('Optimization run completed for the active scenario.', 'הרצת האופטימיזציה הושלמה לתרחיש הפעיל.');
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
      window.setTimeout(() => setSaveState('idle'), 1800);
    } catch {
      setSaveState('error');
    }
  }

  function renderActiveWorkspace() {
    const common = { overview, schedule, copy, locale, compliance };

    if (activeView === 'Overview') {
      return <OverviewPage {...common} files={files} setActiveView={setActiveView} />;
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
      return <SchedulePage {...common} />;
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
      return <ForecastsPage forecasts={forecasts} overview={overview} copy={copy} locale={locale} />;
    }

    if (activeView === 'Reports') {
      return <ReportsPage reports={reports} files={files} copy={copy} locale={locale} />;
    }

    if (activeView === 'Data Hub') {
      return <DataHubPage files={files} impact={impact} overview={overview} copy={copy} locale={locale} />;
    }

    return (
      <SettingsPanel
        settings={settings}
        copy={copy}
        locale={locale}
        saveState={saveState}
        onSave={persistSettings}
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
            <span className="freshness">{copy.data} {new Date(overview.data_freshness).toLocaleTimeString(locale === 'he' ? 'he-IL' : [], { hour: '2-digit', minute: '2-digit' })}</span>
            <IconButton className="icon-button" type="button" aria-label={copy.refresh} size="small" onClick={handleRefresh}>
              <RefreshCcw size={15} />
            </IconButton>
            <IconButton
              className="icon-button"
              type="button"
              aria-label={copy.notifications}
              size="small"
              onClick={() => notify('No open operational alerts.', 'אין התראות תפעוליות פתוחות.')}
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
            <Button className="run-button" type="button" variant="contained" onClick={handleRunOptimization}>
              <Play size={15} fill="currentColor" />
              {copy.runOptimization}
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
  return recommendation?.title_he || 'בדיקת מיקום ברייק';
}

function recommendationRationale(recommendation, locale) {
  if (locale !== 'he') {
    return recommendation?.rationale || 'Recommendation rationale unavailable.';
  }
  return (
    recommendation?.rationale_he ||
    'המערכת מזהה הזדמנות הכנסה, אך ההחלטה נשמרת לבקרה אנושית מול מגבלות שימור ותאימות.'
  );
}

function Metric({ label, value, delta, icon: Icon, positive = false, tone }) {
  return (
    <div className="metric">
      <span className={`metric-icon ${tone || ''}`}>
        <Icon size={17} strokeWidth={1.8} />
      </span>
      <span className="metric-copy">
        <span>{label}</span>
        <strong><Numeric>{value}</Numeric></strong>
      </span>
      <span className={positive ? 'delta positive' : tone === 'risk' ? 'delta risk' : 'delta negative'}>
        {positive ? <ArrowUp size={12} /> : tone === 'risk' ? null : <ArrowDown size={12} />}
        <Numeric>{delta}</Numeric>
      </span>
    </div>
  );
}

function SummaryMetrics({ overview, copy, locale }) {
  const summary = overview.summary || fallbackOverview.summary;
  return (
    <section className="metric-strip" aria-label="Optimization summary">
      <Metric label={copy.metrics[0]} value={formatCurrency(summary.projected_revenue, locale)} delta="+7.6%" icon={CircleDollarSign} positive />
      <Metric label={copy.metrics[1]} value={formatPercent(summary.average_retention, locale)} delta="-1.8pp" icon={Users} />
      <Metric label={copy.metrics[2]} value={formatMinutes(summary.total_ad_seconds, locale)} delta="+120" icon={Clock3} positive />
      <Metric label={copy.metrics[3]} value={copy.risk[riskLabel(summary.risk_score)]} delta={`${summary.risk_score}/100`} icon={ShieldCheck} tone="risk" />
    </section>
  );
}

function OptimizerWorkspace({
  overview,
  schedule,
  compliance,
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
    ['daypart', copy.toolbar[1]],
    ['inventory', copy.toolbar[2]],
  ];

  return (
    <>
      <SummaryMetrics overview={overview} copy={copy} locale={locale} />

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
                label={copy.toolbar[3]}
              />
              <FormControlLabel
                className="check-control"
                control={<Checkbox checked={showBreaks} onChange={(event) => onToggleBreaks(event.target.checked)} size="small" />}
                label={copy.toolbar[4]}
              />
              <Button
                className={showMetrics ? 'secondary-button compact active' : 'secondary-button compact'}
                type="button"
                variant="outlined"
                aria-pressed={showMetrics}
                onClick={onToggleMetrics}
              >
                <SlidersHorizontal size={14} />
                {copy.toolbar[5]}
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
          <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} />
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

function StatusBadge({ status, locale }) {
  const normalized = String(status || 'ready').toLowerCase();
  const labelMap = {
    ready: pageText(locale, 'Ready', 'מוכן'),
    compliant: pageText(locale, 'Compliant', 'תקין'),
    at_risk: pageText(locale, 'Needs review', 'דורש בדיקה'),
    error: pageText(locale, 'Error', 'שגיאה'),
  };
  return <span className={`status-badge ${normalized}`}>{labelMap[normalized] || status}</span>;
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
    renderCell: (params) => {
      const isNumeric = column.numeric || numericKeys.has(column.key);
      const value = column.render
        ? column.render(params.row, params.api.getRowIndexRelativeToVisibleRows?.(params.id) || 0)
        : params.value ?? '';
      return <span className={isNumeric ? 'grid-cell-content numeric-cell' : 'grid-cell-content'}>{value}</span>;
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

function OverviewPage({ overview, compliance, files, copy, locale, setActiveView }) {
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
                  <span>{item.program_type || pageText(locale, 'Mixed', 'מעורב')}</span>
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
            <div><span>{pageText(locale, 'Available source files', 'קבצי מקור זמינים')}</span><strong>{existingFiles} / {fileRows.length || 8}</strong></div>
          </div>
        </section>
      </div>
      <div className="page-grid even">
        <ComplianceLedger compliance={compliance} copy={copy} locale={locale} />
        <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} />
      </div>
    </section>
  );
}

function SchedulePage({ schedule, copy, locale }) {
  const rows = normalizeRows(schedule.break_schedule);
  const [scheduleMode, setScheduleMode] = useState('grid');
  const [scheduleAxis, setScheduleAxis] = useState(gridAxisFromLocation);
  const [selectedProgramKey, setSelectedProgramKey] = useState(null);
  function handleSelectProgram(program) {
    setSelectedProgramKey(program.key);
  }
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Schedule control"
        titleHe="בקרת לוח שידורים"
        bodyEn="Review the weekly break plan by programme type, day, length, expected revenue, and retention guardrail."
        bodyHe="בדיקת תוכנית הברייקים השבועית לפי סוג תוכנית, יום, אורך, הכנסה צפויה ושמירת צפייה."
      />
      <section className="planner-surface compact-surface">
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
              {copy.toolbar[1]}
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
        ) : (
          <DaypartView
            rows={schedule.rows || []}
            locale={locale}
            selectedProgramKey={selectedProgramKey}
            onSelectProgram={handleSelectProgram}
          />
        )}
      </section>
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Break plan rows', 'שורות תוכנית ברייקים')}</h2>
          <span>{rows.length} {pageText(locale, 'rows', 'שורות')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No scheduled breaks were found.', 'לא נמצאו ברייקים מתוכננים.')}
          rows={rows}
          columns={[
            { key: 'day', label: pageText(locale, 'Day', 'יום') },
            { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית') },
            { key: 'position', label: pageText(locale, 'Position', 'מיקום') },
            { key: 'break_type', label: pageText(locale, 'Break type', 'סוג ברייק') },
            { key: 'num_breaks', label: pageText(locale, 'Breaks', 'ברייקים'), render: (row) => formatNumber(row.num_breaks, locale) },
            { key: 'total_break_time', label: pageText(locale, 'Ad minutes', 'דקות פרסום'), render: (row) => formatMinutes(row.total_break_time, locale) },
            { key: 'predicted_revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.predicted_revenue, locale) },
            { key: 'predicted_retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.predicted_retention || 0) * 100, locale) },
          ]}
        />
      </section>
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
        <Metric label={pageText(locale, 'Booked value', 'ערך מוזמן')} value={formatCurrency(inventory.summary?.revenue, locale)} delta="+4.1%" icon={CircleDollarSign} positive />
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
            { key: 'status', label: pageText(locale, 'Status', 'סטטוס'), render: (row) => <StatusBadge status={row.status} locale={locale} /> },
            { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית') },
            { key: 'position', label: pageText(locale, 'Position', 'מיקום') },
            { key: 'break_type', label: pageText(locale, 'Type', 'סוג') },
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
    </section>
  );
}

function ForecastsPage({ forecasts, overview, copy, locale }) {
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
                <span>{item.name}</span>
                <i style={{ '--bar': Number(item.revenue || 0) / maxRevenue }} />
                <strong>{formatCurrency(item.revenue, locale)}</strong>
                <small>{formatPercent(item.retention, locale)}</small>
              </div>
            ))}
          </div>
        </section>
        <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} />
      </div>
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
            { key: 'day', label: pageText(locale, 'Day', 'יום') },
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
            { key: 'exists', label: pageText(locale, 'State', 'מצב'), render: (row) => <StatusBadge status={row.exists ? 'ready' : 'error'} locale={locale} /> },
            { key: 'size', label: pageText(locale, 'Size', 'גודל'), render: (row) => `${formatNumber(Number(row.size || 0) / 1024, locale)} KB` },
            { key: 'modified', label: pageText(locale, 'Modified', 'עודכן'), render: (row) => (row.modified ? new Date(row.modified).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US') : '-') },
          ]}
        />
      </section>
    </section>
  );
}

function DataHubPage({ files, impact, overview, copy, locale }) {
  const fileRows = normalizeRows(files.files);
  const programImpacts = normalizeRows(impact.program_type_impacts);
  const positionImpacts = normalizeRows(impact.position_impacts);
  const lengthImpacts = normalizeRows(impact.length_impacts);
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
        <Metric label={pageText(locale, 'Sources online', 'מקורות זמינים')} value={`${fileRows.filter((file) => file.exists).length}/${fileRows.length || 8}`} delta={copy.data} icon={Database} positive />
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
              { key: 'exists', label: pageText(locale, 'State', 'מצב'), render: (row) => <StatusBadge status={row.exists ? 'ready' : 'error'} locale={locale} /> },
              { key: 'size', label: pageText(locale, 'Size', 'גודל'), render: (row) => `${formatNumber(Number(row.size || 0) / 1024, locale)} KB` },
            ]}
          />
        </section>
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Model explainability', 'הסבריות מודל')}</h2>
            <span>{pageText(locale, 'Impact extracts', 'תוצרי השפעה')}</span>
          </div>
          <div className="impact-stack">
            <ImpactPreview title={pageText(locale, 'Programme type impact', 'השפעת סוג תוכנית')} rows={programImpacts} locale={locale} />
            <ImpactPreview title={pageText(locale, 'Position impact', 'השפעת מיקום')} rows={positionImpacts} locale={locale} />
            <ImpactPreview title={pageText(locale, 'Length impact', 'השפעת אורך')} rows={lengthImpacts} locale={locale} />
          </div>
        </section>
      </div>
    </section>
  );
}

function ImpactPreview({ title, rows, locale }) {
  const first = normalizeRows(rows).slice(0, 3);
  return (
    <div className="impact-preview">
      <strong>{title}</strong>
      {first.length === 0 ? (
        <span>{pageText(locale, 'No extract', 'אין קובץ')}</span>
      ) : (
        first.map((row, index) => {
          const entries = Object.entries(row).slice(0, 3);
          return (
            <span key={`${title}-${index}`}>
              {entries.map(([key, value]) => `${key}: ${value}`).join(' / ')}
            </span>
          );
        })
      )}
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
  const options = ['day', 'daypart', 'hour'];
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
            <small>{programs[0]?.program_type || 'Mixed'}</small>
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
    : `${program.time} / ${program.duration_minutes}m`;
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
        <span className="program-title muted-title">{program.program_type || pageText(locale, 'Program hidden', 'תוכנית מוסתרת')}</span>
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
            {locale === 'he' ? 'ברייק 2 מתוך 4' : 'break 2 of 4'}
          </small>
        </div>
        <span className={rejected ? 'approval rejected' : approved ? 'approval approved' : 'approval'}>{approvalLabel}</span>
      </div>

      <dl className="detail-list">
        <div><dt>{copy.detail[0]}</dt><dd>{formatCurrency(selectedProgram?.revenue, locale)}</dd></div>
        <div><dt>{copy.detail[1]}</dt><dd>{selectedProgram?.retention || 72.3}%</dd></div>
        <div><dt>{copy.detail[2]}</dt><dd>2:00</dd></div>
        <div><dt>{copy.detail[3]}</dt><dd>4</dd></div>
      </dl>

      <div className="guardrail-block">
        <h3>{copy.guardrails}</h3>
        {[
          locale === 'he' ? 'דקות פרסום בשעה' : 'Max ads per hour',
          locale === 'he' ? 'אורך ברייק מינימלי' : 'Minimum break length',
          locale === 'he' ? 'הגנת תוכנית' : 'Program protection',
          locale === 'he' ? 'רף שימור' : 'Retention floor',
        ].map((item, index) => (
          <div className="guardrail-row" key={item}>
            <span>{item}</span>
            <strong>{index === 3 ? copy.atRisk : copy.compliant}</strong>
            {index === 3 ? <span className="guardrail-warning">-0.4pp</span> : <Check size={14} />}
          </div>
        ))}
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

function FrontierPanel({ data, copy, locale }) {
  const width = 420;
  const height = 190;
  const pad = 28;
  const retentions = data.map((point) => point.retention);
  const revenues = data.map((point) => point.revenue);
  const minRetention = Math.min(...retentions, 65);
  const maxRetention = Math.max(...retentions, 80);
  const minRevenue = Math.min(...revenues, 0);
  const maxRevenue = Math.max(...revenues, 1);
  const xFor = (retention) =>
    pad + ((retention - minRetention) / Math.max(maxRetention - minRetention, 1)) * (width - pad * 2);
  const yFor = (revenue) =>
    height - pad - ((revenue - minRevenue) / Math.max(maxRevenue - minRevenue, 1)) * (height - pad * 2);
  const path = data
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(point.retention).toFixed(1)} ${yFor(point.revenue).toFixed(1)}`)
    .join(' ');

  return (
    <div className="analytics-panel frontier-panel chart-ltr" dir="ltr">
      <div className="panel-head">
        <h2>{copy.frontier}</h2>
        <span>{copy.frontierMode}</span>
      </div>
      <svg className="frontier-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Revenue retention frontier">
        {[0, 1, 2, 3].map((line) => {
          const y = pad + line * ((height - pad * 2) / 3);
          return <line key={`h-${line}`} x1={pad} x2={width - pad} y1={y} y2={y} />;
        })}
        {[0, 1, 2, 3, 4].map((line) => {
          const x = pad + line * ((width - pad * 2) / 4);
          return <line key={`v-${line}`} x1={x} x2={x} y1={pad} y2={height - pad} />;
        })}
        <path d={path} />
        {data.map((point) => (
          <g key={`${point.retention}-${point.revenue}`}>
            <circle
              className={point.selected ? 'selected-point' : ''}
              cx={xFor(point.retention)}
              cy={yFor(point.revenue)}
              r={point.selected ? 6 : 4}
            />
            {point.selected && (
              <text x={xFor(point.retention) + 10} y={yFor(point.revenue) - 8}>
                {formatCurrency(point.revenue, locale)} / {point.retention}%
              </text>
            )}
          </g>
        ))}
        <text className="axis-label" x={pad} y={height - 5}>{Math.round(minRetention)}%</text>
        <text className="axis-label" x={width - pad - 24} y={height - 5}>{Math.round(maxRetention)}%</text>
        <text className="axis-label" x={4} y={pad + 4}>{formatCurrency(maxRevenue, locale)}</text>
      </svg>
    </div>
  );
}

function InventoryHeatmap({ copy, locale }) {
  const rows = [
    ['Early fringe', 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.5],
    ['Prime access', 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 0.9],
    ['Primetime', 6.0, 6.2, 6.4, 6.3, 6.1, 6.8, 6.0],
    ['Late', 0.7, 0.7, 0.8, 0.8, 0.7, 0.9, 0.8],
    ['Overnight', 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
  ];
  return (
    <div className="analytics-panel heatmap-panel chart-ltr" dir="ltr">
      <div className="panel-head">
        <h2>{copy.heatmap}</h2>
        <span>{copy.opportunity}</span>
      </div>
      <div className="heatmap">
        <span />
        {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day) => <b key={day}>{day}</b>)}
        {rows.map(([label, ...values]) => (
          <React.Fragment key={label}>
            <strong>{label}</strong>
            {values.map((value, index) => (
              <span key={`${label}-${index}`} style={{ '--heat': value / 7 }}>
                {formatCurrency(value * 1000000, locale)}
              </span>
            ))}
          </React.Fragment>
        ))}
      </div>
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
        {checks.map((check) => (
          <div className="ledger-row" key={check.id}>
            <span>{locale === 'he' ? check.label_he : check.label_en}</span>
            <strong className={check.status === 'at_risk' ? 'at-risk' : ''}>
              {check.status === 'at_risk' ? copy.atRisk : copy.compliant}
            </strong>
            <small>{Number(check.observed).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US')} / {Number(check.limit).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US')}</small>
          </div>
        ))}
        <p className="ledger-note">{compliance?.disclaimer}</p>
      </div>
    </div>
  );
}

function SettingsPanel({ settings, copy, locale, saveState, onSave }) {
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

  const protectedTypes = (draft.protected_program_types || []).join(', ');
  const statusText =
    saveState === 'saved'
      ? copy.saved
      : saveState === 'saving'
        ? copy.saving
        : saveState === 'error'
          ? copy.saveFailed
          : copy.saveSettings;

  return (
    <section className="settings-workspace">
      <div className="settings-hero">
        <div>
          <span className="settings-kicker">{copy.nav.Settings}</span>
          <h1>{copy.settingsTitle}</h1>
          <p>{copy.settingsIntro}</p>
        </div>
        <Button className="run-button" type="button" variant="contained" onClick={() => onSave(draft)}>
          <Save size={15} />
          {statusText}
        </Button>
      </div>

      <div className="settings-grid">
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
              <p>{locale === 'he' ? 'חסויות, ברייקי זהב ואישור אנושי' : 'Sponsorships, gold breaks, and approval flow'}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="settings-toggle-grid">
            <ToggleControl label={copy.sponsorships} checked={draft.sponsorships_enabled} onChange={(value) => updateField('sponsorships_enabled', value)} />
            <ToggleControl label={copy.gold} checked={draft.gold_breaks_enabled} onChange={(value) => updateField('gold_breaks_enabled', value)} />
            <ToggleControl label={copy.approval} checked={draft.require_manual_approval} onChange={(value) => updateField('require_manual_approval', value)} />
            <NumberControl label={locale === 'he' ? 'מקסימום ברייקי זהב ביום' : 'Max gold breaks per day'} value={draft.gold_breaks_max_per_day} onChange={(value) => updateNumber('gold_breaks_max_per_day', value)} suffix="/day" />
            <NumberControl label={copy.dailyCap} value={draft.max_daily_ad_minutes} onChange={(value) => updateNumber('max_daily_ad_minutes', value)} suffix="min" />
          </div>
        </section>
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
