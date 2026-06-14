import React, { useEffect, useMemo, useState } from 'react';
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
    saveSettings: 'Save settings',
    saved: 'Saved',
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
    saveSettings: 'שמירת הגדרות',
    saved: 'נשמר',
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

function useKairosData() {
  const [state, setState] = useState({
    overview: fallbackOverview,
    schedule: fallbackSchedule,
    online: false,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let active = true;
    async function load() {
      const [overviewResult, scheduleResult] = await Promise.all([
        fetchJson('/api/overview', fallbackOverview),
        fetchJson('/api/schedule', fallbackSchedule),
      ]);
      if (!active) return;
      setState({
        overview: overviewResult.data,
        schedule: scheduleResult.data,
        online: overviewResult.online && scheduleResult.online,
        loading: false,
        error: overviewResult.error || scheduleResult.error,
      });
    }
    load();
    return () => {
      active = false;
    };
  }, []);

  return state;
}

function TVBreakDashboard() {
  const { overview, schedule, online, loading, error } = useKairosData();
  const [activeRecommendation, setActiveRecommendation] = useState('rec-1');
  const [approved, setApproved] = useState(new Set(['rec-1']));
  const [scenario, setScenario] = useState('Balanced');
  const [activeView, setActiveView] = useState('Optimizer');
  const [settings, setSettings] = useState(overview.settings || fallbackSettings);
  const [saveState, setSaveState] = useState('idle');

  useEffect(() => {
    const nextSettings = overview.settings || fallbackSettings;
    setSettings((current) => ({ ...current, ...nextSettings }));
  }, [overview.settings]);

  const locale = settings.locale === 'en' ? 'en' : 'he';
  const isHebrew = locale === 'he';
  const copy = copyByLocale[locale];
  const compliance = overview.compliance || fallbackCompliance;

  const selectedProgram = useMemo(() => {
    for (const row of schedule.rows || []) {
      const selected = row.programs?.find((program) => program.selected);
      if (selected) return { channel: row.channel, ...selected };
    }
    const firstRow = schedule.rows?.[0];
    return firstRow?.programs?.[0] ? { channel: firstRow.channel, ...firstRow.programs[0] } : null;
  }, [schedule]);

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

  return (
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

        <nav className="primary-nav">
          {navItems.map(([label, Icon]) => (
            <button
              key={label}
              className={label === activeView ? 'nav-item active' : 'nav-item'}
              type="button"
              onClick={() => setActiveView(label)}
            >
              <Icon size={16} strokeWidth={1.8} />
              <span>{copy.nav[label]}</span>
            </button>
          ))}
        </nav>

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
            <button className="date-control" type="button">
              {copy.dateRange}
              <ChevronDown size={14} />
            </button>
          </div>

          <div className="command-group">
            <label className="scenario-select">
              <span>{copy.scenario}</span>
              <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
                <option value="Balanced">{copy.scenarios[0]}</option>
                <option value="Revenue priority">{copy.scenarios[1]}</option>
                <option value="Retention guardrail">{copy.scenarios[2]}</option>
              </select>
            </label>
            <button className="secondary-button" type="button">
              <GitCompare size={15} />
              {copy.compare}
            </button>
          </div>

          <div className="status-group">
            <span className={online ? 'api-state online' : 'api-state offline'}>
              {online ? copy.liveApi : copy.snapshot}
            </span>
            <span className="freshness">{copy.data} {new Date(overview.data_freshness).toLocaleTimeString(locale === 'he' ? 'he-IL' : [], { hour: '2-digit', minute: '2-digit' })}</span>
            <button className="icon-button" type="button" aria-label={copy.refresh}>
              <RefreshCcw size={15} />
            </button>
            <button className="icon-button" type="button" aria-label={copy.notifications}>
              <Bell size={15} />
            </button>
            <button
              className="secondary-button compact"
              type="button"
              onClick={() => persistSettings({ ...settings, locale: locale === 'he' ? 'en' : 'he', direction: locale === 'he' ? 'ltr' : 'rtl' })}
            >
              <Languages size={14} />
              {locale === 'he' ? copy.english : copy.hebrew}
            </button>
            <button className="run-button" type="button">
              <Play size={15} fill="currentColor" />
              {copy.runOptimization}
            </button>
          </div>
        </header>

        {activeView === 'Settings' ? (
          <SettingsPanel
            settings={settings}
            copy={copy}
            locale={locale}
            saveState={saveState}
            onSave={persistSettings}
          />
        ) : (
          <>
        <section className="metric-strip" aria-label="Optimization summary">
          <Metric label={copy.metrics[0]} value={formatCurrency(overview.summary.projected_revenue, locale)} delta="+7.6%" icon={CircleDollarSign} positive />
          <Metric label={copy.metrics[1]} value={`${overview.summary.average_retention}%`} delta="-1.8pp" icon={Users} />
          <Metric label={copy.metrics[2]} value={formatMinutes(overview.summary.total_ad_seconds, locale)} delta="+120" icon={Clock3} positive />
          <Metric label={copy.metrics[3]} value={copy.risk[riskLabel(overview.summary.risk_score)]} delta={`${overview.summary.risk_score}/100`} icon={ShieldCheck} tone="risk" />
        </section>

        <div className="work-grid">
          <section className="planner-surface" aria-label={copy.canvas}>
            <div className="surface-toolbar">
              <div className="toolbar-left">
                <button className="segmented active" type="button">{copy.toolbar[0]}</button>
                <button className="segmented" type="button">{copy.toolbar[1]}</button>
                <button className="segmented" type="button">{copy.toolbar[2]}</button>
              </div>
              <div className="toolbar-right">
                <label className="check-control">
                  <input type="checkbox" defaultChecked />
                  {copy.toolbar[3]}
                </label>
                <label className="check-control">
                  <input type="checkbox" defaultChecked />
                  {copy.toolbar[4]}
                </label>
                <button className="secondary-button compact" type="button">
                  <SlidersHorizontal size={14} />
                  {copy.toolbar[5]}
                </button>
              </div>
            </div>

            <PlanningCanvas rows={schedule.rows || []} copy={copy} locale={locale} />
          </section>

          <Inspector
            selectedProgram={selectedProgram}
            recommendation={activeRec}
            approved={approved.has(activeRec?.id)}
            onApprove={() => activeRec && toggleApproval(activeRec.id)}
            copy={copy}
            locale={locale}
          />
        </div>

        <section className="analytics-strip" aria-label="Analytics and constraint ledger">
          <FrontierPanel data={overview.frontier || []} copy={copy} locale={locale} />
          <InventoryHeatmap copy={copy} locale={locale} />
          <ComplianceLedger compliance={compliance} copy={copy} locale={locale} />
        </section>
          </>
        )}

        {loading && <div className="toast">{copy.loading}</div>}
        {!loading && error && <div className="toast muted">{copy.apiUnavailable}</div>}
      </main>
    </div>
  );
}

function riskLabel(score) {
  if (score >= 68) return 'High';
  if (score >= 38) return 'Medium';
  return 'Low';
}

function Metric({ label, value, delta, icon: Icon, positive = false, tone }) {
  return (
    <div className="metric">
      <span className={`metric-icon ${tone || ''}`}>
        <Icon size={17} strokeWidth={1.8} />
      </span>
      <span className="metric-copy">
        <span>{label}</span>
        <strong>{value}</strong>
      </span>
      <span className={positive ? 'delta positive' : tone === 'risk' ? 'delta risk' : 'delta negative'}>
        {positive ? <ArrowUp size={12} /> : tone === 'risk' ? null : <ArrowDown size={12} />}
        {delta}
      </span>
    </div>
  );
}

function PlanningCanvas({ rows, copy, locale }) {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const dayLabels = locale === 'he' ? ['ב׳', 'ג׳', 'ד׳', 'ה׳', 'ו׳', 'ש׳', 'א׳'] : days;
  return (
    <div className="planning-canvas">
      <div className="canvas-header">
        <span>{copy.channelProgram}</span>
        {days.map((day, index) => (
          <span key={day}>{dayLabels[index]}</span>
        ))}
      </div>
      {rows.map((row) => (
        <div className="channel-row" key={row.channel}>
          <div className="channel-name">
            <span>{row.channel.replace('ערוץ', 'K')}</span>
            <small>{row.programs?.[0]?.program_type || 'Mixed'}</small>
          </div>
          {days.map((day) => {
            const program = row.programs?.find((item) => item.day === day) || row.programs?.[days.indexOf(day) % row.programs.length];
            return <ProgramCell key={`${row.channel}-${day}`} program={program} locale={locale} />;
          })}
        </div>
      ))}
    </div>
  );
}

function ProgramCell({ program, locale }) {
  if (!program) return <div className="program-cell empty" />;
  const markers = Array.from({ length: Math.max(1, Math.min(10, program.break_markers || 1)) });
  return (
    <button className={program.selected ? 'program-cell selected' : 'program-cell'} type="button">
      <span className="program-title">{program.title}</span>
      <span className="program-meta">{program.time} / {program.duration_minutes}m</span>
      <span className="break-markers">
        {markers.map((_, index) => (
          <i key={index} className={index % 3 === 0 ? 'marker revenue' : 'marker'} />
        ))}
      </span>
      <span className="cell-metrics">
        <span>{formatCurrency(program.revenue, locale)}</span>
        <span>{program.retention}%</span>
      </span>
    </button>
  );
}

function Inspector({ selectedProgram, recommendation, approved, onApprove, copy, locale }) {
  return (
    <aside className="inspector" aria-label="Selected break inspector">
      <div className="inspector-head">
        <span>{copy.selectedBreak}</span>
        <button className="icon-button small" type="button" aria-label="Close inspector">
          <X size={14} />
        </button>
      </div>

      <div className="selected-program">
        <span className="channel-badge">{selectedProgram?.channel?.slice(0, 2) || 'K1'}</span>
        <div>
          <strong>{selectedProgram?.title || 'Selected program'}</strong>
          <small>{selectedProgram?.channel || 'KAI 1'} / {selectedProgram?.time || '20:00'} / break 2 of 4</small>
        </div>
        <span className={approved ? 'approval approved' : 'approval'}>{approved ? copy.approved : copy.pending}</span>
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
        <strong>{locale === 'he' ? recommendation?.title_he || recommendation?.title : recommendation?.title || 'Review placement'}</strong>
        <p>{locale === 'he' ? recommendation?.rationale_he || recommendation?.rationale : recommendation?.rationale || 'Recommendation rationale unavailable.'}</p>
        <div className="recommendation-meta">
          <span>{copy.risk[recommendation?.risk || 'Medium'] || recommendation?.risk}</span>
          <span>{formatCurrency(recommendation?.impact || 0, locale)}</span>
        </div>
      </div>

      <div className="inspector-actions">
        <button className="primary-action" type="button" onClick={onApprove}>
          {approved ? copy.approved : copy.approve}
        </button>
        <button className="secondary-button" type="button">{copy.reject}</button>
        <button className="secondary-button" type="button">{copy.applySimilar}</button>
      </div>

      <div className="export-row">
        <select aria-label="Export scope" defaultValue="Break detail">
          <option value="Break detail">{copy.exportOptions[0]}</option>
          <option value="Weekly traffic plan">{copy.exportOptions[1]}</option>
          <option value="Guardrail report">{copy.exportOptions[2]}</option>
        </select>
        <button className="secondary-button" type="button">
          <Download size={14} />
          {copy.export}
        </button>
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
  const statusText = saveState === 'saved' ? copy.saved : saveState === 'error' ? 'Error' : copy.saveSettings;

  return (
    <section className="settings-workspace">
      <div className="settings-hero">
        <div>
          <span className="settings-kicker">{copy.nav.Settings}</span>
          <h1>{copy.settingsTitle}</h1>
          <p>{copy.settingsIntro}</p>
        </div>
        <button className="run-button" type="button" onClick={() => onSave(draft)}>
          <Save size={15} />
          {statusText}
        </button>
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
            <label>
              <span>{copy.profile}</span>
              <input value={draft.profile_name || ''} onChange={(event) => updateField('profile_name', event.target.value)} />
            </label>
            <label>
              <span>{copy.effectiveDate}</span>
              <input type="date" value={draft.effective_date || ''} onChange={(event) => updateField('effective_date', event.target.value)} />
            </label>
            <label>
              <span>{copy.language}</span>
              <select
                value={draft.locale || 'he'}
                onChange={(event) => updateField('locale', event.target.value)}
              >
                <option value="he">{copy.hebrew}</option>
                <option value="en">{copy.english}</option>
              </select>
            </label>
            <label>
              <span>{copy.source}</span>
              <input value={draft.regulatory_source_url || ''} onChange={(event) => updateField('regulatory_source_url', event.target.value)} />
            </label>
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
            <label>
              <span>{copy.protectedTypes}</span>
              <textarea
                rows={3}
                value={protectedTypes}
                onChange={(event) =>
                  updateField(
                    'protected_program_types',
                    event.target.value.split(',').map((item) => item.trim()).filter(Boolean),
                  )
                }
              />
            </label>
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
    <label className="number-control">
      <span>{label}</span>
      <div>
        <input type="number" value={value ?? 0} onChange={(event) => onChange(event.target.value)} />
        <small>{suffix}</small>
      </div>
    </label>
  );
}

function ToggleControl({ label, checked, onChange }) {
  return (
    <label className="toggle-control">
      <span>{label}</span>
      <input type="checkbox" checked={Boolean(checked)} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

export default TVBreakDashboard;
