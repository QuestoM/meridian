import { readFileSync } from 'node:fs';

const source = readFileSync('src/TVBreakDashboard.jsx', 'utf8');
const styles = readFileSync('src/styles.css', 'utf8');
const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));

const requiredSourceFragments = [
  ['MUI theme', 'createKairosTheme'],
  ['MUI ThemeProvider', 'ThemeProvider'],
  ['MUI RTL cache', '@mui/stylis-plugin-rtl'],
  ['MUI DataGrid', 'DataGrid'],
  ['MUI Button', '<Button'],
  ['MUI Select', '<Select'],
  ['MUI TextField', '<TextField'],
  ['MUI Switch', '<Switch'],
  ['Overview page', 'function OverviewPage'],
  ['Optimizer page', 'function OptimizerWorkspace'],
  ['Grid axis control', 'function GridAxisControl'],
  ['Planner axis grouping', 'function buildPlannerColumns'],
  ['Timeline view', 'function TimelineView'],
  ['Schedule page', 'function SchedulePage'],
  ['Inventory page', 'function InventoryPage'],
  ['Break library page', 'function BreakLibraryPage'],
  ['Campaigns page', 'function CampaignsPage'],
  ['Forecasts page', 'function ForecastsPage'],
  ['Reports page', 'function ReportsPage'],
  ['Data hub page', 'function DataHubPage'],
  ['Inventory API', "/api/inventory"],
  ['Break library API', "/api/break-library"],
  ['Campaigns API', "/api/campaigns"],
  ['Forecasts API', "/api/forecasts"],
  ['Reports API', "/api/reports"],
  ['Break operations API', "/api/break-operations"],
  ['Break decisions API', "/api/break-decisions"],
];

const requiredStyleFragments = [
  ['Kairos shell', 'kairos-shell'],
  ['MUI grid wrapper', 'mui-grid-wrap'],
  ['page workspace', 'page-workspace'],
  ['Timeline styles', 'timeline-view'],
  ['LTR chart contract', 'chart-ltr'],
];

const requiredDependencies = [
  '@mui/material',
  '@mui/x-data-grid',
  '@emotion/react',
  '@emotion/styled',
  '@emotion/cache',
  '@mui/stylis-plugin-rtl',
  'stylis',
];

const bannedNativeControls = ['<button', '<select', '<input', '<textarea'];
const missing = [
  ...requiredSourceFragments.filter(([, fragment]) => !source.includes(fragment)),
  ...requiredStyleFragments.filter(([, fragment]) => !styles.includes(fragment)),
  ...requiredDependencies.filter((name) => !packageJson.dependencies?.[name]).map((name) => [`Dependency ${name}`, name]),
  ...bannedNativeControls.filter((fragment) => source.includes(fragment)).map((fragment) => [`Native control ${fragment}`, fragment]),
];

if (missing.length > 0) {
  console.error('Kairos dashboard system smoke failed.');
  for (const [label, fragment] of missing) {
    console.error(`- Missing or disallowed ${label}: ${fragment}`);
  }
  process.exit(1);
}

console.log('Kairos dashboard system smoke passed.');
