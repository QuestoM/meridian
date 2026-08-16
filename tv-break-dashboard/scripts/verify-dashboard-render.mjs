// The system smoke: the pages exist, the shell is MUI, and screens instantiate
// controls through one canonical boundary.
//
// This check has been DEAD SINCE WAVE ZERO and two separate critics reported it
// without anyone owning the failure. It read `src/TVBreakDashboard.jsx`, the
// 4,102 line monolith that wave zero split into modules, so every run since has
// died on ENOENT before asserting anything.
//
// What that cost is not the ENOENT. It is the check underneath it: this script
// is the only thing that ever banned a native <button>, <select>, <input> or
// <textarea> in favour of the MUI control the shell themes for right-to-left.
// While it was dead the tree drifted to 254 of them across 71 files, and nobody
// saw a single one, because the run failed on line 3 every time.
//
// So the fix was not a path. Three things changed:
//
//   * It searches the whole source tree instead of one file, because the thing
//     it was written to guard no longer lives in one file.
//   * The native-control debt first became a ratchet and was then paid down in
//     Phase 3. Screens now contain zero raw controls. The only four tags live in
//     shell/dom-controls.jsx, where Pressable/InputControl/SelectControl/
//     TextAreaControl preserve native semantics behind the shared contract.
//   * `function DataHubPage` is dropped from the required list. It is genuinely
//     gone: the Data Hub and the Data Center were merged into one tabbed Data
//     page. Every other required fragment was re-checked against the tree and
//     found.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const SRC = 'src';

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

const files = walk(SRC);
const sourceFiles = files.filter((f) => f.endsWith('.jsx') || f.endsWith('.js'));
const styleFiles = files.filter((f) => f.endsWith('.css'));
const read = (f) => readFileSync(f, 'utf8');
const source = sourceFiles.map(read).join('\n');
const styles = styleFiles.map(read).join('\n');
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
  ['Inventory API', '/api/inventory'],
  ['Break library API', '/api/break-library'],
  ['Campaigns API', '/api/campaigns'],
  ['Forecasts API', '/api/forecasts'],
  ['Reports API', '/api/reports'],
  ['Break operations API', '/api/break-operations'],
  ['Break decisions API', '/api/break-decisions'],
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

const CONTROL_BOUNDARY = 'src/shell/dom-controls.jsx';
const EXPECTED_BOUNDARY_TAGS = 4;
const NATIVE_CONTROL = /<(button|select|input|textarea)[\s>/]/g;

const nativeByFile = sourceFiles
  .map((file) => [file, (read(file).match(NATIVE_CONTROL) || []).length])
  .filter(([, count]) => count > 0)
  .sort((a, b) => b[1] - a[1]);
const nativeTotal = nativeByFile.reduce((sum, [, count]) => sum + count, 0);
const boundaryTotal = (read(CONTROL_BOUNDARY).match(NATIVE_CONTROL) || []).length;
const screenNativeByFile = nativeByFile.filter(([file]) => file !== CONTROL_BOUNDARY);
const screenNativeTotal = screenNativeByFile.reduce((sum, [, count]) => sum + count, 0);

const missing = [
  ...requiredSourceFragments.filter(([, fragment]) => !source.includes(fragment)),
  ...requiredStyleFragments.filter(([, fragment]) => !styles.includes(fragment)),
  ...requiredDependencies
    .filter((name) => !packageJson.dependencies?.[name])
    .map((name) => [`Dependency ${name}`, name]),
];

let failed = false;

if (missing.length > 0) {
  failed = true;
  console.error('Kairos dashboard system smoke failed.');
  for (const [label, fragment] of missing) {
    console.error(`- Missing ${label}: ${fragment}`);
  }
}

if (screenNativeTotal !== 0) {
  failed = true;
  console.error(`\nScreens contain ${screenNativeTotal} raw native controls; expected zero.`);
  console.error('Use Button/Pressable/InputControl/SelectControl/TextAreaControl from the canonical Studio boundary.');
  for (const [file, count] of screenNativeByFile.slice(0, 12)) console.error(`  ${count} x ${file}`);
}

if (boundaryTotal !== EXPECTED_BOUNDARY_TAGS || nativeTotal !== EXPECTED_BOUNDARY_TAGS) {
  failed = true;
  console.error(
    `\nCanonical control boundary has ${boundaryTotal} raw tags and the tree has ${nativeTotal}; expected exactly ${EXPECTED_BOUNDARY_TAGS}.`,
  );
  console.error('Keep one native tag for each canonical bridge and no raw tags anywhere else.');
}

if (failed) process.exit(1);

console.log(
  `Kairos dashboard system smoke passed. Screens use zero raw controls; ${nativeTotal} canonical DOM bridges live in ${CONTROL_BOUNDARY}.`,
);
