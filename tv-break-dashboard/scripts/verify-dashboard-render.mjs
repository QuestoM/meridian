// The system smoke: the pages exist, the shell is MUI, and native controls only
// ever go down.
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
// So the fix is not a path. Three things change:
//
//   * It searches the whole source tree instead of one file, because the thing
//     it was written to guard no longer lives in one file.
//   * The native-control ban becomes a RATCHET rather than an absolute. 254 is
//     a real count and pretending it is zero would just get the check switched
//     off again. Going up fails. Going down below the budget also fails, so the
//     number in this file has to follow the tree down and cannot hide a
//     regression behind slack.
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

// The ratchet. Measured on 2026-08-08 with the guard revived after being dead
// since wave zero. It may only go down. Lower it in the same commit that removes
// controls; a count below it fails just as loudly as a count above it, because
// slack is how a budget stops being a guard.
//
// The first number written here was 254, from a hand grep that searched only
// .jsx and used a narrower character class. The guard itself, searching .js as
// well, answered 384. The guard was right and the grep was wrong, which is the
// argument for the guard: a number nobody re-derives is a number nobody checks.
const NATIVE_CONTROL_BUDGET = 350;
const NATIVE_CONTROL = /<(button|select|input|textarea)[\s>/]/g;

const nativeByFile = sourceFiles
  .map((file) => [file, (read(file).match(NATIVE_CONTROL) || []).length])
  .filter(([, count]) => count > 0)
  .sort((a, b) => b[1] - a[1]);
const nativeTotal = nativeByFile.reduce((sum, [, count]) => sum + count, 0);

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

if (nativeTotal > NATIVE_CONTROL_BUDGET) {
  failed = true;
  const over = nativeTotal - NATIVE_CONTROL_BUDGET;
  console.error(
    `\nNative controls went UP: ${nativeTotal} against a budget of ${NATIVE_CONTROL_BUDGET}, ${over} over.`,
  );
  console.error('Use the MUI control instead, which the shell themes for right-to-left.');
  for (const [file, count] of nativeByFile.slice(0, 12)) console.error(`  ${count} x ${file}`);
} else if (nativeTotal < NATIVE_CONTROL_BUDGET) {
  failed = true;
  console.error(
    `\nNative controls are down to ${nativeTotal} but the budget still says ${NATIVE_CONTROL_BUDGET}.`,
  );
  console.error('Lower NATIVE_CONTROL_BUDGET in this file so the slack cannot hide a regression.');
}

if (failed) process.exit(1);

console.log(
  `Kairos dashboard system smoke passed. Native controls at ${nativeTotal}, and the budget only goes down.`,
);
