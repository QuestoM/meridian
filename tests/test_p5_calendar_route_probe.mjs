// Run the shipped calendar-banner control and the shipped section-routing
// function, as the Rules workspace runs them.
//
// Two things this measures. First, `CalendarPricingBanner.jsx`: the banner
// used to render its own control only when `typeof setActiveView ===
// 'function'`, a prop nothing ever supplied, so the control compiled out of
// every render and the banner named a destination with no way to it. This
// renders the shipped component to static markup, with and without a callback
// supplied, and counts the control that lands in the markup. Second,
// `nextRulesSection` from `rules-lib.js`: the section RulesWorkspace shows was
// read from the `?rules` query only once, at mount, so a query rewritten by
// something else after that (the old Pricing bookmark redirect, or the
// browser's own back and forward buttons) left the visible section stuck
// while the address bar claimed another. This calls the shipped function
// directly with the query values that scenario produces.
//
// Bundled with the bundler the product builds with, so every relative import
// resolves exactly as it does in the browser build; only the one accessor
// Node lacks, import.meta.env, is rewritten, and only in the emitted bundle.
//
// Usage: node test_p5_calendar_route_probe.mjs   (prints JSON on stdout)
import { createRequire, registerHooks } from 'node:module';
import { mkdtempSync } from 'node:fs';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const APP = join(ROOT, 'tv-break-dashboard');
const RULES = join(APP, 'src', 'rules');
const BANNER = join(RULES, 'CalendarPricingBanner.jsx');
const ENTRY_ID = join(RULES, 'calendar-route-probe-entry.jsx');
const outDir = mkdtempSync(join(tmpdir(), 'p5-calendar-route-'));

const require_ = createRequire(join(APP, 'package.json'));
const MAP = {};
for (const bare of ['react', 'react/jsx-runtime', 'react-dom', 'react-dom/server', 'rolldown']) {
  MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) return { url: MAP[specifier], shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const ENTRY_SOURCE = `
export { default as CalendarPricingBanner } from '${BANNER}';
export { nextRulesSection } from '${RULES}/rules-lib';
`;

const { build } = await import('rolldown');
await build({
  input: ENTRY_ID,
  external: ['react', 'react-dom', 'react-dom/server', 'react/jsx-runtime'],
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  platform: 'node',
  logLevel: 'silent',
  plugins: [{
    name: 'calendar-route-under-test',
    resolveId(id) { return id === ENTRY_ID ? ENTRY_ID : null; },
    load(id) { return id === ENTRY_ID ? ENTRY_SOURCE : null; },
  }],
});

// node has no Vite import.meta.env. Only that accessor is rewritten, only in
// the emitted bundle, and never in the shipped source.
const built = join(outDir, 'surface.mjs');
fs.writeFileSync(built, fs.readFileSync(built, 'utf8').replaceAll('import.meta.env', '({})'), 'utf8');

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const surface = await import(pathToFileURL(built).href);

const banner = {};
for (const locale of ['he', 'en']) {
  for (const eventsPricing of [null, true, false]) {
    for (const withCallback of [true, false]) {
      const key = `${locale}_${String(eventsPricing)}_${withCallback ? 'callback' : 'no_callback'}`;
      banner[key] = renderToStaticMarkup(React.createElement(surface.CalendarPricingBanner, {
        locale,
        eventsPricing,
        onOpenRateCard: withCallback ? () => {} : undefined,
      }));
    }
  }
}

// The exact scenario measured live: the section a workspace mounted with
// ('calendar'), then a query rewritten by something else after mount ('rate_card'),
// then that same query read again unchanged (must not move a second time), then
// an invalid or missing query (must never clear a real section).
const route = {
  external_change_is_followed: surface.nextRulesSection('calendar', 'rate_card'),
  unchanged_query_is_a_no_op: surface.nextRulesSection('rate_card', 'rate_card'),
  empty_query_keeps_the_current_section: surface.nextRulesSection('licence', ''),
};

process.stdout.write(JSON.stringify({ banner, route }));
