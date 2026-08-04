// Render the shipped price-slot tester and report the week it offers.
//
// The defect this measures is invisible to a source reader and to the payload:
// the rate card and the tester sit on one screen and each read the week from a
// different place, so only rendering both in the same run can compare them.
//
// The component is bundled with the bundler the product builds with, where it
// lives, so its own imports resolve exactly as they do in the browser build, and
// rendered by React to static markup. The version under test is supplied through
// the loader rather than by copying the file, which is what keeps a mutant
// honest. The entry is virtual and carries an id inside the surface's own
// directory so its bare imports resolve through the application's node_modules;
// nothing is ever written into the source tree.
//
// Usage: node test_p5_tester_probe.mjs <bundle dir> <out.json> <tester source> <pricing.json>
import { createRequire, registerHooks } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const APP = `${ROOT}tv-break-dashboard`;
const RULES = `${APP}/src/rules`;
const TESTER = `${RULES}/PricingSlotTester.jsx`;
const ENTRY_ID = `${RULES}/pricing-slot-tester-probe-entry.jsx`;

const [outDir, outFile, testerSource, pricingFile] = process.argv.slice(2);
const require_ = createRequire(`${APP}/package.json`);
const MAP = {};
for (const bare of ['react', 'react/jsx-runtime', 'react-dom', 'react-dom/client', 'react-dom/server', 'rolldown']) {
  MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) return { url: MAP[specifier], shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

// The tester, the reader the rate card above it uses, and the style cache the
// design system needs on a server render.
const ENTRY_SOURCE = `
export { default as PricingSlotTester } from '${TESTER}';
export { layerEntries, keyLabel } from '${RULES}/pricing-layers-lib';
export { CacheProvider } from '@emotion/react';
export { default as createCache } from '@emotion/cache';
`;

const { build } = await import('rolldown');
await build({
  input: ENTRY_ID,
  external: ['react', 'react-dom', 'react-dom/client', 'react/jsx-runtime', 'react-dom/server'],
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  platform: 'node',
  logLevel: 'silent',
  plugins: [{
    name: 'pricing-slot-tester-under-test',
    resolveId(id) { return id === ENTRY_ID ? ENTRY_ID : null; },
    load(id) {
      if (id === ENTRY_ID) return ENTRY_SOURCE;
      return id === TESTER ? fs.readFileSync(testerSource, 'utf8') : null;
    },
  }],
});

// node has no Vite import.meta.env. Only that accessor is rewritten, only in the
// emitted bundle, and never in the shipped source.
const built = `${outDir}/surface.mjs`;
fs.writeFileSync(built, fs.readFileSync(built, 'utf8').replaceAll('import.meta.env', '({})'), 'utf8');

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const surface = await import(pathToFileURL(built).href);
const pricing = JSON.parse(fs.readFileSync(pricingFile, 'utf8'));

const html = {};
for (const locale of ['he', 'en']) {
  html[locale] = renderToStaticMarkup(React.createElement(
    surface.CacheProvider,
    { value: surface.createCache({ key: 'p5' }) },
    React.createElement(surface.PricingSlotTester, {
      state: pricing, locale, notify: () => {}, currency: 'ILS',
    }),
  ));
}

// The same screen's other week: the day layer of the rate card, read through the
// reader that page renders it with.
const dayLayer = (pricing.layers || []).find((layer) => layer.name === 'day') || null;
const card = dayLayer ? surface.layerEntries(dayLayer) : [];
fs.writeFileSync(outFile, JSON.stringify({
  html,
  card_day_keys: card.map(([key]) => String(key)),
  card_day_labels_he: card.map(([key]) => surface.keyLabel('day', key, 'he')),
}), 'utf8');
