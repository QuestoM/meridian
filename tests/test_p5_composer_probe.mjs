// Render the shipped night picker, and the composer that mounts it.
//
// The defect this measures is invisible to the payload and to a source reader
// who trusts the head line: the endpoint returned every airing and the picker
// rendered the first twelve, so the count above the chips was not the count of
// choices below them. Only rendering the component against a real payload and
// counting what came out can catch that.
//
// The components are bundled with the bundler the product builds with, where
// they live, so their own imports resolve exactly as they do in the browser
// build, and rendered by React to static markup. The picker's version under
// test is supplied through the loader rather than by copying the file, which is
// what keeps a mutant honest. The entry is virtual and carries an id inside the
// surface's own directory so its bare imports resolve through the application's
// node_modules; nothing is ever written into the source tree.
//
// Usage: node test_p5_composer_probe.mjs <bundle dir> <out.json> <nights source> <airings.json>
import { createRequire, registerHooks } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const APP = `${ROOT}tv-break-dashboard`;
const RULES = `${APP}/src/rules`;
const NIGHTS = `${RULES}/AiringNights.jsx`;
const ENTRY_ID = `${RULES}/airing-nights-probe-entry.jsx`;

const [outDir, outFile, nightsSource, airingsFile] = process.argv.slice(2);
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

// The picker, the composer that mounts it, and the style cache the design
// system needs on a server render.
const ENTRY_SOURCE = `
export { default as AiringNights } from '${NIGHTS}';
export { default as RestrictionComposer } from '${RULES}/RestrictionComposer';
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
    name: 'airing-nights-under-test',
    resolveId(id) { return id === ENTRY_ID ? ENTRY_ID : null; },
    load(id) {
      if (id === ENTRY_ID) return ENTRY_SOURCE;
      return id === NIGHTS ? fs.readFileSync(nightsSource, 'utf8') : null;
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
const airings = JSON.parse(fs.readFileSync(airingsFile, 'utf8'));

const wrap = (element) => renderToStaticMarkup(React.createElement(
  surface.CacheProvider,
  { value: surface.createCache({ key: 'p5' }) },
  element,
));

const nights = {};
const composer = {};
for (const locale of ['he', 'en']) {
  nights[locale] = wrap(React.createElement(surface.AiringNights, {
    locale,
    airings: airings.count,
    nights: airings.nights,
    day: '',
    onPick: () => {},
  }));
  composer[locale] = wrap(React.createElement(surface.RestrictionComposer, {
    locale, notify: () => {}, onSaved: () => {},
  }));
}

// The same picker with one night already chosen, so the selected state is
// measured rather than assumed from the class name of the default.
const chosen = (airings.nights || []).at(-1);
const picked = chosen ? wrap(React.createElement(surface.AiringNights, {
  locale: 'he',
  airings: airings.count,
  nights: airings.nights,
  day: chosen.day,
  onPick: () => {},
})) : '';

fs.writeFileSync(outFile, JSON.stringify({
  nights, composer, picked, picked_day: chosen ? chosen.day : '',
}), 'utf8');
