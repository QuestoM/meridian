// One loader hook, so a probe never has to learn about a shell primitive again.
//
// The problem this exists to end. Every browser probe in this repository runs a
// piece of the dashboard under bare node, and each one resolves a subtree it
// listed by hand when it was written. On 2026-08-08 two shell primitives were
// added, src/shell/bidi.jsx and src/shell/dates.js, and the components those
// probes drive now import them. Forty-three test files went red at once with
//
//     Error [ERR_MODULE_NOT_FOUND]: Cannot find module '.../src/shell/bidi'
//
// and none of them said anything about the product any more. Five were fixed one
// at a time, four different ways, before it was clear that the population was
// not five. Teaching forty-three harnesses the same lesson individually is how
// the forty-fourth still gets it wrong.
//
// So the knowledge lives here once. A probe adds one flag:
//
//     node --import <this file> <its own entry> ...
//
// and any specifier ending shell/bidi or shell/dates resolves to the REAL
// module, compiled with the bundler's own transform. Not a stub: a probe that
// asserts against a fake primitive proves nothing about what ships.
//
// Why the compiled copy lands inside tv-break-dashboard rather than in a
// temporary directory: bidi.jsx imports react, and an ES module resolves a bare
// specifier from the importing FILE's own location, not from the working
// directory. A copy anywhere outside the dashboard cannot find node_modules.
// That cost two runs to learn; it is written here so it costs nobody a third.

import { createRequire, registerHooks } from 'node:module';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DASHBOARD = join(HERE, '..', '..', 'tv-break-dashboard');
const SHELL = join(DASHBOARD, 'src', 'shell');
// Inside the dashboard, and named so it is obviously not source. A probe may run
// concurrently with another, so the directory is shared and the files are
// written whole rather than appended.
const CACHE = join(DASHBOARD, '.probe-shell');

// Each primitive, and every specifier shape a real importer uses for it.
// dates.js imports bidi as './bidi', with no shell segment, so matching on
// 'shell/bidi' alone silently misses it and looks like it worked.
const MODULES = [
  { name: 'bidi', file: join(SHELL, 'bidi.jsx'), jsx: true, tails: ['shell/bidi', '/bidi'] },
  { name: 'dates', file: join(SHELL, 'dates.js'), jsx: false, tails: ['shell/dates', '/dates'] },
];

const require_ = createRequire(join(DASHBOARD, 'resolve-from-here.js'));
const { transformWithOxc } = await import(pathToFileURL(require_.resolve('vite')).href);

mkdirSync(CACHE, { recursive: true });

const built = new Map();
for (const module of MODULES) {
  const source = readFileSync(module.file, 'utf8');
  // dates.js imports './bidi', which resolves inside the cache directory to the
  // sibling written on the previous pass, so the two stay a pair.
  const code = module.jsx ? (await transformWithOxc(source, module.file)).code : source;
  const target = join(CACHE, `${module.name}.mjs`);
  writeFileSync(target, code.replace(/(from\s*['"])\.\/bidi(['"])/g, '$1./bidi.mjs$2'));
  built.set(module.name, pathToFileURL(target).href);
}

registerHooks({
  resolve(specifier, context, next) {
    if (!specifier.endsWith('.css')) {
      for (const module of MODULES) {
        const bare = specifier.replace(/\.(jsx?|mjs)$/, '');
        if (module.tails.some((tail) => bare.endsWith(tail))) {
          return { url: built.get(module.name), shortCircuit: true };
        }
      }
    }
    return next(specifier, context);
  },
});
