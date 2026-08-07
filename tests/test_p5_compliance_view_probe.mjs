// Run the shipped Today-ledger view logic, as ComplianceLedger.jsx runs it.
//
// `complianceViewState` is the one function that decides whether the Today
// compliance card prints its own scoped verdict, a scoped fallback, an honest
// "no channel declared" note, or an honest "basis not stated" note in place of
// a market-wide figure mislabelled as the operator's own. Node cannot import
// rules-lib.js directly because it reaches `import.meta.env` through
// shell/api.js, which only Vite rewrites, so this bundles the real module with
// the bundler the product builds with, which resolves every relative import
// straight off disk, and only rewrites the one accessor Node lacks. It tests
// the shipped source, not a restatement of it.
//
// Usage: node test_p5_compliance_view_probe.mjs   (prints JSON on stdout)
import { createRequire, registerHooks } from 'node:module';
import fs from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const APP = join(ROOT, 'tv-break-dashboard');
const RULES_LIB = join(APP, 'src', 'rules', 'rules-lib.js');
const outDir = mkdtempSync(join(tmpdir(), 'p5-compliance-view-'));

const require_ = createRequire(join(APP, 'package.json'));
const MAP = { rolldown: pathToFileURL(require_.resolve('rolldown')).href };
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) return { url: MAP[specifier], shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: RULES_LIB,
  output: { dir: outDir, format: 'esm', entryFileNames: 'lib.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  platform: 'node',
  logLevel: 'silent',
});

// node has no Vite import.meta.env. Only that accessor is rewritten, only in
// the emitted bundle, and never in the shipped source.
const built = join(outDir, 'lib.mjs');
fs.writeFileSync(built, fs.readFileSync(built, 'utf8').replaceAll('import.meta.env', '({})'), 'utf8');

const { complianceViewState, complianceScopeSentence } = await import(pathToFileURL(built).href);

const SCOPED_OWN = {
  scope: { scoped: true, scope_channel: 'ערוץ 13', rows_out: 736, competitor_rows_excluded: 8290 },
  checks: [{ id: 'break_spacing', observed: 7.01, limit: 7, unit: 'minutes', violations: 736 }],
  disclaimer: 'own',
};
const UNSCOPED_MARKET_WIDE = {
  // The exact defect the critic measured: /api/overview's compliance block
  // carries no scope key at all.
  checks: [{ id: 'break_spacing', observed: 7.0, limit: 7, unit: 'minutes', violations: 3584 }],
  disclaimer: 'market',
};
const NO_CHANNEL = {
  scope: { scoped: false, reason_en: 'no channel', reason_he: 'אין ערוץ' },
  checks: [],
};

const cases = {
  // Own fetch has not answered yet and has not failed: an honest loading note,
  // never the unscoped prop.
  still_loading: complianceViewState(null, false, UNSCOPED_MARKET_WIDE),
  // Own fetch answered with the scoped route: own wins even though a fallback
  // prop was also supplied.
  own_wins_over_fallback: complianceViewState(SCOPED_OWN, false, UNSCOPED_MARKET_WIDE),
  // Own fetch failed and the fallback carries no scope key at all: the exact
  // shape GET /api/overview returns. Must not print the market-wide numbers as
  // the operator's own.
  unscoped_fallback_is_basis_missing: complianceViewState(null, true, UNSCOPED_MARKET_WIDE),
  // Own fetch failed but the fallback is itself properly scoped: safe to show.
  scoped_fallback_is_shown: complianceViewState(null, true, SCOPED_OWN),
  // Own fetch failed and the fallback names no operator channel: the honest
  // "no population" note, not an empty checks list presented as compliant.
  no_channel_fallback: complianceViewState(null, true, NO_CHANNEL),
};

// The exact scope line the critic measured mispainting: a channel name whose
// own script is Hebrew, read on the English page. The English sentence must
// resolve left to right regardless of that name, and the Hebrew sentence
// still right to left, on the same payload. Channel and figures match the
// live GET /api/compliance response measured against the reference data.
const CRITIC_SCOPE = { scope_channel: 'רשת 13', rows_out: 2391, competitor_rows_excluded: 6635 };
cases.scope_sentence_en = complianceScopeSentence('en', CRITIC_SCOPE);
cases.scope_sentence_he = complianceScopeSentence('he', CRITIC_SCOPE);

process.stdout.write(JSON.stringify(cases));
