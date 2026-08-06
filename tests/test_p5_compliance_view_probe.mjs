// Run the shipped Today-ledger view logic, as ComplianceLedger.jsx runs it.
//
// `complianceViewState` is the one function that decides whether the Today
// compliance card prints its own scoped verdict, a scoped fallback, an honest
// "no channel declared" note, or an honest "basis not stated" note in place of
// a market-wide figure mislabelled as the operator's own. Node cannot import
// rules-lib.js directly because it reaches `import.meta.env` through
// shell/api.js, which only Vite resolves, so this copies the real file into a
// temporary tree and supplies a plain stub for that one import. It tests the
// shipped source, not a restatement of it.
//
// Usage: node test_p5_compliance_view_probe.mjs   (prints JSON on stdout)
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const RULES = join(HERE, '..', 'tv-break-dashboard', 'src', 'rules');

const root = mkdtempSync(join(tmpdir(), 'p5-compliance-view-'));
mkdirSync(join(root, 'rules'));
mkdirSync(join(root, 'shell'));
writeFileSync(join(root, 'shell', 'api.js'), "export const API_BASE = '';\n");
writeFileSync(join(root, 'rules', 'rules-lib.js'), readFileSync(join(RULES, 'rules-lib.js'), 'utf8'));

const lib = await import(pathToFileURL(join(root, 'rules', 'rules-lib.js')).href);
const { complianceViewState } = lib;

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

process.stdout.write(JSON.stringify(cases));
