// Run the shipped draft helpers, as the rate card runs them.
//
// The rate card's boxes are bound through three functions in `rules-lib.js`:
// `mergeOverrides` stages an edit, `dropOverride` takes one back out, and
// `draftValueAt` decides what a box shows. Node cannot import that module
// directly because its specifiers are extensionless and one of them reaches the
// shell, so this copies the two real modules into a temporary tree, adds the
// extensions, stubs the one shell import, and runs the sequence a person
// performs on the surface. It tests the shipped source, not a restatement of it.
//
// Usage: node test_p5_draft_probe.mjs   (prints JSON on stdout)
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const RULES = join(HERE, '..', 'tv-break-dashboard', 'src', 'rules');

const root = mkdtempSync(join(tmpdir(), 'p5-draft-'));
mkdirSync(join(root, 'rules'));
mkdirSync(join(root, 'shell'));
writeFileSync(join(root, 'shell', 'api.js'), "export const API_BASE = '';\n");
for (const name of ['rules-lib.js', 'rules-bidi.js']) {
  const source = readFileSync(join(RULES, name), 'utf8')
    .replace(/from '(\.[^']*?)'/g, (whole, target) => (target.endsWith('.js') ? whole : `from '${target}.js'`));
  writeFileSync(join(root, 'rules', name), source);
}

const lib = await import(pathToFileURL(join(root, 'rules', 'rules-lib.js')).href);
const { draftValueAt, dropOverride, mergeOverrides } = lib;

// The three moves the component makes, named the way it names them.
const stage = (pending, patch) => mergeOverrides(pending || {}, patch);
const unstage = (pending, path) => dropOverride(pending, path);
const shown = (pending, path, saved) => draftValueAt(pending, path) ?? saved;

const BASE = ['base_price_per_second_per_tvr_point'];
const NEWS = ['premiums', 'program_type', 'News'];
const REALITY = ['premiums', 'program_type', 'Reality'];
const SHOW_ON = ['pricing_activation', 'show'];

const savedBase = 60;
const out = {};

// The measured defect: type 80, blur, then press "discard the edit".
let pending = null;
out.on_load = shown(pending, BASE, savedBase);
pending = stage(pending, { base_price_per_second_per_tvr_point: 80 });
out.after_edit = shown(pending, BASE, savedBase);
pending = null; // what "Discard the edit" does
out.after_discard = shown(pending, BASE, savedBase);

// Typing the saved figure back is a revert, not a no-op.
pending = stage(null, { base_price_per_second_per_tvr_point: 80 });
pending = unstage(pending, BASE);
out.after_typing_the_saved_figure_back = { shown: shown(pending, BASE, savedBase), draft: pending };

// A premium, and its parents pruned when the last leaf goes.
pending = stage(null, { premiums: { program_type: { News: 1.65 } } });
out.premium_after_edit = shown(pending, NEWS, 1.15);
pending = unstage(pending, NEWS);
out.premium_after_revert = { shown: shown(pending, NEWS, 1.15), draft: pending };

// A sibling edit survives a revert beside it.
pending = stage(null, { premiums: { program_type: { News: 1.65, Reality: 2.0 } } });
pending = unstage(pending, NEWS);
out.sibling_survives = { draft: pending, reality: shown(pending, REALITY, 1.0) };

// A staged zero is a value, not an absence: the ad-type promo multiplier is 0.
pending = stage(null, { premiums: { program_type: { News: 0 } } });
out.staged_zero = shown(pending, NEWS, 1.15);

// An activation switch shows what the draft asks for while the draft exists.
pending = stage(null, { pricing_activation: { show: true } });
out.switch_after_click = draftValueAt(pending, SHOW_ON) ?? false;
pending = null;
out.switch_after_discard = draftValueAt(pending, SHOW_ON) ?? false;

process.stdout.write(JSON.stringify(out));
