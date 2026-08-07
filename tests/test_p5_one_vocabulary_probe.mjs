// Run every word table this destination's surface reads, in BOTH languages.
//
// The class this guards is one display string frozen in ONE language at the
// place it is produced, then printed verbatim to a reader of the other. On the
// surface side it produces itself three ways, and this drives all three against
// the shipped modules rather than a restatement of them:
//
//   1. A config key printed as a label. The rate card's categories come from
//      config/optimization_weights.yaml, which is written once in one language:
//      `program_type` is keyed in Latin, `ad_type` in Hebrew. Each language got
//      the half that is not its own.
//   2. A description taken half from a table and half from the wire. `descHe`
//      came from the table and the English came from the API, so a layer the
//      table did not name printed English prose under a Hebrew heading.
//   3. A sentence the server authored in one language and the surface printed
//      as it arrived: a wall refusal, a rejected save, an empty-basis reason.
//
// Every label the surface can print for a key the shipped rate card can carry
// is produced here in both locales; the Python wrapper asserts the script.
//
// Bundled with the bundler the product builds with, so every relative import
// resolves exactly as it does in the browser build; only import.meta.env, which
// node lacks, is rewritten, and only in the emitted bundle.
//
// Usage: node test_p5_one_vocabulary_probe.mjs   (prints JSON on stdout)
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
const ENTRY_ID = join(RULES, 'one-vocabulary-probe-entry.js');
const outDir = mkdtempSync(join(tmpdir(), 'p5-one-vocabulary-'));

const require_ = createRequire(join(APP, 'package.json'));
const MAP = {};
for (const bare of ['rolldown']) MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) return { url: MAP[specifier], shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const ENTRY_SOURCE = `
export * from '${RULES}/pricing-layers-lib';
export { basisReason, detailWords, limitBoundsRefusal, refusalSentence, refusalWords } from '${RULES}/rules-words';
export { WALLS } from '${join(APP, 'src', 'session.js')}';
`;

const { build } = await import('rolldown');
await build({
  input: ENTRY_ID,
  output: { dir: outDir, format: 'esm', entryFileNames: 'words.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  platform: 'node',
  logLevel: 'silent',
  plugins: [{
    name: 'one-vocabulary-under-test',
    resolveId(id) { return id === ENTRY_ID ? ENTRY_ID : null; },
    load(id) { return id === ENTRY_ID ? ENTRY_SOURCE : null; },
  }],
});

const built = join(outDir, 'words.mjs');
fs.writeFileSync(built, fs.readFileSync(built, 'utf8').replaceAll('import.meta.env', '({})'), 'utf8');
const words = await import(pathToFileURL(built).href);

const LOCALES = ['en', 'he'];
const both = (fn) => Object.fromEntries(LOCALES.map((locale) => [locale, fn(locale)]));

// Every key the shipped rate card can carry, per layer, straight off the
// vocabulary the surface itself holds plus the config keys the engine ships.
const KEYS = {
  program: ['News', 'PrimeShow1', 'PrimeShow2', 'Other'],
  ad_type: ['פרסומת', 'חסות', 'פרומו'],
  day: ['1', '2', '3', '4', '5', '6', '7'],
  position: ['1', '2', '3', '4', '5', 'L', 'default_middle', 'last'],
};

const labels = {};
for (const [layer, keys] of Object.entries(KEYS)) {
  for (const key of keys) {
    labels[`${layer}::${key}`] = both((locale) => words.keyLabel(layer, key, locale));
  }
}

// A programme title is a proper noun and is not translated in either direction.
// It is asserted to pass through unchanged rather than asserted on its script.
const passthrough = both((locale) => words.keyLabel('show', 'האח הגדול', locale));

const layers = {};
for (const name of Object.keys(words.LAYER_TEXT)) {
  layers[name] = {
    title: both((locale) => words.layerLabel(name, locale)),
    description: both((locale) => words.layerDescription({ name, description: 'FROM THE WIRE' }, locale)),
    has_both_halves: Boolean(words.LAYER_TEXT[name].descEn) === Boolean(words.LAYER_TEXT[name].descHe),
  };
}
// A layer no table names: the producer's own words, the same for both readers.
layers.__unknown__ = {
  title: both((locale) => words.layerLabel('brand_new_layer', locale)),
  description: both((locale) => words.layerDescription({ name: 'brand_new_layer', description: 'FROM THE WIRE' }, locale)),
  has_both_halves: true,
};

// The zero-multiplier warning names its categories through the same table the
// rows do, so the sentence and the row cannot call one category two things.
const warning = both((locale) => words.categoryList('ad_type', ['פרומו', 'חסות'], locale));

// The licence refusal, in the three states a payload can arrive in: the pair the
// store now stamps, an older payload with only the Hebrew, and a gate the
// session decided with no payload at all.
const refusals = {
  pair: both((locale) => words.refusalWords({
    can_edit: false,
    can_edit_reason: words.WALLS.guardrails.detail,
    can_edit_reason_en: 'Only company staff change the regulatory limits.',
    can_edit_reason_he: 'שינוי מגבלות הרגולציה שמור לצוות החברה',
  }, words.WALLS.guardrails.detail, locale)),
  hebrew_only_payload: both((locale) => words.refusalWords(
    { can_edit: false, can_edit_reason: words.WALLS.guardrails.detail },
    words.WALLS.guardrails.detail, locale,
  )),
  session_gate: both((locale) => words.refusalWords(null, words.WALLS.readOnlyRole.detail, locale)),
  every_wall: Object.fromEntries(Object.entries(words.WALLS).map(
    ([name, wall]) => [name, both((locale) => words.refusalSentence(wall.detail, locale))],
  )),
};

// A rejected save: the pair the rules routes now put on the wire, and a route
// that still sends one string, which stays what both readers get.
const details = {
  pair: both((locale) => words.detailWords(
    { message: 'x', words: { en: 'There is no restriction of that sort to write.', he: 'אין סוג הגבלה כזה לכתוב.' } },
    locale,
  )),
  single_string: both((locale) => words.detailWords({ message: '503 Service Unavailable' }, locale)),
};

// The out-of-bounds refusal this surface makes itself, and the value it accepts.
const bounds = { max_breaks_per_hour: { min: 1, max: 20 } };
const limits = {
  refused: both((locale) => words.limitBoundsRefusal(locale, 'max_breaks_per_hour', 99, bounds)),
  not_a_number: both((locale) => words.limitBoundsRefusal(locale, 'max_breaks_per_hour', '', bounds)),
  accepted: both((locale) => words.limitBoundsRefusal(locale, 'max_breaks_per_hour', 4, bounds)),
};

// Every empty-basis reason the two frozen producers this surface reads can emit.
const BASIS = [
  'No saved weekly plan with segment ids is on disk to price.',
  'No saved weekly schedule with segment ids on disk.',
  'No saved weekly schedule on disk.',
  'The saved plan carries no rows for the declared operator channel.',
  'The saved plan carries no rows for the configured operator channel.',
  'Plan segment rebuild failed; see the server log.',
  'The optimization engine is unavailable.',
  'Saved plan no longer joins the EPG rebuild; recompute the schedule.',
  'Saved schedule has no ad-seconds to monetize.',
  'Band computation failed; see the server log.',
];
const basis = Object.fromEntries(BASIS.map((reason) => [reason, both((locale) => words.basisReason(reason, locale))]));

process.stdout.write(JSON.stringify({ labels, passthrough, layers, warning, refusals, details, limits, basis }));
