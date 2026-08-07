// Run the shipped word table, as the two lists that read it run it.
//
// `rules-words.js` is what turns a stored engine key into the sentence a person
// reads: the effect a saved constraint row holds, and the refusal a walled
// control prints before the click. Node cannot import it directly because it
// reaches the frozen session module for the wall details, so this copies the
// real file into a temporary tree and supplies those details from the Python
// constants, which is what makes a drift between the two visible here rather
// than on somebody's screen. It tests the shipped source, not a restatement.
//
// Usage: node test_p5_words_probe.mjs <walls.json>   (prints JSON on stdout)
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const RULES = join(HERE, '..', 'tv-break-dashboard', 'src', 'rules');
const walls = JSON.parse(readFileSync(process.argv[2], 'utf8'));

const root = mkdtempSync(join(tmpdir(), 'p5-words-'));
mkdirSync(join(root, 'rules'));
writeFileSync(join(root, 'session.js'), `export const WALLS = ${JSON.stringify(walls.walls)};\n`);
writeFileSync(join(root, 'rules', 'rules-words.js'), readFileSync(join(RULES, 'rules-words.js'), 'utf8'));
writeFileSync(join(root, 'rules', 'rules-bidi.js'), readFileSync(join(RULES, 'rules-bidi.js'), 'utf8'));

const words = await import(pathToFileURL(join(root, 'rules', 'rules-words.js')).href);

const effects = {};
for (const key of walls.effect_keys) {
  effects[key] = { he: words.effectLabel(key, 'he'), en: words.effectLabel(key, 'en') };
}
const refusals = {};
for (const [name, detail] of Object.entries(walls.refusals)) {
  refusals[name] = { he: words.refusalSentence(detail, 'he'), en: words.refusalSentence(detail, 'en') };
}
const scheduled = {};
for (const count of [1, 2]) {
  scheduled[count] = { he: words.scheduledChangesSentence('he', count), en: words.scheduledChangesSentence('en', count) };
}
process.stdout.write(JSON.stringify({ effects, refusals, scheduled }));
