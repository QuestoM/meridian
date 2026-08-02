// Which facts print on a source card is the reader's choice, not the builder's.
//
// Frame.io's device: 32 metadata fields exist per asset and a dropdown decides
// which of them print on the card itself, so technical truth is a display
// choice on the object rather than something behind a click. The same idea,
// scoped to what this product actually knows about a file.
//
// Four things are never toggleable, because they are the reason the card
// exists: the state, the remedy that goes with it, the file the engine really
// reads, and what an upload here would do. A reader can hide a size; they
// cannot hide the verdict or its consequence.

export const FIELDS = [
  { key: 'rows', en: 'Rows', he: 'שורות' },
  { key: 'columns', en: 'Columns', he: 'עמודות' },
  { key: 'size', en: 'Size', he: 'גודל' },
  { key: 'updated', en: 'Updated', he: 'עודכן' },
  { key: 'path', en: 'Path', he: 'נתיב' },
  { key: 'cadence', en: 'Arrives', he: 'מגיע' },
  { key: 'lastChecked', en: 'Last checked', he: 'נבדק לאחרונה' },
];

export const DEFAULT_FIELDS = ['rows', 'columns', 'size', 'updated'];

const STORAGE_KEY = 'meridian.sources.fields';

export function readFields() {
  if (typeof window === 'undefined' || !window.localStorage) return DEFAULT_FIELDS;
  try {
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || 'null');
    if (!Array.isArray(stored)) return DEFAULT_FIELDS;
    const known = stored.filter((key) => FIELDS.some((field) => field.key === key));
    return known.length ? known : DEFAULT_FIELDS;
  } catch {
    return DEFAULT_FIELDS;
  }
}

export function writeFields(keys) {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
  } catch {
    // A browser with storage disabled keeps the session's choice and forgets it
    // on reload, which is better than refusing the choice.
  }
}

export function fieldLabel(key, locale) {
  const field = FIELDS.find((entry) => entry.key === key);
  if (!field) return key;
  return locale === 'he' ? field.he : field.en;
}
