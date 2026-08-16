import createCache from '@emotion/cache';
import rtlPlugin from '@mui/stylis-plugin-rtl';
import { prefixer } from 'stylis';

// MUI X emits one structural `*:first-child` selector for the grid's main
// container. Every child in that container is a div, so `:first-of-type` is
// equivalent and avoids Emotion's SSR warning after RTL transformation.
function safeDataGridFirstSelector(element) {
  if (element.type !== 'rule' || !Array.isArray(element.props)) return;
  element.props = element.props.map((selector) => (
    selector.includes('MuiDataGrid-main')
      ? selector.replaceAll(':first-child', ':first-of-type')
      : selector
  ));
}

// Emotion's warning plug-in runs before caller-provided Stylis transforms.
// Suppress it only while inserting the one known grid rule that the transform
// above makes safe; every other unsafe selector keeps the normal diagnostic.
function withSafeGridCompatibility(cache) {
  const insert = cache.insert;
  cache.insert = (selector, serialized, sheet, shouldCache) => {
    const styles = String(serialized?.styles || '');
    const knownGridRule = styles.includes('MuiDataGrid-main') && styles.includes(':first-child');
    if (!knownGridRule) return insert(selector, serialized, sheet, shouldCache);
    const previous = cache.compat;
    cache.compat = true;
    try {
      return insert(selector, serialized, sheet, shouldCache);
    } finally {
      cache.compat = previous;
    }
  };
  return cache;
}

export const ltrCache = withSafeGridCompatibility(createCache({
  key: 'mui',
  stylisPlugins: [prefixer, safeDataGridFirstSelector],
}));

export const rtlCache = withSafeGridCompatibility(createCache({
  key: 'muirtl',
  stylisPlugins: [prefixer, safeDataGridFirstSelector, rtlPlugin],
}));
