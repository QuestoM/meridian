import { useCallback, useEffect, useState } from 'react';
import { readFiles } from './plan-week-api';

export const OPTIMIZER_INVENTORY_PATH = 'data/Spots - inventory.csv';

function normalizedPath(value) {
  return String(value || '').replace(/\\/g, '/');
}

export function inventoryReadinessFromFiles(payload) {
  if (!payload || !Array.isArray(payload.also_read)) {
    return {
      status: 'error', code: 'unverified', slots: null, mode: null,
      path: OPTIMIZER_INVENTORY_PATH, note: null, signature: null,
      error: 'the source audit did not report its engine-read files',
    };
  }
  const record = payload.also_read.find((entry) => {
    const path = normalizedPath(entry?.path);
    return path === OPTIMIZER_INVENTORY_PATH || path.endsWith(`/${OPTIMIZER_INVENTORY_PATH}`);
  });
  // This input is optional. A successful audit that does not list it verifies
  // the identity/no-placement mode; a failed or malformed audit never does.
  if (!record) {
    return {
      status: 'ready', code: 'absent', slots: 0, mode: 'identity',
      path: OPTIMIZER_INVENTORY_PATH, note: null, error: null, signature: 'absent',
    };
  }
  const slots = Number(record.yielded_items);
  const common = {
    path: normalizedPath(record.path) || OPTIMIZER_INVENTORY_PATH,
    note: record.note && typeof record.note === 'object' ? record.note : null,
    signature: [record.modified, record.size, record.read_state, record.yielded_items].join('|'),
  };
  if (record.read_state === 'read_yielding_nothing') {
    return { ...common, status: 'blocked', code: 'empty', slots: 0, mode: null, error: null };
  }
  if (record.read_state === 'read_yielding' && Number.isInteger(slots) && slots > 0) {
    return { ...common, status: 'ready', code: 'slots', slots, mode: 'inventory', error: null };
  }
  return {
    ...common, status: 'error', code: 'unverified', slots: null, mode: null,
    error: 'the source audit did not report a usable inventory read state and slot count',
  };
}

const INITIAL = {
  status: 'loading', code: 'checking', slots: null, mode: null,
  path: OPTIMIZER_INVENTORY_PATH, note: null, error: null, signature: null,
};

export function usePlanInventoryReadiness() {
  const [readiness, setReadiness] = useState(INITIAL);

  const verify = useCallback(async (options = {}) => {
    // Automatic comparison preparation verifies immediately before its stream.
    // Keep the last ready snapshot mounted during that quiet read so the read
    // does not cancel its own effect; an operator-triggered retry still shows
    // the checking state.
    if (options?.announce !== false) {
      setReadiness((current) => ({ ...current, status: 'loading', code: 'checking', error: null }));
    }
    const result = await readFiles();
    const next = result.ok
      ? inventoryReadinessFromFiles(result.data)
      : { ...INITIAL, status: 'error', code: 'unverified', error: result.error || 'the source audit could not be read' };
    setReadiness(next);
    return next;
  }, []);

  useEffect(() => { verify(); }, [verify]);

  return { ...readiness, verify, retry: verify };
}

export default usePlanInventoryReadiness;
