import { API_BASE } from './api';
import { finiteNumber } from './format';

// The shell owns only confirmed settings persistence. Plan owns every preview
// and rewrite, beside its readiness checks and consequence review.
export function createPlanActions({
  settings,
  setSettings,
  notify,
  setRefreshKey,
  setSaveState,
  setApplyWeightState,
  settingsAvailable,
}) {
  async function persistSettings(nextSettings) {
    if (!settingsAvailable) {
      setSaveState('error');
      notify(
        'Saved settings are unavailable. Nothing was written; refresh and try again.',
        'ההגדרות השמורות אינן זמינות. דבר לא נכתב; יש לרענן ולנסות שוב.',
      );
      return { ok: false, data: null };
    }
    setSaveState('saving');
    try {
      const response = await fetch(`${API_BASE}/api/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(nextSettings),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const saved = await response.json();
      setSettings(saved);
      setSaveState('saved');
      // Bump the refresh key so dependent views refetch against the saved state
      // instead of leaving stale numbers behind a success toast.
      setRefreshKey((k) => k + 1);
      window.setTimeout(() => setSaveState('idle'), 1800);
      return { ok: true, data: saved };
    } catch {
      setSaveState('error');
      notify(
        'Settings were not saved. The last confirmed values remain in effect.',
        'ההגדרות לא נשמרו. הערכים האחרונים שאושרו נשארו בתוקף.',
      );
      return { ok: false, data: null };
    }
  }

  async function handleApplyFrontierFloor(floor) {
    const nextFloor = finiteNumber(floor);
    if (nextFloor === null) return;
    setApplyWeightState('saving');
    try {
      const result = await persistSettings({ ...settings, min_retention_floor: nextFloor });
      if (!result.ok) return false;
      const pct = Math.round(nextFloor * 100);
      notify(
        `Saved retention floor set to ${pct} percent.`,
        `רף השימור השמור עודכן ל־${pct} אחוז.`,
      );
      return true;
    } finally {
      setApplyWeightState('idle');
    }
  }

  return {
    persistSettings,
    handleApplyFrontierFloor,
  };
}
