import { useCallback } from 'react';
import { optimizationBlockedReason } from './PlanActionSafety';

function sameInventory(review, checked) {
  return review?.inventoryMode === checked.mode
    && Number(review?.inventorySlots) === Number(checked.slots)
    && String(review?.inventoryPath) === String(checked.path)
    && String(review?.inventorySignature) === String(checked.signature);
}

export function usePlanOptimizationActions({ surface, inventory, locale, notify, go, setReview }) {
  const optimizationAllowed = surface.settingsReady && inventory.status === 'ready';
  const blockedReason = optimizationBlockedReason(surface.settingsState, inventory, locale);

  const blockedNotice = useCallback((checked = inventory) => {
    notify?.(
      `Run and comparison are locked: ${optimizationBlockedReason(surface.settingsState, checked, 'en')}.`,
      `ההרצה וההשוואה נעולות: ${optimizationBlockedReason(surface.settingsState, checked, 'he')}.`,
    );
  }, [notify, surface.settingsState, inventory]);

  const verify = useCallback(async () => {
    if (!surface.settingsReady) {
      blockedNotice();
      return null;
    }
    const checked = await inventory.verify();
    if (checked.status !== 'ready') {
      blockedNotice(checked);
      return null;
    }
    return checked;
  }, [surface.settingsReady, inventory, blockedNotice]);

  const runNow = useCallback(async () => {
    go('run');
    const checked = await verify();
    if (!checked) return false;
    setReview({
      kind: 'run',
      inventorySlots: checked.slots,
      inventoryMode: checked.mode,
      inventoryPath: checked.path,
      inventorySignature: checked.signature,
    });
    return true;
  }, [go, verify, setReview]);

  const compareNow = useCallback(async () => {
    go('compare');
    const checked = await verify();
    if (!checked) return false;
    return surface.compare(checked);
  }, [go, verify, surface]);

  const confirmRun = useCallback(async (review) => {
    const checked = await verify();
    if (!checked) return false;
    if (!sameInventory(review, checked)) {
      notify?.(
        'The optimizer inventory changed after the review. Review the weekly rewrite again before running it.',
        'מלאי האופטימייזר השתנה לאחר הבדיקה. יש לבדוק שוב את כתיבת השבוע לפני ההרצה.',
      );
      return false;
    }
    return surface.runPlan();
  }, [verify, notify, surface]);

  return { optimizationAllowed, blockedReason, runNow, compareNow, confirmRun };
}

export default usePlanOptimizationActions;
