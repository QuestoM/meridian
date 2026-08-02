// Every write the advertiser records panel performs, in one place.
//
// Split out of AdvertiserRecordsPanel.jsx to keep that file inside the project's
// file-size law, and named for its parent so the pair is obvious. Nothing here
// holds state: the factory takes the panel's context and returns the handlers,
// so the panel keeps every decision and this module keeps every request. The
// messages are the ones the panel already showed, unchanged.

import { toConditionPayload, toPayload } from './advertisers-helpers';

function conditionUrl(base, advertiserId, ruleId) {
  const advertiser = encodeURIComponent(advertiserId);
  const rule = ruleId === undefined ? '' : `/${encodeURIComponent(ruleId)}`;
  return `${base}/api/advertisers/${advertiser}/conditions${rule}`;
}

async function send(url, method, body) {
  const response = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response;
}

export function recordWrites({ base, notify, reload, refreshGlobal, rowsRef, pending, setPending, setOpenId, setShowAdd }) {
  async function after() {
    await reload();
    refreshGlobal?.();
  }

  async function saveBaseline(draft) {
    try {
      await send(`${base}/api/advertisers/${encodeURIComponent(draft.advertiser_id)}`, 'PUT', toPayload(draft));
      notify(`Advertiser ${draft.advertiser_id} saved.`, `המפרסם ${draft.advertiser_id} נשמר.`);
      await after();
    } catch (error) {
      notify(`Save failed for ${draft.advertiser_id} (${error.message}).`, `השמירה נכשלה עבור ${draft.advertiser_id} (${error.message}).`);
    }
  }

  async function create(draft) {
    try {
      const response = await fetch(`${base}/api/advertisers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ advertiser_id: draft.advertiser_id, ...toPayload(draft) }),
      });
      if (response.status === 409) {
        notify(`Advertiser ${draft.advertiser_id} already exists.`, `המפרסם ${draft.advertiser_id} כבר קיים.`);
        return false;
      }
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${draft.advertiser_id} created.`, `המפרסם ${draft.advertiser_id} נוצר.`);
      setShowAdd(false);
      await after();
      return true;
    } catch (error) {
      notify(`Create failed (${error.message}).`, `היצירה נכשלה (${error.message}).`);
      return false;
    }
  }

  // Two-step delete interlock: the first confirmed click arms the id and warns
  // that the advertiser's scoped rules die with it; only the second deletes.
  async function remove(advertiserId) {
    if (pending !== advertiserId) {
      setPending(advertiserId);
      const row = (rowsRef() || []).find((entry) => entry.advertiser_id === advertiserId) || null;
      const ruleCount = Array.isArray(row?.conditions) ? row.conditions.length : 0;
      if (ruleCount > 0) {
        const rulesEn = ruleCount === 1 ? 'its one scoped rule' : `its ${ruleCount} scoped rules`;
        const rulesHe = ruleCount === 1 ? 'הכלל הממוקד שלו' : `${ruleCount} הכללים הממוקדים שלו`;
        notify(`Deleting ${advertiserId} also deletes ${rulesEn}. Select confirm delete again to proceed.`, `מחיקת ${advertiserId} תמחק גם את ${rulesHe}. לחצו שוב על אישור המחיקה כדי להמשיך.`);
      } else {
        notify(`Select confirm delete again to delete ${advertiserId}.`, `לחצו שוב על אישור המחיקה כדי למחוק את ${advertiserId}.`);
      }
      return;
    }
    setPending(null);
    try {
      await send(`${base}/api/advertisers/${encodeURIComponent(advertiserId)}`, 'DELETE');
      notify(`Advertiser ${advertiserId} deleted. It can be restored from the Restore changes page.`, `המפרסם ${advertiserId} נמחק. ניתן לשחזר מעמוד שחזור שינויים.`);
      setOpenId(null);
      await after();
    } catch (error) {
      notify(`Delete failed for ${advertiserId} (${error.message}).`, `המחיקה נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  async function createCondition(advertiserId, draft) {
    try {
      const ruleId = `rule_${Date.now().toString(36)}`;
      await send(conditionUrl(base, advertiserId), 'POST', { rule_id: ruleId, ...toConditionPayload(draft) });
      notify(`Scoped rule added to ${advertiserId}.`, `כלל ממוקד נוסף ל${advertiserId}.`);
      await after();
      return true;
    } catch (error) {
      notify(`Add rule failed for ${advertiserId} (${error.message}).`, `הוספת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
      return false;
    }
  }

  async function updateCondition(advertiserId, ruleId, draft) {
    try {
      await send(conditionUrl(base, advertiserId, ruleId), 'PUT', toConditionPayload(draft));
      notify(`Scoped rule saved for ${advertiserId}.`, `כלל ממוקד נשמר עבור ${advertiserId}.`);
      await after();
      return true;
    } catch (error) {
      notify(`Save rule failed for ${advertiserId} (${error.message}).`, `שמירת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
      return false;
    }
  }

  async function deleteCondition(advertiserId, ruleId) {
    try {
      await send(conditionUrl(base, advertiserId, ruleId), 'DELETE');
      notify(`Scoped rule removed from ${advertiserId}.`, `כלל ממוקד הוסר מ${advertiserId}.`);
      await after();
    } catch (error) {
      notify(`Delete rule failed for ${advertiserId} (${error.message}).`, `מחיקת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  return { saveBaseline, create, remove, createCondition, updateCondition, deleteCondition };
}
