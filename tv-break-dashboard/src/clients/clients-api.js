// The Clients destination's own reads and writes, in one place.
//
// Every call returns a discriminated state rather than throwing, because a
// failed read and an empty result are different things on a money surface and
// the surface has to say which one happened. `status` is 'ready', 'empty' or
// 'error'; a caller never has to guess from a falsy value.

import { API_BASE } from '../shell/api';

async function readJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

// A refusal in both languages. The campaign endpoints send `detail` as an
// `{message_en, message_he}` pair, because a Hebrew flow that refuses in English
// is a refusal the person it is addressed to cannot act on. An endpoint that
// still sends a plain string is carried through unchanged and reads the same in
// both, which is honest: it is one sentence and this layer will not invent a
// translation for it.
function refusalPair(payload, response) {
  const detail = payload ? payload.detail : null;
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const en = String(detail.message_en || detail.message || '');
    return { en, he: String(detail.message_he || en) };
  }
  const text = detail ? String(detail) : `${response.status} ${response.statusText}`;
  return { en: text, he: text };
}

async function sendJson(path, method, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const pair = refusalPair(payload, response);
    const error = new Error(pair.en);
    error.messageEn = pair.en;
    error.messageHe = pair.he;
    error.status = response.status;
    throw error;
  }
  return payload;
}

export async function loadClients() {
  return readJson('/api/clients');
}

export async function loadMoney() {
  return readJson('/api/clients/money');
}

export async function loadCampaigns() {
  return readJson('/api/clients/campaigns');
}

export async function loadRollup() {
  return readJson('/api/campaigns');
}

// The pricing store, read whole. A client's own rule is the row whose name cell
// carries the client's name, which is a join this destination performs itself
// because the rules read is keyed on the row and the client read is keyed on the
// advertiser.
export async function loadAdvertiserRules() {
  return readJson('/api/advertisers');
}

// Who each advertiser is, what it is bound to, and what it earned, from the one
// read that already joins the name space, the rules store and the daily ledger.
export async function loadAdvertiserIdentity() {
  return readJson('/api/advertisers/identity');
}

// Create the pricing row for a client. The name is what binds it: a row without
// one prices nothing, which is the state all forty five shipped rows are in.
export async function createAdvertiserRule(payload) {
  return sendJson('/api/advertisers', 'POST', payload);
}

// Edit one pricing row. Used from the client record for the two identity fields
// only; the premium and the scoped rules are edited on the rule card itself.
export async function updateAdvertiserRule(advertiserId, payload) {
  return sendJson(`/api/advertisers/${encodeURIComponent(advertiserId)}`, 'PUT', payload);
}

export async function loadOnboardingOptions() {
  return readJson('/api/clients/onboarding/options');
}

export async function onboardClient(payload) {
  return sendJson('/api/clients/onboarding', 'POST', payload);
}

export async function createCampaign(payload) {
  return sendJson('/api/clients/campaigns', 'POST', payload);
}

export async function updateCampaign(campaignId, payload) {
  return sendJson(`/api/clients/campaigns/${encodeURIComponent(campaignId)}`, 'PUT', payload);
}

export async function endCampaign(campaignId) {
  return sendJson(`/api/clients/campaigns/${encodeURIComponent(campaignId)}/deactivate`, 'POST');
}

export async function addFlight(campaignId, payload) {
  return sendJson(`/api/clients/campaigns/${encodeURIComponent(campaignId)}/flights`, 'POST', payload);
}

export async function updateFlight(campaignId, flightId, payload) {
  const campaign = encodeURIComponent(campaignId);
  const flight = encodeURIComponent(flightId);
  return sendJson(`/api/clients/campaigns/${campaign}/flights/${flight}`, 'PUT', payload);
}

export async function removeFlight(campaignId, flightId) {
  const campaign = encodeURIComponent(campaignId);
  const flight = encodeURIComponent(flightId);
  return sendJson(`/api/clients/campaigns/${campaign}/flights/${flight}`, 'DELETE');
}
