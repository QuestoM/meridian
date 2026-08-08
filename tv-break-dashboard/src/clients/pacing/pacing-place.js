// Clients, pacing: the reader's place on the board, kept across leaving it.
//
// A campaign name on a row opens that campaign, which is a different view of the
// destination this panel is mounted in, so this panel unmounts and its state
// goes with it. Measured: clicking the name on the first row switched the
// destination tab from pacing to campaigns, and coming back cost a click on the
// Pacing tab and a rescroll through 56 rows to find the row that was left. The
// outward trip was built a round ago and the return trip was not.
//
// So the campaign a reader left by is written down, and the next time this panel
// mounts it reads it once and focuses that row. It is taken rather than read, so
// a reader who comes back a second time is not sent anywhere they did not ask to
// go, and it lives in sessionStorage rather than in a store because it is about
// one browser tab in one sitting and it is not data about the business.
//
// Storage is not always there. A private window, a locked-down profile or a
// server-side render all answer by throwing, and a place marker is never worth a
// blank screen, so every access is guarded and a failure is the same answer as
// nowhere to go back to.

const KEY = 'meridian.pacing.return-to-campaign';

function store() {
  try {
    return typeof sessionStorage === 'undefined' ? null : sessionStorage;
  } catch (error) {
    return null;
  }
}

export function rememberCampaign(campaignId) {
  const held = store();
  if (!held) return;
  try {
    held.setItem(KEY, String(campaignId || ''));
  } catch (error) {
    // A place marker is not worth an exception on the way out of a screen.
  }
}

// The campaign to return to, once. Reading it clears it, because a mark that
// survives its own use would drag a reader back to the same row every time they
// opened this board for the rest of the sitting.
export function takeRememberedCampaign() {
  const held = store();
  if (!held) return '';
  try {
    const found = String(held.getItem(KEY) || '');
    if (found) held.removeItem(KEY);
    return found;
  } catch (error) {
    return '';
  }
}
