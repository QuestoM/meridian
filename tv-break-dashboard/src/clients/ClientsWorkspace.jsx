import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WALLS, fetchSession, payloadCanEdit } from '../session';
import { AdvertiserRecordsPanel } from './AdvertiserRecordsPanel';
import { AgencyRecordsPanel } from './AgencyRecordsPanel';
import CampaignBoard from './CampaignBoard';
import CampaignRollupPanel from './CampaignRollupPanel';
import ClientRecord from './ClientRecord';
import ClientTree from './ClientTree';
import { ClientsHeader, ClientsLoadFailure, ClientsViewStrip, VIEW_LABELS } from './ClientsChrome';
import MoneyBoard from './MoneyBoard';
import OnboardClientFlow from './OnboardClientFlow';
import PacingWorkspace from './pacing/PacingWorkspace';
import {
  createAdvertiserRule,
  loadAdvertiserRules,
  loadCampaigns,
  loadClients,
  loadMoney,
  updateAdvertiserRule,
} from './clients-api';
import { campaignDeliveryIndex } from './delivery-helpers';
import { joinAliases, splitAliases } from './clients-rule-helpers';
import {
  NO_DRILL,
  RECORD_PARAM,
  VIEW_PARAM,
  agencyIdsByName,
  agencyIndex,
  campaignIdsByName,
  flattenClients,
  initialView,
  ledgerCampaignKeys,
  moneyTarget,
  readParam,
  refusalText,
  requestedView,
  writeParams,
} from './clients-money-helpers';
import './clients-workspace.css';
import './clients-tree.css';
import './clients-record.css';
import './clients-rule-card.css';
import './clients-delivery.css';
import { isolate } from '../shell/bidi';

// Clients: one destination for the commercial spine, agency to advertiser to
// campaign to flight, reached from any of the three navigation entries that
// used to be three separate pages over the same four objects.
//
// The view is a control inside the content rather than a destination of its
// own, and it lives in the query string so a view and an open record are both
// addressable without touching the hash, which is the shell's router. The two
// record panels that existed before this destination are rendered unchanged, so
// nothing reachable in one click before is further away now.
//
// A navigation entry names the view it opens on, and it wins over the view the
// control last stored, or pressing an entry would leave the previous panel on
// screen while the chrome moved. Only an address someone supplied outranks it.

export default function ClientsWorkspace({
  view = 'clients',
  campaigns,
  copy,
  locale = 'he',
  notify = () => {},
  onGlobalRefresh = () => {},
  setActiveView,
  refreshKey = 0,
}) {
  const [active, setActive] = useState(() => initialView(view));
  const [tree, setTree] = useState(null);
  const [money, setMoney] = useState(null);
  const [board, setBoard] = useState(null);
  // The pricing store, read whole so a record can find its own rule without a
  // second round trip. Null means not read yet, which the record says out loud
  // rather than showing an unbound state it has not verified.
  const [ruleRows, setRuleRows] = useState(null);
  const [ruleBusy, setRuleBusy] = useState(false);
  // Which pricing row the records tab should open on. It is set only by a
  // control that already knows the row exists, so that tab is never entered
  // pointing at a card that cannot be there.
  const [openRuleId, setOpenRuleId] = useState('');
  // The same device for the agency records tab, so the agency named on a client
  // record opens as that record rather than as a grid to search again.
  const [openAgencyId, setOpenAgencyId] = useState('');
  // And for the campaign board, so a figure headed by a campaign name opens the
  // booking that carries it instead of naming it.
  const [openCampaignId, setOpenCampaignId] = useState('');
  // An array of section names that failed to load, or an empty array when all
  // loaded. Null while the initial fetch is still in flight.
  const [failed, setFailed] = useState(null);
  const [openClient, setOpenClient] = useState(() => readParam(RECORD_PARAM));
  // Which row the money board has open. It lives here because two surfaces open
  // it: the board's own rows, and a client record asking for its own money.
  const [drill, setDrill] = useState(NO_DRILL);
  const [onboarding, setOnboarding] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [session, setSession] = useState(null);
  const opened = useRef(view);
  const he = locale === 'he';

  const reload = useCallback(() => setReloadKey((key) => key + 1), []);

  // A navigation entry that names a view wins over whatever the view control
  // last stored. The first run is skipped because the mount has already
  // resolved the view, and skipping it is what keeps a supplied address open.
  useEffect(() => {
    if (opened.current === view) {
      return;
    }
    opened.current = view;
    setActive(requestedView(view));
  }, [view]);

  useEffect(() => {
    let alive = true;
    fetchSession().then((result) => {
      if (alive) setSession(result.session);
    });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    setFailed(null);
    Promise.allSettled([loadClients(), loadMoney(), loadCampaigns(), loadAdvertiserRules()]).then((results) => {
      if (!alive) {
        return;
      }
      const [clients, ledger, booked, rules] = results;
      if (clients.status === 'fulfilled') setTree(clients.value);
      if (ledger.status === 'fulfilled') setMoney(ledger.value);
      if (booked.status === 'fulfilled') setBoard(booked.value);
      if (rules.status === 'fulfilled') setRuleRows(rules.value.advertisers || []);
      // VIEW_LABELS[0..3] match the four Promise slots in order (clients,
      // money, campaigns, advertisers). agencies is the fifth label and has
      // no slot, so it is never in the broken list.
      const brokenNames = results
        .map((result, i) => (result.status === 'rejected' ? VIEW_LABELS[i] : null))
        .filter(Boolean);
      setFailed(brokenNames);
    });
    return () => { alive = false; };
  }, [locale, refreshKey, reloadKey]);

  useEffect(() => {
    writeParams({ [VIEW_PARAM]: active, [RECORD_PARAM]: openClient });
  }, [active, openClient]);

  const rows = useMemo(
    () => (tree
      ? flattenClients(tree.agencies || [], tree.unlinked || [], tree.clients_booked_without_spots || [])
      : []),
    [tree],
  );
  // The campaign board stores an agency id and the tree holds the name for it,
  // so the name is resolved once here rather than printed as a key on a row.
  const agencies = useMemo(() => agencyIndex(tree), [tree]);
  // The two indexes the money board's head needs, because the ledger knows an
  // agency and a campaign by the name in the daily file while both records are
  // keyed by an id. They are built here, once, from the reads this surface
  // already holds, and a name that resolves to nothing simply is not in them.
  const agencyIds = useMemo(() => agencyIdsByName(tree), [tree]);
  const campaignIds = useMemo(() => campaignIdsByName(board), [board]);
  // Which campaign names the ledger holds a row for, so a record only offers to
  // open the ones that really open something.
  const ledgerCampaigns = useMemo(() => ledgerCampaignKeys(money), [money]);
  // The delivery ledger the campaign board already read, keyed by campaign, so a
  // client record shows what its campaigns delivered from the same read the
  // board shows it from. Two reads of the same ledger could drift; one cannot.
  const deliveryByCampaign = useMemo(() => campaignDeliveryIndex(board), [board]);
  const record = openClient ? rows.find((row) => row.advertiser === openClient) : null;
  // The endpoint is the authority on whether this account may change anything
  // here, so a read-only account sees the reason instead of a control that
  // would answer 403 after it was pressed.
  const gate = payloadCanEdit(board, session, WALLS.readOnlyRole);

  // The record's money control opens that client's own rows. A client with no
  // row in the ledger opens nothing at all: its record already prints the reason
  // there is no money, and a board that cannot contain the row would answer with
  // the money of whoever leads the ranking instead.
  function openMoneyFor(advertiser) {
    const target = moneyTarget(rows.find((row) => row.advertiser === advertiser));
    if (!target) {
      return;
    }
    setDrill(target);
    setActive('money');
    setOpenClient(advertiser);
  }

  // The campaign the daily file carries for this client, opened as the row the
  // money board already holds for it. The record lists what aired; this is the
  // money behind one of those names, which is the same ledger re-grouped.
  function openCampaignMoney(name) {
    setDrill({ group: 'campaigns', key: String(name) });
    setActive('money');
  }

  // The rule card, opened on the row the record already resolved. The records
  // tab is only ever entered this way with an id in hand, so the reader lands on
  // one named card rather than on the whole store.
  function openRuleCard(advertiserId) {
    setOpenRuleId(advertiserId);
    setActive('advertisers');
  }

  // The agency behind the client, opened as its own record with its terms, its
  // contacts and its linked advertisers, rather than as a line that names it.
  function openAgencyRecord(agencyId) {
    setOpenAgencyId(agencyId);
    setActive('agencies');
  }

  // The booked campaign behind a name, opened on the board that holds its terms
  // and its flights. The board is told which row to open the same way the two
  // record panels are, so the reader lands on one campaign rather than on the
  // whole list to search again.
  function openCampaignRecord(campaignId) {
    setOpenCampaignId(campaignId);
    setActive('campaigns');
  }

  // A refusal that names an object hands over that object's address, so the
  // control beside the sentence goes to the thing the sentence names. The flow
  // closes first, because what it names is behind it.
  function openRefused(target) {
    setOnboarding(null);
    if (target.kind === 'agency') {
      openAgencyRecord(target.id);
    }
    if (target.kind === 'campaign') {
      openCampaignRecord(target.id);
    }
  }

  async function refreshRules() {
    const payload = await loadAdvertiserRules();
    setRuleRows(payload.advertisers || []);
  }

  // Create the client's pricing row. The name is what binds it, so the client is
  // priced by it from the next daily pricing run, and the tree is re-read
  // because the client's premium and its bound state both come from that read.
  async function createRuleFor(draft) {
    setRuleBusy(true);
    try {
      await createAdvertiserRule(draft);
      await refreshRules();
      reload();
      onGlobalRefresh();
      notify(
        `Pricing rule ${draft.advertiser_id} created for ${draft.name}.`,
        `כלל התמחור ${isolate(draft.advertiser_id)} נוצר עבור ${isolate(draft.name)}.`,
      );
    } catch (error) {
      notify(
        `The pricing rule could not be created. ${refusalText(error, 'en')}`,
        `לא ניתן היה ליצור את כלל התמחור. ${isolate(refusalText(error, 'he'))}`,
      );
    } finally {
      setRuleBusy(false);
    }
  }

  // Add a spelling to the client's own pricing row. The store refuses a spelling
  // another row already holds, and that refusal is surfaced verbatim rather than
  // being reported as a generic failure.
  async function addSpellingFor(row, spelling) {
    setRuleBusy(true);
    try {
      await updateAdvertiserRule(row.advertiser_id, {
        aliases: joinAliases([...splitAliases(row.aliases), spelling]),
      });
      await refreshRules();
      reload();
      notify(
        `${spelling} is now priced as ${row.name}.`,
        `${isolate(spelling)} מתומחר מעכשיו כ${isolate(row.name)}.`,
      );
    } catch (error) {
      notify(
        `The spelling could not be added. ${refusalText(error, 'en')}`,
        `לא ניתן היה להוסיף את הכתיב. ${isolate(refusalText(error, 'he'))}`,
      );
    } finally {
      setRuleBusy(false);
    }
  }

  return (
    <section className="page-workspace clients-workspace">
      <ClientsHeader locale={locale} gate={gate} onOnboard={() => setOnboarding({})} />
      <ClientsViewStrip locale={locale} active={active} onSelect={setActive} />
      <ClientsLoadFailure locale={locale} failed={failed} onRetry={reload} />

      <div className="clients-body">
        <div className="clients-main">
          {active === 'clients' ? (
            <ClientTree
              tree={tree}
              locale={locale}
              canEdit={gate.canEdit}
              onOpenClient={setOpenClient}
              onOnboard={() => setOnboarding({})}
            />
          ) : null}
          {active === 'money' ? (
            <MoneyBoard
              money={money}
              locale={locale}
              drill={drill}
              onDrill={setDrill}
              onOpenClient={setOpenClient}
              openers={{
                agencyIds,
                campaignIds,
                onOpenAgency: openAgencyRecord,
                onOpenCampaignRecord: openCampaignRecord,
              }}
            />
          ) : null}
          {active === 'campaigns' ? (
            <>
              <CampaignBoard
                board={board}
                locale={locale}
                notify={notify}
                gate={gate}
                agencies={agencies}
                openCampaignId={openCampaignId}
                onOpened={() => setOpenCampaignId('')}
                onOnboard={() => setOnboarding({})}
                onOpenClient={setOpenClient}
                onOpenAgency={openAgencyRecord}
                onReload={reload}
              />
              <CampaignRollupPanel campaigns={campaigns} locale={locale} refreshKey={refreshKey} />
            </>
          ) : null}
          {/* The delivery pace of what the campaigns view booked. It sits beside
              campaigns rather than under a destination of its own because it
              answers a question about the same object: what was promised, and
              what has aired against it. */}
          {active === 'pacing' ? (
            <PacingWorkspace
              locale={locale}
              notify={notify}
              refreshKey={refreshKey}
              onOpenCampaign={openCampaignRecord}
            />
          ) : null}
          {active === 'advertisers' ? (
            <AdvertiserRecordsPanel
              copy={copy}
              locale={locale}
              notify={notify}
              openAdvertiserId={openRuleId}
              onOpened={() => setOpenRuleId('')}
              onGlobalRefresh={onGlobalRefresh}
            />
          ) : null}
          {active === 'agencies' ? (
            <AgencyRecordsPanel
              copy={copy}
              locale={locale}
              notify={notify}
              setActiveView={setActiveView}
              openAgencyId={openAgencyId}
              onOpened={() => setOpenAgencyId('')}
              onGlobalRefresh={onGlobalRefresh}
            />
          ) : null}
        </div>

        {record ? (
          <ClientRecord
            client={record}
            rows={rows}
            locale={locale}
            basis={tree ? tree.basis : null}
            deliveryByCampaign={deliveryByCampaign}
            airStates={board && board.delivery ? board.delivery.air_state_vocabulary : []}
            statuses={board ? board.status_vocabulary : []}
            goalWords={board ? board.goal_kind_vocabulary : []}
            canEdit={gate.canEdit}
            ruleRows={ruleRows}
            ruleBusy={ruleBusy}
            editRefusal={gate.reason}
            ledgerCampaigns={ledgerCampaigns}
            onClose={() => setOpenClient('')}
            onStep={(next) => next && setOpenClient(next)}
            onOpenMoney={openMoneyFor}
            onCreateRule={createRuleFor}
            onAddSpelling={addSpellingFor}
            onOpenRuleCard={openRuleCard}
            onBookCampaign={() => setOnboarding({ advertiser: record.advertiser, agencyId: record.agency_id })}
            onOpenAgency={openAgencyRecord}
            onOpenCampaignMoney={openCampaignMoney}
          />
        ) : null}

        {onboarding ? (
          <OnboardClientFlow
            locale={locale}
            prefill={onboarding}
            onOpenRefused={openRefused}
            onClose={() => setOnboarding(null)}
            onDone={(advertiser) => {
              setOnboarding(null);
              setActive('clients');
              setOpenClient(advertiser);
              reload();
              onGlobalRefresh();
            }}
          />
        ) : null}
      </div>
    </section>
  );
}
