import React, { Suspense, lazy } from 'react';
import { LoadingState } from '../studio';
import { pageText } from '../shell/format';
import { AdvertiserRecordsPanel } from './AdvertiserRecordsPanel';
import { AgencyRecordsPanel } from './AgencyRecordsPanel';
import CampaignBoard from './CampaignBoard';
import CampaignRollupPanel from './CampaignRollupPanel';
import ClientTree from './ClientTree';
import MoneyBoard from './MoneyBoard';
import PacingWorkspace from './pacing/PacingWorkspace';

// The seven panels of the Commercial destination, and the tab chrome each one
// sits in.
//
// Split out of ClientsWorkspace.jsx to keep that file inside the project's
// file-size law, for the same reason ClientsChrome.jsx was: this module holds no
// state, performs no read and makes no decision. Everything it renders arrives as
// a prop, so the workspace keeps every decision and this file keeps the markup.
//
// Only the ACTIVE panel mounts. That is a rule rather than an optimization: a
// hidden panel that mounts still fetches, still runs a Data Grid, and still
// answers to the keyboard, and the trade-agreement panel behind a closed tab
// would otherwise be reading agreements and measuring obligations for nobody.

// The trade-agreement surface is the heaviest context this destination holds - a
// PDF pane, the term vocabulary and the whole review machinery - and most sessions
// never open it, so it loads on demand instead of riding in the Commercial chunk.
const AgreementsPanel = lazy(() => import('../trade/AgreementsPanel'));

function Panel({ view, children }) {
  return (
    <div
      id={`commercial-panel-${view}`}
      role="tabpanel"
      aria-labelledby={`commercial-tab-${view}`}
      tabIndex={0}
    >
      {children}
    </div>
  );
}

export default function ClientsPanels({
  active, locale, copy, notify, gate, refreshKey, data, open, on,
}) {
  return (
    <div className="clients-main">
      {active === 'clients' ? (
        <Panel view="clients">
          <ClientTree tree={data.tree} locale={locale} onOpenClient={on.openClient} />
        </Panel>
      ) : null}

      {active === 'money' ? (
        <Panel view="money">
          <MoneyBoard
            money={data.money}
            locale={locale}
            drill={data.drill}
            onDrill={on.drill}
            onOpenClient={on.openClient}
            openers={{
              agencyIds: data.agencyIds,
              campaignIds: data.campaignIds,
              onOpenAgency: on.openAgency,
              onOpenCampaignRecord: on.openCampaign,
            }}
          />
        </Panel>
      ) : null}

      {active === 'campaigns' ? (
        <Panel view="campaigns">
          <CampaignBoard
            board={data.board}
            locale={locale}
            notify={notify}
            gate={gate}
            agencies={data.agencies}
            openCampaignId={open.campaignId}
            onOpened={on.campaignOpened}
            onOnboard={on.onboard}
            onOpenClient={on.openClient}
            onOpenAgency={on.openAgency}
            onReload={on.reload}
          />
          <CampaignRollupPanel campaigns={data.campaigns} locale={locale} refreshKey={refreshKey} />
        </Panel>
      ) : null}

      {/* The delivery pace of what the campaigns view booked. It sits beside
          campaigns rather than under a destination of its own because it answers a
          question about the same object: what was promised, and what has aired
          against it. */}
      {active === 'pacing' ? (
        <Panel view="pacing">
          <PacingWorkspace
            locale={locale}
            notify={notify}
            refreshKey={refreshKey}
            onOpenCampaign={on.openCampaign}
          />
        </Panel>
      ) : null}

      {active === 'advertisers' ? (
        <Panel view="advertisers">
          <AdvertiserRecordsPanel
            copy={copy}
            locale={locale}
            notify={notify}
            refreshKey={refreshKey}
            openAdvertiserId={open.ruleId}
            onOpened={on.ruleOpened}
            onGlobalRefresh={on.globalRefresh}
          />
        </Panel>
      ) : null}

      {active === 'agencies' ? (
        <Panel view="agencies">
          <AgencyRecordsPanel
            copy={copy}
            locale={locale}
            notify={notify}
            setActiveView={on.setActiveView}
            openAgencyId={open.agencyId}
            onOpened={on.agencyOpened}
            onGlobalRefresh={on.globalRefresh}
          />
        </Panel>
      ) : null}

      {/* The signed commercial agreements behind every rule the engine applies.
          The lazy chunk's fallback names what is loading rather than showing a
          bare spinner, because on a cold cache this is a real wait. */}
      {active === 'agreements' ? (
        <Panel view="agreements">
          <Suspense
            fallback={(
              <LoadingState
                title={pageText(locale, 'Loading the agreements surface', 'טוען את מסך ההסכמים')}
                description={pageText(
                  locale,
                  'The review machinery loads on demand, so it is fetched the first time this tab is opened.',
                  'מכלול הסקירה נטען לפי דרישה, ולכן הוא מובא בפעם הראשונה שהלשונית נפתחת.',
                )}
              />
            )}
          >
            <AgreementsPanel
              locale={locale}
              notify={notify}
              canEdit={gate.canEdit}
              editRefusal={gate.reason}
              refreshKey={refreshKey}
            />
          </Suspense>
        </Panel>
      ) : null}
    </div>
  );
}
