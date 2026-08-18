import React, { useEffect, useState } from 'react';
import { Button } from '../studio/actions';
import { Code, Figure } from '../shell/bidi';
import { CalendarRange, MousePointerClick, PanelRightClose, Plus } from 'lucide-react';
import { pageText } from '../shell/format';
import { refusalText, vocabularyLabel, vocabularyRemedy, windowLabel } from './clients-money-helpers';
import { endCampaign, loadOnboardingOptions } from './clients-api';
import CampaignDetail from './CampaignDetail';
import CampaignTerms from './CampaignTerms';
import { DeliveryCell } from './DeliveryState';
import { DeliveryBasis } from './DeliveryBasisNotes';
import DemoBadge from './DemoBadge';
import './clients-campaigns.css';
import { isolate } from '../shell/bidi';

const CAMPAIGN_WINDOW = 12;

// Every campaign booked in this product, with its flights, what each flight
// committed to, and what the delivery ledger has counted against it. Google Ads
// puts budget on exactly one layer and states the containment; this does the
// same with the two layers that exist here, and it carries the third column that
// surface has, because this product does derive delivery, per campaign and per
// broadcast day, from the traffic log.
//
// The delivered column is a state and a figure at once, never a figure alone: a
// campaign whose flight still holds days with no per-spot source reads as a
// floor and says how many of its days were counted. When no day of any campaign
// carries a source, the same column is the unknown state and the block under the
// board names the missing feed and the path that would supply it.
//
// Every row is a way in rather than a readout. The campaign name opens its
// terms and its flights and both can be changed there, the client name opens
// the client, and the agency reads as its name with its id underneath, because
// AGY_10 is a database key and not something a person calls anybody. The agency
// cell is a control too: the name it prints is a record with terms, contacts and
// linked clients, and the client record opens exactly that record from the line
// that names the same agency.

export default function CampaignBoard({
  board,
  locale,
  notify,
  gate,
  agencies = {},
  openCampaignId = '',
  onOpened = () => {},
  onOnboard,
  onOpenClient = () => {},
  onOpenAgency,
  onReload,
}) {
  const [openId, setOpenId] = useState(() => openCampaignId || '');
  const [focusId, setFocusId] = useState(() => openCampaignId || '');
  const [pendingEnd, setPendingEnd] = useState('');
  const [booking, setBooking] = useState(false);
  const [options, setOptions] = useState(null);
  const [optionsError, setOptionsError] = useState('');
  const [visibleCount, setVisibleCount] = useState(CAMPAIGN_WINDOW);
  const he = locale === 'he';
  const canEdit = !gate || gate.canEdit;

  // The forms below offer real agencies, real clients and the real weekday
  // vocabulary, so the choices are fetched once for the board rather than on
  // each open, which is what keeps an amend a two-click act.
  useEffect(() => {
    if (!canEdit) {
      return undefined;
    }
    let alive = true;
    loadOnboardingOptions()
      .then((payload) => { if (alive) setOptions(payload); })
      .catch(() => {
        if (alive) {
          setOptionsError(pageText(
            locale,
            'The form choices could not be loaded from the server, so editing is blocked.',
            'אפשרויות הטופס לא נטענו מהשרת, ולכן העריכה חסומה.',
          ));
        }
      });
    return () => { alive = false; };
  }, [canEdit, locale]);

  // A caller that already resolved a campaign opens it here, which is how the
  // head of a money figure becomes the booking behind it. The board is cut to
  // that one row as well, the way the agency grid is cut to the one card, and
  // the request is cleared so closing the row does not reopen it.
  useEffect(() => {
    if (!openCampaignId) {
      return;
    }
    setOpenId(openCampaignId);
    setFocusId(openCampaignId);
    onOpened();
  }, [openCampaignId, onOpened]);

  if (!board) {
    return <div className="clients-loading">{pageText(locale, 'Loading campaigns', 'טוען קמפיינים')}</div>;
  }

  const campaigns = board.campaigns || [];
  // The one campaign a caller asked for, and only while the board really holds
  // it. A filter that empties the table would answer a figure with a blank
  // screen, so an id the board does not carry leaves the whole list on show.
  const focus = campaigns.some((campaign) => campaign.campaign_id === focusId) ? focusId : '';
  const visible = focus ? campaigns.filter((campaign) => campaign.campaign_id === focus) : campaigns;
  const shown = focus ? visible : visible.slice(0, visibleCount);
  const statuses = board.status_vocabulary || [];
  // The three air states and their words, as the payload names them, so the
  // delivered column reads in the ledger's own vocabulary rather than in one
  // this file invented for it.
  const airStates = (board.delivery && board.delivery.air_state_vocabulary) || [];
  const demoCount = board.demo_count ?? campaigns.filter((campaign) => campaign.is_demo).length;
  const bookedCount = board.booked_count ?? (campaigns.length - demoCount);
  const countLine = demoCount > 0
    ? pageText(
      locale,
      `${campaigns.length} campaigns in the ledger: ${bookedCount} booked, ${demoCount} demo data`,
      `${isolate(campaigns.length)} קמפיינים בספר: ${isolate(bookedCount)} הוזמנו, ${isolate(demoCount)} נתוני הדגמה`,
    )
    : pageText(locale, `${campaigns.length} campaigns booked`, `${isolate(campaigns.length)} קמפיינים הוזמנו`);
  const selected = campaigns.find((campaign) => campaign.campaign_id === openId) || null;

  async function end(campaignId) {
    try {
      await endCampaign(campaignId);
      setPendingEnd('');
      notify('Campaign ended.', 'הקמפיין הסתיים.');
      onReload();
    } catch (error) {
      notify(`Could not end the campaign. ${refusalText(error, 'en')}`, `לא ניתן היה לסיים את הקמפיין. ${refusalText(error, 'he')}`);
    }
  }

  // The agency behind the campaign. It opens the agency record when this board
  // was given an opener and the id really resolves to an agency on file, because
  // that is the same condition the record panel needs to find the card. An id
  // with nothing behind it stays a stated state rather than becoming a control
  // that would land the reader on an empty search.
  function agencyCell(campaign) {
    if (!campaign.agency_id) {
      return <span className="clients-unset">{pageText(locale, 'none', 'אין')}</span>;
    }
    const name = agencies[campaign.agency_id];
    if (name && onOpenAgency) {
      return (
        <Button
          type="button"
          className="clients-link clients-cell-name clients-cell-open"
          onClick={() => onOpenAgency(campaign.agency_id)}
        >
          <strong>{name}</strong>
          <small className="clients-campaign-id">{campaign.agency_id}</small>
        </Button>
      );
    }
    return (
      <span className="clients-cell-name">
        {name
          ? <strong>{name}</strong>
          : <span className="clients-unknown">{pageText(locale, 'name not on file', 'השם אינו רשום')}</span>}
        <small className="clients-campaign-id">{campaign.agency_id}</small>
      </span>
    );
  }

  return (
    <section className="clients-campaigns">
      <div className="clients-toolbar">
        <span className="clients-counts">{countLine}</span>
        {canEdit ? (
          <>
            <Button type="button" className="clients-primary" onClick={onOnboard}>
              <Plus size={14} aria-hidden="true" />
              {pageText(locale, 'Book a campaign', 'הזמינו קמפיין')}
            </Button>
            <Button type="button" className="clients-secondary" onClick={() => setBooking(true)}>
              {pageText(locale, 'Book for a client on file', 'הזמינו ללקוח שכבר רשום')}
            </Button>
          </>
        ) : (
          <p className="clients-refusal">{gate.reason}</p>
        )}
      </div>

      {focus ? (
        <p className="clients-basis-note">
          {pageText(
            locale,
            `Showing the requested campaign, of ${campaigns.length} in the ledger.`,
            `מוצג הקמפיין שהתבקש, מתוך ${isolate(campaigns.length)} שבספר.`,
          )}
          <Button type="button" className="clients-inline-action" onClick={() => setFocusId('')}>
            {pageText(locale, 'Show every campaign', 'הציגו את כל הקמפיינים')}
          </Button>
        </p>
      ) : null}

      <DeliveryBasis delivery={board.delivery} locale={locale} />
      {optionsError ? <p className="clients-error" role="alert">{optionsError}</p> : null}

      {booking && !options ? (
        <p className="clients-reason">
          {optionsError || pageText(locale, 'Loading the choices this form offers', 'טוען את האפשרויות שהטופס מציע')}
        </p>
      ) : null}
      {booking && options ? (
        <CampaignTerms
          mode="create"
          options={options}
          terms={board.terms}
          locale={locale}
          onCancel={() => setBooking(false)}
          onSaved={(record) => {
            setBooking(false);
            setOpenId(record.campaign_id);
            notify(
              `Campaign booked as ${record.campaign_id}. Add its flights below.`,
              `הקמפיין הוזמן כ־${record.campaign_id}. הוסיפו את טיסות השידור שלו למטה.`,
            );
            onReload();
          }}
        />
      ) : null}

      {campaigns.length === 0 ? (
        <div className="clients-empty">
          <CalendarRange size={22} aria-hidden="true" />
          <strong>{pageText(locale, 'No campaign is booked yet', 'לא הוזמן קמפיין עדיין')}</strong>
          <p>
            {pageText(
              locale,
              'A campaign is created from a signed insertion order. Onboard a client to book the first campaign.',
              'קמפיין נוצר מתוך הזמנת רכש חתומה. קלטו לקוח כדי להזמין את הקמפיין הראשון.',
            )}
          </p>
          {canEdit ? (
            <Button type="button" className="clients-primary" onClick={onOnboard}>
              {pageText(locale, 'Onboard a client', 'קליטת לקוח')}
            </Button>
          ) : null}
        </div>
      ) : (
        <div className="campaign-control-room">
          <div className="campaign-ledger">
            <table
              id="campaign-ledger-rows"
              className="clients-table"
              aria-label={pageText(locale, 'Booked campaigns and delivery state', 'קמפיינים שהוזמנו ומצב האספקה')}
              aria-rowcount={visible.length}
            >
              <thead>
                <tr>
                  <th scope="col">{pageText(locale, 'Campaign and client', 'קמפיין ולקוח')}</th>
                  <th scope="col">{pageText(locale, 'Window', 'חלון')}</th>
                  {/* What this column holds, which is not what it used to say.
                      It was headed Commitment and its cell renders the delivery
                      ledger: what aired, over how many sourced days, in the
                      state the ledger recorded. A reader comparing rows under
                      that heading was reading delivery as though it were the
                      goal. The commitment itself is per flight and lives on the
                      client record, where each flight's booked figure stands
                      beside what was counted against it. */}
                  <th scope="col">{pageText(locale, 'Delivered', 'סופק')}</th>
                  <th scope="col">{pageText(locale, 'State', 'מצב')}</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((campaign) => {
                  const open = openId === campaign.campaign_id;
                  return (
                    <tr key={campaign.campaign_id} className={open ? 'is-selected' : undefined}>
                      <td>
                        <div className="campaign-ledger-identity">
                          <Button
                            type="button"
                            className="campaign-ledger-open"
                            onClick={() => setOpenId(campaign.campaign_id)}
                            aria-pressed={open}
                          >
                            <strong>{campaign.name}</strong>
                            <Code>{campaign.campaign_id}</Code>
                          </Button>
                          <Button
                            type="button"
                            variant="text"
                            className="clients-link campaign-ledger-client"
                            onClick={() => onOpenClient(campaign.advertiser)}
                          >
                            {campaign.advertiser}
                          </Button>
                          <DemoBadge demo={campaign.demo} locale={locale} />
                        </div>
                      </td>
                      <td className="numeric"><Figure>{windowLabel(campaign.starts_on, campaign.ends_on, locale)}</Figure></td>
                      <td>
                        <DeliveryCell delivery={campaign.delivery} vocabulary={airStates} locale={locale} />
                      </td>
                      <td>
                        <span className={`clients-state ${campaign.status}`}>
                          {vocabularyLabel(statuses, campaign.status, locale)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {shown.length < visible.length ? (
              <div className="clients-window-more" role="status">
                <span>{pageText(locale, `Showing ${shown.length} of ${visible.length} campaigns`, `מוצגים ${isolate(shown.length)} מתוך ${isolate(visible.length)} קמפיינים`)}</span>
                <a href="#campaign-ledger-rows" role="button" className="clients-secondary"
                   onClick={(event) => { event.preventDefault(); setVisibleCount((count) => count + CAMPAIGN_WINDOW); }}
                   onKeyDown={(event) => { if (event.key === ' ') { event.preventDefault(); setVisibleCount((count) => count + CAMPAIGN_WINDOW); } }}>
                  {pageText(locale, 'Show the next campaigns', 'הציגו את הקמפיינים הבאים')}
                </a>
              </div>
            ) : null}
          </div>

          <aside className="card card-dense campaign-inspector" aria-live="polite">
            {selected ? (
              <>
                <header className="campaign-inspector-head">
                  <div>
                    <small className="campaign-inspector-code"><Code>{selected.campaign_id}</Code></small>
                    <h3>{selected.name}</h3>
                    <Button type="button" className="clients-link" onClick={() => onOpenClient(selected.advertiser)}>
                      {selected.advertiser}
                    </Button>
                  </div>
                  <Button type="button" className="clients-icon-button" onClick={() => setOpenId('')} aria-label={pageText(locale, 'Close campaign inspector', 'סגירת פרטי הקמפיין')}>
                    <PanelRightClose size={18} aria-hidden="true" />
                  </Button>
                </header>

                <dl className="campaign-inspector-facts">
                  <div><dt>{pageText(locale, 'Agency', 'סוכנות')}</dt><dd>{agencyCell(selected)}</dd></div>
                  <div><dt>{pageText(locale, 'Window', 'חלון')}</dt><dd className="numeric"><Figure>{windowLabel(selected.starts_on, selected.ends_on, locale)}</Figure></dd></div>
                  <div><dt>{pageText(locale, 'Flights', 'טיסות')}</dt><dd className="numeric"><Figure>{selected.flights.length}</Figure></dd></div>
                  <div><dt>{pageText(locale, 'State', 'מצב')}</dt><dd><span className={`clients-state ${selected.status}`}>{vocabularyLabel(statuses, selected.status, locale)}</span></dd></div>
                </dl>

                <p className="campaign-inspector-remedy">{vocabularyRemedy(statuses, selected.status, locale)}</p>
                {canEdit && selected.status === 'active' && pendingEnd !== selected.campaign_id ? (
                  <Button type="button" className="clients-secondary" onClick={() => setPendingEnd(selected.campaign_id)}>
                    {pageText(locale, 'End campaign', 'סיום הקמפיין')}
                  </Button>
                ) : null}
                {pendingEnd === selected.campaign_id ? (
                  <div className="clients-confirm">
                    <small>{pageText(locale, 'The campaign is marked ended and remains in the ledger.', 'הקמפיין יסומן כהסתיים ויישאר בספר.')}</small>
                    <div className="clients-row-actions">
                      <Button type="button" className="clients-primary" onClick={() => end(selected.campaign_id)}>{pageText(locale, 'Confirm end', 'אישור סיום')}</Button>
                      <Button type="button" className="clients-secondary" onClick={() => setPendingEnd('')}>{pageText(locale, 'Keep active', 'השאירו פעיל')}</Button>
                    </div>
                  </div>
                ) : null}

                <CampaignDetail
                  campaign={selected}
                  board={board}
                  options={options}
                  optionsError={optionsError}
                  locale={locale}
                  canEdit={canEdit}
                  notify={notify}
                  onChanged={onReload}
                />
              </>
            ) : (
              <div className="campaign-inspector-empty">
                <MousePointerClick size={28} aria-hidden="true" />
                <strong>{pageText(locale, 'Select a campaign', 'בחרו קמפיין')}</strong>
                <p>{pageText(locale, 'Commitments, delivery state and flights open here while the campaign ledger remains available.', 'כאן יוצגו ההתחייבויות, מצב האספקה וטיסות השידור. ספר הקמפיינים יישאר פתוח.')}</p>
              </div>
            )}
          </aside>
        </div>
      )}
    </section>
  );
}
