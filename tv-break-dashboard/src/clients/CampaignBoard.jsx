import React, { useEffect, useState } from 'react';
import { CalendarRange, ChevronDown, ChevronUp, Plus } from 'lucide-react';
import { pageText } from '../shell/format';
import { refusalText, vocabularyLabel, vocabularyRemedy, windowLabel } from './clients-money-helpers';
import { endCampaign, loadOnboardingOptions } from './clients-api';
import CampaignDetail from './CampaignDetail';
import CampaignTerms from './CampaignTerms';
import { DeliveryBasis, DeliveryCell } from './DeliveryState';
import DemoBadge from './DemoBadge';
import './clients-campaigns.css';

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
  const [openId, setOpenId] = useState('');
  const [focusId, setFocusId] = useState('');
  const [pendingEnd, setPendingEnd] = useState('');
  const [booking, setBooking] = useState(false);
  const [options, setOptions] = useState(null);
  const [optionsError, setOptionsError] = useState('');
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
            'The choices these forms offer could not be loaded, so editing is unavailable rather than guessed.',
            'האפשרויות שהטפסים מציעים לא נטענו, ולכן העריכה אינה זמינה ולא מנוחשת.',
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
      `${campaigns.length} campaigns on this board: ${bookedCount} booked, ${demoCount} demo seed data`,
      `⁦${campaigns.length}⁩ קמפיינים על הלוח: ⁦${bookedCount}⁩ הוזמנו, ⁦${demoCount}⁩ נתוני זרע הדגמה`,
    )
    : pageText(locale, `${campaigns.length} campaigns booked`, `⁦${campaigns.length}⁩ קמפיינים הוזמנו`);

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
        <button
          type="button"
          className="clients-link clients-cell-name clients-cell-open"
          onClick={() => onOpenAgency(campaign.agency_id)}
        >
          <strong>{name}</strong>
          <small className="clients-campaign-id">{campaign.agency_id}</small>
        </button>
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
    <section className="clients-campaigns" dir={he ? 'rtl' : 'ltr'}>
      <div className="clients-toolbar">
        <span className="clients-counts">{countLine}</span>
        {canEdit ? (
          <>
            <button type="button" className="clients-primary" onClick={onOnboard}>
              <Plus size={14} aria-hidden="true" />
              {pageText(locale, 'Book a campaign', 'הזמינו קמפיין')}
            </button>
            <button type="button" className="clients-secondary" onClick={() => setBooking(true)}>
              {pageText(locale, 'Book for a client on file', 'הזמינו ללקוח שכבר רשום')}
            </button>
          </>
        ) : (
          <p className="clients-refusal">{gate.reason}</p>
        )}
      </div>

      {focus ? (
        <p className="clients-basis-note">
          {pageText(
            locale,
            `Showing the one campaign that was asked for, of ${campaigns.length} on this board.`,
            `מוצג הקמפיין שהתבקש בלבד, מתוך ⁦${campaigns.length}⁩ שעל הלוח.`,
          )}
          <button type="button" className="clients-inline-action" onClick={() => setFocusId('')}>
            {pageText(locale, 'Show every campaign', 'הציגו את כל הקמפיינים')}
          </button>
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
              'A campaign is the commercial object a signed insertion order becomes. Onboard a client to create the first one.',
              'קמפיין הוא האובייקט המסחרי שאליו הופכת הזמנה חתומה. קלטו לקוח כדי ליצור את הראשון.',
            )}
          </p>
          {canEdit ? (
            <button type="button" className="clients-primary" onClick={onOnboard}>
              {pageText(locale, 'Onboard a client', 'קליטת לקוח')}
            </button>
          ) : null}
        </div>
      ) : (
        <table className="clients-table">
          <thead>
            <tr>
              <th scope="col">{pageText(locale, 'Campaign', 'קמפיין')}</th>
              <th scope="col">{pageText(locale, 'Client', 'לקוח')}</th>
              <th scope="col">{pageText(locale, 'Agency', 'סוכנות')}</th>
              <th scope="col">{pageText(locale, 'Window', 'חלון')}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Flights', 'טיסות')}</th>
              <th scope="col">{pageText(locale, 'Delivered', 'סופק')}</th>
              <th scope="col">{pageText(locale, 'State', 'מצב')}</th>
              <th scope="col">{pageText(locale, 'What to do', 'מה לעשות')}</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((campaign) => {
              const open = openId === campaign.campaign_id;
              const toggle = () => setOpenId(open ? '' : campaign.campaign_id);
              return (
                <React.Fragment key={campaign.campaign_id}>
                  <tr>
                    <td>
                      <button type="button" className="clients-link" onClick={toggle} aria-expanded={open}>
                        {campaign.name}
                        {open ? <ChevronUp size={13} aria-hidden="true" /> : <ChevronDown size={13} aria-hidden="true" />}
                      </button>
                      <DemoBadge demo={campaign.demo} locale={locale} />
                      <small className="clients-campaign-id">{campaign.campaign_id}</small>
                    </td>
                    <td>
                      <button type="button" className="clients-link" onClick={() => onOpenClient(campaign.advertiser)}>
                        {campaign.advertiser}
                      </button>
                    </td>
                    <td>{agencyCell(campaign)}</td>
                    <td className="numeric" dir="ltr">{windowLabel(campaign.starts_on, campaign.ends_on, locale)}</td>
                    <td className="numeric" dir="ltr">
                      <button type="button" className="clients-link" onClick={toggle} aria-expanded={open}>
                        {campaign.flights.length}
                      </button>
                    </td>
                    <td>
                      <DeliveryCell
                        delivery={campaign.delivery}
                        vocabulary={airStates}
                        locale={locale}
                      />
                    </td>
                    <td>
                      <span className={`clients-state ${campaign.status}`}>
                        {vocabularyLabel(statuses, campaign.status, locale)}
                      </span>
                    </td>
                    <td>
                      <small className="clients-remedy">{vocabularyRemedy(statuses, campaign.status, locale)}</small>
                      {canEdit && pendingEnd !== campaign.campaign_id ? (
                        <span className="clients-row-actions">
                          <button type="button" className="clients-inline-action" onClick={toggle}>
                            {pageText(locale, 'Amend it', 'תקנו')}
                          </button>
                          {campaign.status === 'active' ? (
                            <button type="button" className="clients-inline-action" onClick={() => setPendingEnd(campaign.campaign_id)}>
                              {pageText(locale, 'End it', 'סיימו')}
                            </button>
                          ) : null}
                        </span>
                      ) : null}
                      {pendingEnd === campaign.campaign_id ? (
                        <span className="clients-confirm">
                          <small>{pageText(locale, 'It is marked ended, never deleted.', 'הוא מסומן כהסתיים, לעולם אינו נמחק.')}</small>
                          <button type="button" className="clients-inline-action" onClick={() => end(campaign.campaign_id)}>
                            {pageText(locale, 'Confirm', 'אישור')}
                          </button>
                          <button type="button" className="clients-inline-action" onClick={() => setPendingEnd('')}>
                            {pageText(locale, 'Keep it', 'השאירו')}
                          </button>
                        </span>
                      ) : null}
                    </td>
                  </tr>
                  {open ? (
                    <tr className="clients-subrow">
                      <td colSpan={8}>
                        <CampaignDetail
                          campaign={campaign}
                          board={board}
                          options={options}
                          optionsError={optionsError}
                          locale={locale}
                          canEdit={canEdit}
                          notify={notify}
                          onChanged={onReload}
                        />
                      </td>
                    </tr>
                  ) : null}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
