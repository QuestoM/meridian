import React, { useEffect, useState } from 'react';
import { CalendarRange, ChevronDown, ChevronUp, Plus } from 'lucide-react';
import { pageText } from '../shell/format';
import { localized, refusalText, vocabularyLabel, vocabularyRemedy, windowLabel } from './clients-money-helpers';
import { endCampaign, loadOnboardingOptions } from './clients-api';
import CampaignDetail from './CampaignDetail';
import CampaignTerms from './CampaignTerms';
import './clients-campaigns.css';

// Every campaign booked in this product, with its flights and what each flight
// committed to. Google Ads puts budget on exactly one layer and states the
// containment; this does the same with the two layers that exist here, and it
// refuses to render the third column that surface has, pace, because nothing in
// this repository observes delivery.
//
// The delivery state is a state, not a blank: it names the missing feed and the
// path that would supply it, on the board and on every row.
//
// Every row is a way in rather than a readout. The campaign name opens its
// terms and its flights and both can be changed there, the client name opens
// the client, and the agency reads as its name with its id underneath, because
// AGY_10 is a database key and not something a person calls anybody.

export default function CampaignBoard({
  board,
  locale,
  notify,
  gate,
  agencies = {},
  onOnboard,
  onOpenClient = () => {},
  onReload,
}) {
  const [openId, setOpenId] = useState('');
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

  if (!board) {
    return <div className="clients-loading">{pageText(locale, 'Loading campaigns', 'טוען קמפיינים')}</div>;
  }

  const campaigns = board.campaigns || [];
  const statuses = board.status_vocabulary || [];

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

  function agencyCell(campaign) {
    if (!campaign.agency_id) {
      return <span className="clients-unset">{pageText(locale, 'none', 'אין')}</span>;
    }
    const name = agencies[campaign.agency_id];
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
        <span className="clients-counts">
          {pageText(locale, `${campaigns.length} campaigns booked`, `⁦${campaigns.length}⁩ קמפיינים הוזמנו`)}
        </span>
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

      <p className="clients-basis-note">{localized(board.delivery, 'reason', locale)}</p>
      <p className="clients-basis-path">{localized(board.delivery, 'path_forward', locale)}</p>
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
              <th scope="col">{pageText(locale, 'State', 'מצב')}</th>
              <th scope="col">{pageText(locale, 'What to do', 'מה לעשות')}</th>
            </tr>
          </thead>
          <tbody>
            {campaigns.map((campaign) => {
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
                      <td colSpan={7}>
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
