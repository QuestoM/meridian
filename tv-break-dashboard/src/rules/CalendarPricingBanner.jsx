import React from 'react';
import { Info } from 'lucide-react';
import { calendarPricingBannerSentence, rateCardTabLinkLabel } from './rules-lib';

// The calendar's price-multiplier banner. Its control used to be gated on
// whether a legacy prop happened to be threaded through from the shell, which
// compiled it out on every render because nothing ever supplied that prop:
// a sentence naming a live pricing layer with no way to reach it. Navigation
// inside this workspace is always the workspace's own to give, so the button
// renders every time and calls back into the section that opens the rate
// card, never a page name that can drift out of date.
function CalendarPricingBanner({ locale, eventsPricing, onOpenRateCard }) {
  return (
    <div className="cal-banner">
      <Info size={16} aria-hidden="true" />
      <p>
        {calendarPricingBannerSentence(locale, eventsPricing)}{' '}
        <button type="button" className="cal-banner-link" onClick={() => onOpenRateCard?.()}>
          {rateCardTabLinkLabel(locale)}
        </button>
      </p>
    </div>
  );
}

export default CalendarPricingBanner;
