import React, { useEffect, useState } from 'react';
import { pageText } from '../shell/format';
import { API_BASE } from '../shell/api';
import { isolate, rate } from './rules-lib';

// The revenue owner's own question, on the door they walk through. What is a
// second of airtime worth, on the plan of record, on their channel, computed by
// the engine rather than asserted. The figure and its scope are printed
// together; the basis the payload carries is printed under it rather than
// hidden in a tooltip, because a rate is meaningless without the thing it was
// measured over.
//
// The basis printed here is the one that belongs to THIS figure, which is
// totals.basis. It is not the payload's top-level basis: that one is
// frame_revenue_net's retention-cost model and it belongs to revenue_net_ils
// and retention_cost_ils, which are not on this card. Measured on the shipped
// surface, printing it here put a formula under 142.0920 whose own five named
// inputs produce 36,783,099.42, so the caption stated a provenance the number
// never had. The arithmetic is now printed substituted as well, in the payload's
// own figures, so the division can be checked on the screen that claims it.

// A plain grouped number, no currency. The division line has to be checkable
// digit by digit against the two figures beside it, and the shekel sign belongs
// to the headline rather than to the arithmetic that produced it.
function plain(value, locale, digits) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '--';
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(number);
}

export default function WorthOfASecond({ locale }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    fetch(`${API_BASE}/api/yield-per-second`)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`${response.status}`))))
      .then((body) => { if (alive) setState(body); })
      .catch((problem) => { if (alive) setError(problem.message); });
    return () => { alive = false; };
  }, []);

  if (error) {
    return (
      <section className="rules-card rules-worth">
        <span className="rules-figure-label">{pageText(locale, 'A second of airtime is worth', 'שנייה של זמן שידור שווה')}</span>
        <span className="rules-figure-reason">
          {pageText(locale, `The yield figure is unreachable (${error}).`, `נתון התשואה אינו זמין (${error}).`)}
        </span>
      </section>
    );
  }
  if (!state) return null;
  if (!state.available) {
    return (
      <section className="rules-card rules-worth">
        <span className="rules-figure-label">{pageText(locale, 'A second of airtime is worth', 'שנייה של זמן שידור שווה')}</span>
        <span className="rules-figure-reason">{state.reason}</span>
      </section>
    );
  }

  const totals = state.totals || {};
  // With no declared channel the payload's own scope is every channel in the
  // loaded plan, which is a market figure and not this operator's. It is named
  // as one rather than printed under the operator's own heading, and the
  // channels are counted and never listed.
  const declared = Boolean(state.scope_channel);
  const where = declared
    ? state.scope_channel
    : pageText(
      locale,
      `all ${state.n_channels_total} channels in the loaded plan`,
      `כל ${state.n_channels_total} הערוצים בתוכנית שנטענה`,
    );
  return (
    <section className="rules-card rules-worth">
      <div>
        <span className="rules-figure-label">{pageText(locale, 'A second of airtime is worth', 'שנייה של זמן שידור שווה')}</span>
        {/* Isolated for the same reason the rate card's own figures are. The
            rate card sits directly under this one and prints the same currency,
            and measured in a browser this figure unisolated painted
            "142.2122" with the shekel sign against the last digit and its space
            stranded, while the pair below it painted the sign in front. One
            screen, one currency, one rendering. */}
        <strong className="rules-worth-value" dir="ltr">{isolate(rate(totals.yield_per_second, locale))}</strong>
        {/* The count is the payload's own: segments with ad seconds on them,
            which is fewer than the plan's rows. The rate-card delta beside this
            counts every planned row, so the two say what they each counted
            rather than sharing one word for two numbers. */}
        <span className="rules-figure-scope">
          {pageText(
            locale,
            `${where}, ${state.date_from} to ${state.date_to}, ${totals.segment_count} segments carrying breaks`,
            `${where}, ${state.date_from} עד ${state.date_to}, ${totals.segment_count} מקטעים שנושאים ברייקים`,
          )}
        </span>
        {!declared && (
          <span className="rules-figure-reason">
            {pageText(
              locale,
              'No operator channel is declared, so this is the whole loaded plan and not your channel. Declare your channel under Channel and model.',
              'לא הוצהר ערוץ מפעיל, ולכן זו כל התוכנית שנטענה ולא הערוץ שלכם. הצהירו על הערוץ במדור ערוץ ומודל.',
            )}
          </span>
        )}
      </div>
      <p className="rules-figure-basis">
        <span>{pageText(locale, 'How it is computed', 'איך זה מחושב')}</span>
        {totals.basis?.formula ? (
          <>
            <span>
              {pageText(
                locale,
                'Expected revenue on the scope named above, divided by the seconds of ad time it carries.',
                'ההכנסה הצפויה בהיקף שצוין למעלה, חלקי שניות הפרסום שהוא נושא.',
              )}
            </span>
            <code dir="ltr">{totals.basis.formula}</code>
            <code dir="ltr">{`${plain(totals.revenue, locale, 2)} / ${plain(totals.ad_seconds, locale, 0)} = ${plain(totals.yield_per_second, locale, 4)}`}</code>
          </>
        ) : (
          <span>
            {pageText(
              locale,
              'The server did not state how this figure was computed, so nothing is claimed for it here.',
              'השרת לא מסר איך חושב הנתון הזה, ולכן לא נטענת כאן טענה לגביו.',
            )}
          </span>
        )}
      </p>
    </section>
  );
}
