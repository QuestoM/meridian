import React from 'react';
import { pageText } from '../shell/surface-helpers';
import { keyLabel } from './pricing-layers-lib';

// Which positions count as PREFERRED, and under which counting method a
// preferred-position percentage would be quoted. The trade agrees the set per
// client and per agreement, so it is configuration and never a constant, and it
// is tri-state: while nobody has set one, this card says so plainly instead of
// showing a default that looks agreed. Two parties audit each other with the
// percentage, so the methods are named here even before one is computed.
function PricingPreferredPositions({ state, locale }) {
  const block = state && typeof state.preferred_positions === 'object' ? state.preferred_positions : null;
  if (!block) return null;
  const positions = Array.isArray(block.positions) ? block.positions : null;
  const methods = Array.isArray(block.counting_methods) ? block.counting_methods : [];
  const scopeWords = {
    agreement: pageText(locale, 'from the agreement', 'מתוך ההסכם'),
    advertiser: pageText(locale, 'from the client', 'מתוך הלקוח'),
    channel_default: pageText(locale, 'the channel default', 'ברירת המחדל של הערוץ'),
    unset: pageText(locale, 'not configured', 'לא הוגדר'),
  };
  return (
    <div className="pricing-layer-card">
      <div className="pricing-layer-head">
        <div>
          <span className="pricing-layer-title">{pageText(locale, 'Preferred positions', 'מיקומים מועדפים')}</span>
          <p className="pricing-layer-desc">{pageText(locale,
            'Which of the positions 1 to 5 and L count as preferred. It is agreed per client and per agreement, so it is configuration, not a fixed list.',
            'אילו מהמיקומים 1 עד 5 ו-L נחשבים מועדפים. הדבר מוסכם לכל לקוח ולכל הסכם, ולכן זו הגדרה ולא רשימה קבועה.')}</p>
        </div>
        <span className={`pricing-chip ${positions === null ? 'empty' : 'live'}`}>
          {positions === null ? pageText(locale, 'Not set', 'לא הוגדר') : scopeWords[block.scope] || block.scope}
        </span>
      </div>
      {positions === null ? (
        <p className="pricing-empty">{pageText(locale,
          'No preferred set is configured, so no preferred-position percentage is computed anywhere. A guessed percentage is worse than none, because the channel and the agency audit each other with this number.',
          'לא הוגדרה קבוצת מיקומים מועדפים, ולכן לא מחושב בשום מקום אחוז מיקומים מועדפים. אחוז מנוחש גרוע יותר מהיעדר אחוז, מכיוון שהערוץ והמשרד מבקרים זה את זה לפי המספר הזה.')}</p>
      ) : (
        <div className="pricing-multipliers">
          {positions.map((key) => (
            <div className="pricing-mult" key={`preferred-${key}`}>
              <span className="pricing-mult-label">{keyLabel('position', key, locale)}</span>
            </div>
          ))}
        </div>
      )}
      <p className="pricing-base-note">{pageText(locale,
        'Any preferred-position percentage must name the method that counted it:',
        'כל אחוז מיקומים מועדפים חייב לנקוב בשיטה שלפיה נספר:')}</p>
      {methods.map((method) => (
        <p className="pricing-base-note" key={`method-${method.key}`}>
          {locale === 'he' ? method.he : method.en}
        </p>
      ))}
    </div>
  );
}

export default PricingPreferredPositions;
