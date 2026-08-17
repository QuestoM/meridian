// Plain language for the terms that decide money: the price basis (family B)
// and the discounts, commissions and incentives that move it (family C).
//
// Each describer returns {headline, rows, table} and nothing else. `headline` is
// the sentence a commercial director reads instead of the parameter object;
// `rows` are the fields that do not belong in the sentence; `table` is a rate
// card, ladder or coefficient set rendered as the table it is in the document.
// A field the document did not supply becomes a row marked missing, which the
// review card shows as an incompleteness rather than as a zero.

import { formatNumber, pageText } from '../shell/format';
import {
  basisLabel,
  contractMoney,
  isMissing,
  listPhrase,
  moneyPhrase,
  percentPhrase,
  periodLabel,
  secondsPhrase,
} from './term-language';

function row(label, value, extra = {}) {
  if (isMissing(value)) return { label, missing: true, ...extra };
  return { label, value: String(value), ...extra };
}

function quoted(label, value) {
  return row(label, value, { quote: true });
}

export const MONEY_DESCRIBERS = {
  'cpp-daypart-table': (params, locale) => {
    const rows = Array.isArray(params.rows) ? params.rows : [];
    const audience = params.audience;
    const base = params.base_length_seconds;
    return {
      headline: pageText(
        locale,
        `The price of a rating point, by daypart, for ${audience || 'an audience the document does not name'}. ${rows.length} daypart rates, quoted against a ${base || 30} second spot; other lengths follow the length factors.`,
        `מחיר נקודת רייטינג לפי רצועה, עבור ${audience || 'קהל שהמסמך אינו נוקב בשמו'}. ${formatNumber(rows.length, locale)} תעריפי רצועה, לתשדיר בסיס של ${formatNumber(base || 30, locale)} שניות; אורכים אחרים נגזרים ממקדמי האורך.`,
      ),
      rows: [
        row(pageText(locale, 'Audience the rate is quoted for', 'הקהל שהתעריף נקוב לו'), audience),
        row(pageText(locale, 'Base spot length', 'אורך תשדיר הבסיס'), secondsPhrase(base, locale)),
      ],
      table: {
        caption: pageText(locale, 'Cost per rating point by daypart', 'עלות לנקודת רייטינג לפי רצועה'),
        columns: [
          { key: 'daypart', label: pageText(locale, 'Daypart, as the document names it', 'הרצועה, בשמה במסמך') },
          { key: 'cpp', label: pageText(locale, 'Cost per point', 'עלות לנקודה'), numeric: true },
        ],
        rows: rows.map((entry) => ({
          daypart: entry.daypart,
          cpp: contractMoney(entry.cpp, locale),
        })),
      },
    };
  },

  'target-cpp': (params, locale) => ({
    headline: pageText(
      locale,
      `A single price per rating point in the target audience: ${contractMoney(params.cpp, locale) || 'a rate the document does not state'} for ${params.audience || 'an audience the document does not name'}.`,
      `מחיר אחד לנקודת רייטינג בקהל היעד: ${contractMoney(params.cpp, locale) || 'תעריף שאינו נקוב במסמך'} עבור ${params.audience || 'קהל שהמסמך אינו נוקב בשמו'}.`,
    ),
    rows: [
      row(pageText(locale, 'Target audience', 'קהל היעד'), params.audience),
      row(pageText(locale, 'Price per point', 'מחיר לנקודה'), contractMoney(params.cpp, locale)),
      quoted(pageText(locale, 'Rating vintage the price is read against', 'גרסת הרייטינג שלפיה נמדד'), params.vintage),
    ],
  }),

  'length-factor-table': (params, locale) => {
    const rows = Array.isArray(params.rows) ? params.rows : [];
    return {
      headline: pageText(
        locale,
        `How a spot's length changes its price: ${rows.length} stated lengths, each with its multiplier against the base rate.`,
        `כיצד אורך התשדיר משנה את מחירו: ${formatNumber(rows.length, locale)} אורכים נקובים, לכל אחד המקדם שלו מול תעריף הבסיס.`,
      ),
      rows: [quoted(pageText(locale, 'What happens to a length the table does not list', 'מה קורה לאורך שאינו בטבלה'), params.rounding_rule)],
      table: {
        caption: pageText(locale, 'Length factors', 'מקדמי אורך'),
        columns: [
          { key: 'length', label: pageText(locale, 'Length', 'אורך'), numeric: true },
          { key: 'factor', label: pageText(locale, 'Multiplier', 'מקדם'), numeric: true },
        ],
        rows: rows.map((entry) => ({
          length: secondsPhrase(entry.length_seconds, locale),
          factor: `x${formatNumber(entry.factor, locale)}`,
        })),
      },
    };
  },

  'ratecard-index': (params, locale) => ({
    headline: pageText(
      locale,
      `Prices are tied to the channel's rate card rather than quoted outright: ${percentPhrase(params.percent_of_ratecard, locale) || 'a share the document does not state'} of it.`,
      `המחירים צמודים למחירון הערוץ ולא נקובים בפני עצמם: ${percentPhrase(params.percent_of_ratecard, locale) || 'שיעור שאינו נקוב במסמך'} ממנו.`,
    ),
    rows: [
      row(pageText(locale, 'Share of the rate card', 'שיעור מהמחירון'), percentPhrase(params.percent_of_ratecard, locale)),
      quoted(pageText(locale, 'Which rate card version', 'איזו גרסת מחירון'), params.ratecard_version),
      quoted(pageText(locale, 'What happens when the rate card changes', 'מה קורה כשהמחירון משתנה'), params.change_rule),
    ],
  }),

  'fixed-spot-pricing': (params, locale) => {
    const rows = Array.isArray(params.rows) ? params.rows : [];
    return {
      headline: pageText(
        locale,
        `${rows.length} slots priced as a flat amount per spot, outside the cost-per-point mechanism.`,
        `${formatNumber(rows.length, locale)} שיבוצים מתומחרים בסכום קבוע לתשדיר, מחוץ למנגנון העלות לנקודה.`,
      ),
      rows: [],
      table: {
        caption: pageText(locale, 'Fixed prices per spot', 'מחירים קבועים לתשדיר'),
        columns: [
          { key: 'what', label: pageText(locale, 'Programme or slot', 'תוכנית או שיבוץ') },
          { key: 'length', label: pageText(locale, 'Length', 'אורך'), numeric: true },
          { key: 'price', label: pageText(locale, 'Price per spot', 'מחיר לתשדיר'), numeric: true },
        ],
        rows: rows.map((entry) => ({
          what: entry.programme || entry.slot_note,
          length: secondsPhrase(entry.length_seconds, locale),
          price: moneyPhrase(entry.price, locale),
        })),
      },
    };
  },

  'sponsorship-terms': (params, locale) => ({
    headline: pageText(
      locale,
      `Sponsorship of ${params.programme || 'a programme the document does not name'}: ${formatNumber(params.airings, locale)} sponsor announcements at ${moneyPhrase(params.price_per_airing, locale)} each.`,
      `חסות לתוכנית ${params.programme || 'שהמסמך אינו נוקב בשמה'}: ${formatNumber(params.airings, locale)} הודעות חסות, ${moneyPhrase(params.price_per_airing, locale)} להודעה.`,
    ),
    rows: [
      row(pageText(locale, 'Programme', 'התוכנית'), params.programme),
      row(pageText(locale, 'Announcements bought', 'מספר ההודעות'), formatNumber(params.airings, locale)),
      row(pageText(locale, 'Price per announcement', 'מחיר להודעה'), moneyPhrase(params.price_per_airing, locale)),
      row(pageText(locale, 'Announcement length', 'אורך ההודעה'), secondsPhrase(params.notice_length_seconds, locale)),
      quoted(pageText(locale, 'Season or period', 'עונה או תקופה'), params.period),
    ],
  }),

  'gold-break-rates': (params, locale) => {
    const fixed = Array.isArray(params.fixed_prices) ? params.fixed_prices : [];
    return {
      headline: pageText(
        locale,
        `Gold breaks are priced above the ordinary rate: a ${percentPhrase(params.surcharge_percent, locale)} surcharge${fixed.length ? `, plus ${fixed.length} slots at a flat price` : ''}.`,
        `ברייקי זהב מתומחרים מעל התעריף הרגיל: תוספת של ${percentPhrase(params.surcharge_percent, locale)}${fixed.length ? `, ובנוסף ${formatNumber(fixed.length, locale)} שיבוצים במחיר קבוע` : ''}.`,
      ),
      rows: [row(pageText(locale, 'Surcharge over the applicable rate', 'תוספת מעל התעריף החל'), percentPhrase(params.surcharge_percent, locale))],
      table: fixed.length ? {
        caption: pageText(locale, 'Gold breaks at a flat price', 'ברייקי זהב במחיר קבוע'),
        columns: [
          { key: 'what', label: pageText(locale, 'Slot', 'השיבוץ') },
          { key: 'price', label: pageText(locale, 'Price', 'מחיר'), numeric: true },
        ],
        rows: fixed.map((entry) => ({
          what: entry.scope_note,
          price: moneyPhrase(entry.price, locale),
        })),
      } : null,
    };
  },

  'volume-discount-ladder': (params, locale) => {
    const tiers = Array.isArray(params.tiers) ? params.tiers : [];
    const mechanics = String(params.mechanics || 'unstated');
    const mechanicsCopy = {
      retroactive: {
        he: 'רטרואקטיבי: השיעור של המדרגה שהושגה חל על מלוא ההיקף, כולל מה שנרכש לפניה.',
        en: 'Retroactive: the rate of the tier reached applies to the whole volume, including what was bought before it.',
      },
      marginal: {
        he: 'שולי: כל מדרגה מתמחרת את הפלח שלה בלבד, ולא את מה שמתחתיה.',
        en: 'Marginal: each tier prices only its own band, not what sits below it.',
      },
      unstated: {
        he: 'המסמך אינו קובע אם המדרגות חלות רטרואקטיבית או שולית, וההפרש בין השתיים הוא כסף אמיתי.',
        en: 'The document does not say whether the ladder is retroactive or marginal, and the difference between the two is real money.',
      },
    }[mechanics] || { he: mechanics, en: mechanics };
    return {
      headline: pageText(
        locale,
        `A discount that grows with volume: ${tiers.length} tiers. ${mechanicsCopy.en}`,
        `הנחה שגדלה עם ההיקף: ${formatNumber(tiers.length, locale)} מדרגות. ${mechanicsCopy.he}`,
      ),
      rows: [
        row(pageText(locale, 'How the tiers apply', 'אופן החלת המדרגות'), locale === 'he' ? mechanicsCopy.he : mechanicsCopy.en),
        row(pageText(locale, 'Volume the tiers are measured on', 'ההיקף שלפיו נמדדות המדרגות'), params.measured_on ? basisLabel(params.measured_on, locale) : null),
      ],
      table: {
        caption: pageText(locale, 'Discount tiers', 'מדרגות ההנחה'),
        columns: [
          { key: 'threshold', label: pageText(locale, 'From', 'מהיקף של'), numeric: true },
          { key: 'discount', label: pageText(locale, 'Discount', 'הנחה'), numeric: true },
        ],
        rows: tiers.map((tier) => ({
          threshold: contractMoney(tier.threshold, locale),
          discount: percentPhrase(tier.discount_percent, locale),
        })),
      },
    };
  },

  'agency-commission': (params, locale) => ({
    headline: pageText(
      locale,
      `The agency keeps ${percentPhrase(params.percent, locale)}, calculated on ${basisLabel(params.base, locale)}. This one is live: approving it changes the net revenue this channel reports.`,
      `הסוכנות מקבלת ${percentPhrase(params.percent, locale)}, מחושב על ${basisLabel(params.base, locale)}. הסעיף הזה פעיל: אישורו משנה את ההכנסה נטו שהערוץ מדווח.`,
    ),
    rows: [
      row(pageText(locale, 'Commission rate', 'שיעור העמלה'), percentPhrase(params.percent, locale)),
      row(pageText(locale, 'What it is calculated on', 'על מה מחושב'), basisLabel(params.base, locale)),
      row(pageText(locale, 'How it is settled', 'אופן ההתחשבנות'), params.form === 'invoice_deduction'
        ? pageText(locale, 'deducted on the invoice', 'מנוכה בחשבונית')
        : params.form),
    ],
  }),

  'seasonal-coefficients': (params, locale) => {
    const rows = Array.isArray(params.rows) ? params.rows : [];
    const blackouts = rows.filter((entry) => entry.discount_blackout);
    return {
      headline: blackouts.length && rows.length === blackouts.length
        ? pageText(
          locale,
          `No discount applies in ${listPhrase(blackouts.map((b) => b.period_label), locale)}. The rate stands at full price for that period.`,
          `לא תחול הנחה ב${listPhrase(blackouts.map((b) => b.period_label), locale)}. התעריף עומד במלואו לתקופה הזאת.`,
        )
        : pageText(
          locale,
          `The rate moves by season: ${rows.length} periods with their own multiplier${blackouts.length ? `, and ${blackouts.length} where no discount applies at all` : ''}.`,
          `התעריף נע לפי עונה: ${formatNumber(rows.length, locale)} תקופות ולכל אחת מקדם${blackouts.length ? `, ובתוכן ${formatNumber(blackouts.length, locale)} שבהן לא תחול הנחה כלל` : ''}.`,
        ),
      rows: [],
      table: {
        caption: pageText(locale, 'Seasonal coefficients', 'מקדמי עונתיות'),
        columns: [
          { key: 'period', label: pageText(locale, 'Period', 'תקופה') },
          { key: 'effect', label: pageText(locale, 'Effect on the rate', 'ההשפעה על התעריף'), numeric: true },
        ],
        rows: rows.map((entry) => ({
          period: entry.period_label,
          effect: entry.discount_blackout
            ? pageText(locale, 'no discount at all', 'ללא הנחה כלל')
            : `x${formatNumber(entry.coefficient, locale)}`,
        })),
      },
    };
  },

  'cash-discount': (params, locale) => ({
    headline: pageText(
      locale,
      `A further ${percentPhrase(params.percent, locale)} off for paying early. Recorded and tracked; it does not change a placed spot's price.`,
      `${percentPhrase(params.percent, locale)} נוספים על תשלום מוקדם. נרשם ונמצא במעקב; אינו משנה את מחירו של תשדיר משובץ.`,
    ),
    rows: [
      row(pageText(locale, 'Discount', 'ההנחה'), percentPhrase(params.percent, locale)),
      quoted(pageText(locale, 'What qualifies for it', 'מה מזכה בה'), params.qualifying_terms),
    ],
  }),

  'share-bonus': (params, locale) => ({
    headline: pageText(
      locale,
      `A reward for share of the advertiser's television spend: reach ${percentPhrase(params.share_threshold_percent, locale)} and a further ${percentPhrase(params.award_discount_percent, locale)} follows. Standing is measured; the payout stays a human decision.`,
      `תמריץ על נתח מהשקעת הטלוויזיה של המפרסם: הגעה ל${percentPhrase(params.share_threshold_percent, locale)} מזכה בעוד ${percentPhrase(params.award_discount_percent, locale)}. העמידה נמדדת; ההענקה נשארת החלטה אנושית.`,
    ),
    rows: [
      row(pageText(locale, 'Share to reach', 'הנתח הנדרש'), percentPhrase(params.share_threshold_percent, locale)),
      row(pageText(locale, 'Reward', 'ההטבה'), percentPhrase(params.award_discount_percent, locale)),
      row(pageText(locale, 'Measured over', 'נמדד על פני'), periodLabel(params.period, locale)),
      quoted(pageText(locale, 'Where the total spend figure comes from', 'מהיכן מגיע נתון סך ההשקעה'), params.denominator_source),
    ],
  }),

  'success-deal': (params, locale) => ({
    headline: pageText(
      locale,
      `The channel takes ${percentPhrase(params.share_percent, locale)} of a measured business outcome rather than a media price alone. Measured and shown; nothing settles automatically.`,
      `הערוץ מקבל ${percentPhrase(params.share_percent, locale)} מתוצאה עסקית מדודה, ולא מחיר מדיה בלבד. נמדד ומוצג; דבר אינו מתחשבן אוטומטית.`,
    ),
    rows: [
      row(pageText(locale, 'Share of the outcome', 'הנתח מהתוצאה'), percentPhrase(params.share_percent, locale)),
      quoted(pageText(locale, 'What is measured, and by whom', 'מה נמדד ובידי מי'), params.measurement_basis),
      quoted(pageText(locale, 'When it settles', 'מתי מתחשבנים'), params.settlement_cycle),
    ],
  }),

  'added-value-media': (params, locale) => ({
    headline: pageText(
      locale,
      `Bonus airtime worth ${percentPhrase(params.percent, locale)} of what is actually bought, granted on top of the schedule and drawn down later.`,
      `מדיה נוספת בשווי ${percentPhrase(params.percent, locale)} מהרכישות בפועל, מוענקת מעל ללוח ומנוצלת בהמשך.`,
    ),
    rows: [
      row(pageText(locale, 'Rate', 'השיעור'), percentPhrase(params.percent, locale)),
      quoted(pageText(locale, 'Measured on', 'נמדד על'), params.basis),
      quoted(pageText(locale, 'By when it must be used', 'עד מתי יש לנצל'), params.delivery_window),
      quoted(pageText(locale, 'What quality of airtime it buys', 'איזו איכות מלאי'), params.quality_note),
    ],
  }),

  'new-business-incentive': (params, locale) => ({
    headline: pageText(
      locale,
      'A first-year incentive for a brand new to television, on top of every other discount.',
      'תמריץ שנה ראשונה למותג חדש בטלוויזיה, מעל לכל הנחה אחרת.',
    ),
    rows: [
      quoted(pageText(locale, 'What is awarded', 'מה מוענק'), params.award_note),
      quoted(pageText(locale, 'What counts as new', 'מה נחשב חדש'), params.qualification),
    ],
  }),

  'package-bundle': (params, locale) => {
    const components = Array.isArray(params.components) ? params.components : [];
    const outside = components.filter((c) => c.in_product === false);
    return {
      headline: pageText(
        locale,
        `One price for ${components.length} components at a ${percentPhrase(params.bundle_discount_percent, locale)} bundle discount. ${outside.length} of them are not television inventory this system plans, so their delivery cannot be counted here.`,
        `מחיר אחד ל${formatNumber(components.length, locale)} מרכיבים בהנחת חבילה של ${percentPhrase(params.bundle_discount_percent, locale)}. ${formatNumber(outside.length, locale)} מהם אינם מלאי טלוויזיה שהמערכת מתכננת, ולכן אספקתם אינה נספרת כאן.`,
      ),
      rows: [row(pageText(locale, 'Bundle discount', 'הנחת החבילה'), percentPhrase(params.bundle_discount_percent, locale))],
      table: {
        caption: pageText(locale, 'What is in the bundle', 'מה כלול בחבילה'),
        columns: [
          { key: 'component', label: pageText(locale, 'Component', 'מרכיב') },
          { key: 'inProduct', label: pageText(locale, 'Planned and counted here', 'מתוכנן ונספר כאן') },
        ],
        rows: components.map((entry) => ({
          component: entry.component,
          inProduct: entry.in_product
            ? pageText(locale, 'yes', 'כן')
            : pageText(locale, 'no, it is outside this system', 'לא, מחוץ למערכת הזאת'),
        })),
      },
    };
  },
};
