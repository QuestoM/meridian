// Plain language for the terms that are not about the price itself: who the
// agreement covers (A), what the advertiser undertakes (D), what the channel
// guarantees and how it makes good (E), what may and may not be placed (F),
// process and law (G), and how delivery is measured and settled (H).
//
// Same contract as term-language-money.js: {headline, rows, table}. The clause's
// own words are carried through as quoted rows wherever the schema stores free
// text, because on a placement constraint or a make-good rule the exact wording
// is the operative fact and a paraphrase would be a second, weaker agreement.

import { formatDay, formatSpan } from '../shell/dates';
import { formatNumber, pageText } from '../shell/format';
import {
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

const COUNTERPARTY_COPY = {
  agency: { he: 'סוכנות מדיה', en: 'a media agency' },
  advertiser: { he: 'מפרסם ישיר', en: 'a direct advertiser' },
  advertiser_via_agency: { he: 'מפרסם באמצעות סוכנות', en: 'an advertiser through an agency' },
};

const LEVEL_COPY = {
  agency_framework: { he: 'הסכם מסגרת עם סוכנות', en: 'an agency framework' },
  advertiser: { he: 'הסכם מפרסם', en: 'an advertiser agreement' },
  campaign: { he: 'נספח קמפיין', en: 'a campaign appendix' },
};

const CURE_COPY = {
  credit: { he: 'זיכוי כספי', en: 'a cash credit' },
  bonus_spots: { he: 'שידורי בונוס', en: 'bonus spots' },
  rate_adjustment: { he: 'תיקון תעריף', en: 'a rate adjustment' },
};

const TREATMENT_COPY = {
  charged: { he: 'עודף האספקה מחויב', en: 'over-delivery is charged for' },
  absorbed: { he: 'עודף האספקה נספג ואינו מחויב', en: 'over-delivery is absorbed and not charged' },
  banked: { he: 'עודף האספקה נצבר לזכות עתידית', en: 'over-delivery is banked for later' },
};

function pick(map, key, locale) {
  const entry = map[String(key || '')];
  if (!entry) return key ? String(key) : null;
  return locale === 'he' ? entry.he : entry.en;
}

export const TERM_DESCRIBERS = {
  'agreement-parties': (params, locale) => {
    const who = [params.agency, params.advertiser].filter(Boolean).join(' / ');
    const represented = Array.isArray(params.represented_advertisers) ? params.represented_advertisers : [];
    return {
      headline: pageText(
        locale,
        `The counterparty is ${pick(COUNTERPARTY_COPY, params.counterparty_type, locale)}${who ? `: ${who}` : ''}${represented.length ? `, covering ${represented.length} named advertisers` : ''}.`,
        `הצד להסכם הוא ${pick(COUNTERPARTY_COPY, params.counterparty_type, locale)}${who ? `: ${who}` : ''}${represented.length ? `, ובמסגרתו ${formatNumber(represented.length, locale)} מפרסמים נקובים בשם` : ''}.`,
      ),
      rows: [
        row(pageText(locale, 'Agency', 'הסוכנות'), params.agency),
        row(pageText(locale, 'Advertiser', 'המפרסם'), params.advertiser),
        row(pageText(locale, 'Advertisers this agreement covers', 'המפרסמים שההסכם חל עליהם'),
          represented.length ? represented.join(', ') : null),
        row(pageText(locale, 'Signatories', 'החותמים'),
          Array.isArray(params.signatories) ? params.signatories.join(' · ') : null),
      ],
    };
  },

  'brand-scope': (params, locale) => {
    const included = Array.isArray(params.included_brands) ? params.included_brands : [];
    const excluded = Array.isArray(params.excluded_brands) ? params.excluded_brands : [];
    return {
      headline: pageText(
        locale,
        `The agreement covers ${included.length ? included.join(', ') : 'brands the document does not list'}${excluded.length ? `, and explicitly excludes ${excluded.join(', ')}` : ''}.`,
        `ההסכם חל על ${included.length ? included.join(', ') : 'מותגים שאינם נקובים במסמך'}${excluded.length ? `, ואינו חל במפורש על ${excluded.join(', ')}` : ''}.`,
      ),
      rows: [
        row(pageText(locale, 'Covered brands', 'מותגים בהיקף'), included.length ? included.join(', ') : null),
        row(pageText(locale, 'Excluded brands', 'מותגים מחוץ להיקף'), excluded.length ? excluded.join(', ') : null),
      ],
    };
  },

  'channel-scope': (params, locale) => {
    const channels = Array.isArray(params.channels) ? params.channels : [];
    const nonTv = Array.isArray(params.non_tv_assets) ? params.non_tv_assets : [];
    return {
      headline: pageText(
        locale,
        `Applies to ${listPhrase(channels, locale)}${nonTv.length ? `. It also names ${nonTv.length} non-television assets, which this system does not plan or count` : ''}.`,
        `חל על ${listPhrase(channels, locale)}${nonTv.length ? `. בנוסף נקובים ${formatNumber(nonTv.length, locale)} נכסים שאינם טלוויזיה, שהמערכת אינה מתכננת או סופרת` : ''}.`,
      ),
      rows: [
        row(pageText(locale, 'Channels', 'ערוצים'), channels.length ? channels.join(', ') : null),
        row(pageText(locale, 'Assets outside television', 'נכסים מחוץ לטלוויזיה'), nonTv.length ? nonTv.join(', ') : null),
      ],
    };
  },

  'effective-window': (params, locale) => ({
    headline: params.starts_on && params.ends_on
      ? pageText(
        locale,
        `In force ${formatSpan(params.starts_on, params.ends_on, locale)}${params.auto_renewal ? ', and it renews itself unless notice is given' : ', with no automatic renewal'}.`,
        `בתוקף ${formatSpan(params.starts_on, params.ends_on, locale)}${params.auto_renewal ? ', ומתחדש מעצמו אלא אם תינתן הודעה' : ', ללא חידוש אוטומטי'}.`,
      )
      : pageText(locale, 'The document does not state both ends of the term.', 'המסמך אינו נוקב בשני קצות התקופה.'),
    rows: [
      row(pageText(locale, 'Starts', 'תחילה'), params.starts_on ? formatDay(params.starts_on) : null),
      row(pageText(locale, 'Ends', 'סיום'), params.ends_on ? formatDay(params.ends_on) : null),
      row(pageText(locale, 'Renews automatically', 'מתחדש אוטומטית'),
        params.auto_renewal === undefined ? null : (params.auto_renewal
          ? pageText(locale, 'yes', 'כן') : pageText(locale, 'no', 'לא'))),
      row(pageText(locale, 'Notice before renewal', 'הודעה מוקדמת לפני חידוש'),
        isMissing(params.renewal_notice_days) ? null
          : pageText(locale, `${formatNumber(params.renewal_notice_days, locale)} days`, `${formatNumber(params.renewal_notice_days, locale)} ימים`)),
    ],
  }),

  'agreement-level': (params, locale) => ({
    headline: pageText(
      locale,
      `This is ${pick(LEVEL_COPY, params.level, locale)}${params.parent_agreement ? ', sitting under an agreement above it' : ''}. The level decides which agreement wins when two of them say different things.`,
      `זה ${pick(LEVEL_COPY, params.level, locale)}${params.parent_agreement ? ', היושב תחת הסכם שמעליו' : ''}. הרמה קובעת מי גובר כאשר שני הסכמים אומרים דברים שונים.`,
    ),
    rows: [
      row(pageText(locale, 'Level', 'הרמה'), pick(LEVEL_COPY, params.level, locale)),
      quoted(pageText(locale, 'The agreement above it', 'ההסכם שמעליו'), params.parent_agreement),
    ],
  }),

  'precedence-clause': (params, locale) => ({
    headline: pageText(
      locale,
      `When two documents disagree, ${params.winner || 'one of them'} wins over ${params.loser || 'the other'}${params.scope_note ? ` — ${params.scope_note}` : ''}.`,
      `כאשר שני מסמכים חולקים, ${params.winner || 'אחד מהם'} גובר על ${params.loser || 'האחר'}${params.scope_note ? ` — ${params.scope_note}` : ''}.`,
    ),
    rows: [
      row(pageText(locale, 'Wins', 'הגובר'), params.winner),
      row(pageText(locale, 'Loses', 'הנדחה'), params.loser),
      quoted(pageText(locale, 'Only in these matters', 'רק בעניינים אלה'), params.scope_note),
      quoted(pageText(locale, 'The clause, as written', 'הסעיף, כלשונו'), params.verbatim),
    ],
  }),

  definitions: (params, locale) => {
    const entries = Array.isArray(params.entries) ? params.entries : [];
    return {
      headline: pageText(
        locale,
        `${entries.length} definitions this agreement sets for itself. A daypart defined here overrides the channel's own usage for everything priced under this agreement.`,
        `${formatNumber(entries.length, locale)} הגדרות שההסכם קובע לעצמו. רצועה שמוגדרת כאן גוברת על השימוש הנוהג בערוץ לכל מה שמתומחר תחת ההסכם.`,
      ),
      rows: [],
      table: {
        caption: pageText(locale, 'Definitions the agreement sets', 'ההגדרות שההסכם קובע'),
        columns: [
          { key: 'term', label: pageText(locale, 'Term', 'מונח') },
          { key: 'definition', label: pageText(locale, 'As the document defines it', 'כהגדרתו במסמך') },
          { key: 'bounds', label: pageText(locale, 'Hours', 'שעות'), numeric: true },
        ],
        rows: entries.map((entry) => ({
          term: entry.term,
          definition: entry.definition,
          bounds: entry.daypart_bounds
            ? `${entry.daypart_bounds.start}-${entry.daypart_bounds.end}`
            : null,
        })),
      },
    };
  },

  'amendment-layer': (params, locale) => {
    const modifies = Array.isArray(params.modifies) ? params.modifies : [];
    return {
      headline: pageText(
        locale,
        `An amendment that changes ${modifies.length} clauses of an agreement already in force${params.effective_on ? `, from ${formatDay(params.effective_on)}` : ''}.`,
        `תיקון שמשנה ${formatNumber(modifies.length, locale)} סעיפים בהסכם שכבר בתוקף${params.effective_on ? `, מיום ${formatDay(params.effective_on)}` : ''}.`,
      ),
      rows: [
        row(pageText(locale, 'Takes effect', 'נכנס לתוקף'), params.effective_on ? formatDay(params.effective_on) : null),
        row(pageText(locale, 'Clauses it changes', 'הסעיפים שהוא משנה'), modifies.length ? modifies.join(', ') : null),
        quoted(pageText(locale, 'What it changes', 'מה הוא משנה'), params.summary),
      ],
    };
  },

  'budget-commitment': (params, locale) => ({
    headline: pageText(
      locale,
      `The advertiser undertakes to buy ${moneyPhrase(params.amount, locale)} ${periodLabel(params.period, locale)}. This is the figure every volume tier is measured against.`,
      `המפרסם מתחייב לרכוש ${moneyPhrase(params.amount, locale)} ${periodLabel(params.period, locale)}. זה הסכום שכל מדרגת היקף נמדדת מולו.`,
    ),
    rows: [
      row(pageText(locale, 'Amount committed', 'ההיקף המתחייב'), moneyPhrase(params.amount, locale)),
      row(pageText(locale, 'Over', 'על פני'), periodLabel(params.period, locale)),
    ],
  }),

  'share-commitment': (params, locale) => ({
    headline: pageText(
      locale,
      `The advertiser undertakes a share of its television spend rather than an amount: ${percentPhrase(params.share_percent, locale)}. Standing depends on a figure this channel does not own.`,
      `המפרסם מתחייב לנתח מהשקעת הטלוויזיה שלו ולא לסכום: ${percentPhrase(params.share_percent, locale)}. העמידה תלויה בנתון שאינו בידי הערוץ.`,
    ),
    rows: [
      row(pageText(locale, 'Share committed', 'הנתח המתחייב'), percentPhrase(params.share_percent, locale)),
      quoted(pageText(locale, 'Where the denominator comes from', 'מהיכן מגיע המכנה'), params.denominator_source),
      row(pageText(locale, 'Over', 'על פני'), periodLabel(params.period, locale)),
    ],
  }),

  'daypart-mix': (params, locale) => {
    const rows = Array.isArray(params.rows) ? params.rows : [];
    const basis = params.basis === 'rating_points'
      ? pageText(locale, 'rating points', 'נקודות רייטינג')
      : pageText(locale, 'money', 'כסף');
    return {
      headline: pageText(
        locale,
        `How the buy must be spread across dayparts: ${rows.length} floors and ceilings, measured in ${basis}.`,
        `כיצד חייבת הרכישה להתפרס על הרצועות: ${formatNumber(rows.length, locale)} רצפות ותקרות, נמדדות ב${basis}.`,
      ),
      rows: [row(pageText(locale, 'Measured in', 'נמדד ב'), basis)],
      table: {
        caption: pageText(locale, 'Daypart mix', 'תמהיל הרצועות'),
        columns: [
          { key: 'daypart', label: pageText(locale, 'Daypart, as the document names it', 'הרצועה, בשמה במסמך') },
          { key: 'bound', label: pageText(locale, 'Floor or ceiling', 'רצפה או תקרה'), numeric: true },
        ],
        rows: rows.map((entry) => ({
          daypart: entry.daypart,
          bound: entry.min_percent !== undefined
            ? pageText(locale, `at least ${percentPhrase(entry.min_percent, locale)}`, `לפחות ${percentPhrase(entry.min_percent, locale)}`)
            : pageText(locale, `at most ${percentPhrase(entry.max_percent, locale)}`, `לכל היותר ${percentPhrase(entry.max_percent, locale)}`),
        })),
      },
    };
  },

  'flighting-obligation': (params, locale) => {
    const rules = Array.isArray(params.rules) ? params.rules : [];
    return {
      headline: pageText(
        locale,
        `Continuity the advertiser undertakes: ${rules.length} rules about when activity may not stop.`,
        `רציפות שהמפרסם מתחייב לה: ${formatNumber(rules.length, locale)} כללים לגבי מתי הפעילות אינה עוצרת.`,
      ),
      rows: rules.map((entry, index) => quoted(
        pageText(locale, `Rule ${index + 1}`, `כלל ${formatNumber(index + 1, locale)}`),
        entry.rule,
      )),
    };
  },

  'length-mix': (params, locale) => ({
    headline: pageText(
      locale,
      'A required mix of spot lengths. Measured and shown; it does not constrain the plan by itself.',
      'תמהיל אורכים נדרש. נמדד ומוצג; אינו מגביל את התוכנית מעצמו.',
    ),
    rows: (Array.isArray(params.rows) ? params.rows : []).map((entry) => row(
      secondsPhrase(entry.length_seconds, locale),
      entry.min_percent !== undefined
        ? pageText(locale, `at least ${percentPhrase(entry.min_percent, locale)}`, `לפחות ${percentPhrase(entry.min_percent, locale)}`)
        : percentPhrase(entry.max_percent, locale),
    )),
  }),

  'cancellation-terms': (params, locale) => {
    const windows = Array.isArray(params.windows) ? params.windows : [];
    return {
      headline: pageText(
        locale,
        `What cancelling costs, by how late it comes: ${windows.length} bands.`,
        `מה עולה ביטול, לפי מועדו: ${formatNumber(windows.length, locale)} מדרגות.`,
      ),
      rows: [quoted(pageText(locale, 'How the days are counted', 'כיצד נמנים הימים'), params.notes)],
      table: {
        caption: pageText(locale, 'Cancellation fees', 'דמי ביטול'),
        columns: [
          { key: 'when', label: pageText(locale, 'Notice before air', 'הודעה לפני שידור'), numeric: true },
          { key: 'fee', label: pageText(locale, 'Fee', 'חיוב'), numeric: true },
        ],
        rows: windows.map((entry) => ({
          when: entry.days_before_air === 0
            ? pageText(locale, 'on the day', 'ביום השידור')
            : pageText(locale, `${formatNumber(entry.days_before_air, locale)} days or more`, `${formatNumber(entry.days_before_air, locale)} ימים ומעלה`),
          fee: percentPhrase(entry.fee_percent, locale),
        })),
      },
    };
  },

  'trp-delivery-guarantee': (params, locale) => ({
    headline: isMissing(params.points)
      ? pageText(
        locale,
        `The channel guarantees delivery in ${params.audience || 'an audience the document does not name'}, but the document does not state how many points. Standing cannot be measured until that number exists.`,
        `הערוץ מתחייב לאספקה ב${params.audience || 'קהל שהמסמך אינו נוקב בשמו'}, אך המסמך אינו נוקב בכמה נקודות. אי אפשר למדוד עמידה עד שהמספר הזה יימצא.`,
      )
      : pageText(
        locale,
        `The channel guarantees ${formatNumber(params.points, locale)} rating points in ${params.audience || 'an audience the document does not name'}, with ${percentPhrase(params.tolerance_percent, locale)} tolerance.`,
        `הערוץ מתחייב ל${formatNumber(params.points, locale)} נקודות רייטינג ב${params.audience || 'קהל שהמסמך אינו נוקב בשמו'}, בסטייה מותרת של ${percentPhrase(params.tolerance_percent, locale)}.`,
      ),
    rows: [
      row(pageText(locale, 'Points guaranteed', 'הנקודות המתחייבות'),
        isMissing(params.points) ? null : formatNumber(params.points, locale)),
      row(pageText(locale, 'Audience', 'הקהל'), params.audience),
      row(pageText(locale, 'Tolerance before it is a shortfall', 'סטייה מותרת לפני שזה חוסר'), percentPhrase(params.tolerance_percent, locale)),
      row(pageText(locale, 'Measured over', 'נמדד על פני'), periodLabel(params.window, locale)),
      quoted(pageText(locale, 'Rating vintage', 'גרסת הרייטינג'), params.vintage),
      quoted(pageText(locale, 'Checkpoints along the way', 'נקודות ביקורת בדרך'), params.checkpoints),
    ],
  }),

  'effective-cpp-cap': (params, locale) => ({
    headline: pageText(
      locale,
      `Whatever the rate card says, the effective cost per point may not exceed ${contractMoney(params.cap, locale)} in ${params.audience || 'the stated audience'}.`,
      `מה שלא יאמר המחירון, העלות האפקטיבית לנקודה לא תעלה על ${contractMoney(params.cap, locale)} ב${params.audience || 'הקהל הנקוב'}.`,
    ),
    rows: [
      row(pageText(locale, 'Cap per point', 'התקרה לנקודה'), contractMoney(params.cap, locale)),
      row(pageText(locale, 'Audience', 'הקהל'), params.audience),
      row(pageText(locale, 'Measured over', 'נמדד על פני'), periodLabel(params.window, locale)),
      quoted(pageText(locale, 'What happens if it is breached', 'מה קורה בחריגה'), params.true_up_form),
    ],
  }),

  'preferred-position-guarantee': (params, locale) => ({
    headline: pageText(
      locale,
      `${percentPhrase(params.target_percent, locale)} of spots must land in a preferred position: ${listPhrase(params.preferred_positions, locale)}. This one is live and will steer real placement.`,
      `${percentPhrase(params.target_percent, locale)} מהתשדירים חייבים ליפול במיקום מועדף: ${listPhrase(params.preferred_positions, locale)}. הסעיף הזה פעיל ויטה שיבוץ אמיתי.`,
    ),
    rows: [
      row(pageText(locale, 'Positions that count', 'המיקומים הנחשבים'), listPhrase(params.preferred_positions, locale)),
      row(pageText(locale, 'Share required', 'השיעור הנדרש'), percentPhrase(params.target_percent, locale)),
      row(pageText(locale, 'Counted per', 'נספר לפי'), params.counting_method),
      row(pageText(locale, 'Measured over', 'נמדד על פני'), periodLabel(params.window, locale)),
    ],
  }),

  'gold-break-allocation': (params, locale) => ({
    headline: pageText(
      locale,
      `${formatNumber(params.count, locale)} gold breaks reserved ${periodLabel(params.period, locale)}${params.first_refusal ? ', with first refusal on them' : ''}.`,
      `${formatNumber(params.count, locale)} ברייקי זהב מוקצים ${periodLabel(params.period, locale)}${params.first_refusal ? ', עם זכות סירוב ראשונה עליהם' : ''}.`,
    ),
    rows: [
      row(pageText(locale, 'Gold breaks allocated', 'ברייקי זהב מוקצים'), formatNumber(params.count, locale)),
      row(pageText(locale, 'Per', 'לכל'), periodLabel(params.period, locale)),
      row(pageText(locale, 'First refusal', 'זכות סירוב ראשונה'),
        params.first_refusal === undefined ? null
          : (params.first_refusal ? pageText(locale, 'yes', 'כן') : pageText(locale, 'no', 'לא'))),
    ],
  }),

  'makegood-accrual-policy': (params, locale) => {
    const accruals = Array.isArray(params.accruals) ? params.accruals : [];
    return {
      headline: pageText(
        locale,
        `A credit balance the counterparty accrues and spends later: ${accruals.length} accrual rules.`,
        `יתרת זכות שהצד השני צובר ומנצל בהמשך: ${formatNumber(accruals.length, locale)} כללי צבירה.`,
      ),
      rows: [
        quoted(pageText(locale, 'How it may be used', 'כיצד ניתן לנצל'), params.utilisation),
        quoted(pageText(locale, 'When it expires', 'מתי פג'), params.expiry),
        quoted(pageText(locale, 'What quality of airtime it buys', 'איזו איכות מלאי'), params.quality_note),
      ],
      table: accruals.length ? {
        caption: pageText(locale, 'Accrual rules', 'כללי הצבירה'),
        columns: [
          { key: 'trigger', label: pageText(locale, 'What triggers it', 'מה מפעיל') },
          { key: 'rate', label: pageText(locale, 'Rate', 'שיעור'), numeric: true },
          { key: 'level', label: pageText(locale, 'Held at', 'מוחזק ברמת') },
        ],
        rows: accruals.map((entry) => ({
          trigger: entry.trigger,
          rate: percentPhrase(entry.rate_percent, locale),
          level: entry.level,
        })),
      } : null,
    };
  },

  'shortfall-cure': (params, locale) => ({
    headline: pageText(
      locale,
      `If delivery falls short, the channel cures it with ${pick(CURE_COPY, params.cure_form, locale)}. Which form it is decides whether the shortfall costs money or inventory.`,
      `אם האספקה תחסר, הערוץ משלים ב${pick(CURE_COPY, params.cure_form, locale)}. צורת ההשלמה קובעת אם החוסר עולה כסף או מלאי.`,
    ),
    rows: [
      row(pageText(locale, 'Form of cure', 'צורת ההשלמה'), pick(CURE_COPY, params.cure_form, locale)),
      quoted(pageText(locale, 'What quality it must be', 'באיזו איכות'), params.quality_rule),
      quoted(pageText(locale, 'By when', 'עד מתי'), params.cure_window),
      quoted(pageText(locale, 'How a missing point is valued', 'כיצד מוערכת נקודה חסרה'), params.valuation_basis),
    ],
  }),

  'underspend-true-up': (params, locale) => ({
    headline: pageText(
      locale,
      'If the committed volume is not reached, the discount already given is recalculated and the difference comes back.',
      'אם ההיקף המתחייב לא הושג, ההנחה שניתנה מחושבת מחדש וההפרש מוחזר.',
    ),
    rows: [
      quoted(pageText(locale, 'What triggers it', 'מה מפעיל'), params.trigger_note),
      quoted(pageText(locale, 'How it is recalculated', 'כיצד מחושב מחדש'), params.re_rating_rule),
    ],
  }),

  'overdelivery-treatment': (params, locale) => ({
    headline: pageText(
      locale,
      `Delivering more than promised: ${pick(TREATMENT_COPY, params.treatment, locale)}.`,
      `אספקה מעל למובטח: ${pick(TREATMENT_COPY, params.treatment, locale)}.`,
    ),
    rows: [
      row(pageText(locale, 'Treatment', 'הטיפול'), pick(TREATMENT_COPY, params.treatment, locale)),
      quoted(pageText(locale, 'Up to what point', 'עד לאיזה גבול'), params.banking_cap),
    ],
  }),

  'preemption-compensation': (params, locale) => ({
    headline: pageText(
      locale,
      'What the channel owes when it pulls a spot of its own accord: a news special or a schedule change does not cancel the obligation.',
      'מה חייב הערוץ כשהוא מוריד תשדיר מיוזמתו: מהדורה מיוחדת או שינוי לוח אינם מבטלים את החובה.',
    ),
    rows: [
      quoted(pageText(locale, 'What counts as pre-emption', 'מה נחשב הורדה'), params.qualifying_events),
      quoted(pageText(locale, 'The remedy', 'הסעד'), params.remedy_form),
      quoted(pageText(locale, 'Within', 'בתוך'), params.window),
      quoted(pageText(locale, 'Quality the replacement must match', 'איכות שהחלופה חייבת לשמור'), params.quality_rule),
    ],
  }),

  'competitive-separation': (params, locale) => ({
    headline: params.separation_unit === 'same_break'
      ? pageText(
        locale,
        `No competitor in ${params.category || 'the stated category'} may sit in the same break. ${params.hard ? 'A hard rule: the planner will not place one.' : 'A soft rule: the planner will avoid it and say when it could not.'}`,
        `אף מתחרה ב${params.category || 'הקטגוריה הנקובה'} לא ישובץ באותו מקבץ. ${params.hard ? 'כלל קשיח: המתכנן לא ישבץ כך.' : 'כלל רך: המתכנן ימנע, ויאמר כשלא הצליח.'}`,
      )
      : pageText(
        locale,
        `At least ${formatNumber(params.separation_quantity, locale)} spots must separate this advertiser from a competitor in ${params.category || 'the stated category'}.`,
        `לפחות ${formatNumber(params.separation_quantity, locale)} תשדירים יפרידו בין המפרסם לבין מתחרה ב${params.category || 'הקטגוריה הנקובה'}.`,
      ),
    rows: [
      row(pageText(locale, 'Category', 'הקטגוריה'), params.category),
      row(pageText(locale, 'Separation', 'ההפרדה'), params.separation_unit === 'same_break'
        ? pageText(locale, 'never in the same break', 'לא באותו מקבץ')
        : pageText(locale, `${formatNumber(params.separation_quantity, locale)} spots`, `${formatNumber(params.separation_quantity, locale)} תשדירים`)),
      row(pageText(locale, 'Hard or soft', 'קשיח או רך'), params.hard === undefined ? null
        : (params.hard ? pageText(locale, 'hard, never broken', 'קשיח, לא נשבר')
          : pageText(locale, 'soft, avoided and reported', 'רך, נמנע ומדווח'))),
    ],
  }),

  'category-exclusivity': (params, locale) => ({
    headline: pageText(
      locale,
      `${params.category || 'A category'} is sold to this advertiser alone within the stated scope${isMissing(params.premium_percent) ? '' : `, at a ${percentPhrase(params.premium_percent, locale)} premium`}. Exclusivity removes inventory from every other buyer in that category.`,
      `${params.category || 'קטגוריה'} נמכרת למפרסם הזה בלבד בהיקף הנקוב${isMissing(params.premium_percent) ? '' : `, בתוספת של ${percentPhrase(params.premium_percent, locale)}`}. בלעדיות מוציאה מלאי מכל קונה אחר באותה קטגוריה.`,
    ),
    rows: [
      row(pageText(locale, 'Category', 'הקטגוריה'), params.category),
      row(pageText(locale, 'Premium paid for it', 'התוספת עבורה'), percentPhrase(params.premium_percent, locale)),
      quoted(pageText(locale, 'Where it applies', 'היכן חלה'), params.exclusivity_scope),
    ],
  }),

  'content-adjacency-exclusion': (params, locale) => {
    const excluded = Array.isArray(params.excluded_content) ? params.excluded_content : [];
    return {
      headline: pageText(
        locale,
        `The spot must be kept away from ${excluded.length} kinds of content. ${params.hard ? 'A hard rule: the planner will not place it there.' : 'A soft rule: avoided where possible.'}`,
        `יש להרחיק את התשדיר מ${formatNumber(excluded.length, locale)} סוגי תוכן. ${params.hard ? 'כלל קשיח: המתכנן לא ישבץ שם.' : 'כלל רך: נמנע במידת האפשר.'}`,
      ),
      rows: [
        row(pageText(locale, 'Content to keep away from', 'התוכן שיש להרחיק ממנו'), excluded.length ? excluded.join(', ') : null),
        row(pageText(locale, 'How far', 'באיזה טווח'), params.radius === 'same_break'
          ? pageText(locale, 'not in the same break', 'לא באותו מקבץ') : params.radius),
      ],
    };
  },

  'adjacency-purchase': (params, locale) => ({
    headline: pageText(
      locale,
      `Placement bought for what it sits beside: ${params.target_content || 'content the document names'}.`,
      `שיבוץ שנרכש בשל מה שהוא צמוד אליו: ${params.target_content || 'תוכן שהמסמך נוקב בו'}.`,
    ),
    rows: [
      quoted(pageText(locale, 'Beside what', 'צמוד למה'), params.target_content),
      quoted(pageText(locale, 'Which break', 'איזה מקבץ'), params.break_relation),
      quoted(pageText(locale, 'What it costs extra', 'מה התוספת'), params.premium_note),
    ],
  }),

  'programme-daypart-restrictions': (params, locale) => ({
    headline: params.mode === 'allow'
      ? pageText(
        locale,
        'The spot may run ONLY inside the scope marked below. Everything outside it is off limits.',
        'התשדיר ישודר רק בהיקף המסומן להלן. כל מה שמחוצה לו אסור.',
      )
      : pageText(
        locale,
        'The spot may NOT run inside the scope marked below. Everywhere else is open.',
        'התשדיר לא ישודר בהיקף המסומן להלן. כל היתר פתוח.',
      ),
    rows: [
      row(pageText(locale, 'Rule', 'הכלל'), params.mode === 'allow'
        ? pageText(locale, 'only here', 'רק כאן') : pageText(locale, 'not here', 'לא כאן')),
      row(pageText(locale, 'Hard or soft', 'קשיח או רך'), params.hard === undefined ? null
        : (params.hard ? pageText(locale, 'hard, never broken', 'קשיח, לא נשבר')
          : pageText(locale, 'soft, avoided and reported', 'רך, נמנע ומדווח'))),
    ],
  }),

  'position-entitlements': (params, locale) => ({
    headline: pageText(
      locale,
      `Positions inside the break this advertiser is entitled to: ${listPhrase(params.positions, locale)}${params.top_and_tail ? ', including the pair that opens and closes it' : ''}.`,
      `מיקומים בתוך המקבץ שהמפרסם זכאי להם: ${listPhrase(params.positions, locale)}${params.top_and_tail ? ', ובכללם הצמד הפותח והסוגר' : ''}.`,
    ),
    rows: [
      row(pageText(locale, 'Positions', 'המיקומים'), listPhrase(params.positions, locale)),
      row(pageText(locale, 'Opens and closes the break', 'פותח וסוגר את המקבץ'),
        params.top_and_tail === undefined ? null
          : (params.top_and_tail ? pageText(locale, 'yes', 'כן') : pageText(locale, 'no', 'לא'))),
    ],
  }),

  'creative-constraints': (params, locale) => {
    const rules = Array.isArray(params.rules) ? params.rules : [];
    return {
      headline: pageText(
        locale,
        `${rules.length} rules about the material itself: delivery deadlines, house numbers and clearance. A spot with no valid clearance does not air.`,
        `${formatNumber(rules.length, locale)} כללים על החומר עצמו: מועדי מסירה, מספרי בית ואישור שידור. תשדיר בלי אישור שידור בתוקף אינו משודר.`,
      ),
      rows: [
        ...rules.map((entry, index) => quoted(
          pageText(locale, `Rule ${index + 1}`, `כלל ${formatNumber(index + 1, locale)}`), entry.note)),
        row(pageText(locale, 'Clearance gate', 'שער אישור שידור'),
          params.qc_gate === undefined ? null
            : (params.qc_gate ? pageText(locale, 'yes, clearance is required before air', 'כן, נדרש אישור לפני שידור')
              : pageText(locale, 'no', 'לא'))),
      ],
    };
  },

  'spot-length-constraints': (params, locale) => ({
    headline: pageText(
      locale,
      `Only these lengths may be placed: ${(params.allowed_lengths_seconds || []).map((v) => formatNumber(v, locale)).join(', ')} seconds.`,
      `רק אורכים אלה ישובצו: ${(params.allowed_lengths_seconds || []).map((v) => formatNumber(v, locale)).join(', ')} שניות.`,
    ),
    rows: [
      row(pageText(locale, 'Allowed lengths', 'אורכים מותרים'),
        (params.allowed_lengths_seconds || []).length
          ? (params.allowed_lengths_seconds || []).map((v) => formatNumber(v, locale)).join(', ') : null),
      quoted(pageText(locale, 'Anything else', 'כל אורך אחר'), params.notes),
    ],
  }),

  'frequency-caps': (params, locale) => {
    const unit = {
      break: { he: 'מקבץ', en: 'break' },
      hour: { he: 'שעה', en: 'hour' },
      day: { he: 'יום', en: 'day' },
      week: { he: 'שבוע', en: 'week' },
    }[String(params.unit || '')] || { he: params.unit, en: params.unit };
    return {
      headline: pageText(
        locale,
        `At most ${formatNumber(params.cap, locale)} spots per ${unit.en}. This one is live and the planner will not exceed it.`,
        `לכל היותר ${formatNumber(params.cap, locale)} תשדירים ל${unit.he}. הסעיף הזה פעיל והמתכנן לא יחרוג ממנו.`,
      ),
      rows: [
        row(pageText(locale, 'Cap', 'התקרה'), formatNumber(params.cap, locale)),
        row(pageText(locale, 'Per', 'לכל'), locale === 'he' ? unit.he : unit.en),
      ],
    };
  },

  'payment-terms': (params, locale) => ({
    headline: pageText(
      locale,
      `Payment terms: ${params.terms || 'the document does not state them'}.`,
      `תנאי תשלום: ${params.terms || 'המסמך אינו נוקב בהם'}.`,
    ),
    rows: [
      quoted(pageText(locale, 'Terms', 'התנאים'), params.terms),
      quoted(pageText(locale, 'Billing cycle', 'מחזור החיוב'), params.billing_cycle),
    ],
  }),

  'settlement-mechanics': (params, locale) => ({
    headline: pageText(
      locale,
      `How the invoice is actually computed: ${params.grain || 'the document does not state the grain'}. The order the discounts and the commission are applied in changes the amount.`,
      `כיצד מחושבת החשבונית בפועל: ${params.grain || 'המסמך אינו נוקב בגרעין המדידה'}. הסדר שבו מוחלות ההנחות והעמלה משנה את הסכום.`,
    ),
    rows: [
      quoted(pageText(locale, 'What is measured, at what grain', 'מה נמדד, באיזה גרעין'), params.grain),
      quoted(pageText(locale, 'How often', 'באיזו תכיפות'), params.cadence),
      quoted(pageText(locale, 'In what order the terms apply', 'באיזה סדר מוחלים התנאים'), params.application_order),
    ],
  }),

  'measurement-source': (params, locale) => ({
    headline: pageText(
      locale,
      `Whose rating figure settles this agreement: ${params.source || 'the document does not name a source'}, on ${params.audience_basis || 'a basis it does not state'}.`,
      `נתון הרייטינג של מי מכריע בהסכם הזה: ${params.source || 'המסמך אינו נוקב במקור'}, על בסיס ${params.audience_basis || 'שאינו נקוב'}.`,
    ),
    rows: [
      row(pageText(locale, 'Source', 'המקור'), params.source),
      quoted(pageText(locale, 'Audience basis', 'בסיס הקהל'), params.audience_basis),
      quoted(pageText(locale, 'Which vintage of the figure', 'איזו גרסה של הנתון'), params.vintage),
      quoted(pageText(locale, 'Which figure is final', 'איזה נתון סופי'), params.final_rule),
    ],
  }),

  'delivery-truth-source': (params, locale) => ({
    headline: pageText(
      locale,
      'Which record decides what actually aired, when two records disagree.',
      'איזה רשומה קובעת מה שודר בפועל, כששתי רשומות חולקות.',
    ),
    rows: [quoted(pageText(locale, 'Order of authority', 'סדר הקדימות'), params.source_order)],
  }),

  'term-effective-windows': (params, locale) => ({
    headline: pageText(
      locale,
      'One clause has its own window, different from the agreement it sits in. Two versions of the same rate never apply on the same day.',
      'לסעיף אחד חלון תוקף משלו, שונה מזה של ההסכם שבתוכו. שתי גרסאות של אותו תעריף אינן חלות באותו יום.',
    ),
    rows: [
      quoted(pageText(locale, 'Which clause', 'איזה סעיף'), params.applies_to),
      quoted(pageText(locale, 'For how long', 'למשך מתי'), params.window_note),
    ],
  }),
};

// The process and legal terms all carry {summary, deadlines, details}: what the
// clause says and when it bites. One describer serves all of them rather than
// eight that differ only in their heading.
export const PROCESS_TERMS = [
  'payment-indexation', 'audit-rights', 'termination', 'confidentiality',
  'credit-security', 'dispute-resolution', 'reporting-obligations', 'force-majeure',
  'regional-feed-splits', 'coop-invoicing', 'barter-inquiry',
];

export function processDescriber(params, locale) {
  const deadlines = Array.isArray(params.deadlines) ? params.deadlines : [];
  const reports = Array.isArray(params.reports) ? params.reports : [];
  const rows = [
    quoted(pageText(locale, 'What the clause says', 'מה הסעיף אומר'), params.summary || params.details),
    quoted(pageText(locale, 'What qualifies', 'מה מקים את הסעיף'), params.qualifying_events),
    quoted(pageText(locale, 'What follows', 'מה נובע'), params.relief),
    row(pageText(locale, 'Notice required', 'הודעה נדרשת'),
      isMissing(params.notice_days) ? null
        : pageText(locale, `${formatNumber(params.notice_days, locale)} days`, `${formatNumber(params.notice_days, locale)} ימים`)),
    row(pageText(locale, 'What survives the end', 'מה שורד את הסיום'),
      Array.isArray(params.survival) && params.survival.length ? params.survival.join(' · ') : null),
    ...reports.map((entry) => quoted(entry.cadence || pageText(locale, 'Report', 'דוח'), entry.report)),
    ...deadlines.map((entry) => row(entry.label, entry.on ? formatDay(entry.on) : null)),
  ];
  // A process schema holds eight optional fields and a given clause uses two or
  // three of them. Printing the other five as gaps would turn "this clause has
  // no deadline" into "a deadline is missing", so an absent optional field is
  // dropped here rather than reported. The REQUIRED field is the summary, and
  // the headline names its absence when the extraction has none.
  return {
    headline: params.summary
      ? String(params.summary)
      : pageText(locale, 'A process clause with no summary extracted.', 'סעיף תהליכי שלא חולץ לו תקציר.'),
    headlineIsQuote: Boolean(params.summary),
    rows: rows.filter((entry) => entry.value !== undefined && entry.value !== null),
  };
}
