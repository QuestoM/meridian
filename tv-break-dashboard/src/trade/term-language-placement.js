// Plain language for the terms that decide WHERE a spot may go and HOW delivery
// is proven: placement constraints (F), process and law (G), and measurement and
// settlement (H).
//
// Split from term-language-terms.js at the project's file-size boundary, along
// the seam the taxonomy already draws: everything here answers "may this spot be
// placed, and how do we agree on what aired", while its sibling answers "who is
// covered and what did each side undertake". Both halves keep the same
// {headline, rows, table} contract and the same row helpers.
//
// The clause's own words are carried through as quoted rows wherever the schema
// stores free text, because on a placement constraint the exact wording is the
// operative fact and a paraphrase would be a second, weaker agreement.

import { formatDay } from '../shell/dates';
import { formatNumber, pageText } from '../shell/format';
import { isMissing, listPhrase, percentPhrase } from './term-language';
import {
  quoted,
  row,
} from './term-language-rows';


export const PLACEMENT_DESCRIBERS = {
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

