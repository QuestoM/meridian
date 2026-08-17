// The trade-term vocabulary, mirrored from kairos/trade/taxonomy.py.
//
// The API sends a term_id and nothing else: the Hebrew name a reviewer reads,
// the family it is grouped under, and the honesty status that says whether
// approving it will change behaviour all live in the Python registry. This
// module is that registry's twin on this side of the wire, generated from it so
// a name cannot be mistyped, and tests/test_trade_ui_vocabulary.py pins the two
// to the same term list so neither drifts silently.
//
// STATUS is the vocabulary that keeps the review honest, and it is the reason
// this file carries more than names:
//
//   BINDS           representable AND changes behaviour through live machinery
//   REPRESENTABLE   held faithfully; the binding path is not wired yet
//   TRACKED         stored, measured and surfaced; consequences stay human
//   RECORDED        stored and shown with deadline tracking only
//   NOT_APPLICABLE  a known foreign structure with no Israeli evidence
//
// RANK is provenance: IL is an Israeli primary source, TRADE is the owner's
// media-professional transcript, STD is standard practice not yet attested
// locally. Interface copy must never assert an STD term as local market fact.

export const TERM_FAMILIES = {
  A: "זהות, היקף ומסמך",
  B: "בסיס הכסף",
  C: "הנחות, עמלות ותמריצים",
  D: "התחייבויות המפרסם",
  E: "התחייבויות הערוץ והשלמות",
  F: "אילוצי שיבוץ",
  G: "תהליך ומשפט",
  H: "מדידה והתחשבנות",
  NA: "לא רלוונטי לשוק הישראלי",
};

// Clause classes that are commercially irrelevant by design. The list is closed
// so "irrelevant" can never become a silent dumping ground.
export const IRRELEVANT_CLASSES = {
  'signature-block': "בלוק חתימות",
  'notice-addresses': "כתובות למשלוח הודעות",
  'counterparts-execution': "עותקים וחתימה",
  'severability': "הפרדת סעיפים בטלים",
  'headings-interpretation': "כותרות לנוחות בלבד",
  'page-furniture': "כותרות עמוד, מספור וסימון",
  'table-of-contents': "תוכן עניינים",
  'preamble-recitals': "מבוא והואיל",
};

export const TERMS = {
  'agreement-parties': { family: 'A', he: "הצדדים להסכם", en: "Agreement parties", status: 'REPRESENTABLE', rank: 'TRADE' },
  'brand-scope': { family: 'A', he: "היקף מותגים", en: "Brand scope", status: 'REPRESENTABLE', rank: 'STD' },
  'channel-scope': { family: 'A', he: "היקף ערוצים", en: "Channel scope", status: 'REPRESENTABLE', rank: 'STD' },
  'effective-window': { family: 'A', he: "תקופת תוקף", en: "Effective window", status: 'REPRESENTABLE', rank: 'STD' },
  'agreement-level': { family: 'A', he: "רמת ההסכם", en: "Agreement level", status: 'REPRESENTABLE', rank: 'TRADE' },
  'precedence-clause': { family: 'A', he: "סעיף עדיפות", en: "Precedence clause", status: 'REPRESENTABLE', rank: 'STD' },
  'definitions': { family: 'A', he: "הגדרות", en: "Definitions", status: 'REPRESENTABLE', rank: 'STD' },
  'amendment-layer': { family: 'A', he: "תיקון / נספח", en: "Amendment / appendix", status: 'REPRESENTABLE', rank: 'TRADE' },
  'cpp-daypart-table': { family: 'B', he: "טבלת CPP לפי רצועה", en: "CPP by daypart", status: 'REPRESENTABLE', rank: 'IL' },
  'target-cpp': { family: 'B', he: "CPP לקהל יעד", en: "Target-audience CPP", status: 'REPRESENTABLE', rank: 'IL' },
  'length-factor-table': { family: 'B', he: "מקדמי אורך", en: "Length factor table", status: 'REPRESENTABLE', rank: 'IL' },
  'ratecard-index': { family: 'B', he: "הצמדה למחירון", en: "Rate-card index", status: 'REPRESENTABLE', rank: 'STD' },
  'fixed-spot-pricing': { family: 'B', he: "מחיר קבוע לספוט", en: "Fixed spot pricing", status: 'REPRESENTABLE', rank: 'IL' },
  'sponsorship-terms': { family: 'B', he: "תנאי חסות", en: "Sponsorship terms", status: 'REPRESENTABLE', rank: 'IL' },
  'gold-break-rates': { family: 'B', he: "מחירון ברייק זהב", en: "Gold-break rates", status: 'REPRESENTABLE', rank: 'IL' },
  'payment-indexation': { family: 'B', he: "הצמדה וריבית", en: "Indexation and interest", status: 'RECORDED', rank: 'STD' },
  'volume-discount-ladder': { family: 'C', he: "מדרגות הנחת היקף", en: "Volume discount ladder", status: 'REPRESENTABLE', rank: 'STD' },
  'share-bonus': { family: 'C', he: "תמריץ נתח", en: "Share bonus", status: 'TRACKED', rank: 'STD' },
  'seasonal-coefficients': { family: 'C', he: "מקדמי עונתיות", en: "Seasonal coefficients", status: 'REPRESENTABLE', rank: 'IL' },
  'agency-commission': { family: 'C', he: "עמלת סוכנות", en: "Agency commission", status: 'BINDS', rank: 'TRADE' },
  'cash-discount': { family: 'C', he: "הנחת מזומן", en: "Cash discount", status: 'RECORDED', rank: 'STD' },
  'success-deal': { family: 'C', he: "עסקת הצלחה", en: "Success deal", status: 'TRACKED', rank: 'TRADE' },
  'added-value-media': { family: 'C', he: "מדיה נוספת קבועה", en: "Added-value media", status: 'REPRESENTABLE', rank: 'TRADE' },
  'new-business-incentive': { family: 'C', he: "תמריץ לקוח חדש", en: "New-business incentive", status: 'REPRESENTABLE', rank: 'STD' },
  'package-bundle': { family: 'C', he: "חבילה משולבת", en: "Package bundle", status: 'REPRESENTABLE', rank: 'IL' },
  'budget-commitment': { family: 'D', he: "התחייבות תקציב", en: "Budget commitment", status: 'REPRESENTABLE', rank: 'TRADE' },
  'share-commitment': { family: 'D', he: "התחייבות נתח", en: "Share commitment", status: 'TRACKED', rank: 'STD' },
  'daypart-mix': { family: 'D', he: "תמהיל רצועות", en: "Daypart mix", status: 'REPRESENTABLE', rank: 'STD' },
  'flighting-obligation': { family: 'D', he: "התחייבות רציפות", en: "Flighting obligation", status: 'TRACKED', rank: 'STD' },
  'length-mix': { family: 'D', he: "תמהיל אורכים", en: "Length mix", status: 'TRACKED', rank: 'STD' },
  'cancellation-terms': { family: 'D', he: "תנאי ביטול", en: "Cancellation terms", status: 'RECORDED', rank: 'TRADE' },
  'trp-delivery-guarantee': { family: 'E', he: "התחייבות נקודות רייטינג", en: "TRP delivery guarantee", status: 'REPRESENTABLE', rank: 'IL' },
  'effective-cpp-cap': { family: 'E', he: "תקרת CPP אפקטיבי", en: "Effective-CPP cap", status: 'REPRESENTABLE', rank: 'STD' },
  'preferred-position-guarantee': { family: 'E', he: "התחייבות מיקומים מועדפים", en: "Preferred-position guarantee", status: 'BINDS', rank: 'TRADE' },
  'gold-break-allocation': { family: 'E', he: "הקצאת ברייקי זהב", en: "Gold-break allocation", status: 'REPRESENTABLE', rank: 'IL' },
  'makegood-accrual-policy': { family: 'E', he: "מדיניות צבירת מייק גוד", en: "Make-good accrual policy", status: 'REPRESENTABLE', rank: 'TRADE' },
  'shortfall-cure': { family: 'E', he: "מנגנון השלמה", en: "Shortfall cure", status: 'REPRESENTABLE', rank: 'TRADE' },
  'underspend-true-up': { family: 'E', he: "התחשבנות חוסר ניצול", en: "Under-spend true-up", status: 'REPRESENTABLE', rank: 'STD' },
  'overdelivery-treatment': { family: 'E', he: "טיפול בעודף אספקה", en: "Over-delivery treatment", status: 'REPRESENTABLE', rank: 'STD' },
  'preemption-compensation': { family: 'E', he: "פיצוי על הקדמת שידור", en: "Pre-emption compensation", status: 'REPRESENTABLE', rank: 'TRADE' },
  'competitive-separation': { family: 'F', he: "הפרדה תחרותית", en: "Competitive separation", status: 'BINDS', rank: 'TRADE' },
  'category-exclusivity': { family: 'F', he: "בלעדיות קטגוריה", en: "Category exclusivity", status: 'REPRESENTABLE', rank: 'STD' },
  'content-adjacency-exclusion': { family: 'F', he: "הרחקה מתוכן", en: "Content adjacency exclusion", status: 'BINDS', rank: 'STD' },
  'adjacency-purchase': { family: 'F', he: "רכישת סמיכות", en: "Adjacency purchase", status: 'REPRESENTABLE', rank: 'IL' },
  'programme-daypart-restrictions': { family: 'F', he: "הגבלות תוכניות ורצועות", en: "Programme/daypart restrictions", status: 'BINDS', rank: 'TRADE' },
  'position-entitlements': { family: 'F', he: "זכויות מיקום בברייק", en: "Position entitlements", status: 'BINDS', rank: 'TRADE' },
  'creative-constraints': { family: 'F', he: "אילוצי חומרים", en: "Creative constraints", status: 'BINDS', rank: 'TRADE' },
  'spot-length-constraints': { family: 'F', he: "אילוצי אורך", en: "Spot-length constraints", status: 'BINDS', rank: 'TRADE' },
  'frequency-caps': { family: 'F', he: "תקרות תדירות", en: "Frequency caps", status: 'BINDS', rank: 'STD' },
  'payment-terms': { family: 'G', he: "תנאי תשלום", en: "Payment terms", status: 'RECORDED', rank: 'STD' },
  'reporting-obligations': { family: 'G', he: "חובות דיווח", en: "Reporting obligations", status: 'RECORDED', rank: 'STD' },
  'audit-rights': { family: 'G', he: "זכויות ביקורת", en: "Audit rights", status: 'RECORDED', rank: 'STD' },
  'termination': { family: 'G', he: "סיום ההסכם", en: "Termination", status: 'RECORDED', rank: 'STD' },
  'force-majeure': { family: 'G', he: "כוח עליון", en: "Force majeure", status: 'RECORDED', rank: 'TRADE' },
  'confidentiality': { family: 'G', he: "סודיות", en: "Confidentiality", status: 'RECORDED', rank: 'IL' },
  'credit-security': { family: 'G', he: "בטחונות ואשראי", en: "Credit and security", status: 'RECORDED', rank: 'STD' },
  'dispute-resolution': { family: 'G', he: "יישוב מחלוקות", en: "Dispute resolution", status: 'RECORDED', rank: 'STD' },
  'settlement-mechanics': { family: 'H', he: "מנגנון התחשבנות", en: "Settlement mechanics", status: 'REPRESENTABLE', rank: 'IL' },
  'measurement-source': { family: 'H', he: "מקור מדידה", en: "Measurement source", status: 'REPRESENTABLE', rank: 'IL' },
  'delivery-truth-source': { family: 'H', he: "מקור אמת לשידור", en: "Delivery truth source", status: 'REPRESENTABLE', rank: 'TRADE' },
  'term-effective-windows': { family: 'H', he: "חלונות תוקף לסעיפים", en: "Per-term effective windows", status: 'REPRESENTABLE', rank: 'STD' },
  'regional-feed-splits': { family: 'NA', he: "פיצול שידור אזורי", en: "Regional feed splits", status: 'NOT_APPLICABLE', rank: 'STD' },
  'coop-invoicing': { family: 'NA', he: "חיוב משותף יצרן-קמעונאי", en: "Co-op invoicing", status: 'NOT_APPLICABLE', rank: 'STD' },
  'barter-inquiry': { family: 'NA', he: "עסקת ברטר / תשלום לפי פנייה", en: "Barter / per-inquiry", status: 'NOT_APPLICABLE', rank: 'STD' },
};

const STATUS_COPY = {
  BINDS: {
    he: 'משנה התנהגות',
    en: 'Changes behaviour',
    tone: 'positive',
    heNote: 'אישור ההסכם יחבר את הסעיף למנגנון פעיל: הוא ישנה תמחור, שיבוץ או התחשבנות.',
    enNote: 'Approval wires this clause into live machinery: it will change pricing, placement or settlement.',
  },
  REPRESENTABLE: {
    he: 'נשמר במלואו',
    en: 'Held in full',
    tone: 'info',
    heNote: 'הסעיף נשמר על כל פרטיו וניתן למדידה, אך אין עדיין נתיב שמפעיל אותו אוטומטית.',
    enNote: 'The clause is stored in full and can be measured, but no path activates it automatically yet.',
  },
  TRACKED: {
    he: 'נמדד, ההחלטה אנושית',
    en: 'Measured, decided by a person',
    tone: 'info',
    heNote: 'המערכת תמדוד את העמידה בסעיף ותציג אותה; מה לעשות בעקבותיה נשאר החלטה אנושית.',
    enNote: 'Standing against this clause is measured and shown; what to do about it stays a human decision.',
  },
  RECORDED: {
    he: 'נרשם עם מועדים',
    en: 'Recorded with deadlines',
    tone: 'neutral',
    heNote: 'הסעיף נרשם והמועדים שבו נמצאים במעקב; הוא אינו משנה תמחור או שיבוץ.',
    enNote: 'The clause is recorded and its deadlines are tracked; it does not change pricing or placement.',
  },
  NOT_APPLICABLE: {
    he: 'לא רלוונטי לשוק הישראלי',
    en: 'Not applicable in this market',
    tone: 'warning',
    heNote: 'מבנה מוכר משוק אחר, שאין לו עדות בשוק הישראלי. הוא מסומן בשמו כדי שסעיף כזה יסורב עם נימוק ולא יושמט בשקט.',
    enNote: 'A structure known from another market with no Israeli evidence. It is classified by name so a clause matching it is refused with a reason rather than dropped in silence.',
  },
};

const RANK_COPY = {
  IL: { he: 'מקור ישראלי ראשוני', en: 'Israeli primary source' },
  TRADE: { he: 'מתוך תמלול אנשי המקצוע', en: 'From the trade transcript' },
  STD: { he: 'נוהג מקובל, לא מאומת בשוק המקומי', en: 'Standard practice, not attested locally' },
};

export function termSpec(termId) {
  return TERMS[termId] || null;
}

// The name a reviewer reads. An unknown id returns the id itself rather than a
// blank: a term the server knows and this table does not is a drift to see, not
// a gap to paper over.
export function termName(termId, locale) {
  const spec = TERMS[termId];
  if (!spec) return String(termId || '');
  return locale === 'he' ? spec.he : spec.en;
}

export function familyName(family, locale) {
  const he = TERM_FAMILIES[family];
  if (!he) return String(family || '');
  return locale === 'he' ? he : FAMILY_EN[family] || he;
}

const FAMILY_EN = {
  A: 'Identity, scope and document',
  B: 'The money basis',
  C: 'Discounts, commissions and incentives',
  D: 'What the advertiser commits to',
  E: 'What the channel commits to',
  F: 'Placement constraints',
  G: 'Process and law',
  H: 'Measurement and settlement',
  NA: 'Not applicable in this market',
};

export const FAMILY_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'NA'];

export function statusCopy(termId, locale) {
  const spec = TERMS[termId];
  const entry = spec ? STATUS_COPY[spec.status] : null;
  if (!entry) return null;
  return {
    status: spec.status,
    tone: entry.tone,
    label: locale === 'he' ? entry.he : entry.en,
    note: locale === 'he' ? entry.heNote : entry.enNote,
  };
}

export function rankCopy(termId, locale) {
  const spec = TERMS[termId];
  const entry = spec ? RANK_COPY[spec.rank] : null;
  if (!entry) return null;
  return { rank: spec.rank, label: locale === 'he' ? entry.he : entry.en };
}

export function irrelevantClassName(key, locale) {
  const he = IRRELEVANT_CLASSES[key];
  if (!he) return String(key || '');
  return locale === 'he' ? he : IRRELEVANT_EN[key] || he;
}

const IRRELEVANT_EN = {
  'signature-block': 'Signature block',
  'notice-addresses': 'Addresses for notices',
  'counterparts-execution': 'Counterparts and execution',
  'severability': 'Severability',
  'headings-interpretation': 'Headings for convenience only',
  'page-furniture': 'Page furniture, numbering and marks',
  'table-of-contents': 'Table of contents',
  'preamble-recitals': 'Preamble and recitals',
};

// Every term id, grouped, for the add-a-missed-term picker.
export function termsByFamily() {
  const groups = FAMILY_ORDER.map((family) => ({
    family,
    terms: Object.entries(TERMS)
      .filter(([, spec]) => spec.family === family)
      .map(([id, spec]) => ({ id, ...spec })),
  }));
  return groups.filter((group) => group.terms.length > 0);
}
