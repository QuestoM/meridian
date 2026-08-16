// The product's words, in both languages, in one place.
//
// Every label a surface renders comes from here, so a concept has exactly one
// name per language. Measured before this file existed: the optimizer balance
// wore five names across four surfaces, and one idea called "programme type"
// carried two value sets sharing only two members, so the same words named
// different things depending on the page.
//
// Two rules hold this file honest. One concept, one key, one word per
// language: a surface that wants a different word for the same thing has not
// agreed with the rest of the product yet, and the disagreement belongs here.
// And retired words are absent, not commented out and not kept as aliases,
// because an alias is how a retired word comes back.
//
// Keys are namespaced by kind: activity, action, place, object, concept,
// state, role. Read one with word(key, locale); read both with WORDS[key].

export const LOCALES = ['he', 'en'];

export const DEFAULT_LOCALE = 'he';

export const WORDS = {
  // --- the four activity classes, and the verb and output of each
  'activity.training': {
    en: 'training',
    he: 'אימון',
  },
  'activity.training.verb': {
    en: 'train',
    he: 'לאמן',
  },
  'activity.training.output': {
    en: 'model version',
    he: 'גרסת מודל',
  },
  'activity.run': {
    en: 'run',
    he: 'הרצה',
  },
  'activity.run.verb': {
    en: 'run',
    he: 'להריץ',
  },
  'activity.run.output': {
    en: 'plan version',
    he: 'גרסת תוכנית',
  },
  'activity.change': {
    en: 'change',
    he: 'שינוי',
  },
  'activity.change.verb': {
    en: 'save',
    he: 'לשמור',
  },
  'activity.publish': {
    en: 'publish',
    he: 'הפצה',
  },
  'activity.publish.verb': {
    en: 'publish',
    he: 'להפיץ',
  },
  // --- the buttons those verbs produce
  'action.run_plan': {
    en: 'Run the plan',
    he: 'הריצו את התוכנית',
  },
  'action.run_weekly_plan': {
    en: 'Run the weekly plan',
    he: 'הרצת הלוח השבועי',
  },
  'action.save_and_run': {
    en: 'Save and run',
    he: 'שמרו והריצו',
  },
  'action.preview': {
    en: 'Preview',
    he: 'תצוגה מקדימה',
  },
  'action.publish': {
    en: 'Publish',
    he: 'הפצה',
  },
  'action.train': {
    en: 'Train the model',
    he: 'לאמן את המודל',
  },
  // --- the nine addressable places
  'place.today': {
    en: 'Today',
    he: 'היום',
  },
  'place.plan': {
    en: 'Plan',
    he: 'תוכנית',
  },
  'place.plan.week': {
    en: 'Week',
    he: 'שבוע',
  },
  'place.plan.day': {
    en: 'Day',
    he: 'יום',
  },
  'place.plan.break': {
    en: 'Break',
    he: 'ברייק',
  },
  'place.clients': {
    en: 'Clients',
    he: 'לקוחות',
  },
  'place.rules': {
    en: 'Rules',
    he: 'כללים',
  },
  'place.sources': {
    en: 'Sources',
    he: 'מקורות',
  },
  'place.kai': {
    en: 'Mabat',
    he: 'מבט',
  },
  'place.history': {
    en: 'History',
    he: 'היסטוריה',
  },
  'place.account_settings': {
    en: 'Account settings',
    he: 'הגדרות חשבון',
  },
  'place.model_console': {
    en: 'Model console',
    he: 'קונסולת המודל',
  },
  // --- the objects a person opens
  'object.weekly_plan': {
    en: 'weekly plan',
    he: 'תוכנית שבועית',
  },
  'object.plan_version': {
    en: 'plan version',
    he: 'גרסת תוכנית',
  },
  'object.model_version': {
    en: 'model version',
    he: 'גרסת מודל',
  },
  'object.broadcast_day': {
    en: 'broadcast day',
    he: 'יום שידור',
  },
  'object.break': {
    en: 'break',
    he: 'ברייק',
  },
  'object.breaks': {
    en: 'breaks',
    he: 'ברייקים',
  },
  'object.break_contents': {
    en: 'break contents',
    he: 'תוכן הברייק',
  },
  'object.gold_break': {
    en: 'gold break',
    he: 'ברייק זהב',
  },
  'object.gold_breaks': {
    en: 'gold breaks',
    he: 'ברייקי זהב',
  },
  'object.tonight_breaks': {
    en: "tonight's breaks",
    he: 'הברייקים של הערב',
  },
  'object.spot': {
    en: 'spot',
    he: 'תשדיר',
  },
  'object.house_number': {
    en: 'House Number',
    he: 'House Number',
  },
  'object.broadcast_strip': {
    en: 'broadcast strip',
    he: 'רצועת שידור',
  },
  'object.pin': {
    en: 'pin',
    he: 'נעיצה',
  },
  'object.restriction': {
    en: 'restriction',
    he: 'הגבלה',
  },
  'object.target': {
    en: 'target',
    he: 'יעד',
  },
  'object.make_good': {
    en: 'make-good',
    he: 'פיצוי שידור',
  },
  'object.advertiser': {
    en: 'advertiser',
    he: 'מפרסם',
  },
  'object.agency': {
    en: 'agency',
    he: 'סוכנות',
  },
  'object.campaign': {
    en: 'campaign',
    he: 'קמפיין',
  },
  'object.operator': {
    en: 'operator',
    he: 'מפעיל',
  },
  'object.channel': {
    en: 'channel',
    he: 'ערוץ',
  },
  'object.source_file': {
    en: 'source file',
    he: 'קובץ מקור',
  },
  // --- the concepts that wore several names each
  'concept.revenue_balance': {
    en: 'Revenue and retention balance',
    he: 'איזון הכנסה מול צפייה',
  },
  'concept.caution': {
    en: 'Uncertainty caution',
    he: 'זהירות מול אי-ודאות',
  },
  'concept.retention_floor': {
    en: 'Minimum retention floor',
    he: 'רצפת צפייה מינימלית',
  },
  'concept.programme_genre': {
    en: 'Programme genre',
    he: "ז'אנר תוכנית",
  },
  'concept.pricing_class': {
    en: 'Pricing class',
    he: 'מחלקת תמחור',
  },
  'concept.expected_revenue': {
    en: 'Expected revenue',
    he: 'הכנסה צפויה',
  },
  'concept.retention_cost': {
    en: 'Retention cost',
    he: 'עלות שימור',
  },
  'concept.yield_per_second': {
    en: 'Yield per second',
    he: 'תשואה לשנייה',
  },
  'concept.supply': {
    en: 'Supply',
    he: 'היצע',
  },
  'concept.pacing': {
    en: 'Delivery pacing',
    he: 'קצב אספקה',
  },
  'concept.guardrail': {
    en: 'Regulatory guardrail',
    he: 'מגבלת רגולציה',
  },
  'concept.rate_card': {
    en: 'Rate card',
    he: 'מחירון',
  },
  // --- states a person reads on a control
  'state.plan_current': {
    en: 'Plan version, run at',
    he: 'גרסת תוכנית, הורצה ב',
  },
  'state.model_trained_at': {
    en: 'Model version, trained at',
    he: 'גרסת מודל, אומנה ב',
  },
  'state.newer_model_version': {
    en: 'A newer model version exists',
    he: 'קיימת גרסת מודל חדשה יותר',
  },
  'state.plan_out_of_date': {
    en: 'The saved plan is out of date',
    he: 'התוכנית השמורה אינה עדכנית',
  },
  // --- who the account is
  'role.admin': {
    en: 'Admin',
    he: 'ניהול',
  },
  'role.operator': {
    en: 'Operator',
    he: 'תפעול',
  },
  'role.viewer': {
    en: 'Viewer',
    he: 'צפייה',
  },
  'affiliation.company': {
    en: 'Company',
    he: 'חברה',
  },
  'affiliation.channel': {
    en: 'Channel',
    he: 'ערוץ',
  },
};

// The programme's editorial genre, as the plan file carries it. Fifteen values,
// measured on output/weekly_break_schedule.csv. This is what a scheduler reads.
export const PROGRAMME_GENRES = {
  Children: {
    en: 'Children',
    he: 'ילדים',
  },
  Comedy: {
    en: 'Comedy',
    he: 'קומדיה',
  },
  Digital: {
    en: 'Digital',
    he: 'דיגיטל',
  },
  Documentary: {
    en: 'Documentary',
    he: 'דוקומנטרי',
  },
  Drama: {
    en: 'Drama',
    he: 'דרמה',
  },
  Lifestyle: {
    en: 'Lifestyle',
    he: 'לייפסטייל',
  },
  'Morning Program': {
    en: 'Morning show',
    he: 'תוכנית בוקר',
  },
  Music: {
    en: 'Music',
    he: 'מוזיקה',
  },
  News: {
    en: 'News',
    he: 'חדשות',
  },
  Other: {
    en: 'Other',
    he: 'אחר',
  },
  Promo: {
    en: 'Promo',
    he: 'פרומו',
  },
  Reality: {
    en: 'Reality',
    he: 'ריאליטי',
  },
  Religious: {
    en: 'Religious',
    he: 'תוכן דתי',
  },
  'Special Event': {
    en: 'Special event',
    he: 'אירוע מיוחד',
  },
  'Talk Show': {
    en: 'Talk show',
    he: 'תוכנית אירוח',
  },
};

// The four-value axis the rate card and the retention model price on, measured
// in config/optimization_weights.yaml and models/tv_break_coefficients.json. It
// shares only News and Other with the genre set above, which is why the two
// carry different names here.
export const PRICING_CLASSES = {
  News: {
    en: 'News',
    he: 'חדשות',
  },
  PrimeShow1: {
    en: 'First prime show',
    he: 'תוכנית פריים ראשונה',
  },
  PrimeShow2: {
    en: 'Second prime show',
    he: 'תוכנית פריים שנייה',
  },
  Other: {
    en: 'Other',
    he: 'אחר',
  },
};

function pick(entry, locale) {
  if (!entry) return '';
  return locale === 'en' ? entry.en : entry.he;
}

// The one lookup. An unknown key returns an empty string rather than the key
// itself, so a missing word renders as a gap a reviewer notices instead of a
// snake_case token a person has to decode.
export function word(key, locale = DEFAULT_LOCALE) {
  return pick(WORDS[key], locale);
}

export function hasWord(key) {
  return Object.prototype.hasOwnProperty.call(WORDS, key);
}

// A programme's genre, by its value in the plan file. An unmapped value is
// returned unchanged, because the plan's own word is more honest than a blank.
export function genreLabel(value, locale = DEFAULT_LOCALE) {
  const entry = PROGRAMME_GENRES[value];
  return entry ? pick(entry, locale) : String(value || '');
}

export function pricingClassLabel(value, locale = DEFAULT_LOCALE) {
  const entry = PRICING_CLASSES[value];
  return entry ? pick(entry, locale) : String(value || '');
}

export function genreValues() {
  return Object.keys(PROGRAMME_GENRES);
}

export function pricingClassValues() {
  return Object.keys(PRICING_CLASSES);
}
