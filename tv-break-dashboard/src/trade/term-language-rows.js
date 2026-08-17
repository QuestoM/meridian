// The row vocabulary the term describers share.
//
// Every describer returns {headline, rows, table} and every row is built by one
// of the two helpers here, so a missing value is reported the same way on all
// sixty-three terms: `row` marks an absent value as a GAP rather than printing a
// blank or a zero, and `quoted` marks a value as the document's own words so the
// surface can render it as a quotation instead of as interface copy.
//
// The closed vocabularies below are the other half of the same job. A schema
// stores `cure_form: "bonus_spots"`, and a reviewer has to read "bonus spots".
// `pick` performs that lookup and, for a key no vocabulary covers, returns the
// key itself rather than a blank: a value the extraction produced and this table
// does not know is drift to see, not a gap to paper over.

export function row(label, value, extra = {}) {
  if (value === null || value === undefined || value === '') {
    return { label, missing: true, ...extra };
  }
  return { label, value: String(value), ...extra };
}

export function quoted(label, value) {
  return row(label, value, { quote: true });
}

export function pick(map, key, locale) {
  const entry = map[String(key || '')];
  if (!entry) return key ? String(key) : null;
  return locale === 'he' ? entry.he : entry.en;
}

export const COUNTERPARTY_COPY = {
  agency: { he: 'סוכנות מדיה', en: 'a media agency' },
  advertiser: { he: 'מפרסם ישיר', en: 'a direct advertiser' },
  advertiser_via_agency: { he: 'מפרסם באמצעות סוכנות', en: 'an advertiser through an agency' },
};

export const LEVEL_COPY = {
  agency_framework: { he: 'הסכם מסגרת עם סוכנות', en: 'an agency framework' },
  advertiser: { he: 'הסכם מפרסם', en: 'an advertiser agreement' },
  campaign: { he: 'נספח קמפיין', en: 'a campaign appendix' },
};

export const CURE_COPY = {
  credit: { he: 'זיכוי כספי', en: 'a cash credit' },
  bonus_spots: { he: 'שידורי בונוס', en: 'bonus spots' },
  rate_adjustment: { he: 'תיקון תעריף', en: 'a rate adjustment' },
};

export const TREATMENT_COPY = {
  charged: { he: 'עודף האספקה מחויב', en: 'over-delivery is charged for' },
  absorbed: { he: 'עודף האספקה נספג ואינו מחויב', en: 'over-delivery is absorbed and not charged' },
  banked: { he: 'עודף האספקה נצבר לזכות עתידית', en: 'over-delivery is banked for later' },
};
