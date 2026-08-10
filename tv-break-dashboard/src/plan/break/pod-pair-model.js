function words(locale, en, he) {
  return locale === 'he' ? he : en;
}

function advertiserText(spot, locale) {
  const field = (spot && spot.advertiser) || {};
  return field.value || words(locale, 'unknown advertiser', 'מפרסם לא ידוע');
}

// Pair separation changes when a spot moves, so the screen judges the order it
// is showing rather than replaying the verdict the server took before the drag.
export function pairVerificationList(pairBlock, spots, locale) {
  const list = Array.isArray(spots) ? spots : [];
  const index = new Map(list.map((spot, spotIndex) => [spot.spot_key, spotIndex]));
  const bySpot = new Map(list.map((spot) => [spot.spot_key, spot]));
  return (Array.isArray(pairBlock && pairBlock.verdicts) ? pairBlock.verdicts : [])
    .flatMap((verdict, verdictIndex) => {
      if (!index.has(verdict.lead_key) || !index.has(verdict.closer_key)) return [];
      const between = Math.abs(index.get(verdict.lead_key) - index.get(verdict.closer_key)) - 1;
      const low = Number(verdict.allowed_min);
      const high = Number(verdict.allowed_max);
      if (Number.isFinite(low) && Number.isFinite(high) && between >= low && between <= high) return [];
      const key = verdict.closer_key || verdict.lead_key;
      const spot = bySpot.get(key);
      return [{
        key: `pair-separation-${verdict.rule_id || verdictIndex}`,
        spotKey: key,
        advertiser: advertiserText(spot, locale),
        detail: words(locale,
          `This pair has ${between} other advertisements between its two creatives; the rule allows ${low} to ${high}`,
          `בצמד הזה יש ${between} תשדירים אחרים בין שני התשדירים; הכלל מתיר ${low} עד ${high}`),
        kind: 'pair_separation',
      }];
    });
}
