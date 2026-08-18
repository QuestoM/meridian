import { isolate } from '../shell/bidi';
// The client's own pricing rule: which stored row prices this client, what that
// row is worth, and every spelling that resolves to it.
//
// Two stores answer two different questions and the record has to keep them
// apart. `data/advertiser_names.csv` is the observed name space: who the
// advertiser is, written from the daily files the channel already produces, and
// nothing in the product writes it. `data/advertiser_rules.csv` is the pricing
// store: one of its rows becomes this client's rule the moment its `name` cell
// carries the client's name, and only then does that row's premium price the
// client's spots on the daily pricing path.
//
// The fold below mirrors `kairos/optimize/advertiser_rules_identity.py` exactly:
// surrounding and repeated whitespace, letter case, and the Hebrew geresh and
// gershayim against their ASCII equivalents. Nothing else, and never a fuzzy or
// edit-distance match, because a wrong advertiser is a wrong price.

// Typographic pairs that differ only in how the same Hebrew name was typed.
const PUNCTUATION_FOLD = [
  ['׳', "'"],
  ['״', '"'],
  ['‘', "'"],
  ['’', "'"],
  ['“', '"'],
  ['”', '"'],
];

export function normalizeName(raw) {
  let text = raw === null || raw === undefined ? '' : String(raw);
  PUNCTUATION_FOLD.forEach(([from, to]) => {
    text = text.split(from).join(to);
  });
  return text.split(/\s+/).filter(Boolean).join(' ').toLowerCase();
}

// Split a pipe-joined alias cell into its trimmed, non-empty parts.
export function splitAliases(raw) {
  return String(raw === null || raw === undefined ? '' : raw)
    .split('|')
    .map((part) => part.trim())
    .filter(Boolean);
}

export function joinAliases(aliases) {
  return (aliases || []).map((alias) => String(alias).trim()).filter(Boolean).join('|');
}

// Every token a rules row claims: its id, its name, its display name, its aliases.
export function rowTokens(row) {
  return [row?.advertiser_id, row?.name, row?.display_name, ...splitAliases(row?.aliases)]
    .map(normalizeName)
    .filter(Boolean);
}

// True when nothing binds this row to an advertiser, so it prices nothing.
//
// A row is identified by ANY of its tokens, not by the name column alone. The
// shipped rows carry their identity in advertiser_id and leave name empty, so
// the old test called every one of them unbound - and the client drawer then
// told the operator that the pricing store and the name store were out of
// step about a client whose row was sitting in the file.
export function isUnboundRow(row) {
  return rowTokens(row).length === 0;
}

// The rules row bound to this client, or null when no row carries its name.
//
// THE SERVER'S ANSWER WINS when it has one. It resolves identity through the
// name store - id, name, display name, every alias - and it sends the row it
// landed on as ``rules_row_id``. Matching here as well was how the surface
// came to contradict the server and report the contradiction as a data fault;
// an identity resolved twice is an identity that will disagree with itself.
// The token match below stays as the fallback for a client record that predates
// the field, so an older payload still finds its row.
export function ruleRowFor(client, rows) {
  const resolvedId = normalizeName(client?.rules_row_id);
  if (resolvedId) {
    const named = (rows || []).find(
      (row) => normalizeName(row?.advertiser_id) === resolvedId,
    );
    if (named) {
      return named;
    }
  }
  const wanted = new Set(
    [client?.advertiser, client?.shown_name, ...(client?.aliases || [])]
      .map(normalizeName)
      .filter(Boolean),
  );
  if (!wanted.size) {
    return null;
  }
  return (rows || []).find((row) => {
    if (isUnboundRow(row)) {
      return false;
    }
    return rowTokens(row).some((token) => wanted.has(token));
  }) || null;
}

// Every spelling of this client, each one carrying where it came from. The
// observed ones are what the daily files spelled; the rule ones are what the
// operator typed onto the pricing row. The client's own name is not repeated.
export function spellingsFor(client, row) {
  const seen = new Set([normalizeName(client?.advertiser)]);
  const out = [];
  const push = (text, source) => {
    const key = normalizeName(text);
    if (!key || seen.has(key)) {
      return;
    }
    seen.add(key);
    out.push({ text: String(text).trim(), source });
  };
  (client?.aliases || []).forEach((alias) => push(alias, 'observed'));
  if (row) {
    push(row.display_name, 'rule');
    splitAliases(row.aliases).forEach((alias) => push(alias, 'rule'));
  }
  return out;
}

// The next free rules-row id in the ADV_nn shape the store already uses. The id
// is an internal handle; the name is what makes the row this client's rule.
export function nextRuleId(rows) {
  let max = 0;
  (rows || []).forEach((row) => {
    const match = /^ADV_(\d+)$/i.exec(String(row?.advertiser_id || ''));
    if (match && Number(match[1]) > max) {
      max = Number(match[1]);
    }
  });
  return `ADV_${String(max + 1).padStart(2, '0')}`;
}

// A premium as the reader meets it elsewhere in the product: the multiplier and
// what it does to the rate card. A missing value reads as a dash, never as 1.
export function premiumText(value) {
  const premium = Number(value);
  if (value === null || value === undefined || !Number.isFinite(premium)) {
    return { multiplier: '-', delta: '' };
  }
  const percent = Math.round((premium - 1) * 100);
  return {
    multiplier: `${premium.toFixed(2)}x`,
    delta: percent === 0 ? '' : `${percent > 0 ? '+' : '−'}${Math.abs(percent)}%`,
  };
}

// Whether a typed spelling can be added: non-empty, not already held here, and
// not held by another rules row, which the API refuses with 409 anyway.
export function spellingRefusal(candidate, client, row, rows, locale) {
  const wanted = normalizeName(candidate);
  const he = locale === 'he';
  if (!wanted) {
    return he ? 'הקלידו כתיב.' : 'Type a spelling.';
  }
  const held = new Set([
    normalizeName(client?.advertiser),
    ...spellingsFor(client, row).map((entry) => normalizeName(entry.text)),
  ]);
  if (held.has(wanted)) {
    return he ? 'הכתיב הזה כבר רשום ללקוח.' : 'This spelling is already on this client.';
  }
  const other = (rows || []).find(
    (candidateRow) => candidateRow !== row && rowTokens(candidateRow).includes(wanted),
  );
  if (other) {
    return he
      ? `הכתיב הזה כבר שייך לשורת תמחור אחרת (${isolate(other.advertiser_id)}).`
      : `Another pricing row already holds this spelling (${other.advertiser_id}).`;
  }
  return '';
}

// The premium a typed value would store, or null when it is not a number the
// store may hold. Nothing is coerced silently: a bad value stops the write.
export function parsePremium(raw) {
  const text = String(raw === null || raw === undefined ? '' : raw).trim();
  if (!text) {
    return null;
  }
  const value = Number(text);
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }
  return value;
}

// ------------------------------------------------------ the rule that removed

// Which rule left these spots out of the money, in one line.
//
// A count and the bare word "rule" was what every summary carried: a reader was
// told money had been left out of a total, and not told what to change to get it
// back. The rule was always reachable — one drawer, one campaign, one spot down —
// and being reachable is not the same as being told.
//
// A level can involve more than one rule. When it does the largest is named and
// the rest are counted, because a tile is one line and "several rules" tells a
// reader nothing at all.
export function droppingRuleLine(rules, locale) {
  const named = (rules || []).filter((rule) => rule && (rule.sentence_he || rule.sentence_en));
  if (!named.length) {
    return '';
  }
  const first = named[0];
  const sentence = (locale === 'he' ? first.sentence_he : first.sentence_en) || '';
  if (named.length === 1) {
    return sentence;
  }
  const rest = named.length - 1;
  if (locale === 'he') {
    return rest === 1
      ? `${sentence} כלל נוסף אחד הוציא תשדירים נוספים.`
      : `${sentence} ${isolate(rest)} כללים נוספים הוציאו תשדירים נוספים.`;
  }
  return `${sentence} ${rest} further ${rest === 1 ? 'rule' : 'rules'} removed spots too.`;
}
