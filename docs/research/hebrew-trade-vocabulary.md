NOT A BUILD INSTRUCTION. This file is primary-source research into Hebrew and
Israeli television advertising vocabulary. It records what Israeli regulators,
Israeli trade bodies and Israeli commercial pages actually say, with the Hebrew
quoted and the URL given. It changes no field, screen, rule or engine behaviour
on its own. See `docs/audits/research-scope-ruling.md` for the standing rule.

**`docs/media-domain-from-the-trade.md` OUTRANKS THIS FILE AND EVERY SOURCE IN
IT.** That document is the owner and the media professional describing the
market they work in. This one is the open web. Where they disagree, the trade
document wins and this file is the thing that is wrong. Nothing recorded here
overturns anything in the trade document, and nothing here should be cited to
argue against it.

# Hebrew trade vocabulary for television advertising — Israeli sources

Researched 2026-08-08/09. Unlike the foreign-systems files in this directory,
almost all of this one is Part 1 — relevant to Israel, with Israeli evidence —
because the sources ARE Israeli: the Second Authority's regulations, the Israeli
Marketing Association's media guide, a Hebrew advertising glossary, Hebrew
Wikipedia, Israeli media agencies and Israeli trade press.

Every claim gets its own verdict word and its own URL. ATTESTED means the Hebrew
was found in the named Israeli source; NOT CONFIRMED means it was searched for
and not found, with the queries listed so nobody repeats them.

## The three things that stop this file being misread

### 1. Absence of web evidence is WEAK evidence of absence here

Israeli channel rate cards are not published. There is no Israeli equivalent of
the RTÉ rate card PDF or the Thinkbox glossary. The commercial detail of this
market lives in closed-door negotiation, and the open web sees only the
marketing pages of small media buyers reselling into it.

So a NOT CONFIRMED here means "the open web does not say this", not "this is not
true in Israel". The case that matters: `docs/media-domain-from-the-trade.md`
gives preferred positions as first through fifth plus L, from the owner, who
works in this market. This file could only attest first, second, third and last.
**The ground truth stands. The web simply cannot see positions four and five.**
Do not resolve this in the web's favour, and do not open a task to reconcile it.

### 2. Nothing here is a source of default numbers

Two stacking premium percentages are recorded below — a gold-break surcharge of
up to 25% and a position-in-break premium of 5% to 20%. Both come from an
Israeli trade body's public media guide. They describe **what the open Israeli
market publishes**, which is a real fact about the shape of the market: Israel
prices break-type and position-in-break as two separate, stacking multipliers.

They are **not this operator's numbers**. This product reads premiums from the
channel's own configuration and never guesses them — that is the point of the
pricing hierarchy in `docs/pricing-hierarchy-design.md`, and a percentage lifted
from an industry guide is precisely the guess that rule exists to forbid.
Neither figure may be used as a default, seed, fallback or placeholder anywhere.

### 3. Where a claim touches something already shipped

Said inline, per claim, so a reader can compare rather than re-derive. Three do:
break size bears on pod size; the gold break is already a modelled break type;
the quarter-hour settlement basis corroborates `docs/quarter-hour-billing.md`.

## Part 1 — ATTESTED, with Israeli evidence

### 1.1 The core nouns

The single most important vocabulary finding: **the Israeli word for a
commercial break is `מקבץ פרסומות` (regulator) or `ברייק` (trade)**. Not
`הפסקת פרסומות`, which is the lay descriptive phrase and appears in no
regulation as a defined term.

**`מקבץ פרסומות` / `מקבץ פרסומת` — ATTESTED, legal definition.** 1992 rules §1 —
https://www.nevo.co.il/law_html/law00/4941.htm — `מספר תשדירי פרסומת המשודרים ברצף`.
The public broadcaster's parallel rules define it slightly wider, explicitly
admitting a cluster of one (Kan rules §1,
https://www.nevo.co.il/law_html/law00/201776.htm):

> `תשדיר פרסום בודד או מספר תשדירים פרסומיים המשודרים ברצף`

**`ברייק` — ATTESTED, the professional term.** Hebrew advertising glossary
`א-ב בפרסום` — https://pirsum.wordpress.com/א-ב-בפרסום/ — glosses it
`הכינוי השגור להפסקת הפרסומות בערוצים המסחריים`. Corroborated by Hebrew
Wikipedia `סרטון פרסומת` — https://he.wikipedia.org/wiki/סרטון_פרסומת:

> `סרטוני פרסומת מסחריים משודרים בדרך כלל בהפסקת פרסומות מיוחדת (המכונה בלשון המקצועית "ברייק")`

**`תשדיר פרסומת` — ATTESTED, legal definition.** 1992 rules §1, same URL:

> `כל תשדיר, פינה או משדר קצר אחר, לרבות אם הם בעלי ערך תכניתי, המכילים אזכור של גורם מסחרי`

**`ספוט` — ATTESTED.** `א-ב בפרסום` glosses it as
`כינוי לשידור אחד של פרסומת בטלוויזיה`. Hebrew Wikipedia uses it for the
purchased air time itself: `המפרסם משלם לבעל זכויות השידור על זמן האוויר ("הספוט")`.

**`שיבוץ` — ATTESTED, and load-bearing.** The title noun of the placement
regulation itself (`שיבוץ תשדירי פרסומת בשידורי טלויזיה`), used operationally in
the Kan rules: `שיבוץ תשדיר פרסום במקבץ פרסום`. The Israeli word for
placement/slotting.

**`אתנחתה` — ATTESTED, legal definition, and genuinely useful.** 1992 rules §1:

> `הפסקה בהמשכיות התכנית שהיתה מתרחשת מאליה, גם ללא שיבוץ תשדיר הפרסומת`

A natural pause that would have happened anyway. The regulatory concept of a
legitimate break OPPORTUNITY, distinct from the break itself — Israel has
separate words for the slot and the thing in it.

**`קדימון` — ATTESTED, legal definition.** 1992 rules §1: a promo,
`תשדיר המודיע על לוח השידורים או על שידור אחד או יותר שישודרו בעתיד`. Promos
share the cluster with ads and are regulated alongside them.

**`רצועה` — ATTESTED.** The Israeli word for a daypart strip; CPP is set per
רצועה — ishivuk.co.il, `מכפלת ה-CPP שנקבע לרצועה ברייטינג הממוצע...`

### 1.2 A break inside a programme versus a break adjoining its end

This is the Israeli analogue of what the UK calls CVE (centre versus end) and
what the Irish rate card indexes at 1.1. **The regulatory distinction is
ATTESTED. The commercial price index is NOT CONFIRMED — see §2.5.**

1992 rules — https://www.nevo.co.il/law_html/law00/4941.htm. §10(b), the
mid-programme cluster, legal ONLY at a natural pause:

> `לא ישובץ תשדיר פרסומת וקדימונים במהלכה של תכנית אלא באתנחתה`

§34, the cluster adjoining the programme's end, carrying no אתנחתה requirement:

> `ואולם רשאי בעל זכיון לשדרו במקבץ הפרסומת הגובל בסיומה של התכנית`

So Israel distinguishes the two, and the regulator's word for the end-adjoining
case is `גובל` — adjoining, bordering. What Israel does NOT have is a coined
noun for either: the regulation describes the position, it does not name it the
way "centre break" and "end break" do. There is no Hebrew `מקבץ אמצע` or
`מקבץ בין תכניות` in any source found.

The practical consequence is real even without a price index: a mid-programme
break needs a defensible אתנחתה and an end-adjoining one does not — a constraint
on where breaks may legally go, not a price on where they sit.

### 1.3 What makes a break premium in Israel: adjacency to the news

**ATTESTED.** Israeli Marketing Association media guide —
https://www.ishivuk.co.il/מדריך-מדיה-2/

> `מקבצי הפרסומות החזקים הם לרוב אלו הממוקמים צמוד למקבצי החדשות`

The strongest ad clusters are those adjacent to the news clusters. This is
Israel's answer to "which break is the premium break", and the axis is
adjacency-to-news, not centre-versus-end — worth knowing before anyone models
break quality on an imported CVE axis that Israel does not price.

### 1.4 Break size

**ATTESTED.** Hebrew advertising glossary `א-ב בפרסום` —
https://pirsum.wordpress.com/א-ב-בפרסום/ — names three sizes:
`הברייק הממוצע נע בין 12-15 פרסומות`; `'ברייק קצר' = כשלוש פרסומות`;
`ברייק 'זהב' = פרסומת אחת וחזרנו`.

**Bears on pod size** — anything in the optimiser or the break board that assumes
a pod holds a handful of spots should be compared against a 12-to-15 average.
This is an open-web average across Israeli commercial television, not this
operator's measured distribution; compare it to the operator's own data before
treating it as a prior.

### 1.5 The gold break

**ATTESTED, heavily, and already a modelled break type in this product.**

The Hebrew name is **`ברייק זהב`** (also `ברייק הזהב`). Definition, from a media
agency explainer — https://www.astrateg.co.il/פרסום-בטלוויזיה/

> `בטלוויזיה המסחרית נהוג להפריד בין ברייק פרסומות רגיל בו משדרים מקבץ פרסומות ארוך לבין ברייק הזהב`

Confirmed independently in trade press reporting real transactions:
ice.co.il/advertising-marketing/news/article/750850 —
`רכישת ברייק הזהב בגמר הריאליטי האח הגדול` at `בכ-300 אלף שקלים`; and
globes.co.il/news/article.aspx?did=1001351815 — `תשווק רק 10 ברייקים זהב`.

Kairos already models the gold break. What this adds is the Hebrew name and the
attested trade definition of one to three spots.

### 1.6 The two stacking premiums the open market publishes

**ATTESTED as published market practice. NOT to be used as default values — see
§3 of the preamble.** Both from https://www.ishivuk.co.il/מדריך-מדיה-2/

**Position-in-break premium, 5% to 20%:**

> `המיקומים המועדפים בברייק הינם: ראשון, שני, שלישי ואחרון`

with `תוספת תשלום הנעה בין 5% לעד 20%`, stated on the page as varying by
position, by broadcaster and by time slot, and NOT broken out position by
position. The rationale given:

> `כיוון שברייק הפרסומות מורכב לרוב מפרסומות רבות, יש חשיבות למיקומו של הספוט בתוך הברייק`

> `אשר משפיע על הרייטינג (אשר יורד במהלך מקבץ הפרסומות), על בולטות ועל זכירות`

Two things matter beyond the number. First, **last is a named position alongside
the numbered ones** — the Hebrew reads "first, second, third **and last**", not
"fourth" — independently corroborating the trade document's treatment of L as a
distinct position rather than an index. Second, the stated mechanism is that
**rating decays through the break**, an Israeli source asserting the same
intra-break decay the retention model already deals in.

**Gold-break surcharge, up to 25%:** same source, for a break of one to three
ads, `עבורו לרוב נגבית תוספת של עד 25% לפרסומת בודדת`.

The structural finding, which survives the specific numbers being ignored:
**Israel prices break-type and position-in-break as two separate multipliers
that stack.** Same shape as the Irish rate card's centre-break index multiplied
by its position premium; Israel indexes on gold-versus-regular where Ireland
indexes on centre-versus-end.

### 1.7 First and last cost more — stated independently, twice

**ATTESTED**, by two Israeli commercial sources unconnected to the Marketing
Association guide.

https://natanelimelech.co.il/כמה-עולה-דקת-פרסום-בטלוויזיה/

> `מיקום הפרסומת בתוכנית או במהלך הפסקות פרסומות משפיע גם הוא. פרסומות שפותחות את הפסקת הפרסומות או מסיימות אותה יהיו יקרות יותר`

https://mediasgroup.co.il/2024/07/30/פרסום-בזמן-שיא-בטלוויזיה-אסטרטגיה-עלו/

> `פרסומות בתחילת או בסוף הבלוק יקרות יותר מאלו באמצע`

Note the form: Israel describes this with **verbs** — `פותחות` (open),
`מסיימות` (close) — not with a product noun. See §2.2.

### 1.8 `בלוק` as an informal agency synonym

**ATTESTED** — and this corrects an earlier verdict of mine, see §4. The
mediasgroup.co.il sentence quoted immediately above uses `הבלוק` for the break.
It is informal and secondary: `מקבץ` is the regulator's term and `ברייק` is the
trade's. The compound `בלוק פרסומות` was still not found anywhere.

### 1.9 Position in break as a regulatory lever, and where it is NOT sellable

**ATTESTED.** The regulator's phrase for position-in-break is
**`מקום תשדיר הפרסומת במקבץ`**. 1992 rules §38 —
https://www.nevo.co.il/law_html/law00/4941.htm — empowers the Director to impose
separation arrangements `לרבות בדרך של קביעת מרווחי מעבר, קביעת מקום תשדיר הפרסומת או הקדימון במקבץ`.
One hard first-position rule exists, §34: an ad containing clips from a
programme may run in the cluster adjoining it
`ובלבד שלא יהיה תשדיר הפרסומת הראשון במקבץ`.

**The counter-case that matters:** for the public broadcaster, position is
explicitly NOT sellable — Kan rules, https://www.nevo.co.il/law_html/law00/201776.htm

> `שיבוץ מקבץ פרסום בלוח השידורים ושיבוץ תשדיר פרסום במקבץ פרסום ייעשו לפי שיקול דעתו הבלעדי של התאגיד, בלי שהמזמין היה מעורב בהחלטה זו`

Position premium is a commercial-channel practice, not a universal Israeli one.

### 1.10 Volume limits, and what is NOT limited

**ATTESTED.** Commercial channels, 1992 rules §3 —
https://www.nevo.co.il/law_html/law00/4941.htm

> `זמן השידור המרבי לתשדירי פרסומת שבעל זיכיון רשאי להקצות בכל שעה, לא יעלה על 10 דקות`

with 20:00–24:00 allowed to be redistributed at the licensee's discretion
`ובלבד שסך זמן תשדירי הפרסומת בשעות אלה לא יעלה על 40 דקות`, and a daily cap of
10% of broadcast time. Breaks are fenced by a `מרווח מעבר` of at least 3 seconds
(§20(b)). Kan: 9 minutes per broadcast hour (§38(a)); a single spot capped at 90
seconds (§43(a)); sponsorships within a cluster capped at one minute (§45(a)).

**What is NOT limited, and this matters to the optimiser:** no cap was found on
the length of a single `מקבץ`, nor on the number of `מקבצים` per hour or per
programme, in either regime. Israeli law constrains **total minutes**, not
**break count**. Breaks-per-hour is therefore a commercial and audience-retention
constraint in Israel, not a legal one, and must not be presented in the UI as a
regulatory rule.

### 1.11 Quarter-hour settlement — corroborates what is already shipped

**ATTESTED.** https://www.ishivuk.co.il/מדריך-מדיה-2/

> `אם שודר ספוט בשעה 8:03, עלותו בפועל תהיה מכפלת ה-CPP שנקבע לרצועה ברייטינג הממוצע שהיה בין 8:00-8:14`

An Israeli trade body stating that a spot settles on the average rating of the
quarter-hour containing it. **Independent open-web corroboration of
`docs/quarter-hour-billing.md`**, which was derived from the owner and the data.
Related, same source: length is priced **pro-rata against a 30-second base**,
`תשדירים ארוכים מ"30 יחושבו על פי פרורטה, כלומר- לפי החלק היחסי שלהם` — Israel
does not price per-second off a rate card, it prices a 30" base and scales.

### 1.12 Rating vocabulary, sponsorship, and the rest

All ATTESTED.

- **`רייטינג` / `נקודת רייטינג` / CPP** — ishivuk.co.il: `רייטינג` =
  `אחוז הצופים (בערוץ או בתוכנית) במשך פרק זמן מסוים, מתוך סך האוכלוסייה הרלוונטית`;
  `עלות לנקודת רייטינג (CPP)`. Standard in trade press.
- **`חסות` / `הודעת חסות`** — legal, with its own regulation (תש"ע-2009,
  https://www.nevo.co.il/law_html/law00/73408.htm). Commercial form is a 6-second
  on-screen mention with voiceover — astrateg.co.il:
  `תשדירי חסות באורך 6 שניות בתוספת קריינות`. Distinct inventory from a spot,
  sold at a pre-agreed fixed rate rather than by position.
- **`פריים טיים`** — in Hebrew script, astrateg.co.il:
  `התכנית שודרה בלב הפריים-טיים של ערוץ 2, בין השעה 21:05 ל-23:00`.
- **`לוח שידורים`** — the schedule, in the regulator's definition of `קדימון`
  and in the Kan rules' `שיבוץ מקבץ פרסום בלוח השידורים`.
- **`ג'ינגל`** — the working trade word across Israeli production studios;
  Hebrew Wikipedia files it under the coinage `זמריר`, relating it to
  `תשדירי פרסומת`. A creative-asset term, not a scheduling one.

## Part 2 — NOT CONFIRMED, with the queries run

### 2.1 No Hebrew noun for a "closer"

The Dutch broadcaster RTL sells a separately-priced product literally called
"Block closer". Israel appears to have no noun for it. Verified term by term
against the Hebrew glossary `א-ב בפרסום`
(https://pirsum.wordpress.com/א-ב-בפרסום/) and Hebrew Wikipedia `תשדיר פרסומת`.
**ABSENT from both:** `סוגר`, `ספוט סוגר`, `תשדיר סוגר`, `קלוזר`.

What Israel has instead is the verb form, attested twice in §1.7: `פותחות` /
`מסיימות`, `בתחילת או בסוף`. The premium is described, not nominalised.

Queries: `"טופ אנד טייל" OR "סוגר ברייק" OR "תשדיר סוגר" OR "ספוט סוגר" פרסומת טלוויזיה`;
`"פותח וסוגר" OR "פותח את הברייק" OR "סוגר את הברייק" פרסומת טלוויזיה מפרסם`.

### 2.2 No Hebrew name for Top and Tail

**ABSENT** from the same two sources: `טופ אנד טייל`, `פתיחה וסגירה`. The
mechanism is described in `docs/media-domain-from-the-trade.md` under its English
name and with a sharper constraint than any public source gives; the Hebrew
trade name for it, if one exists, is not on the open web. Same queries as §2.1.

### 2.3 Positions four and five

**NOT CONFIRMED.** The ishivuk source was interrogated directly rather than
re-read: `רביעי` and `חמישי` do not appear in its discussion of break positions;
its list is `ראשון, שני, שלישי ואחרון`. No channel rate card, agency explainer,
trade-press piece or tender document enumerating beyond third was found.

Queries: `מדריך מדיה ברייק מיקומים "ראשון, שני, שלישי" תוספת תשלום תשדיר טלוויזיה`;
`מחירון פרסום טלוויזיה ישראל מקדם מיקום בברייק אחרון תוספת אחוזים רשת קשת`;
`"המיקומים המועדפים בברייק" ספוט מיקום תוספת תשלום`.

**This does not contradict the trade document.** See §1 of the preamble.

### 2.4 No letter code for the last position

**NOT CONFIRMED.** No Israeli source uses `L`, or any Hebrew letter code, for
the last position; they use the word `אחרון`. The `L` coding in
`docs/media-domain-from-the-trade.md` is the operator's and the market's, simply
not written down publicly.

### 2.5 No centre-versus-end price index

**NOT CONFIRMED.** The regulatory distinction is real (§1.2), but no Israeli
source prices a mid-programme break differently from an end-adjoining one, and
the ishivuk guide, asked directly whether it distinguishes breaks inside a
programme from breaks between programmes, returns no. Israel's published premium
axes are break-type (gold) and position-in-break, not CVE.

Queries: `"מקבץ פנימי" OR "הפסקה פנימית" OR "מקבץ בין תכניות" פרסומות טלוויזיה שיבוץ`;
`"ברייק פנימי" OR "ברייק בין תכניות" OR "מקבץ הצמוד" פרסומות טלוויזיה תמחור`;
`פרסום טלוויזיה ישראל מקבץ באמצע התוכנית לעומת בין תוכניות מחיר שונה ברייק צמוד לחדשות`.

### 2.6 Other terms searched and not found

`ראש הפסקה`, `סוף הפסקה` — NOT CONFIRMED. Query:
`"ראש ההפסקה" OR "סוף ההפסקה" OR "ראש הברייק" פרסומת טלוויזיה תשדיר`.

`מיקום בהפסקה` as a fixed phrase — NOT CONFIRMED; direct page verification
returned DOES NOT APPEAR. The concept is attested as
`מיקומו של הספוט בתוך הברייק` (trade) and `מקום תשדיר הפרסומת במקבץ`
(regulator). Use one of those, not the invented compound.

`שנייה פרסומית`, `מחיר לשנייה` — NOT CONFIRMED as trade terms. The real
mechanism is `פרו רטה` against a 30-second base (§1.11). Query:
`"מחיר לשנייה" OR "שניות פרסום" תשדיר טלוויזיה ישראל תמחור לפי שניות מקדם אורך`.

**Competitor separation — NOT CONFIRMED in any regulation.** The words
`מתחרה` / `מתחרים` do not appear in the 1992 placement rules, and no rule in the
1992 rules, the 2021 Rashut content rules or the 2021 Kan rules bars competing
advertisers from the same cluster. The only hook is the generic §38 separation
discretion, which is aimed at viewer confusion, not competitive protection.
Trade queries (`"הפרדה בין מתחרים"`, `"בלעדיות קטגוריה"`) found nothing.
**Consequence: the competitor-separation constraint in this product is a
commercial and contractual practice, not an Israeli legal requirement, and must
not be presented in the UI as a regulatory rule.**

## Part 3 — The method note this file exists to preserve

**A summarising layer will conflate your query terms with the source's content.
Discard the result; do not report it.**

This happened concretely during this research and is the reason the file was
written. A search for `"ספוט סוגר" OR "תשדיר סוגר"` returned what looked like
clean definitions:

> ספוט סוגר — כינוי לשידור אחד של פרסומת בטלוויזיה
> תשדיר סוגר — תשדיר רדיו קצר, המעביר מסר פרסומי...

Both were fabrications. The glossary's actual entry is for `ספוט` alone
(`כינוי לשידור אחד של פרסומת בטלוויזיה`); the summariser had taken the entry for
a term that exists and attached it to the compound term in the query, which does
not exist. Had it been believed, this file would have invented a Hebrew trade
term and handed it to the product.

It was caught by going back to the page and asking a different question: not
"quote the sentences containing X" — which invites the model to produce a
sentence containing X — but **"for each of these terms, say APPEARS or ABSENT"**,
which makes absence a first-class answer. That re-interrogation returned ABSENT
for all four closer variants.

Three rules follow, for anyone doing primary-source research through a
fetch-and-summarise layer:

1. **Ask for APPEARS/ABSENT, not for quotes.** A prompt that asks for sentences
   containing a term pressures the layer to manufacture one. A prompt that offers
   ABSENT as a legitimate answer gets honest negatives.
2. **Corroborate any load-bearing quote through a second, independent path.**
   The position-premium sentence in §1.6 was obtained twice — once by fetching the
   page, once via a search snippet of the same page — and the two matched. That is
   why it is recorded as ATTESTED rather than as probable.
3. **A negative that arrives too conveniently is as suspect as a positive.** The
   fabricated definitions arrived because the query implied the term existed. Note
   what your own query assumed before believing the answer.

## Part 4 — Corrections to this file's own earlier findings

Kept visible rather than silently fixed.

**`בלוק` was reported NOT CONFIRMED. It is ATTESTED.** The first pass searched
for the compound `בלוק פרסומות`, did not find it, and wrote off the bare noun
with it. `בלוק` alone is used by Israeli media agencies — see §1.8. The compound
`בלוק פרסומות` remains unfound, which is the narrower and correct claim.

The error points the same way as Part 3: the query shape determined the verdict.
Searching for a compound and concluding against its head noun is the same class
of mistake as searching for a term and being handed a definition of it.

## Sources

Regulator and law, all on nevo.co.il: `law00/4941.htm` — כללי הרשות השניה (שיבוץ
תשדירי פרסומת בשידורי טלויזיה), תשנ"ב-1992, the primary source for most of Part 1;
`law00/204875.htm` — תוכן פרסומי בטלוויזיה, תשפ"ב-2021; `law00/73408.htm` — חסות
לתכניות טלוויזיה, תש"ע-2009; `law00/201776.htm` — כללי מועצת תאגיד השידור
הישראלי, תשפ"א-2021 (Kan).

Trade body and glossaries: ishivuk.co.il/מדריך-מדיה-2/ (מדריך מדיה, איגוד השיווק
הישראלי); pirsum.wordpress.com/א-ב-בפרסום/; he.wikipedia.org `סרטון פרסומת` and
`זמריר`.

Agencies and trade press: astrateg.co.il; mediasgroup.co.il; natanelimelech.co.il;
ice.co.il; globes.co.il. Full URLs are given inline at each claim.
