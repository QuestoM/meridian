# Israeli television rating currency: a research record, not a build instruction

Research record, compiled 2026-08-09 from primary web sources. NOTHING IN THIS
FILE IS APPLIED. Nothing here changes the product on its own.

`docs/media-domain-from-the-trade.md` OUTRANKS every source in this file,
including the ones labelled MEASURED. It is a record of how this market actually
works from someone who works in it. Where this file and the trade document
disagree, the trade document wins, and the owner is the authority. Web research
can corroborate the trade document, can supply a citation for something the
trade document asserts, and can surface a question. It cannot overrule it.

The value of this file is the SOURCE-QUALITY LABELS, not the claims. A claim
here without its label is worthless. Do not strip them when quoting.

## How to read this file

Every finding carries one of:

- **ACTIONABLE** — there is Israeli evidence, and the finding bears on something
  we ship. It still requires owner sign-off before any code changes.
- **DOCUMENTATION** — recorded because it is true and may be useful. No work
  follows from it.
- **NOT CONFIRMED** — searched for and not found. Stated so nobody re-derives it
  or, worse, assumes it.

Source-quality labels are separate and always shown: MEASURED (I fetched and read
the document), NEWS INTERVIEW, TRADE PRESS, WIKIPEDIA ONLY, PAYWALLED, INTERNAL.

Where a finding bears on shipped code, the file is named so a reader compares
rather than re-derives.

---

# THE CORRECTION: "overnight plus one" is not confirmed usage

**ACTIONABLE. This corrects a document in this repo.**

`docs/ux-gauntlet/RESUME-HERE.md:127` states the trade settles on "Jewish
households, quarter-hour, overnight plus one". The first two are confirmed by
this research (sections 4 and 5 below). **The third is NOT CONFIRMED as published
Israeli usage.**

The only term the IARB publishes is **OVERNIGHT**, defined as live viewing plus
deferred viewing up to 02:00 on the night of broadcast (finding 6a). No Israeli
source uses a "plus one" form.

Searched, Hebrew and English, zero hits for the term: `אוברנייט +1`,
`overnight+1`, `OVN+1`, אוברנייט + צפייה נדחית, רייטינג ישראל מטבע מסחרי + יום
למחרת, חיוב/התחשבנות מפרסמים + נתונים סופיים + CPP.

**Two readings fit the evidence, and they imply DIFFERENT STORED VINTAGES:**

1. **"+1" is the D+1 publication.** The overnight figure (live + deferred to
   02:00) published the following morning. This matches finding 6c exactly, where
   the morning-published figure is described as the one that matters to
   advertisers. Under this reading the stored value is a single overnight number.
2. **"+1" is one further day of deferred viewing.** An overnight figure plus a
   day of additional deferred accumulation. Under this reading the stored value
   is a later, larger number than the one published on the morning after.

**Why this bears on the code:** these two readings produce different numbers for
the same spot, and the gap is not small. Finding 7 measures deferred accumulation
of +0.1 to +8.1 rating points depending on the programme. **A product that stores
a TVR without recording which vintage it is cannot tell these apart**, cannot
reproduce a settlement, and cannot honestly label a figure provisional or final.

This is not resolved here and the transcript has NOT been corrected. It needs one
sentence from the owner. Until then, treat the vintage as unknown and record it
explicitly rather than assuming either reading.

---

# THE QUALIFIER THAT PREVENTS THE WHOLE FILE BEING MISREAD

**The measured universe is NOT Jewish-only.**

The panel measures all television households in Israel excluding the population
of East Jerusalem and Gaza. The Jewish cut is applied **COMMERCIALLY, on top of
that**, by agreement between the channels and the advertisers. See finding 4e for
the universe definition and 4a for the commercial decision in the IARB CEO's own
words.

Anyone who reads "the currency is Jewish households" as "the panel is Jewish
households" gets the model wrong. Both bases are measured, weighted and published
daily. One of them is what the money settles on.

---

# 1. Who publishes the currency

**1a. DOCUMENTATION. MEASURED.** The publisher is a Joint Industry Committee, not
a regulator. Hebrew `הוועדה הישראלית למדרוג`; English Israel Audience Research
Board (IARB).
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3000`
> "בישראל, כמו ברוב מדינות העולם, נמדדת הצפייה בטלוויזיה, על ידי ועדה משותפת של השחקנים בתעשיית הטלוויזיה. JIC – Joint Industry Committee"

**1b. DOCUMENTATION. MEASURED.** Non-profit, member-funded, and its output is
explicitly named as the trading currency. Same URL.
> "הוועדה פועלת שלא למטרות רווח והיא ממומנת על ידי חבריה."
> "מערכת המדידה נועדה לשרת את מנגנון קביעת המחירים של תשדירי הפרסומות בערוצים המסחריים (Currency)"

**1c. DOCUMENTATION. MEASURED.** Current membership, verbatim. Ten members, no
regulator among them. Same URL.
> "חברי הוועדה הם: ערוץ 9 / כאן 11 / קשת 12 / רשת 13 / עכשיו 14 / ערוץ 15 - i24 News / ערוץ 24 / ה.ל.א טי וי - ערוץ 30 / איגוד השיווק הישראלי / איגוד חברות הפרסום בישראל"

**1d. DOCUMENTATION. WIKIPEDIA ONLY.** Founded February 1995. The founding date
is not on the committee's own site.
URL: `https://he.wikipedia.org/wiki/הוועדה_הישראלית_למדרוג`

**1e. CONTRADICTION TO RECORD.** The separate he.wikipedia article
`מדרוג (רייטינג)` lists הרשות השנייה (Second Authority) as a committee member and
also lists the defunct ערוץ עשר. The committee's own site lists no regulator.
**Use 1c. The Wikipedia membership list is stale.**

# 2. The operator, and its rename

**2a. DOCUMENTATION. Vendor press release** — primary for the vendor, but it is
the vendor announcing itself. Kantar has delivered the Israeli currency since
1998; in December 2019 it was awarded seven more years with an option to 2030.
Panel stated as 700 households; technology Focal Meter plus existing Peoplemeter.
URL: `https://www.kantar.com/press-center/kantar-awarded-7-year-contract-to-continue-delivering-tv-trading-currency-in-israel`

**2b. DOCUMENTATION. MEASURED.** What changed is corporate ownership and name
only: carve-out from Kantar Group, acquisition by H.I.G. Capital (August 2025),
rebrand to Fifty5Blue (February 2026).
URL: `https://www.businesswire.com/news/home/20260225433501/en/Kantar-Media-Rebrands-as-Fifty5Blue-Marking-a-Decisive-Step-Forward-in-Audience-Measurement`

**2c. DOCUMENTATION. TRADE PRESS ONLY.** That the rename applies to the Israeli
operator. TheMarker, 2026-07-22, refers to "חברת קנטאר מדיה" as recently renamed
Fifty5Blue following acquisition by an American fund. **The IARB site still says
קנטאר מדיה.** No IARB statement names Fifty5Blue.
URL: `https://www.themarker.com/news/themedia/2026-07-22/ty-article/0000019f-89ca-d458-a9ff-bbea97310000`

# 3. Panel size

**3a. DOCUMENTATION. MEASURED.** 700 households, approximately 2,200 individuals,
representing receiver-owning households in Israel. Live official figure, fetched
2026-08-09.
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3010`
> "הפאנל מונה 700 משקי בית (כ-2,200 פרטים), המייצגים את אוכלוסיית בעלי המקלטים בישראל."

**3b. NOT CONFIRMED.** A 2018 plan to enlarge the panel to 900 households appears
only in he.wikipedia. Secondary marketing pages variously state 600 and 700-800.
Disregard all of these; the official site says 700.

**3c. DOCUMENTATION. MEASURED.** Establishment survey, same URL as 3a.
> "בתי הפאנל נבחרים ממאגר שנוצר ב'סקר כינון' רחב-היקף, המונה 4,500 משקי בית מדי שנה"

# 4. The Jewish-household trading base

This is the load-bearing section of the file.

**4a. ACTIONABLE. NEWS INTERVIEW — and it is the single most load-bearing claim
in this document.** Channels are paid by advertisers only on Jewish-household
viewing data. Both bases are published daily. It is a commercial decision, not a
measurement limitation. Stated by the IARB's own CEO, Roni Aran.
URL: `https://www.ynet.co.il/entertainment/article/uks7ntuwl`
Author רן בוקר (Ran Boker), 2023-01-13. Verified against raw page source.
> "נתוני הרייטינג שמתפרסמים בכל בוקר מפוצלים לשתי קבוצות: נתון אחד מלמד על הצפייה בקרב כלל האוכלוסייה בישראל, ונתון אחר מצביע על הצפייה בקרב משקי הבית היהודיים. ערוצי הטלוויזיה מקבלים את הכסף מהמפרסמים רק על בסיס נתוני צפייה במשקי הבית היהודיים - כלומר, יש התעלמות מוחלטת מאלו שלא יהודים. 'זו החלטה מסחרית של הערוצים מול המפרסמים', מסביר ארן"

Translation: "The rating data published each morning is split into two groups:
one figure shows viewing among the general population in Israel, and another
indicates viewing among Jewish households. The television channels receive the
money from advertisers only on the basis of viewing data in Jewish households —
that is, there is complete disregard of those who are not Jewish. 'This is a
commercial decision of the channels vis-a-vis the advertisers,' explains Aran."

**Weight comes from the speaker being the IARB CEO. There is no IARB methodology
document stating the commercial base.** It is corroborated by 4b, 4c and 4d, but
the corroboration is of the two-base structure, not of the settlement rule.

**4b. ACTIONABLE. MEASURED.** The IARB's own public report application enumerates
exactly four reporting bases under the field `קהל מטרה` (target audience).
URL: `https://midrug.safenet.co.il/app/`
> "משקי בית בכלל האוכלוסייה" / "פרטים 4+ בכלל האוכלוסייה" / "משקי בית באוכלוסיה היהודית" / "פרטים 4+ באוכלוסיה היהודית"

That is `{households, individuals 4+} x {general population, Jewish population}`.
**Bears on: how an audience is identified at all.** Any figure the product stores
or displays belongs to exactly one of these four bases, and a figure without its
base is not interpretable. See also the money-scoping doctrine already recorded
for this repo (overview/forecasts/yield on the same scope).

**4c. DOCUMENTATION. MEASURED.** The daily report is itself published on both
bases.
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3004`
> "מדי יום מפיקה הוועדה דו"ח המדרג את 20 התכניות הנצפות ביותר באוכלוסייה הכללית ובאוכלוסייה היהודית ביום הקודם."

**4d. DOCUMENTATION. TRADE PRESS.** Independent corroboration that broadcasters
rely on the Jewish-only figures. TheMarker, יסמין גואטה (Yasmin Guetta),
2025-02-03.
URL: `https://www.themarker.com/advertising/2025-02-03/ty-article-magazine/00000194-c533-d533-a3b6-cd3f10760000`
> "חשוב להדגיש כי הנתונים בכתבה זו מתייחסים לאוכלוסייה הכללית בישראל (לרבות ערבים), ולכן הם נמוכים יותר מהנתונים שעליהם נסמכים גופי השידור, המתייחסים ליהודים בלבד."

**4e. ACTIONABLE. MEASURED. This is the qualifier reproduced at the top of the
file.** The measured universe is not Jewish-only.
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3026`
> "האוכלוסייה הרלוונטית בישראל למדידת הצפייה הינה: כלל משקי בית בעלי מקלטים ו/או פרטים בגילאי 4+ במשקי בית בעלי מקלטים, ללא אוכלוסיית מזרח ירושלים ועזה. נתוני הצפייה משוקללים מידי יום עפ"י התפלגות נתוני היוניברס (נתוני האוכלוסייה בישראל) לפי גיל, מין, מס' נפשות במשק בית ועוד."

Translation: "The relevant population in Israel for measuring viewing is: all
households owning receivers and/or individuals aged 4+ in households owning
receivers, excluding the population of East Jerusalem and Gaza. Viewing data is
weighted daily according to the distribution of the universe data by age, sex,
number of persons in the household and more."

The only stated geographic exclusion is East Jerusalem and Gaza.

**4f. DOCUMENTATION. MEASURED.** Religion is an explicit panel weighting
variable, which is what makes the Jewish cut producible at all.
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3010`
> "נתוני הצפייה הנאספים ממנו עוברים שקלול סטטיסטי של משתנים דמוגרפיים (כגון מין, גיל, פלטפורמה, דת, ותק בארץ)"

# 5. Round-quarter-hour settlement, corroborated three ways

**Bears on: `docs/quarter-hour-billing.md`, which is the SSOT for this rule.**
Nothing below requires a change to it. This section exists so the rule can be
defended with external citations instead of re-derived.

**5a. ACTIONABLE. MEASURED. Israeli Marketing Association — itself an IARB member
(see 1c), which makes this quasi-primary.**
URL: `https://www.ishivuk.co.il/מדריך-מדיה-2/`
> "רייטינג רבעי שעות: הרייטינג הממוצע בכל רבע שעה עגולה במהלך היממה. כיום נהוג לתמחר ספוט על פי הרייטינג הממוצע ברבע השעה בה הוא שודר. כלומר, אם שודר ספוט בשעה 8:03, עלותו בפועל תהיה מכפלת ה CPP שנקבע לרצועה ברייטינג הממוצע שהיה בין 8:00-8:14."

The minute-versus-quarter-hour gap, same page. This is the arbitrage the
optimiser exists to exploit, stated by an IARB member organisation:
> "כיוון שהתשלום עבור הספוט מתבצע לפי רייטינג רבעי שעה, וכיוון שבמהלך מקבצי פרסומות יורד הרייטינג ביחס לתכנית עצמה, השאיפה היא למקסם את רייטינג הדקות ביחס לרייטינג רבעי השעות. נתון זה מושפע רבות ממיקום הספוט בתוך מקבץ הפרסומות."

Spot-length factors, same page, and they are ASYMMETRIC:
> "האורך הבסיסי של תשדיר בטלוויזיה הינו 30"... פקטור אורכים קצרים מ-30" מחושבים לפי טבלה שנקבעה בשוק ואינה מבטאת את החלק היחסי של האורך לעומת 30". תשדירים ארוכים מ-30" יחושבו על פי פרורטה"

Thirty seconds is the 100% base. Lengths shorter than 30" use a market-agreed
table that is explicitly NOT proportional. Lengths longer than 30" are pro rata.
**Worth raising against the trade document: if the engine treats length linearly
in both directions, that is wrong for sub-30" spots.**

**STALENESS WARNING on 5a.** This page is dated in parts. It discusses ערוץ 2
franchisees "נכון ליוני 2012", and Channel 2 split in 2017. The quarter-hour
mechanic is independently corroborated by 5b and 5c. **Every other detail on that
page must be re-checked before use.**

**5b. DOCUMENTATION. WIKIPEDIA ONLY.** Corroborates 5a; not usable standalone.
URL: `https://he.wikipedia.org/wiki/מדרוג_(רייטינג)`
> "על מנת לקבוע רייטינג לתשדיר פרסומת נהוג להשתמש בפרק זמן של רבע שעה ולאו דווקא ברייטינג בעת שידור התשדיר בפועל. שיאים של נתוני צפייה מחושבים על פרקי זמן של דקה וזוהי הרזולוציה הנמוכה ביותר אליה ניתן להגיע"

The same article is the source for the GRP/TRP definitions, including the
parenthetical that in Israel the GRP base is Jewish households:
> "GRP ... מייצג צבירה של סך נקודות הרייטינג שצברו כלל הספוטים של הקמפיין בקרב כלל האוכלוסייה (בישראל - משקי בית יהודים)."
> "TRP פירושו Target Rating Point והוא מייצג צבירה של סך נקודות הרייטינג מקרב קהל היעד (לדוגמה גילאי 25–54 בעלי הכנסה מעל הממוצע)."

**5c. INTERNAL — our own measurement, not independent external confirmation.**
`docs/quarter-hour-billing.md:14-18` and its settlement-lane measurement against
the real Reshet 13 2025-04-27 plan file (175 spot rows, 10 breaks; all 8
`planned_tvr` changes bracket a round quarter-hour boundary, two of them
mid-break at :45). Cite as ours.

# 6. The published OVERNIGHT definition

**6a. ACTIONABLE. MEASURED.** The committee's published OVERNIGHT figure is live
viewing plus deferred viewing up to 02:00 on the night of broadcast.
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3004`
> "דו"חות המדרוג מספקים מדי יום נתוני צפיית OVERNIGHT: צפייה חיה + צפייה נדחית עד 2:00 בליל אמש."

**6b. DOCUMENTATION. MEASURED.** Deferred viewing is measured up to seven days
from original broadcast; time-shifted viewing was added in 2011. Same URL.
> "המערכת מודדת צפייה נדחית עד שבעה ימים ממועד השידור המקורי."

**6c. ACTIONABLE. TRADE PRESS.** The morning-published overnight figure is the
commercially relevant one. Source as 4d.
> "הנתונים שמתפרסמים מדי בוקר, מתייחסים לצפייה בזמן השידור (Live) בתוספת צפייה נדחית עד 2:00 בלילה בערב השידור, והם הנתונים שחשובים למפרסמים."

**6d. ACTIONABLE. MEASURED.** As-run is an input to the ratings currency itself,
not only to billing. Broadcasters transmit as-run each morning BEFORE data
distribution, and the currency is classified off it.
URL: `http://www.midrug-tv.org.il/index.php?dir=site&page=content&cs=3010`
> "מדי בוקר לפני הפצת הנתונים מעבירים גופי השידור את לוח השידורים בפועל של אמש as run. מערכת ה-as run קולטת את התכניות מהגופים המשדרים ומבצעת אינטגרציה בינם לבינה. לאחר שהנתונים מזוהים, מתבצע סיווג שלהם לפי זמני השידור לתכניות, פרומואים, פרסומות וכד'."

This raises the stakes on as-run correctness. It does not merely settle the
invoice; it shapes the published rating. Corroborates the As Run section of
`docs/media-domain-from-the-trade.md` from the measurement side.

**6e. DOCUMENTATION. PAYWALLED — HEADLINE ONLY.** Deferred viewing is defined as
starting 20 seconds after broadcast. Haaretz, אופיר בר-זהר, 2010-05-09.
URL: `https://www.haaretz.co.il/misc/2010-05-09/ty-article/0000017f-db22-d3a5-af7f-fbae4bad0000`
> "ועדת המדרוג החליטה: 'צפייה נדחית' היא כזו שמתבצעת לאחר 20 שניות מהשידור"

**I read the headline and lede only. Do NOT present the 20-second rule as read
from the full article.**

# 7. How much a published rating actually moves

**ACTIONABLE. TRADE PRESS.** Source for all six: TheMarker, יסמין גואטה,
2025-02-03, URL as 4d. Article body visible, not paywalled; extracted from raw
page source.

**BASIS WARNING: all figures are on the GENERAL population base**, which the
article states explicitly (see 4d). The Jewish-household equivalents will differ
and are not published.

Columns: live | published at-broadcast (includes deferred to 02:00) | after 7
days | delta from published to 7-day.

| Programme | Live | Published | 7-day | Delta |
|---|---|---|---|---|
| הפטריוטים (daily live) | 6.6% | 7.3% | 7.4% | +0.1 |
| הכוכב הבא (season avg) | 12.6% | 16.9% | 18.9% | +2.0 |
| הישרדות S7 | 4.8% | 8.0% | 11.3% | +3.3 |
| האח הגדול | 6.9% | 11.5% | 15.2% | +3.7 |
| בקרוב אצלי | 9.4% | 14.9% | 20.9% | +6.0 |
| קופה ראשית S5 (season avg) | 4% | 7.9% | 16% | +8.1 |

> "בשבוע שלאחר השידור נוסף לתוכנית בממוצע 0.1% בלבד — והיא הגיעה ל-7.4% צפייה בממוצע." (הפטריוטים)

**Bears on: any figure the product calls final.** The transcript's "one, two,
even three points" sits inside this range, and the range is wider than that and
is GENRE-DEPENDENT. A live daily programme barely moves; scripted and reality
formats nearly double. **A single global uplift constant would be wrong.**

**DO NOT DOUBLE-COUNT.** A separate single-episode datapoint exists in the same
article (קופה ראשית S5E1, aired 2024-11-12: 11.6% then 19% then 34.4%). The
34.4% **includes repeat broadcasts**, which is a different thing from deferred
viewing. Keep them apart.

---

# FOREIGN MATERIAL — DOCUMENTATION ONLY, NOT APPLIED

Recorded because it names mechanisms we could not otherwise name. No work follows
from any of it. Kept short deliberately; the fuller foreign sweep lives in
`docs/research/broadcast-systems-and-terms-2026-08-09.md`.

**8. Nielsen restatement modelling. MEASURED — I fetched and parsed the spec.**
URL: `https://nielsen-production-darportal.apigee.io/portals/api/sites/nielsen-production-darportal/liveportal/apis/mv-n1a/spec`
HTTP 200, 49,220 bytes, OpenAPI 3.0.1, `info.title` "N1 Ads Data Service", 15
endpoints. Verified: `DataRestatementHistory` ("This method returns campaign
restatement history for a given data date or campaigns"), `CampaignDataAvailability`
("...so they can decide when to pull or re-pull the data"), `releaseType` enum
`["Initial","Re-Stated"]`, `releaseStatus` enum `["Released","Held"]`,
`whenInitiallyReleased` enum `["Y","N"]`. Vintage triple verbatim: "for a given
campaign - dataDate - reportingFrequency combination".

**SCOPE CAVEAT: this is Nielsen's DIGITAL AD RATINGS campaign service, not a
linear-TV ratings API. Do not present it as the linear currency.**

Relevance to Israel: it demonstrates that a currency models *initial versus
re-stated* as a first-class stored attribute. Our restatement problem is real
(section 7); this is one vocabulary for it. It is not a schema to copy.

**9. MRC Minimum Standards. MEASURED — I downloaded the PDF and extracted the
text with `pdftotext`.**
URL: `https://mediaratingcouncil.org/sites/default/files/Standards/MRC%20Minimum%20Standards%20-%20December%202011.pdf`
HTTP 200, application/pdf, 173,051 bytes, 12 pages. Title "Minimum Standards For
Media Rating Research", Media Rating Council, Inc. Footer "Revised December 2011".
Section B, item 19:
> "Each service shall keep documentation of errors of any type in published figures for a period of two years. Included in such documentation shall be: the length of time the error affected published figures; the effect of the error in absolute and relative terms; its cause; the corrective action taken; and the disclosures, if any, made to subscribers (copies of notices, etc.). If no disclosure was made, the record should indicate the reason underlying this decision."

Also verified present: C.2.d "Situations of data reissuance due to errors." and
C.3.b "Users are notified timely of errors noted in the System and/or data".

Note the IARB lists the MRC as a peer body on its own links page
(`...&cs=3022`), so this is not an alien standard. It is still foreign and still
not applied.

**10. BXF as-run discrepancy vocabulary. MEASURED — I fetched both schema files.**
The enumeration lives in `SMPTE/st2021-4`, file **`schema/bxftypes.xsd`**
(45,505 bytes), `simpleType name="AsRunStatusType"`. In schema order:

`Aired Without Discrepancy`, `Technical Difficulty`, `Did not air`,
`Aired with Duration Discrepancy`, `Aired with Content Discrepancy`, `Preempted`,
`Joined in Progress`, `Inserted by Operator`, `Unknown`, `Missing Content`

Surrounding type in `schema/asrun.xsd` (2,920 bytes, "Copyright 2023 Society of
Motion Picture and Television Engineers", targetNamespace
`http://smpte-ra.org/schemas/2021/2023/BXF`, version 8.100): `Status` has
`maxOccurs="unbounded"` with documentation "How the event was processed by the
automation system"; `StartDateTime` "The actual time the event started.";
`Duration` "The actual duration of the event as aired."; `EventNotes` "Used by
the operator to indicate what may have happened to an event."; `AsRunEventId`
"References the scheduled event ID unless the event was added manually in which
case this the Null value should be used."

Relevance to Israel: `docs/media-domain-from-the-trade.md` says make good covers
"a spot that did not air or aired wrong". "Aired wrong" is one phrase doing a lot
of work. This is a foreign decomposition of it. **The Hebrew must come from the
owner. Only the distinctions are candidates for transfer, and only with sign-off.**

---

# MY OWN CORRECTIONS

Recorded because a record that shows its author correcting itself is worth more
than one that quietly fixes it. All three were errors in my first-pass reporting,
caught when I independently verified a subagent's findings.

1. **MRC item numbering.** I first cited it as "B.19". The document numbers it
   "19." within section B; "B.19" is not a printed string in the PDF. Cite as
   "MRC Minimum Standards, section B, item 19". I had also **omitted the final
   sentence** ("If no disclosure was made, the record should indicate the reason
   underlying this decision"), which is part of the standard.
2. **BXF file name.** I first attributed `AsRunStatusType` to `asrun.xsd`. The
   enumeration is defined in `bxftypes.xsd`; `asrun.xsd` only references the type.
3. **Nielsen scope.** I first presented the restatement endpoints without stating
   that the spec is Digital Ad Ratings, not linear TV. That distinction matters
   and was missing.

I also reported at one point that the schema research had not returned when it
had. That was my error in reading a background task's state, not a failure of the
research.

---

# REPRODUCTION NOTES

These are what make the file checkable. Without them the sources do not resolve.

- **`midrug-tv.org.il` redirects https to http.** Fetch over `http://`, or a
  redirect-following client will report a cross-host redirect and stop.
- **The report application at `https://midrug.safenet.co.il/app/` is served in
  cp1255, not UTF-8.** Read as UTF-8 it renders as mojibake and the four
  reporting bases in 4b are unreadable. Decode as cp1255.
- **The SMPTE raw path guesses 404.** `raw.githubusercontent.com/SMPTE/st2021-4/...`
  does not resolve for guessed paths. Use the contents API instead:
  `gh api "repos/SMPTE/st2021-4/contents/schema/bxftypes.xsd"` and base64-decode
  the `.content` field. Code search (`gh api "search/code?q=AsRunStatusType+org:SMPTE"`)
  is how the correct path was found.
- The ynet and TheMarker article bodies are readable without a subscription; the
  Haaretz 2010 article is not.

---

# SOURCE-QUALITY SUMMARY: what rests on what

Anything in this list is weaker than a primary document and must be presented as
such.

- **4a, the single most load-bearing claim in this file, is a NEWS INTERVIEW.**
  Mitigated by the speaker being the IARB CEO, and by 4b/4c/4d corroborating the
  two-base structure. No IARB document states the commercial settlement base.
- **2c** (Fifty5Blue rename applies to the Israeli operator): TRADE PRESS ONLY.
- **1d** (founding date February 1995): WIKIPEDIA ONLY.
- **5b** (quarter-hour, and the GRP/TRP definitions including the Jewish-household
  parenthetical): WIKIPEDIA ONLY standalone. The quarter-hour mechanic is
  independently carried by 5a.
- **5a** carries a 2012 staleness warning and discusses a channel that split in
  2017. Only the quarter-hour and length-factor mechanics are corroborated
  elsewhere; the rest of the page needs re-checking.
- **6e** (20-second deferred threshold): PAYWALLED, headline only.
- **Two criticisms I did NOT fetch and which must NOT be quoted as read:** a
  Globes September 2017 report of panel under-representation of Arabs and
  מסורתיים versus over-representation of secular households, and a 2009 Mossawa
  Center allegation that Arabs, Haredim and Russians are removed from rating
  tables. Both are Wikipedia-cited only. I read neither original.

# NOT CONFIRMED

- **"Overnight plus one" as published Israeli usage.** See the correction section
  at the top. Search terms are listed there.
- **Which vintage contractually settles an invoice.** 6c says the morning figure
  is what matters to advertisers; the trade document says "overnight plus one".
  No Israeli contract, rate card or tender document is on the open web.
- **Any IARB methodology PDF, tender document, or data dictionary.** There is
  none. The committee site has no downloadable methodology. Israeli documentation
  is genuinely thin, and the CEO interview at 4a is the most authoritative public
  statement of the commercial base that exists.
- **Any published Israeli average for overnight-to-final revision.** The six
  programme-level figures in section 7 are the best available and come from one
  trade article, not from the measurement body.
- **Whether the panel grew from 700 to 900.** See 3b.
- **Instar Analytics technical documentation** (the software IARB members use to
  cut this data). The public site exposes no technical documentation on
  quarter-hour or target-audience definitions. This is the most likely place a
  foreign manual would name a mechanism Israel uses daily, and it is gated.
