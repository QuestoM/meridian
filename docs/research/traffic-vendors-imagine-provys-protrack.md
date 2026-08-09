NOT APPLIED — this file is documentation of foreign broadcast-traffic vendors only; nothing in the Kairos product changes because of it.

# Foreign traffic/sales vendor research: Imagine Communications, PROVYS, Myers ProTrack

Scope and rule: `docs/audits/research-scope-ruling.md`. Israel is the only market
that matters. This file exists to record foreign vendor documentation so the next
reader does not have to re-derive it, and to record — as a finding in its own
right — which of those foreign terms Israel already has under a different (or the
same) name, and which have no Israeli evidence anywhere. `docs/media-domain-from-the-trade.md`
outranks every claim below wherever they disagree.

Every line is labeled MEASURED (a URL was fetched and the text is quoted or
closely paraphrased from it) or INFERRED (a search-engine synthesis across
several results, not a single fetched quote). Login-gated documentation is
flagged as such and was not read.

## The negative-result mapping (the actual finding)

For each foreign term below: does Israel already have it, under what name, with
what citation — or is there no Israeli evidence anywhere. This is what stops the
next reader from proposing "avail" or "clash" as a gap.

| Foreign term (source) | Result | Israeli citation |
|---|---|---|
| **avail** — Imagine glossary: "A representation of time on a station, cable channel or network offered ('available') for sale." | NO ISRAELI EVIDENCE | Codebase grep for `\bavail(s)?\b` across `kairos_api/`, `kairos/`, `docs/` returns zero hits. `docs/media-domain-from-the-trade.md` never uses the word; it describes the same underlying object only as "orders" and "breaks." |
| **makegood** — ProTrack Radio: "highlights exceptions so make-goods can quickly be executed by Traffic"; Landmark Sales: "automated... as-run reconciliation" | ISRAEL ALREADY HAS IT, same name | `kairos_api/makegood_store.py` (module docstring: "A make-good is the compensating delivery a channel owes a client when a flight..."); `docs/media-domain-from-the-trade.md:125-135` ("Make good... managed at three levels at once: campaign, advertiser and agency... an accrual and utilisation ledger"). |
| **affidavit** — Imagine glossary: "A proof-of-performance document that conveys the spots ordered by the advertiser, the spots aired by the station, the copy that ran for each spot, and any discrepancies." | NO ISRAELI EVIDENCE | Codebase grep for `affidavit` returns zero hits. Trade doc never uses the word. As Run (below) covers overlapping evidentiary ground but is not shown to be issued as a distinct client-facing document under any name — recorded as unconfirmed, not claimed as a match. |
| **As Run File** — Imagine glossary: "The electronic file that is created by the automation system of the events that played out." | ISRAEL ALREADY HAS IT, same name | `docs/media-domain-from-the-trade.md:117-123` ("`As Run` is a JSON file from the broadcast system, second by second, produced after the fact... Billing and delivery must be computed from As Run, never from the plan"); wired in code at `kairos_api/break_api_states.py:96` ("Supply a delivery or as-run feed for this week through Sources."). |
| **clash** — Imagine glossary: "Competitive brands, products, or services that do not advertise in the same commercial break or within a program sponsored by one another." | NO ISRAELI EVIDENCE for this word | The one codebase hit for "clash" (`kairos_api/agencies.py:372-373`) is an unrelated data-integrity duplicate-name check, not the advertising term. The underlying mechanism already has an Israeli name: `docs/media-domain-from-the-trade.md:45` lists "competitive separation" as a run-parameter priority. Nothing is missing; the word "clash" itself has no Israeli source. |
| **co-op invoicing** — OSI: "Electronic co-op invoicing with notary signatures" | NO ISRAELI EVIDENCE | Zero codebase hits, zero trade-doc mentions. A US national-brand/local-retailer cost-split structure with no confirmed Israeli parallel. |
| **TIP standards** — OSI: "spot placement based on TIP standards"; CrossFlight: "integrates seamlessly with customers' existing traffic systems via open APIs, including TIP" | NO ISRAELI EVIDENCE | `docs/media-domain-from-the-trade.md:155-157` states the incumbent Israeli system (Owner, עונר) is "Closed, no public API, changes require a vendor request" — the opposite of an open interoperability standard. No Israeli source names TIP or an equivalent. |
| **GPR** — PROVYS: "air-time software solution... supporting both rate-card and GPR based sales operations" (search-synthesized, INFERRED) | NO ISRAELI EVIDENCE | Trade doc names GRP and TRP explicitly (`docs/media-domain-from-the-trade.md:107-108`) but never GPR. No Israeli source uses this term. |
| **Auto Fill / oversell management** — OSI: "Full 24-hour inventory management, including Auto Fill for playlist generation"; "oversell management for sold-out programs" | NO ISRAELI EVIDENCE | Codebase grep for `oversell` and `auto.?fill` returns zero hits. The two `filler` hits that do exist (`kairos_api/scenario_compare_api.py`, `kairos_api/core.py`) are unrelated statistics terminology ("0-break filler rows"), not an unsold-inventory mechanism. Trade doc does not describe one. |
| **separation violations** — OSI: "visual indicator for separation violations" | Mechanism already named | Same concept as "competitive separation," `docs/media-domain-from-the-trade.md:45`. Not new vocabulary. |
| **Preferred position ranking** — BCM: "Ad sales with multi-currency, ratecard, ratings, and invoice bases"; Landmark Rights & Scheduling position language (both generic, no numbered scheme quoted) | Mechanism already named and already measured in more detail than anything read here | `docs/media-domain-from-the-trade.md:73-75`: "Preferred positions are first, second, third, fourth, fifth and Last, where Last is `L` and is a distinct position, not a number." Lines 85-89 name two live counting methods: "Agency method" (numerator = preferred positions obtained, denominator = breaks appeared in, double-counted if a break is appeared in twice) and "Channel method" (measured out of total broadcasts). This is what was measured; no comparative claim beyond it is made here. |
| **audience / line / spot trading models** — Landmark Sales: "Utilize audience, line, and spot trading models in one system" | "audience" maps to an already-named Israeli concept; "line" is untested | `docs/media-domain-from-the-trade.md:218` ("Goal-based orders (TRP against a named audience) as a first-class order type") and lines 171-179 ("the agency sends a GRP or target-audience goal instead of a spot list") already name and describe the "audience" trading concept under the term goal-based order. "Line" trading has no confirmed Israeli equivalent found either way — left untested, not claimed as a match or a gap. |
| **Mission Control** — xG Linear: "a daily work-to-be-done dashboard" | NO ISRAELI EVIDENCE | UI-feature name, not domain vocabulary. No Israeli parallel found. |
| **Amortization schedules** — Mediagenix WHATS'ON API docs: "creating amortization schedules, selling rights" | NO ISRAELI EVIDENCE | Rights-cost-amortization accounting concept. Not present in the trade doc or the codebase. |
| **Pre-emption** (industry term named in the original research brief's target module list, not a verbatim quote from any single fetched vendor page) | ISRAEL ALREADY HAS IT, under "preemptible" | `kairos_api/campaigns_commitment.py:124-125` (`"value": "preemptible", "label_en": "Preemptible"`); `docs/campaign-reference-notes.md:88` ("priority — `guaranteed`... or `preemptible` (can be displaced by a higher-priority campaign)"). Corroborated, not sourced, by the trade doc's description of campaigns being bumped: control-room delays and phone-in pulls (`docs/media-domain-from-the-trade.md`, As Run section). |
| **Block booking** (also from the brief's target list; no vendor page fetched in this cluster used this exact phrase) | ISRAEL ALREADY HAS IT, named directly by the owner | `docs/media-domain-from-the-trade.md:181-187` ("Stage one, a block booking. An agency reserves capacity ahead, often without naming the client..."); tracked as a known-absent build item independent of this research in `docs/trade-gap-analysis.md:304-347`. |

Net result: zero new vocabulary crossed into Part 1 of the original report. Every
mechanism Israel has, it already names; every foreign term with no Israeli
citation stops here as documentation.

## Imagine Communications (Harris Broadcast heritage)

Full family tree — MEASURED from `imaginecommunications.com/monetize-tv/`:

| Layer | Product | Description (quoted) |
|---|---|---|
| Ad Sales / OMS | **CrossFlight™** | "first fully integrated and real-time order management system (OMS) plus traffic solution for the U.S. broadcast industry" — debuted NAB 2024; "integrates seamlessly with customers' existing traffic systems via open APIs, including TIP." Not itself a traffic system — an OMS layer in front of traffic. |
| Sales/Traffic/Billing | **Landmark™ Sales** | "Multiplatform ad sales and scheduling for global broadcasters"; "over 600 channels in 50 countries." Used by Sky Media (200+ linear channels, 25M+ spots/yr, per [prnewswire](https://www.prnewswire.com/news-releases/sky-media-chooses-imagine-communications-landmark-sales-system-274763021.html)). Named capabilities: "audience, line, and spot trading models," "automated ratings prediction, spot booking, copy allocation, sequencing, and as-run reconciliation." |
| Sales/Traffic/Billing (NA broadcast) | **OSI™ Traffic & Billing** | "Sales, inventory & traffic management for North American Broadcasters." Named capabilities: "Full 24-hour inventory management, including Auto Fill for playlist generation"; "24/7 spot placement engine with visual indicator for separation violations and oversell management"; "break level reporting and oversell control"; "custom log file generation for all playout vendors"; "automated contract and copy imports"; "electronic co-op invoicing with notary signatures"; "mass edit and update capabilities for order lines"; two-way Google Ad Manager integration; "spot placement based on TIP standards." A combined **LandmarkOSI™** branding also exists ("LandmarkOSI Cloud™"); a datasheet exists on a third-party reseller CDN (av-iq.com) but repeated TLS certificate errors prevented reading its full text — its title/snippet only: "sales, inventory, traffic & revenue management for stations and station groups." |
| Sales/Traffic/Billing (MVPD/cable) | **xG Linear™** | "Efficient & scalable traffic suite for MVPDs" — "manages $7B of MVPD ad revenue annually," now integrated with Salesforce Media Cloud. Named features: "Auto-frontloading," "Dynamic ad placement," "Ad copy automation," "Finalization automation," "Mission Control" (daily work-to-be-done dashboard), "Schedule Viewer," "Export CSV from all key UI lists." Could NOT CONFIRM distinct products called "xG Sales" or "xG Traffic" as separate SKUs — xG Linear is the one current xG traffic/billing product found. An older "Eclipse®/xG® Advertising Management for MSO Service Providers" datasheet exists on the same third-party reseller CDN but could not be read (same TLS errors) — COULD NOT CONFIRM current/legacy status. |
| Sales/Traffic/Billing (mid-size) | **BCM** | "(formerly Broadcast Master™)... Mid-size broadcasters can streamline sales, scheduling, media, and finance." Named features: "Ad sales with multi-currency, ratecard, ratings, and invoice bases," "Automated linear and nonlinear presentation scheduling," "Dynamic promo and secondary event creation, versioning and scheduling," "Contracts and rights for linear and nonlinear platforms," "frame-accurate playlist creation." |
| Optimization | **GamePlan™** | Cloud-based inventory/yield optimizer; integrates with Landmark Sales. |
| Video Ad Server | **SureFire™** | Broadcast-quality video ad server; integrates with Landmark Sales. |
| Rights & Scheduling | **Landmark™ Rights & Scheduling** | "Manage contractual and financial data for linear and on-demand agreements"; "Create frame-accurate playlists for single or multichannel schedules"; "Automate workflows using integrations with MAM, sales, playout and financial systems." |
| Automation/Playout (NOT traffic) | **ADC™, Versio™, Nexio+™ AMP** | Confirmed playout/master-control automation and video servers, a separate division ("Make TV" not "Monetize TV"). Not traffic/sales products. |

Documentation access: a public API landing page exists
(`imaginecommunications.com/insights-and-resources/api-landmark-sales/`) but real
API docs require login — quoted: "For the most comprehensive and current APIs
request access to our API Developer Portal" and "The customer portal includes
exclusive APIs not featured here." `support.imaginecommunications.com` is
Salesforce-hosted and could not even be fetched (TLS altname mismatch pointing to
Salesforce `force.com` infrastructure) — **login required, could not access**. A
"Customer Portal Getting Started Guide" PDF is publicly hosted but its text was
not extractable by the fetch tool (binary/font-embedded PDF) — unread, contents
unknown. The glossary (`imaginecommunications.com/glossary/`) confirms real
in-house term definitions: Avail, A/R, Ad-ID, Affidavit, As Run File, Campaign,
Clash — the excerpt fetched did not include Rate Card, Makegood, Log,
Pre-emption, Order, Reconciliation, or Copy Rotation (they may exist elsewhere on
the site; not confirmed absent, just not in the fetched excerpt).

## PROVYS (DCIT, Czech)

Two generations of branding confirmed. **Current** (MEASURED from
`provys.com`): three products — **PROVYS Sphere** (broadcast management suite),
**Stream Circle** (cloud playout automation), **Tweenly** (broadcast graphics
tool). PROVYS Sphere's named modules (from `provys.com/provys-sphere`): **Content
Administration** ("TV program scheduling and EPG data management"), **Finance**
("long-term budgets... cost estimations... cost limits"), **Rights Management**
("evaluate appropriate content... purchase contracts... monitor compliance with
licensing terms"), **On-Demand Content**, **Promotion Management**, **Ads
Planning** ("CRM module... Organize ad campaigns, set pricing strategy, and fully
monetize your broadcast").

**Legacy** (MEASURED from a third-party mirror, silo.tips, of what appears to be
an original PROVYS "TV Office" vendor brochure — provenance not independently
verified against provys.com, flagged accordingly): named modules included
"PROVYS broadcast planning," "PROVYS play-list scheduler," "PROVYS secondary
event editor and scheduler," "PROVYS linear rights management," "PROVYS
non-linear rights management," "PROVYS content library and archive," "PROVYS
media management," "PROVYS media workflow management and run-time," "PROVYS
production project planner/tracking," "PROVYS resource management," **"PROVYS
sales customer relationship management," "PROVYS sales campaign management,"
"PROVYS sales traffic management," "PROVYS sales financial management,"**
"PROVYS annual planning," "PROVYS self-promotion planner," "PROVYS management
reporting."

No public PDF manuals, API refs, or help-centre docs were locatable via search;
provys.com/provys.eu appear to be marketing-only, no public documentation portal
found.

## Myers Information Systems — ProTrack (PBS-dominant)

MEASURED from `myersinfosys.com`. Confirmed: **ProTrack TV** and **ProTrack
Radio/ProRadio** suites, described as products that "interconnect Traffic,
Scheduling, Sales, Engineering and IT departments." Named modules from
`myersinfosys.com/protrack-tv/`: **Metadata**, **Scheduling**, **Sales**
("definitive tool to define sales goals and inventory"), **Traffic** ("finalize
program schedules, ad insertions, interstitial placements"), System
Integrations, Professional Services. From a 2016 PMDMC news post: ProTrack's
sales environment includes "CRM capabilities, proposals, contracts, rate cards,
sales dashboards," "ability to update order entry," workflow "from contract to
reconciliation," and leverages "broadcast and non-broadcast sales inventory."
ProTrack Radio (add-on page): "a full-featured sales environment that allows
clients to set goals, generate contracts based on avails, monitor performance,
and close the contractual loop with invoicing, affidavits, and AR" — the
clearest single confirmation of avails + invoicing + affidavits + AR as named
ProTrack capabilities. **ProRadio** add-on: "highlights exceptions so make-goods
can quickly be executed by Traffic." **ProWeb** add-on is NOT a back-office
module — it is a public-facing schedule-display website widget ("ACCURATE
SCHEDULES & DESCRIPTIONS," "WHAT'S ON NOW & TONIGHT WIDGETS," searchable program
library).

**PBS relevance**: a search-engine-synthesized (INFERRED, not a single fetched
quote) claim states ProTrack is used by "virtually all U.S. PBS member
stations." No distinct branded PBS system called "Merlin" or "PBS Traffic" was
found as a product name — COULD NOT CONFIRM any such product exists; "PBS
Traffic" appears only as a generic descriptive term in job postings, not a
proprietary system name.
