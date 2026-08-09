NOT APPLIED. This file is foreign documentation only. No build recommendation, no
field, screen, rule, or engine behavior in the product may cite this file as its
justification. See `docs/audits/research-scope-ruling.md` for the standing rule
this file obeys.

# Foreign sell-side ad-ops / order-management systems — documentation only

Cluster researched: Operative (AOS, Operative.One, OnAir, IBMS, Broadway, STAQ,
and siblings), FreeWheel (MRM/Streaming Hub, Strata, Beeswax), Mediaocean
(Prisma, Spectra, Lumina, Aura, Radia, Ignitia), Salesforce Media Cloud
(Advertising Sales Management), Google Ad Manager for TV, S4M, Placements.io,
Boostr, FatTail AdBook.

## Part 1 — Relevant to Israel, with Israeli evidence

**None.** Every candidate overlap between this foreign material and
`docs/media-domain-from-the-trade.md` was checked against the test in the
ruling (is there Israeli evidence — the trade document, the owner's words, the
traffic file, the regulator, an Israeli source — not just the foreign source),
and every one failed it because the Israeli document already names the thing
itself, independently, in its own vocabulary:

- **Top and Tail** — the international trade term (confirmed via
  mediafederation.org.au's glossary and themediaant.com: two spots for the same
  product in one break, "at least one other spot for a different product placed
  in-between") describes exactly the mechanism `docs/media-domain-from-the-trade.md`
  calls "Top and Tail" — but that document already uses this exact English name
  and already specifies the harder constraint (separated by exactly one or two
  other advertisements, up to twenty creative versions, validity windows). The
  foreign term adds nothing unnamed; Israel already has the name and a sharper
  spec. Not a finding.
- **Make good** — FreeWheel's Placement API v3 has a `placement_type` field with
  a literal `MAKE_GOOD` enum value (MEASURED, api-docs.freewheel.tv). The
  Israeli document already uses "Make good" as its own working term, and
  describes a mechanism FreeWheel's flat per-placement flag does not capture
  (a three-level accrual-and-utilization ledger across campaign, advertiser, and
  agency). The foreign system is shallower than what Israel already does under
  the same name. Not a finding.
- **Gold break (ברייק זהב)** — searched for in the foreign material and not
  found under any name in Operative, FreeWheel, Mediaocean, or Salesforce
  documentation. No foreign source to attach even if it were needed; the term
  and its separate rate card are already established Israeli vocabulary,
  already guarded by `tests/test_w0_4_vocabulary.py`. Not a finding.
- **As Run** — none of the vendor documentation reached in this research
  described an as-run reconciliation mechanism in enough detail to compare
  (Mediaocean's DARE/Auto-Avails pages were the closest, and those are a
  US-specific EDI protocol between agencies and stations, not a mechanism with
  Israeli evidence behind it — see Part 2). The Israeli document already names
  As Run, already knows it is a second-by-second post-hoc JSON file, and
  already states the policy consequence (billing from As Run, never the plan).
  No gap to fill.
- **Block booking** — a real term in this cluster's domain generally, but not
  found named this way in any of the specific vendor pages fetched. The Israeli
  document already uses "a block booking" as its own term for stage one of the
  negotiation. Not a finding.
- **Preferred-position counting (agency method vs. channel method)** —
  searched for directly; no international standardization or vendor
  documentation of this exact dual-method dispute was found anywhere in this
  research. This appears to be Israeli-market-specific with no foreign paper
  trail at all, which is the opposite of the case the ruling is looking for.

If a genuine case turns up later — a foreign manual naming something Israel
does daily that has no name in `docs/media-domain-from-the-trade.md` and no
Israeli source anywhere — it belongs in a new finding against that document,
with the Israeli evidence attached, not in this file.

## Part 2 — Foreign, not applied (documentation record)

Layer, module, and documentation-reach findings for each vendor, for future
reference only. Every claim below is either MEASURED (exact URL and quoted
phrase actually read) or INFERRED (stated as such). None of it is a build
recommendation.

### Operative (operative.com)

Full-stack sell-side platform: sales/CRM, planning & pricing, order
management, traffic & scheduling, billing/invoicing, reconciliation, across
linear, streaming, and digital.

Corporate history — MEASURED (nocamels.com, tvtechnology.com,
broadcastbeat.com): SintecMedia (Israel-founded ad-tech company) acquired
Operative Media Inc. for $200M in Dec 2016; SintecMedia had earlier acquired
Pilat Media (UK) for ~$103M in 2014, whose flagship product was IBMS, and
which also owned MediaPro and an OTT platform (OTTilus). SintecMedia rebranded
as Operative.

Product line — MEASURED, quoted from https://www.operative.com/products/
(fetched 2026-08-09):
- **AOS** — "AI-powered converged media platform for companies to automate
  advertising sales across digital and linear channels."
- **Operative.One** — "Market-leading advertising management platform for
  digital publishers to streamline inventory, planning and order workflows."
  Modules per its own page: Sales Orders (quote-to-cash), Ad Slots
  (inventory), Invoices (billing).
- **Adeline AI** — assistive AI layer for inventory optimization and workflow
  speed.
- **AOS CloudCore** — integration/normalization layer across applications and
  datasets.
- **OnAir** — "Integrated broadcast traffic management solution that
  streamlines sales operations, powers traffic and billing, and optimizes ad
  placement." Own page: Sales Operations (Ratecard/Ratings Management,
  real-time inventory), Traffic & Billing, content-rights management.
- **IBMS** — "End-to-end broadcast and streaming management platform... with
  integrated content scheduling, media, rights management, sales and traffic
  capabilities." Own page splits it into IBMS Content (content lifecycle,
  rights) and IBMS Sales (inventory/revenue planning, proposals, orders). The
  page never spells out what the initialism stands for; older third-party
  sources disagree ("Integrated Business Management System" vs. "Integrated
  Broadcast Management System") — UNCONFIRMED which is correct.
- **Broadway** — end-to-end platform for cable networks/broadcast stations.
  Own page names three modules: "Power Sales" (ratecards, inventory, packages
  across linear/OTT/FAST), "Speed Traffic & Billing," "Drive Improved
  Stewardship" (delivery/deal-obligation management). Promotions/programming/
  finance as distinct named modules — COULD NOT CONFIRM.
- **SIMS** — program management: scheduling, finances, content, rights for
  networks/broadcasters.
- **MediaPro** — sales administration for Northern European sales
  houses/TV/radio networks (the Pilat-originated product).
- **STAQ** — acquired by Operative (prnewswire.com). Own page: automates
  media data collection, normalizes revenue/spend/inventory data, generates
  "standard revenue and reconciliation reports," "over 400 integrations."
  Data/reconciliation layer, not traffic/order entry.
- **OnTarget** — ML-based forecasting/yield optimization.
- **Medea** — affiliate/retransmission revenue-fee management.
- **Nestor** — MVPD-side content-cost/contract management.
- **Ad Manager** — self-service ad sales/campaign automation.

Documentation reached: operative.com product pages only (MEASURED, quoted
above). A release-notes PDF exists at
`o1.operative.com/release_notes/Production_Release_Notes.pdf` (2025.8, per a
third-party aggregator summary) — not independently fetched. No live
`developer.operative.com` or `docs.operative.com` could be located or reached;
all API-mechanics claims (OAuth2, GraphQL playground, webhooks, Postman
collections) trace only to the third-party site apitracker.io and are
UNCONFIRMED against a primary source.

Operative + Imagine Communications partnership — MEASURED
(imaginecommunications.com, operative.com press, April 2022): combines
Imagine's OSI, XG Linear, GamePlan, and SureFire with Operative's AOS and
OnAir; "bidirectional order data flows, ongoing synchronization of orders and
campaign performance, and real-time inventory management."

### FreeWheel (Comcast)

Layer: sell-side publisher monetization (MRM/Streaming Hub) plus a buy-side
DSP (Beeswax) and a separate buy-side agency platform (Strata), all under one
docs portal.

MRM/Streaming Hub — MEASURED (adexchanger.com): "The FreeWheel Streaming Hub,
which until a 2024 rebrand was referred to as an MRM (short for 'monetization
rights management') platform, sits on the publisher side of the business."

API docs portal — MEASURED, fetched api-docs.freewheel.tv (2026-08-09): four
categories — Publisher APIs, Advertiser APIs (Beeswax), Partner APIs
(Demand), Strata APIs ("retrieve and create essential business components,
such as orders, vendor invoices, and bills").

Endpoints actually fetched and quoted:
- `GET /services/v4/available_television_networks` — "Retrieve MRM IDs of TV
  Networks" — fields `id`/`name`/`status`; params `order_by`, `order`, `page`,
  `per_page`, `status` (ACTIVE/IN_ACTIVE).
- **Placement API V3** — `GET/POST/PUT https://api.freewheel.tv/services/v3/placement/{FW_ID}.xml`
  (XML only). Fields: `id`, `insertion_order_id`, `name`, `status`
  (IN_ACTIVE/ACTIVE/CANCELLED/COMPLETED/TESTING), `placement_type`
  (NORMAL/MAKE_GOOD/PROMO), `schedule`, `price`, `budget`, `delivery`,
  `content_targeting`. "A placement is a sale, or a collection of ad units,
  that share sales specifications like flight dates and content targeting."
- **Campaign API V3** — `POST/GET/PUT/DELETE https://api.freewheel.tv/services/v3/campaign.xml`;
  nested `GET .../campaign/{campaign_id}/insertion_orders.xml`. Create fields:
  `name`, `description`, `advertiser_id`, `agency_id`, `external_id`,
  `assignments`, `delivery` (`value`, `type`, `period`,
  `advanced_fc_identity_level`).
- **Ad Creative Scheduling API V4** — existence confirmed via search-result
  title only; field content not fetched, UNCONFIRMED.
- Financial/Strata reference pages ("List Vendor Invoices," "Create a Vendor
  Invoice Response," "GET Order Status") — existence confirmed via
  search-result titles only; direct fetches returned HTTP 404, so field-level
  content is UNCONFIRMED.

Strata — MEASURED (freewheel.com, nexttv.com): launched 1983, acquired by
Comcast in 2005 (predating FreeWheel), operates as a **buy-side** agency
media-buying platform under FreeWheel's umbrella — freewheel.com's own release
headline calls it "FreeWheel's buyside technology platform." Not sell-side
despite the shared docs portal.

Beeswax — MEASURED (businesswire.com, mediapost.com): acquired by
Comcast/FreeWheel Dec 2020, closed Jan 2021. A DSP ("Bidder-as-a-Service"),
i.e. advertiser/buy-side, not sell-side order management.

"FreeWheel Market/Marketplace" as a distinct named product — COULD NOT
CONFIRM.

### Mediaocean

Important correction: Mediaocean's core named products (Prisma, Spectra,
Lumina, Aura, Radia) are predominantly **buy-side (agency)** tools, not
sell-side order management for broadcasters/publishers. MEASURED: "Mediaocean
provides the foundational software for agencies and advertisers buying
traditional and digital media." Spectra is the system agencies use to
purchase local broadcast/cable/radio inventory; it talks to sellers' own
systems (e.g. WideOrbit) via the **DARE** protocol and **Auto-Avails**, rather
than being the seller's traffic system itself. Radia is explicitly labeled
"(Trading desk workflow)" in Mediaocean's own support-site category taxonomy —
also buy-side.

- **Prisma** — buy-side digital/omnichannel buying and campaign execution.
- **Spectra** — buy-side traditional/linear buying; sub-flavors "Spectra DS"
  and "Spectra OX" per search results.
- **Lumina** — media planning, spend visibility, budget control in buy
  execution systems.
- **Aura** — centralized project time and cost workflows (agency internal
  finance, not ad-sales).
- **Ignitia** — per search snippet, agency-side cross-media bill/pay:
  "functionality for media and creative billing from booking to payment...
  invoice and order linking, invoice matching from vendors, centralized
  client and vendor invoicing." Still buy-side finance.
- **Radia** — buy-side "Trading desk workflow" (Mediaocean's own support-site
  category).
- **PATS** — appears in developer-API search results alongside Prisma and
  Radia as a buy-side workflow system; acronym expansion and exact function
  COULD NOT CONFIRM.

Developer API portal — MEASURED from search-snippet text of
developer.mediaocean.com (direct fetch failed on a certificate error, so this
is snippet-sourced, not a direct read): Prisma Bulk data API, Prisma cost
API, Ad Server API, and an **Automated Avails API** ("enables buyers to
electronically request and receive proposals directly within the Spectra
systems") — the closest thing to a sell-side-facing interface, but still
framed from the buy side.

Bottom line: Mediaocean is the agency/buy-side counterpart that talks TO
seller systems (WideOrbit, Marketron, Imagine Communications OSI, Operative
OnAir/Broadway) via DARE/Auto-Avails — not a sell-side competitor to them.

### Salesforce Media Cloud — Advertising Sales Management (ASM)

Layer: sales/CRM + order management, sell-side, on the Salesforce platform.
Debuted July 2021 (salesforce.com press release).

Object names — MEASURED, quoted from a Salesforce Help article ("Maintain
Advertising Sales Management Object and Field Mapping," help.salesforce.com,
fetched 2026-08-09): Order, Order Ad Placement, Media Plan, Ad Opportunity,
Opportunity, Quote, Quote Line Item, Ad Server, Ad Server User, Ad Server
Account, Media Property, Ad Product, Ad Service, Ad Space, Ad Space
Specification, Ad Space Group, Pricing Procedure, Decision Table, Expression
Set, Context Definition, Inventory Slot, Revenue Schedule, Spot Calendar,
Account, Contact, Campaign, Ad Placement, Ad Availability View Config,
Demographic Code, Rate Card, Publication, Issue. Note: conceptual object
names confirmed, not literal API/schema identifiers.

### Google Ad Manager for TV / "DoubleClick for Broadcast"

Directly fetched admanager.google.com/home/partner-solutions/broadcast/
(2026-08-09): **no real linear-TV traffic/order-management product exists
here.** The page is entirely Dynamic Ad Insertion (DAI) for streaming/CTV and
Programmatic Guaranteed marketplace tooling. No spot scheduling, linear
avails management, or conventional trafficking found.

"DoubleClick for Broadcast" — COULD NOT CONFIRM as a real product name; no
source found using that exact title.

Broadcast Traffic Systems (BTS) — MEASURED (bts.tv): built a GAM integration
"following on from the success of Freewheel integration, allowing
broadcasters to integrate digital and linear advertising campaigns under a
single contract" — confirms GAM's linear-TV touchpoint is via third-party
traffic-system integrations, not a native GAM module.

### S4M (s4m.io)

**Not part of this cluster — excluded.** MEASURED (businesswire.com,
marketingdive.com, mmaglobal.com): S4M is a Paris-founded (2011) drive-to-store
measurement company; its platform FUSIO by S4M measures foot traffic from
digital campaigns. No connection to broadcast, linear TV, or ad-sales order
management found anywhere.

### Placements.io, Boostr, FatTail AdBook — sell-side OMS peer set

- **Placements.io** — "The Operating System for buying and selling media."
  Sell-side product: AdSalesOS (buy-side counterpart: MarketerOS). Custom IO
  templates, omnichannel IO export with DocuSign, workflow/automation,
  ticketing, Kanban. "Dozens of bi-directional API integrations support
  planning, pricing, activation, and billing." Recently launched Python SDK.
- **Boostr** — combines CRM, OMS, and a Proposal Recommendation engine. OMS:
  "order management across insertion order (IO) and programmatic sales,
  enabling margin protection, simplified media planning, efficiencies
  trafficking orders and month end close." Named revenue-management features:
  split reconciliation, commission calculations, revenue actualization,
  automated revenue recognition.
- **FatTail AdBook+** — "manages proposals, trafficking, and billing in one
  system." Two modules: AdBookOMS (inventory access across channels/media
  types) and AdBookPSP (a separate programmatic-demand channel operating
  independently of the publisher OMS).

## Gaps / unresolved in this research

- No reachable primary Operative developer/API portal (`developer.operative.com`
  equivalent) — only third-party aggregator claims.
- FreeWheel's Strata/Financial reference pages 404'd on direct fetch; only
  titles confirmed, not field content.
- Mediaocean's own developer portal could not be fetched directly (cert
  error); its API claims are snippet-sourced.
- IBMS's acronym expansion is unresolved between two disagreeing historical
  sources.
- "FreeWheel Market/Marketplace" as a distinct product name is unconfirmed.
- "DoubleClick for Broadcast" is unconfirmed as a real product name.
