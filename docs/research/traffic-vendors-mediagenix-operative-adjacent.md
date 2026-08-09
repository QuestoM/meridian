NOT APPLIED — this file is documentation of foreign broadcast-traffic and
adjacent (MAM/CDN/graphics/playout) vendors only; nothing in the Kairos product
changes because of it.

# Foreign vendor research: Mediagenix WHATS'ON, Vizrt, Operative/Pilat/SintecMedia, Xytech, Etere, Amagi, Veset, Broadpeak

Companion to `docs/research/traffic-vendors-imagine-provys-protrack.md`, split
out under the same file for the 450-line law. Same rule applies:
`docs/audits/research-scope-ruling.md`. Israel is the only market that matters;
`docs/media-domain-from-the-trade.md` outranks every claim below wherever they
disagree. The negative-result mapping (which foreign terms Israel already has,
under what name, and which have no Israeli evidence at all) lives in the
companion file — nothing found in this half of the cluster added a new row to
that table, because this half is mostly adjacent layers (programme rights, MAM,
CDN, graphics/playout, DAI) rather than the ad-sales traffic layer the mapping
tests against.

Every line is labeled MEASURED (a URL was fetched and the text is quoted or
closely paraphrased from it) or INFERRED (a search-engine synthesis across
several results, not a single fetched quote). Login-gated documentation is
flagged as such and was not read.

## Mediagenix WHATS'ON

MEASURED from `mediagenix.tv/whats-on/` and
`mediagenix.tv/help-and-documentation/api/`. This is a **programme planning /
rights / scheduling** system, explicitly NOT an ad-sales/traffic system — the
fetch confirmed "the content does not mention ad sales, traffic management, rate
cards, or avails capabilities." Named capabilities: "linear (including FAST) and
nonlinear programming and scheduling," VOD scheduling, strategic planning,
content budgeting, title/metadata management, scheduling automation. Customers
per an INFERRED search snippet (not independently fetched): BBC, ARD, ZDF, RTÉ,
VRT, NRK.

**API docs**: real developer documentation exists but is gated. WHATS'ON's own
API page confirms: "REST APIs using JSON payloads," "OpenAPI Specification,"
domains covering rights management, MAM, scheduling, configurable reporting,
change notification, business datasets. A separate INFERRED search result
described WHATS'ON Business APIs as covering "scheduling content, creating
amortization schedules, selling rights" — rights licensing/amortization, not ad
inventory sales. **Login required**: the page states "Get direct access to the
documentation" gated behind a contact-form submission (name/email/company/
consent) — no public API reference accessible without registering.

## Vizrt — confirmed NOT a traffic/sales vendor

MEASURED from `vizrt.com/broadcasting/`: Vizrt's product catalog is exclusively
**Graphics** (Viz Engine, Viz Pilot Edge, Viz Ticker, Viz Artist, Viz Channel
Branding, etc.), **XR & Virtual Sets**, **Production Automation** (Viz Mosart),
**Live Production & Switchers** (TriCaster line), **Cloud & Remote Production**,
**Sports Production**, **Video Conversion**, and **Content & Distribution** (Viz
One, Viz Story). No sales, order entry, avails, or billing product exists.
Vizrt's own documentation for Viz Multichannel confirms the integration posture:
it "receives playlist schedules from 3rd party traffic systems" and integrates
with automation/traffic systems in the master control room rather than providing
its own — one legacy doc page references **"ADC-100 by Harris Broadcast"** as a
third-party automation integration, not a Vizrt product. State plainly: Vizrt is
not a traffic vendor.

## Pilat Media / IBMS / SintecMedia / Operative

A consolidated lineage confirmed across several MEASURED fetches:
- 2016: SintecMedia acquired Operative Media for ~$200M (PRNewswire/AdExchanger/
  NoCamels, cross-corroborated, INFERRED synthesis of consistent reporting).
- ~2018: SintecMedia rebranded as **Operative**.
- The combined company (`operative.com`), per its "Advertising" solutions page
  (MEASURED, directly fetched):
  - **AOS** — "AI-powered converged media platform... automate advertising
    sales across digital and linear channels" (Sales/CRM, Traffic/Scheduling).
    AOS's own product page confirms a named sub-component: "AOS Products,
    Ratecards, and Sales modules" — i.e. Ratecards is a distinct named module.
  - **Operative.One** — "market-leading advertising management platform for
    digital publishers... inventory, planning and order workflows" (Sales/CRM,
    Traffic/Scheduling).
  - **Broadway** — "end-to-end platform for cable networks and broadcast
    stations to manage ad sales, traffic operations, promotions, stewardship,
    programming, and finance" (Sales, Traffic, Billing) — a newer/renamed
    offering not previously documented in prior knowledge of this lineage.
  - **IBMS** (ex-Pilat Media) — "end-to-end broadcast and streaming management
    platform... integrated content scheduling, media, rights management, sales
    and traffic" (Sales, Traffic, Rights, Billing) — still live under its
    original name.
  - **OnAir** (ex-SintecMedia) — "integrated broadcast traffic management
    solution that streamlines sales operations, powers traffic and billing, and
    optimizes ad placement" — still live.
  - **SIMS** — "program management solution... scheduling, finances, content
    and rights" (Traffic/Scheduling, Rights, Billing) — still live.
  - **STAQ** — "cloud-based reporting and analytics... aggregate, normalize, and
    automate revenue data" — the reconciliation/as-run/analytics layer.
  - **Adeline AI** — referenced by name only, no functional description found.
- Pilat's earlier product **MediaPro** (advertising sales system) and the
  **OTTilus** OTT platform were named in a 2018 acquisition news article as part
  of the Pilat portfolio joining SintecMedia's line — NOT independently verified
  as still existing under current Operative branding. COULD NOT CONFIRM current
  status of MediaPro/OTTilus specifically.

## Xytech

MEASURED. Xytech is a **media operations / MAM / transmission-logistics**
vendor, not an ad-sales traffic vendor. Flagship platform **MediaPulse** (with
**MetaVault MAM** module). **Xytech Transmission**
(`fabricdata.com/xytech-transmission`) covers only technical delivery logistics:
"Contribution Resource Scheduling" (satellite uplinks, fiber paths, signal
routing, technical personnel), network visualization, conflict detection,
delivery coordination, schedule propagation. Explicitly does not cover order
management, billing, or ad-sales traffic. **Xytech Operations** — "Production
Scheduling for Media Companies" (found via INFERRED search, not independently
fetched for detail). Xytech occupies the log/playout-adjacent and MAM/resource-
scheduling layer, not the ad-sales/traffic layer that Landmark/OSI/Provys/
WHATS'ON/Operative occupy.

## Etere

MEASURED from `etere.com/DocView/6486/BROADCAST-AIRSALES.aspx`. Product: **Etere
Airsales** (full) and **Etere Airsales Lite**. "Integrated solution for traffic,
accounting and all sales management-related operations... entirely scalable...
from the smallest single-station setup to the largest corporation," with
"automatic invoice logs" and "instant addition or deletion of graphics up to 30
seconds before playout." Named sub-modules/screens: **"Airsales Integrated
Accounting"** (invoicing of pre-paid/post-paid ads, payment tracking, agency/AE
commission management), **"Etere Airsales Scheduling"** (a "Weekly Schedule
application" with drag-and-drop manual schedule creation), **"Etere Promo
Placement"** (promo/campaign management). RAI (Italian public broadcaster) is a
named customer per a page title, "The Traffic Flows Fast: Commercials of RAI
Runs ETERE" (title only, not independently fetched for quote-level detail).
Separate Dynamic/Multiscreen/Multichannel **Ad Insertion** products exist for
SCTE-35/SCTE-104 targeted insertion — DAI, a different (addressable/digital)
layer from traditional linear Airsales.

## Amagi

MEASURED via search snippets (INFERRED, not deep-fetched). **CLOUDPORT** = cloud
channel origination/playout platform (not traffic/sales). **THUNDERSTORM** =
"revolutionizes ad monetization on streaming TV by delivering contextually
relevant and truly personalized ads" — dynamic ad insertion / SSAI, a
digital/addressable ad-tech layer, not a traditional linear traffic-and-sales
order-entry system like Landmark/OSI/Provys. **ADS PLUS** mentioned alongside
THUNDERSTORM for fill-rate/inventory-value optimization. No evidence found of a
distinct "Amagi Ad Sales" product with avails/rate-card/order-entry
functionality — Amagi sits in the playout + addressable-ad-insertion layers, not
the traditional traffic/sales layer this cluster targeted. COULD NOT CONFIRM a
traditional traffic-and-sales module for Amagi.

## Veset

MEASURED via search snippets (INFERRED). **Veset Nimbus** = cloud playout
platform (channel creation/delivery to satellite/cable/CDN), SaaS pay-as-you-go
pricing. Confirmed: "Veset Nimbus offers scheduling and integration with leading
traffic systems" — Veset is a playout vendor that integrates with third-party
traffic systems; it is not itself a traffic/sales vendor.

## Broadpeak — confirmed NOT a traffic vendor

MEASURED via search snippets (INFERRED). Broadpeak is purely a **CDN /
video-streaming-delivery** vendor ("Advanced CDN," "CDNaaS," anti-piracy/token
security). No traffic, sales, or order-management product of any kind. State
plainly: Broadpeak is a CDN vendor, not a traffic vendor.
