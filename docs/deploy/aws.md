# Kairos on AWS — the dedicated account, and why the account is the product

Stood up 2026-08-24 under an owner-delegated mandate. Everything here was
executed, not planned: the account exists, the stack runs, and every claim below
names the thing it can be checked against.

## The isolation model, stated before the parts

The requirement is not "secure hosting". It is that **Questo can hand this to
the channel without retaining access to the channel's data**. Encryption inside
a shared account does not deliver that — whoever holds the account holds the
keys. So the boundary is the ACCOUNT itself:

- **Account `666823182071` ("Kairos")** exists for this product and nothing
  else. It was created as a member of organization `o-j4l92avif2` via the
  Organizations API — no signup form, no separate payment instrument — following
  the house pattern (`netanel+kairos-prod@questo.media`, access through
  `OrganizationAccountAccessRole`).
- **All state lives on one EFS volume inside that account** (`kairos-state`,
  encrypted, `DeletionPolicy: Retain`). Channel data arrives through the app's
  own upload screens — browser → ALB → app → EFS — and never transits a Questo
  machine.
- **Handover is an account transfer, not a data migration.** When the channel is
  ready, the account leaves the organization (or is invited into the channel's
  own organization). At that moment Questo's `OrganizationAccountAccessRole`
  path dies with the org membership, the channel owns root, and nothing about
  the running system changes — the isolation story is the offboarding story.
- Until handover this is demo-grade custody: Questo can administer the account,
  and says so. The claim "we cannot see your data" becomes true at transfer, by
  construction rather than by policy document.

## The AI stays on the operator's machine — deliberately

Owner instruction, 2026-08-24. The deployed container carries **no Anthropic
key and no assistant credential of any kind**. The Kai assistant, and Claude
Code doing development, both live on the operator workstation exactly as before.
Assistant calls against the cloud instance fail with the product's own honest
no-credential refusal (bilingual, already measured live). Consequences:

- No model vendor ever sees channel data on the cloud path.
- The demo can still show the assistant — from the operator's machine, against
  the operator's local instance.
- Wiring the assistant into the cloud later is a secrets decision (Secrets
  Manager + env), not an architecture change.

## What runs where

One region — **il-central-1 (Tel Aviv)**, opted in and enabled for the account.
The channel's data stays in Israel; for an Israeli broadcaster that is part of
the pitch, not a footnote.

```
Internet ──► ALB :80 ──► Fargate task (ARM64, 1 vCPU / 2 GB)
                          ├── node serve-dist.mjs :8080  (static dashboard + /api proxy)
                          └── uvicorn kairos_api :8000   (loopback only)
                          └── /app/state ◄── EFS (encrypted, Retain)
```

- The API binds **loopback only**; the single public doorway is the static
  server's proxy, so the dashboard and API share one origin — no CORS, and the
  app's own auth wall (verified 401 on `/api/*` before deploy) fronts everything.
- The container mirrors the operator's local topology (static server in front of
  uvicorn) because that is the shape every test and demo has run against. The
  cloud is not a novel configuration of the product.
- **First boot seeds, never overwrites**: the image carries `data/`, `output/`
  and `models/` as a demo seed; the entrypoint copies them onto EFS only when
  the volume is empty. Uploaded channel data therefore outlives every deploy.
  Verified locally: second boot logged no seeding.

## The pieces, checkable

| Piece | Where |
|---|---|
| Account | `666823182071`, org `o-j4l92avif2`, profile `kairos` |
| Image | `666823182071.dkr.ecr.il-central-1.amazonaws.com/kairos:v4` (arm64, 821 MB; v1-v3 superseded — see the incident record) |
| Stack | CloudFormation `kairos` — `deploy/kairos-stack.yaml` |
| Container build | `deploy/Dockerfile` (dashboard built in stage 1 from the lockfile; runtime from `requirements-api.txt`, the maintained minimal set) |
| Entrypoint | `deploy/entrypoint.sh` (seed-if-empty, symlink, two processes) |
| Proxy | `deploy/serve-dist.mjs` (static + /api,/auth passthrough incl. Set-Cookie) |

## Runbook

```bash
# Redeploy after a code change
docker build -f deploy/Dockerfile -t kairos:local .
aws ecr get-login-password --profile kairos --region il-central-1 \
  | docker login --username AWS --password-stdin 666823182071.dkr.ecr.il-central-1.amazonaws.com
docker tag kairos:local 666823182071.dkr.ecr.il-central-1.amazonaws.com/kairos:v<N>
docker push 666823182071.dkr.ecr.il-central-1.amazonaws.com/kairos:v<N>
aws cloudformation update-stack --stack-name kairos \
  --template-body file://deploy/kairos-stack.yaml \
  --parameters ParameterKey=ImageUri,ParameterValue=...:v<N> \
  --capabilities CAPABILITY_IAM --profile kairos --region il-central-1

# A shell inside the running task (user seeding, diagnostics)
aws ecs execute-command --cluster kairos --task <task-id> \
  --container kairos --interactive --command /bin/sh \
  --profile kairos --region il-central-1

# Logs
aws logs tail /kairos/app --follow --profile kairos --region il-central-1
```

First operator account: generated BY AWS inside the account
(`kairos/admin-password` in Secrets Manager, created by the stack), injected
into the task as an ECS secret, and consumed on first boot by the product's own
`scripts/init_auth.py` — which never prints a password that arrived via the
environment and refuses to touch an existing store, so a redeploy can never
reset credentials. The password exists in no image, no repository, no chat
transcript and no shell history; the operator reads it from the Secrets Manager
console when signing in the first time, and the dashboard forces a change.

## The access model — two layers, one principle

**The principle, at both layers: an actor holds the authority of the signed-in
human, and not a millimetre beyond.** Stated by the owner, and it was already
the product's own Rule 7 before it became infrastructure.

**Inside the product** (measured — 38 auth/assistant-security tests green):
three roles, `admin` / `operator` / `viewer`, enforced AT THE SERVER, not in
the UI: a viewer session is refused on every mutating method (403), account
management demands the admin role (403 otherwise), anonymous is walled with
401. The full user lifecycle is in-app: `/auth/users` CRUD, per-user
affiliation, password reset, forced change on first login, and the shell's
UserAdminDialog gives the admin all of it on screen. **Kai runs under the
caller's session**: a viewer's ask offers no propose tools and a forced
propose is refused; threads are isolated per user; a proposal that touches a
broadcast-licence limit carries the store's own permission verdict before the
approval card renders. The assistant reaches exactly the controls the person
behind it can reach.

**On AWS** (each boundary verified live, allowed-and-denied both):

| Role | Holds | Verified |
|---|---|---|
| `KairosDeploy` — **the AI's standing profile** (`kairos`) | image push, task-def + service update, stack update, logs; `iam:PassRole` on the task roles only | ECS list ✓, `iam:CreateUser` **AccessDenied** |
| `KairosReadOnly` (`kairos-ro`) | AWS ReadOnlyAccess | describe ✓, `ecs:UpdateService` **AccessDenied** |
| `OrganizationAccountAccessRole` (`kairos-admin`) | break-glass admin, reached by NAME only | — |

The AI's routine authority is deploy-scope; admin is a deliberate human
escalation. And the whole account is fenced by an organization SCP
(`kairos-region-lock`, attached to this account alone): every regional API
outside **il-central-1** meets an explicit deny, so the channel's data cannot
drift out of Israel even by mistake — verified: `ec2 describe-vpcs` answers in
il-central-1 and is denied in eu-central-1. IAM/STS/billing-class global
services are exempted the standard way.

At handover, the layers separate cleanly: the product's admin passes to the
channel with the account; the org roles and the SCP die with the org
membership; and whatever access the channel grants afterwards is theirs to
define inside their own boundary.

## Incident record: the auth store that rode in the image

The first image baked the operator's local `data/auth/users.json` — real
password hashes — into the seed, and v1's first boot copied it onto EFS. Caught
in the pre-live check ("what sensitive files rode into the seed?"), before the
URL was shared anywhere. Three-part fix, all shipped the same hour:
`.dockerignore` now excludes `data/auth` so no image can carry a store again;
the entrypoint removes exactly the leaked file from any volume it reached,
matched by content hash so a legitimately seeded admin store can never be
touched; and first-boot seeding moved to the managed secret above. The
remediation line stays in the entrypoint as the record of the incident.

## Decisions taken along the way, so they are not re-argued

- **Vercel: not used for the product.** The CLI is authenticated (`questom`) and
  available, but the product path stays single-account — the frontend is ~2 MB
  of static files the container already serves, and a second vendor in front of
  the auth wall would dilute the "one account, one boundary" pitch. Vercel
  remains an option for a public marketing shell that carries no data.
- **HTTPS: `kairos.questo.media`, canonical** (owner-refined, 2026-08-24: the
  first request followed the owner's literal `www.` spelling; the owner then
  asked why www at all, and nothing technical requires it -- the apex-CNAME
  limitation applies to `questo.media` itself, not to a subdomain -- so the
  bare name is canonical and `www.` answers with a 301, served over the same
  certificate, which carries both names). ACM certificate
  `9f5ba05d-ffff-4ec3-a2aa-3b2b8d2f9f04` in il-central-1, DNS validation via
  the owner's Wix-managed zone; the www-only request was deleted. The stack takes `CertificateArn`
  as a parameter: empty serves the pre-domain HTTP demo; set, it adds a TLS 1.3
  listener on 443 and turns port 80 into a permanent 301 redirect, so a
  credential can never be typed across plain HTTP by following an old link. A
  watcher applies the parameter automatically the moment the certificate
  validates; if it ever needs re-running:
  `scratchpad/https-when-ready.sh` (session) or the update-stack line in the
  runbook with `ParameterKey=CertificateArn`.
- **No NAT gateway.** Tasks get public IPs inside a security group that admits
  only the ALB. For a single-service demo this removes the standing ~$32/month
  NAT cost without opening a port.
- **Fargate ARM64** — matches the build machine natively, no cross-compilation,
  and it is the cheaper compute.
- Estimated standing cost: Fargate ~$30, ALB ~$20, EFS/ECR/logs single digits —
  **~$55–60/month** until handover.
