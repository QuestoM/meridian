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
- **HTTPS: `www.kairos.questo.media`** (owner-chosen, 2026-08-24). ACM
  certificate `80cd82ae-6dc5-4d3d-b5e6-dad92c543d8b` in il-central-1, DNS
  validation via the owner's Wix-managed zone. The stack takes `CertificateArn`
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
