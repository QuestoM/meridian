# Kairos on AWS: deployment plan and runbook

> תקציר בעברית: המסמך הזה מסביר איך להריץ את מערכת Kairos (שרת FastAPI שמגיש גם
> את הדאשבורד, פלוס קובצי נתונים ומודלים) על AWS בצורה זולה ופשוטה. ההמלצה: להריץ
> את אותו Docker image על ECS Fargate (שירות אחד, תמיד פעיל, תעבורה נמוכה) מאחורי
> Application Load Balancer, לשמור את התיקיות data/models/output על EFS כדי לא
> לשנות קוד, ולהשתמש ב-EventBridge Scheduler להרצה השבועית (אימון/חישוב מקדמים)
> וההרצה היומית (תוכנית הברייקים). המערכת כרגע **בלי שום הזדהות/לוגין**, ולכן לפני
> חשיפה לאינטרנט חובה לשים אותה מאחורי אימות (Cognito / ALB auth / לכל הפחות basic
> auth). שום דבר כאן לא מבצע הקצאה אמיתית של משאבים: צריך את חשבון ה-AWS וההרשאות
> של הבעלים. כל הסכומים הם הערכה גסה בלבד.

This document is a plan and a runbook. It does **not** provision anything. Actual
provisioning needs the owner's AWS account, region choice, and credentials, and
those are owner decisions. The cost numbers below are order-of-magnitude
estimates for a single small internal tool, not a quote.

---

## 1. What we are deploying

- A FastAPI backend: `kairos_api/server.py`, ASGI app object `app`, listens on
  port 8000 (run with `uvicorn kairos_api.server:app --host 0.0.0.0 --port 8000`).
  Python dependencies are in `requirements.txt` (FastAPI, uvicorn, pandas,
  numpy, scikit-learn, TensorFlow + TensorFlow Probability for the impact model,
  etc.).
- A Vite/React dashboard in `tv-break-dashboard/` (build with `npm ci && npm run
  build`, output `dist/`).
- Flat-file state on disk, read and written by the app relative to the repo root:
  - `data/` - input CSVs (`Programmes.csv`, `Spots.csv`, `Dayparts.csv`,
    `kairos_settings.json`), the daily-input folder, and the reference data.
  - `models/` - the trained posterior `tv_break_posterior.pkl` and the measured
    coefficients `tv_break_coefficients.json`.
  - `output/` - generated schedules (`weekly_break_schedule.csv`), impact CSVs,
    and the run log. The decisions/audit log is written to
    `data/kairos_decisions.json`.

The repo now contains three deployment artifacts at the root:
`Dockerfile`, `docker-compose.yml`, `.dockerignore`. The Dockerfile is a
two-stage build (Node builds the dashboard, then a `python:3.11-slim` runtime
installs `requirements.txt`, copies the app, and copies the built `dist/`).

---

## 2. Serving the dashboard (important honesty note)

The simplest topology is to have FastAPI serve the built dashboard as static
files at `/`, so one container and one origin handle everything.

**As of this writing the FastAPI app does NOT mount StaticFiles.** Every route in
`kairos_api/server.py` is under `/api/*`; there is no `app.mount(...)` and no
`StaticFiles` import. The Dockerfile copies the built dashboard to
`/app/tv-break-dashboard/dist`, but nothing serves it yet.

To serve the dashboard from the same container, add this small, one-time change
to `kairos_api/server.py` (after the `app = FastAPI(...)` block and the existing
`/api` routes). This is described here rather than applied, because the task
scope is deployment artifacts only:

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_DIST = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "dist"
if _DIST.is_dir():
    # Mount last, after all /api routes, so it does not shadow the API.
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="dashboard")
```

Notes:
- Mount it **after** every `/api/...` route is declared so the catch-all static
  mount does not intercept API calls.
- `html=True` makes it serve `index.html` for `/`. A pure static mount does not
  do SPA fallback for deep links, which is fine for this dashboard since it is
  served from the root; if client-side deep links are added later, a small
  catch-all route returning `index.html` handles it.

### Alternative: S3 + CloudFront for the dashboard

The more cloud-native option is to upload `dist/` to an S3 bucket and put
CloudFront in front of it, leaving FastAPI to serve only `/api/*`:

- Pros: CDN edge caching, no app process serving static bytes, clean separation.
- Cons: two origins, so you must set CORS (`KAIROS_CORS_ORIGINS`) and point the
  dashboard at the API URL (`VITE_KAIROS_API_URL`) at build time.

Recommendation for a low-traffic internal tool: **start with FastAPI serving the
static `dist/`** (one container, one origin, the smallest moving-parts setup).
Move to S3 + CloudFront only if you want CDN caching or a separate static
release cadence.

---

## 3. Recommended architecture (always-on, low-traffic internal tool)

```
                         Internet
                            |
                  (auth: Cognito / ALB OIDC)
                            |
                 Application Load Balancer (HTTPS, ACM cert)
                            |
                 ECS Service (Fargate, 1 task, always on)
                 container: kairos:latest  (port 8000)
                   |- FastAPI /api/*  + static dashboard /
                   |
                 EFS mount (access points):
                   /app/data    /app/models    /app/output
                            |
       EventBridge Scheduler -> run weekly + daily jobs (see Section 4)
```

- **Compute: ECS Fargate, one service, one always-on task.** A single small
  internal tool with low traffic does not need autoscaling fleets; one task
  behind an ALB is enough and is the cheapest always-on option that still gives
  you health checks, rolling deploys, and logs. (If you preferred scale-to-zero,
  App Runner is an option, but it does not mount EFS, which this app needs for
  its file state - so Fargate is the better fit here.)
- **Ingress: Application Load Balancer** with an ACM TLS certificate and a health
  check against `GET /api/health` (the app already exposes it). The ALB is also
  where authentication is enforced (Section 6).
- **State: EFS, mounted into the Fargate task** at `/app/data`, `/app/models`,
  `/app/output`. This is the recommendation **because the app uses local file
  paths** and EFS lets those paths just work with no code change - the container
  reads and writes EFS exactly as it reads and writes a local disk.
  - The more cloud-native alternative is **S3** for inputs, the posterior, and
    outputs. That is more durable and cheaper at rest, but it **requires a code
    refactor**: every `Path(... ).read_*/write_*` and `pd.read_csv(path)` in
    `kairos_api/server.py`, `kairos/service.py`, and the scripts would need to go
    through `boto3` (or a small storage abstraction). Recommended later, not for
    the first deploy.
- **Image registry: ECR** holds the built `kairos` image.
- **Config and secrets: SSM Parameter Store / Secrets Manager** (Section 5).
- **Logs: CloudWatch Logs** via the `awslogs` driver on the task.

---

## 4. Scheduling the weekly and daily jobs

The engine has two recurring jobs, both already implemented as scripts in the
repo that read and write the same `data/`, `models/`, `output/` state:

- **Weekly programme re-fit / coefficient refresh.** Either
  `python scripts/train_impact_model.py` (full Meridian/TensorFlow posterior fit,
  CPU-heavy, needs Python 3.11 + TensorFlow) or the lighter, Meridian-free
  `PYTHONUTF8=1 python scripts/compute_measured_coefficients.py` which writes
  `models/tv_break_coefficients.json`. For a weekly cadence on the same image,
  the measured-coefficients path is the lean default; the full posterior fit is
  the heavier monthly-or-on-demand option.
- **Daily plan generation.** `python scripts/export_schedule.py` rebuilds
  `output/weekly_break_schedule.csv` (the schedule the dashboard reads) by running
  the engine over the reference data. `python run_optimization.py` is the
  posterior-driven optimizer entry point when a trained model is present.

Trigger these with **EventBridge Scheduler**, in the **Asia/Jerusalem** timezone
(EventBridge Scheduler supports named timezones directly, so no UTC math and no
DST drift). Two clean options:

**Option A - run the script in the existing Fargate task definition (recommended).**
Use an EventBridge Scheduler rule with an **ECS RunTask** target that launches a
one-off task from the same task definition but overrides the container command to
run the script. This reuses the same image, the same EFS mounts, and the same
dependencies, so the scheduled job sees exactly the same files the web service
sees.

- Weekly (Sunday, early morning local time):
  `cron(0 3 ? * SUN *)` with `Timezone: Asia/Jerusalem`
  (Sunday is the start of the broadcast week in Israel.)
- Daily (every morning, early):
  `cron(0 5 * * ? *)` with `Timezone: Asia/Jerusalem`

  (EventBridge cron is `minute hour day-of-month month day-of-week year`; one of
  day-of-month / day-of-week must be `?`.)

**Option B - a Lambda trigger.** A Lambda can invoke the ECS RunTask above, or
do light orchestration. Note the heavy weekly posterior fit must **not** run
inside Lambda directly (15-minute cap and TensorFlow size), so even with a Lambda
trigger the actual training runs on Fargate. For the daily light job, a Lambda
that calls RunTask is fine. Option A keeps everything in one place and is simpler.

Each scheduled run writes back to EFS, so the next web request and the next
scheduled run both see fresh files.

---

## 5. Secrets and config

- Non-secret config (`KAIROS_CORS_ORIGINS`, `VITE_KAIROS_API_URL`,
  `KAIROS_OPTIMIZE_TIMEOUT_SECONDS`) lives in **SSM Parameter Store** and is
  injected into the task definition as environment variables.
- Any real secrets (for example an LLM API key for the optional AI classifier,
  or future DB credentials) live in **Secrets Manager** and are referenced by the
  task definition's `secrets` block, not baked into the image. The image must
  never contain credentials.
- The dashboard's `VITE_KAIROS_API_URL` is a **build-time** value for Vite. If you
  split the dashboard onto S3 + CloudFront, set it before `npm run build`. If
  FastAPI serves the dashboard from the same origin, it can stay empty (the
  dashboard calls relative `/api/...`).

---

## 6. Authentication (must-read before exposing publicly)

**The application currently has NO authentication.** There is no login, no
session, no API key check; every `/api/*` route and the dashboard are open to
anyone who can reach the URL. Some routes also write state (settings, decisions)
and one route (`POST /api/optimize`) shells out to a subprocess.

Before exposing this on the public internet, put it behind authentication. In
order of preference:

1. **ALB + Cognito** (or ALB OIDC against an existing IdP). The ALB authenticates
   the user before the request reaches the container, so no app code changes are
   needed. This is the cleanest fit for an internal tool.
2. **CloudFront + Cognito** if the dashboard is served from CloudFront.
3. **At minimum, HTTP basic auth** at the ALB/proxy layer, or an IP allow-list on
   the ALB security group restricting access to the office/VPN range.

Do not skip this. The combination of open write endpoints and a subprocess call
makes an unauthenticated public deployment a real risk.

---

## 7. First-deploy runbook (AWS CLI)

This is the concrete sequence the owner runs once, with their account and region.
Placeholders in `<...>` are owner-specific. Region `il-central-1` (Tel Aviv) or
`eu-central-1` are natural choices; pick one and keep it consistent.

```bash
# 0) Variables (edit these)
export AWS_REGION=il-central-1
export ACCOUNT_ID=<your-account-id>
export ECR_REPO=kairos
export CLUSTER=kairos-cluster
export SERVICE=kairos-svc

# 1) Create the ECR repo and push the image
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -t "$ECR_REPO:latest" .
docker tag "$ECR_REPO:latest" "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"
docker push "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"

# 2) One-time networking + storage (do via console or IaC; summarized here)
#    - A VPC with 2 private subnets (for Fargate) and 2 public subnets (for the ALB).
#    - An EFS file system with mount targets in the private subnets, and three
#      access points: /data, /models, /output.
#    - An ALB in the public subnets with an HTTPS listener (ACM cert) and a
#      target group (type: ip) health-checking GET /api/health.
#    - Security groups: ALB SG allows 443 from your users; task SG allows 8000
#      from the ALB SG and 2049 (NFS) to the EFS SG.

# 3) Create the ECS cluster
aws ecs create-cluster --cluster-name "$CLUSTER" --region "$AWS_REGION"

# 4) Register the task definition (task-def.json: Fargate, awsvpc, the ECR image,
#    portMappings 8000, the three EFS volumes mounted at /app/data, /app/models,
#    /app/output, awslogs driver, and env/secrets from SSM/Secrets Manager).
aws ecs register-task-definition --cli-input-json file://task-def.json --region "$AWS_REGION"

# 5) Create the service (1 task, attached to the ALB target group)
aws ecs create-service \
  --cluster "$CLUSTER" \
  --service-name "$SERVICE" \
  --task-definition kairos \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<private-subnet-1>,<private-subnet-2>],securityGroups=[<task-sg>],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=<target-group-arn>,containerName=kairos,containerPort=8000" \
  --region "$AWS_REGION"

# 6) Seed the EFS state once (the data/, models/, output/ files the app reads).
#    Easiest: a one-off Fargate task with the same image that copies the repo's
#    seed files onto the mounts, or mount EFS from a small EC2/DataSync job and
#    upload data/, models/, output/ there.

# 7) Create the schedules (Section 4) - EventBridge Scheduler rules with
#    ECS RunTask targets and command overrides for the weekly + daily scripts.

# Redeploys after the first time: rebuild, push :latest, then force a new
# deployment so ECS pulls the new image:
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
  --force-new-deployment --region "$AWS_REGION"
```

For anything beyond the first manual run, express this as Terraform or AWS CDK
(one stack per environment) so it stands up and tears down reproducibly.

---

## 8. Rough monthly cost (order of magnitude)

Single always-on small task, low traffic. Figures are approximate and vary by
region; treat them as a sanity range, not a quote.

| Item | Assumption | Rough monthly |
| --- | --- | --- |
| Fargate task (always on) | ~0.5 vCPU / 1 GB, 24x7 | ~$15-25 |
| Application Load Balancer | 1 ALB, low LCU | ~$16-22 |
| EFS | a few GB, low I/O | ~$1-3 |
| ECR + CloudWatch Logs | small image, modest log volume | ~$1-3 |
| Data transfer | low internal-tool traffic | ~$1-5 |
| Scheduled jobs | short daily + weekly Fargate runs | cents to low $ |
| **Total** | | **~$35-60 / month** |

If the heavy weekly Meridian posterior fit is enabled, bump the scheduled task to
more vCPU/memory for the minutes it runs; that adds cents-to-low-dollars per run,
not a fixed monthly cost. If you later move the dashboard to S3 + CloudFront, add
a few dollars for CloudFront and subtract a little ALB/Fargate static-serving
load. Cognito for a handful of internal users is effectively free at this scale.

---

## 9. What still needs a code change to be fully cloud-native

Honest list of work that is out of scope for these deployment artifacts and would
require touching application code:

1. **Static dashboard serving.** Add the StaticFiles mount described in Section 2
   (one small block in `kairos_api/server.py`) so the single container actually
   serves the dashboard at `/`. Until then the container serves only `/api/*` and
   the dashboard must be hosted separately (S3 + CloudFront).
2. **File paths to S3.** The engine reads and writes local paths
   (`data/`, `models/`, `output/`, `data/kairos_decisions.json`). EFS makes this
   work with zero code change today. Becoming S3-native means routing those reads
   and writes through `boto3` / a storage abstraction in `kairos_api/server.py`,
   `kairos/service.py`, and the scripts.
3. **Authentication.** The app has no login. ALB/Cognito covers the public-exposure
   risk without code changes, but if you want per-user authorization or audit of
   who changed settings/made decisions, that is application work.
4. **Externalize the decisions / audit log to a database.** The operator
   decisions are appended to `data/kairos_decisions.json` (capped at 500 entries).
   For multi-instance safety, real audit history, and concurrency, move this to a
   managed DB (DynamoDB or RDS/Postgres). With a single always-on task on EFS this
   is not urgent, but it is the right move before scaling out or treating the
   decisions log as a system of record.

---

## 10. Scope statement

These artifacts (`Dockerfile`, `docker-compose.yml`, `.dockerignore`, this doc)
let the owner build and run the system locally and give a concrete AWS plan. They
do **not** create any AWS resources. Real provisioning requires the owner's AWS
account, region, and credentials and is intentionally not performed here.
