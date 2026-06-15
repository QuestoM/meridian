# Kairos on AWS: architecture blueprint (design only)

This is a cost-efficient, serverless-first design for running Kairos as an online
product: a dashboard where the channel uploads a month of data, trains the
Meridian impact model in the cloud (with full status, logs and failure handling),
keeps the trained posterior as the basis for the daily break-placement decisions,
and (stage 2) adds an AI classifier and an AI chat assistant.

It is a blueprint. Nothing here provisions real resources. The owner decides the
account, region and budget before any of it is applied.

## Design principles

- Serverless-first, scale-to-zero. Heavy compute (Meridian MCMC, Bedrock) runs
  only when a job is submitted; the rest is Lambda and managed stores that cost
  almost nothing at idle. A single-channel workload is tiny.
- Heavy think only where it earns it. Training and the AI classifier/chat are the
  only expensive steps; everything else is light.
- Every run is observable and reproducible: input checksums, a job record, full
  step logs, and the trained artifact retained as the basis for what follows.
- No fabrication carries over from the engine: a posterior that fails convergence
  is flagged and not promoted; missing data is honest empty, never invented.

## Components and why each one

| Concern | Service | Why this and not the alternative |
| --- | --- | --- |
| Raw data + artifacts + outputs | S3 | Cheap durable store for uploaded Spots/Programmes/Dayparts, the trained posterior pkl, schedules and run logs. Versioned buckets give artifact history for free. |
| Training orchestration | Step Functions | Models the pipeline ingest to validate to train to diagnose to publish as explicit states with retries, timeouts and per-step logs. This is exactly the "handle every failure, show status, keep logs" requirement. |
| Training compute | AWS Batch on Fargate Spot | Meridian MCMC needs Python 3.11/3.12 + TensorFlow, is CPU-heavy and runs minutes. Lambda caps at 15 min and dislikes TF; SageMaker is heavier and pricier than needed. Batch on Fargate Spot is the lean fit, scale-to-zero. |
| Light compute | Lambda | Data validation/enrich (reuses `kairos.data.contracts`), the AI classifier call, aggregation, and the daily optimize (the optimizer is light and fast). |
| Daily optimize endpoint | API Gateway + Lambda | Loads the active posterior from S3, runs `optimize_day_plan`, returns the plan. Fast, cheap, scale-to-zero. |
| App/website DB | DynamoDB | Runs, plans, settings, per-channel coefficients, training-job status. Serverless, near-free at this scale, no idle cost. |
| AI classifier + chat (stage 2) | Bedrock (Claude) | Unclassified programmes to a structured-output classifier with search grounding. Stage 2 adds a chat assistant over the same data. |
| Static dashboard | S3 + CloudFront + WAF | The existing Vite/React dashboard served as static assets on a CDN, protected by WAF. |
| Auth | Cognito | User pools for the dashboard. |
| Schedules | EventBridge | Daily optimize and monthly retrain triggers. |
| Logs/metrics/alarms | CloudWatch | All Step Functions and Lambda logs, training metrics, alarms. Grafana over ECS Fargate is an optional later upgrade, not needed to start. |

## Flow

```
upload (dashboard) -> API GW -> Lambda (validate + enrich) -> S3 raw
                                       |
                                 Step Functions
                ingest -> validate -> train(Batch/Fargate Spot) -> diagnose -> publish
                                       |                              |
                                 CloudWatch logs              posterior.pkl + R-hat/ESS
                                                                      |
                                                                  S3 artifacts + DynamoDB
                                                                      |
daily: EventBridge -> API GW -> Lambda(optimize, loads active posterior) -> S3 schedule
                                                                      |
                                       CloudFront + WAF -> dashboard (static)
stage 2: Bedrock classifier (unclassified programmes) + Bedrock chat assistant
```

## Online training UX (what the dashboard shows)

1. Upload or connect a source. A validation report renders from
   `kairos.data.contracts` before anything trains (errors visible, nothing imputed).
2. A Train button submits a Step Functions execution. The dashboard polls
   DynamoDB for live status: queued, validating, sampling, diagnosing, publishing.
3. Training history lists every run: data window, duration, convergence
   diagnostics (R-hat, divergences, effective sample size), and the resulting
   per-channel coefficients. A posterior that fails convergence is flagged and not
   promoted automatically.
4. The latest successful posterior is stored in S3 and registered in DynamoDB as
   the active model. The daily optimize endpoint loads it. Without a healthy
   model, the engine falls back honestly to the declared assumption.

## Cost shape (order of magnitude, single channel)

- Idle: a few dollars a month (S3 + DynamoDB + CloudWatch retention).
- Per training run: minutes of Fargate Spot vCPU, cents to low single dollars.
- Daily optimize: Lambda invocations, negligible.
- Bedrock (stage 2): per-call, only for unclassified programmes and chat.

The dominant cost is training, which is run monthly, on demand. Everything else is
effectively free at this scale.

## What stays out of v1

- Grafana over ECS Fargate: CloudWatch dashboards cover v1; add Grafana only if
  richer ops visualisation is wanted.
- The AI chat assistant: stage 2.
- Multi-channel/multi-tenant scale-out: the design already supports it (per-tenant
  partition keys in DynamoDB, per-tenant S3 prefixes), but is not needed for the
  single Israeli channel at first.

## Infrastructure-as-code

When the owner is ready to provision, this is expressed as Terraform or AWS CDK
(one stack per environment), so the whole thing stands up and tears down
reproducibly. The IaC is written against the chosen account, region and budget;
those are owner decisions, not defaults.
