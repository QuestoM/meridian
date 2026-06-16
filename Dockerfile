# Kairos TV-break system - multi-stage production image.
#
# Stage 1 builds the Vite/React dashboard into static files.
# Stage 2 installs the Python/FastAPI backend and copies the built dashboard
# into the image so a single container serves both the API (/api/*) and the
# static dashboard (/).
#
# IMPORTANT (verified against kairos_api/server.py at build time):
# The FastAPI app does NOT currently mount StaticFiles - every route is /api/*.
# For the container to actually serve the dashboard at "/", a small one-time
# server change is required (see docs/deploy/AWS_DEPLOYMENT.md, "Serving the
# dashboard"). Until that change lands, the dashboard files are still copied to
# /app/tv-break-dashboard/dist and you can serve them from CloudFront/S3 or a
# sidecar instead. The build itself is correct and self-contained either way.

# ---------------------------------------------------------------------------
# Stage 1: build the dashboard
# ---------------------------------------------------------------------------
FROM node:20-bookworm-slim AS dashboard-build

WORKDIR /dashboard

# Install dependencies first for better layer caching.
# package-lock.json is present in the repo, so `npm ci` is reproducible.
COPY tv-break-dashboard/package.json tv-break-dashboard/package-lock.json ./
RUN npm ci

# Copy the rest of the dashboard source and build it.
# Vite's default outDir is "dist" (vite.config.js does not override it).
COPY tv-break-dashboard/ ./
RUN npm run build

# After this stage, the built static site lives at /dashboard/dist

# ---------------------------------------------------------------------------
# Stage 2: Python / FastAPI runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Keep Python lean and predictable inside the container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1

WORKDIR /app

# Build tooling is occasionally needed by scientific wheels. Install it, use it,
# then remove it to keep the layer small. Most listed deps ship manylinux wheels
# for cpython 3.11, so this is a safety net rather than a hard requirement.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching.
# requirements.txt includes fastapi + uvicorn[standard] plus the engine deps.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential 2>/dev/null || true

# Copy the application source. The .dockerignore keeps node_modules, .git,
# caches, local logs and large local data backups out of the context.
COPY . .

# Bring in the dashboard built in stage 1. We copy it to the path the app
# expects to serve from (tv-break-dashboard/dist). This path is referenced by
# the documented StaticFiles mount in docs/deploy/AWS_DEPLOYMENT.md.
COPY --from=dashboard-build /dashboard/dist ./tv-break-dashboard/dist

# Runtime state directories. These are also intended to be mounted as volumes
# (docker-compose) or EFS access points (Fargate) so state persists across
# container restarts. Creating them here guarantees the app can write even
# when nothing is mounted.
RUN mkdir -p /app/data /app/models /app/output

# CORS origins are read from this env var by kairos_api/server.py.
ENV KAIROS_CORS_ORIGINS="http://localhost:8000,http://127.0.0.1:8000"

EXPOSE 8000

# Single process serving the API (and, after the documented one-line change,
# the static dashboard) on port 8000.
CMD ["uvicorn", "kairos_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
