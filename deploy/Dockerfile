# Stage 1: the dashboard, built exactly as the operator builds it.
FROM node:26-slim AS dashboard
WORKDIR /build
COPY tv-break-dashboard/package.json tv-break-dashboard/package-lock.json ./
RUN npm ci
COPY tv-break-dashboard/ ./
RUN npm run build

# Stage 2: the runtime. requirements-api.txt is the maintained minimal API set.
FROM python:3.12-slim
WORKDIR /app
# libatomic1: the node binary copied below links against it and slim lacks it.
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
# Node for the static server + proxy (mirrors the local topology).
COPY --from=node:26-slim /usr/local/bin/node /usr/local/bin/node
COPY kairos/ kairos/
COPY kairos_api/ kairos_api/
COPY config/ config/
COPY scripts/init_auth.py scripts/
COPY deploy/serve-dist.mjs deploy/entrypoint.sh deploy/
COPY --from=dashboard /build/dist tv-break-dashboard/dist
# Demo data rides as a SEED, not as live state: the entrypoint copies it onto
# the volume only when the volume is empty. Real channel data arrives through
# the app's own upload screens and lands on the volume, never in this image.
COPY data/ seed/data/
COPY output/ seed/output/
COPY models/ seed/models/
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
