#!/bin/sh
# First boot against an empty volume: seed the state directories from the demo
# data baked into the image, then hand the live paths to the app via symlinks.
# A volume that already holds state is NEVER overwritten - the channel's
# uploaded data outlives every deploy, which is the entire point of the volume.
set -e
STATE=/app/state
mkdir -p "$STATE"
# The auth store NEVER rides in the image (.dockerignore data/auth): baking the
# operator's local users.json would ship real password hashes to the cloud --
# which the first image did, was caught in the pre-live check, and was rebuilt
# out. A missing store means the API boots LOCKED (auth.py's fresh-clone rule),
# which is the correct cloud default until an admin is seeded on the volume.
for d in data output models; do
  if [ ! -d "$STATE/$d" ]; then
    echo "seeding $STATE/$d from the image's demo copy"
    cp -r "/app/seed/$d" "$STATE/$d"
  fi
  rm -rf "/app/$d"
  ln -sfn "$STATE/$d" "/app/$d"
done
mkdir -p "$STATE/data/auth"
# REMEDIATION, self-applying. The first image baked the operator's local
# users.json and v1 seeded it onto the volume before the pre-live check caught
# it. This removes exactly that file -- matched by content hash, so a real
# admin store seeded later can never be touched -- from any volume the leak
# reached. Harmless no-op everywhere else, kept as the record of the incident.
LEAKED_SHA="dad1cdd4b3b54c607a728b37ce4498e2c8c6bdaf9e0fa594d8f1563cf6e44f82"
STORE="$STATE/data/auth/users.json"
if [ -f "$STORE" ] && [ "$(sha256sum "$STORE" | cut -d' ' -f1)" = "$LEAKED_SHA" ]; then
  rm -f "$STORE"
  echo "remediation: removed the leaked operator auth store from the volume"
fi
# First-boot admin, hands-free and secret-free-in-transit: when the task
# receives KAIROS_ADMIN_PASSWORD (injected by ECS from the managed secret,
# never from an image, a repository or a shell history) and no store exists
# yet, the product's own init_auth seeds the admin. init_auth never prints a
# password that arrived via the environment, and refuses to touch an existing
# store, so a redeploy can never reset credentials.
if [ -n "$KAIROS_ADMIN_PASSWORD" ] && [ ! -f "$STATE/data/auth/users.json" ]; then
  python scripts/init_auth.py && echo "first admin seeded from the managed secret"
fi

# The API binds loopback only; the ONLY way in from outside is the static
# server's proxy, so the auth wall and the dashboard share one origin.
python -m uvicorn kairos_api.server:app --host 127.0.0.1 --port 8000 &
exec node /app/deploy/serve-dist.mjs /app/tv-break-dashboard/dist 8080 http://127.0.0.1:8000
