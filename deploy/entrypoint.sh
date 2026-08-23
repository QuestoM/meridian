#!/bin/sh
# First boot against an empty volume: seed the state directories from the demo
# data baked into the image, then hand the live paths to the app via symlinks.
# A volume that already holds state is NEVER overwritten - the channel's
# uploaded data outlives every deploy, which is the entire point of the volume.
set -e
STATE=/app/state
mkdir -p "$STATE"
for d in data output models; do
  if [ ! -d "$STATE/$d" ]; then
    echo "seeding $STATE/$d from the image's demo copy"
    cp -r "/app/seed/$d" "$STATE/$d"
  fi
  rm -rf "/app/$d"
  ln -sfn "$STATE/$d" "/app/$d"
done
# The API binds loopback only; the ONLY way in from outside is the static
# server's proxy, so the auth wall and the dashboard share one origin.
python -m uvicorn kairos_api.server:app --host 127.0.0.1 --port 8000 &
exec node /app/deploy/serve-dist.mjs /app/tv-break-dashboard/dist 8080 http://127.0.0.1:8000
