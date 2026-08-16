#!/usr/bin/env bash
# Run the full stack with OrbStack (via its docker CLI), capping the app
# container at 1 GB RAM.
#
#   scripts/run_orbstack.sh          # build + start (postgres + app)
#   scripts/run_orbstack.sh stop     # stop containers (keep the network + volume)
#   scripts/run_orbstack.sh clean    # stop and remove everything
#
# Containers share one bridge network, so the app reaches Postgres at
# aibackend-postgres:5432 (the .env DSN stays localhost:5432 for host-side
# dev runs). The app container is limited to --memory=1g --memory-swap=1g.
set -euo pipefail

IMAGE="ai-backend:latest"
NET="aibackend"
PG="aibackend-postgres"
APP="aibackend-app"
VOLUME="aibackend-pgdata"

ensure_engine() {
  if ! docker info >/dev/null 2>&1; then
    echo "==> Docker (OrbStack) is not running. Start OrbStack and re-run:" >&2
    echo "    open -a OrbStack" >&2
    exit 1
  fi
}

cmd_start() {
  ensure_engine

  echo "==> Building ${IMAGE}"
  docker build -t "$IMAGE" .

  docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET"
  docker rm -f "$PG" "$APP" >/dev/null 2>&1 || true

  echo "==> Starting postgres"
  docker run -d --network "$NET" --name "$PG" \
    -e POSTGRES_USER=aibackend -e POSTGRES_PASSWORD=aibackend -e POSTGRES_DB=aibackend \
    -v "$VOLUME:/var/lib/postgresql/data" \
    postgres:16-alpine

  echo "==> Waiting for postgres"
  for _ in $(seq 1 30); do
    docker exec "$PG" pg_isready -U aibackend >/dev/null 2>&1 && break
    sleep 1
  done

  local envfile=()
  [[ -f .env ]] && envfile=(--env-file .env)

  echo "==> Starting app (memory limit 1 GB)"
  docker run -d --network "$NET" --name "$APP" \
    -p 8000:8000 \
    --memory=1g --memory-swap=1g --pids-limit 512 \
    "${envfile[@]}" \
    -e DATABASE_URI=postgresql://aibackend:aibackend@aibackend-postgres:5432/aibackend \
    "$IMAGE"

  echo "==> Waiting for /health"
  for _ in $(seq 1 60); do
    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
      echo "==> Ready: http://localhost:8000/health"
      curl -s http://localhost:8000/health | python3 -m json.tool
      return 0
    fi
    sleep 1
  done
  echo "App did not become healthy in time; check: docker logs $APP" >&2
  return 1
}

cmd_stop() {
  docker stop "$APP" "$PG" 2>/dev/null || true
  echo "Stopped (containers kept). Start again with scripts/run_orbstack.sh"
}

cmd_clean() {
  cmd_stop
  docker rm -f "$APP" "$PG" 2>/dev/null || true
  docker network rm "$NET" 2>/dev/null || true
  docker volume rm -f "$VOLUME" 2>/dev/null || true
  echo "Cleaned up containers, network ${NET} + volume ${VOLUME}"
}

case "${1:-start}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  clean) cmd_clean ;;
  *) echo "usage: $0 [start|stop|clean]" >&2; exit 2 ;;
esac
