#!/usr/bin/env bash
# Run the full stack with Podman, capping the app container at 1 GB RAM.
#
#   scripts/run_podman.sh          # build + start (postgres + app)
#   scripts/run_podman.sh stop     # stop containers (keep the pod + volume)
#   scripts/run_podman.sh clean    # stop and remove everything
#
# Containers share one pod, so the app reaches Postgres at localhost:5432
# (same DSN as DATABASE_URI in .env). The app container is limited to
# --memory=1g --memory-swap=1g; the podman machine VM itself gets 2 GB.
set -euo pipefail

IMAGE="ai-backend:latest"
POD="aibackend"
PG="aibackend-postgres"
APP="aibackend-app"

ensure_machine() {
  if podman info >/dev/null 2>&1; then
    return
  fi
  if ! podman machine list --format '{{.Name}}' | grep -q .; then
    echo "==> Initializing podman machine (Linux VM, 2 CPUs / 2 GB)"
    podman machine init --cpus 2 --memory 2048 --disk-size 20
  fi
  echo "==> Starting podman machine"
  podman machine start
}

cmd_start() {
  ensure_machine

  echo "==> Building ${IMAGE}"
  podman build -t "$IMAGE" .

  podman pod exists "$POD" && podman pod rm -f "$POD"

  echo "==> Creating pod ${POD} (publishes :8000)"
  podman pod create --name "$POD" --publish 8000:8000

  echo "==> Starting postgres"
  podman run -d --pod "$POD" --name "$PG" \
    -e POSTGRES_USER=aibackend -e POSTGRES_PASSWORD=aibackend -e POSTGRES_DB=aibackend \
    -v aibackend-pgdata:/var/lib/postgresql/data \
    postgres:16-alpine

  echo "==> Waiting for postgres"
  for _ in $(seq 1 30); do
    podman exec "$PG" pg_isready -U aibackend >/dev/null 2>&1 && break
    sleep 1
  done

  local envfile=()
  [[ -f .env ]] && envfile=(--env-file .env)

  echo "==> Starting app (memory limit 1 GB)"
  podman run -d --pod "$POD" --name "$APP" \
    --memory=1g --memory-swap=1g --pids-limit 512 \
    "${envfile[@]}" \
    -e DATABASE_URI=postgresql://aibackend:aibackend@localhost:5432/aibackend \
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
  echo "App did not become healthy in time; check: podman logs $APP" >&2
  return 1
}

cmd_stop() {
  podman stop "$APP" "$PG" 2>/dev/null || true
  echo "Stopped (pod ${POD} kept). Start again with scripts/run_podman.sh"
}

cmd_clean() {
  cmd_stop
  podman pod rm -f "$POD" 2>/dev/null || true
  podman volume rm -f aibackend-pgdata 2>/dev/null || true
  echo "Cleaned up pod ${POD} + volume"
}

case "${1:-start}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  clean) cmd_clean ;;
  *) echo "usage: $0 [start|stop|clean]" >&2; exit 2 ;;
esac
