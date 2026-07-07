#!/usr/bin/env bash
# Bring up the full local stack: neo4j + qdrant (docker) and the tool server.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d

echo -n "waiting for qdrant "
for _ in $(seq 1 60); do
  curl -sf localhost:6333/collections >/dev/null && break
  echo -n .; sleep 2
done
echo " ok"

echo -n "waiting for neo4j "
for _ in $(seq 1 100); do
  docker exec ecs-neo4j cypher-shell -u neo4j -p ecs-local-dev "RETURN 1" >/dev/null 2>&1 && break
  echo -n .; sleep 3
done
echo " ok"

if curl -sf localhost:7801/health >/dev/null 2>&1; then
  echo "toolserver already running on :7801"
else
  mkdir -p data
  echo "starting toolserver on :7801 (log: data/toolserver.log)"
  nohup .venv/bin/uvicorn toolserver.app:app --host 0.0.0.0 --port 7801 \
    >> data/toolserver.log 2>&1 &
  echo $! > data/toolserver.pid
  for _ in $(seq 1 30); do
    curl -sf localhost:7801/health >/dev/null 2>&1 && break
    sleep 1
  done
fi

.venv/bin/python scripts/status.py
