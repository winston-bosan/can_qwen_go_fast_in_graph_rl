#!/usr/bin/env bash
# RunPod pod setup: run neo4j + qdrant + toolserver NATIVELY (no docker).
#
# RunPod GPU pods ARE containers -- docker-in-docker is not available, so the
# repo's docker-compose.yml cannot be used inside a pod. This script installs
# and supervises the three services natively, then (optionally) chains into
# setup_remote.sh for the verl training venv.
#
# Usage (from the repo root on the pod, typically /workspace/entity_component_search):
#   training/launch/setup_runpod.sh [--restore /path/data_bundle.tar.zst]
#                                   [--import-csv] [--skip-train-setup]
#                                   [--stop] [--status]
#
#   --restore FILE    extract a data bundle made by make_data_tarball.sh
#                     (binary restore route -- versions must match, see below)
#   --import-csv      rebuild the neo4j store from data/neo4j/import/*.csv via
#                     neo4j-admin database import full (version-independent route)
#   --skip-train-setup  skip calling setup_remote.sh (services only)
#   --stop            stop all three services and exit
#   --status          print service status and exit
#
# Data restore routes -- pick ONE:
#   1. BINARY restore (fast): tarball contains data/qdrant (storage dir),
#      data/neo4j/data (store files), data/sidecar.db. Requires the pod to run
#      the SAME major.minor versions as the box that produced them:
#      qdrant 1.18.x (local docker ran 1.18.2) and neo4j 5.26.x (local docker
#      ran neo4j:5.26-community). Auth (neo4j/ecs-local-dev) travels with the
#      store. Use QDRANT_VERSION / NEO4J_VERSION below to match.
#   2. CSV re-import (slow, version-proof): tarball only needs
#      data/neo4j/import/{entities.csv,triples.csv} + data/qdrant + sidecar.db.
#      Pass --import-csv. Use when the binary store won't start (version skew,
#      arch mismatch) -- takes ~10-20 min for 21M edges, then the entity_qid
#      index build. Qdrant has no CSV route; its storage dir is the only
#      artifact, so keep qdrant on 1.18.x regardless.
#
# Idempotent: downloads are skipped when present, services are not restarted
# when already listening, config edits are upserts. Safe to re-run after a pod
# restart (RunPod wipes the container layer but keeps /workspace).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${ECS_DATA_DIR:-$ROOT/data}"
SERVICES_DIR="${ECS_SERVICES_DIR:-$ROOT/.services}"
RUN_DIR="$SERVICES_DIR/run"          # pidfiles + logs
VENV="${ECS_TRAIN_VENV:-$ROOT/.venv-train}"

QDRANT_VERSION="${QDRANT_VERSION:-v1.18.2}"     # match the dev box (docker ran 1.18.2)
NEO4J_VERSION="${NEO4J_VERSION:-5.26.0}"        # any 5.26.x; match dev box minor for binary restore
NEO4J_PASSWORD="${ECS_NEO4J_PASSWORD:-ecs-local-dev}"
NEO4J_HEAP="${NEO4J_HEAP:-8G}"
NEO4J_PAGECACHE="${NEO4J_PAGECACHE:-8G}"
TOOLSERVER_WORKERS="${TOOLSERVER_WORKERS:-4}"   # each worker lazily loads its own embedder (harrier-270m, ~0.7GB VRAM)
TOOLSERVER_CUDA="${TOOLSERVER_CUDA_VISIBLE_DEVICES-}"  # e.g. "0" to pin embedders to GPU0 on multi-GPU pods

mkdir -p "$RUN_DIR" "$DATA_DIR"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
log() { echo "[setup_runpod] $*"; }

port_up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

start_service() {  # name port pidfile logfile cmd...
    local name="$1" port="$2" pidfile="$3" logfile="$4"; shift 4
    if port_up "$port"; then log "$name already listening on :$port"; return 0; fi
    if pid_alive "$pidfile"; then log "$name pid alive but port :$port closed -- check $logfile"; return 1; fi
    log "starting $name (log: $logfile)"
    setsid nohup "$@" >>"$logfile" 2>&1 &
    echo $! > "$pidfile"
    for _ in $(seq 1 "${SERVICE_WAIT:-90}"); do
        port_up "$port" && { log "$name up on :$port"; return 0; }
        pid_alive "$pidfile" || { log "FATAL: $name exited early -- tail $logfile:"; tail -20 "$logfile"; return 1; }
        sleep 2
    done
    log "FATAL: $name did not open :$port -- tail $logfile:"; tail -20 "$logfile"; return 1
}

stop_service() {  # name pidfile
    if pid_alive "$2"; then
        log "stopping $1 (pid $(cat "$2"))"
        kill "$(cat "$2")" 2>/dev/null || true
        for _ in $(seq 1 30); do pid_alive "$2" || break; sleep 1; done
        pid_alive "$2" && kill -9 "$(cat "$2")" 2>/dev/null || true
    fi
    rm -f "$2"
}

status() {
    for svc in "qdrant 6333" "neo4j 7687" "toolserver 7801"; do
        set -- $svc
        if port_up "$2"; then echo "$1  UP    :$2"; else echo "$1  DOWN  :$2"; fi
    done
    if [ -x "$VENV/bin/python" ]; then
        (cd "$ROOT" && "$VENV/bin/python" scripts/status.py) || true
    fi
}

# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
RESTORE_TARBALL="" IMPORT_CSV=0 SKIP_TRAIN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --restore) RESTORE_TARBALL="$2"; shift 2 ;;
        --import-csv) IMPORT_CSV=1; shift ;;
        --skip-train-setup) SKIP_TRAIN=1; shift ;;
        --stop)
            stop_service toolserver "$RUN_DIR/toolserver.pid"
            stop_service neo4j "$RUN_DIR/neo4j.pid"
            stop_service qdrant "$RUN_DIR/qdrant.pid"
            exit 0 ;;
        --status) status; exit 0 ;;
        *) log "unknown arg: $1"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 0. OS deps (JRE for neo4j, zstd for the bundle, misc)
# ---------------------------------------------------------------------------
if ! command -v java >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1; then
    log "installing OS deps (headless JDK, zstd)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # neo4j 5.26 supports Java 17 and 21; prefer 21, fall back to 17
    apt-get install -y -qq openjdk-21-jre-headless zstd curl rsync \
        || apt-get install -y -qq openjdk-17-jre-headless zstd curl rsync
fi
java -version 2>&1 | head -1

# ---------------------------------------------------------------------------
# 1. restore data bundle (before services touch the dirs)
# ---------------------------------------------------------------------------
if [ -n "$RESTORE_TARBALL" ]; then
    [ -f "$RESTORE_TARBALL" ] || { log "FATAL: $RESTORE_TARBALL not found"; exit 1; }
    if port_up 6333 || port_up 7687; then
        log "FATAL: stop services before restoring data ($0 --stop)"; exit 1
    fi
    log "extracting $RESTORE_TARBALL into $ROOT (contains data/... paths)"
    tar -I zstd -xf "$RESTORE_TARBALL" -C "$ROOT"
    if [ -f "$DATA_DIR/data_bundle.sha256" ]; then
        log "verifying checksums (skip with ECS_SKIP_VERIFY=1)"
        if [ "${ECS_SKIP_VERIFY:-0}" != "1" ]; then
            (cd "$ROOT" && sha256sum --quiet -c "$DATA_DIR/data_bundle.sha256") \
                || { log "FATAL: checksum mismatch"; exit 1; }
        fi
    fi
    [ -f "$DATA_DIR/data_bundle_manifest.json" ] && { log "bundle manifest:"; cat "$DATA_DIR/data_bundle_manifest.json"; }
fi

# ---------------------------------------------------------------------------
# 2. qdrant (standalone binary; storage dir == the docker volume layout)
# ---------------------------------------------------------------------------
QDRANT_BIN="$SERVICES_DIR/qdrant/qdrant"
if [ ! -x "$QDRANT_BIN" ]; then
    log "downloading qdrant $QDRANT_VERSION"
    mkdir -p "$SERVICES_DIR/qdrant"
    url_base="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION"
    curl -fL "$url_base/qdrant-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/qdrant.tgz \
        || curl -fL "$url_base/qdrant-x86_64-unknown-linux-musl.tar.gz" -o /tmp/qdrant.tgz
    tar -xzf /tmp/qdrant.tgz -C "$SERVICES_DIR/qdrant" && rm -f /tmp/qdrant.tgz
    [ -x "$QDRANT_BIN" ] || { log "FATAL: qdrant binary not found after extract"; exit 1; }
fi
"$QDRANT_BIN" --version
mkdir -p "$DATA_DIR/qdrant"
start_service qdrant 6333 "$RUN_DIR/qdrant.pid" "$RUN_DIR/qdrant.log" \
    env QDRANT__STORAGE__STORAGE_PATH="$DATA_DIR/qdrant" \
        QDRANT__SERVICE__HOST=127.0.0.1 \
        QDRANT__SERVICE__HTTP_PORT=6333 QDRANT__SERVICE__GRPC_PORT=6334 \
        QDRANT__TELEMETRY_DISABLED=true \
        "$QDRANT_BIN"

# ---------------------------------------------------------------------------
# 3. neo4j (tarball install; store dir == the docker /data volume layout)
# ---------------------------------------------------------------------------
NEO4J_HOME="$SERVICES_DIR/neo4j-community-$NEO4J_VERSION"
if [ ! -x "$NEO4J_HOME/bin/neo4j" ]; then
    log "downloading neo4j-community-$NEO4J_VERSION"
    curl -fL "https://dist.neo4j.org/neo4j-community-$NEO4J_VERSION-unix.tar.gz" -o /tmp/neo4j.tgz
    tar -xzf /tmp/neo4j.tgz -C "$SERVICES_DIR" && rm -f /tmp/neo4j.tgz
fi

conf="$NEO4J_HOME/conf/neo4j.conf"
set_conf() {  # key value -- idempotent upsert
    grep -q "^$1=" "$conf" && sed -i "s|^$1=.*|$1=$2|" "$conf" || echo "$1=$2" >> "$conf"
}
set_conf server.directories.data "$DATA_DIR/neo4j/data"
set_conf server.directories.import "$DATA_DIR/neo4j/import"
set_conf server.default_listen_address 127.0.0.1
set_conf server.memory.heap.initial_size "$NEO4J_HEAP"
set_conf server.memory.heap.max_size "$NEO4J_HEAP"
set_conf server.memory.pagecache.size "$NEO4J_PAGECACHE"
mkdir -p "$DATA_DIR/neo4j/data" "$DATA_DIR/neo4j/import"

if [ "$IMPORT_CSV" = "1" ]; then
    [ -f "$DATA_DIR/neo4j/import/entities.csv" ] && [ -f "$DATA_DIR/neo4j/import/triples.csv" ] \
        || { log "FATAL: --import-csv needs data/neo4j/import/{entities,triples}.csv"; exit 1; }
    if port_up 7687; then log "FATAL: stop neo4j before --import-csv"; exit 1; fi
    log "bulk import (destroys any existing store) -- same command as ingest/load_neo4j.py"
    "$NEO4J_HOME/bin/neo4j-admin" database import full neo4j \
        --nodes=Entity="$DATA_DIR/neo4j/import/entities.csv" \
        --relationships="$DATA_DIR/neo4j/import/triples.csv" \
        --overwrite-destination --verbose
fi

# Fresh store (CSV import or empty dir): set the password before first start.
# A binary-restored store carries its auth; set-initial-password would fail -> skip.
if [ ! -d "$DATA_DIR/neo4j/data/dbms" ]; then
    "$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD" || true
fi

SERVICE_WAIT=150 start_service neo4j 7687 "$RUN_DIR/neo4j.pid" "$RUN_DIR/neo4j.log" \
    "$NEO4J_HOME/bin/neo4j" console

if [ "$IMPORT_CSV" = "1" ]; then
    log "creating entity_qid index (as ingest/load_neo4j.py does)"
    "$NEO4J_HOME/bin/cypher-shell" -a bolt://127.0.0.1:7687 -u neo4j -p "$NEO4J_PASSWORD" \
        "CREATE INDEX entity_qid IF NOT EXISTS FOR (n:Entity) ON (n.qid)"
fi

# ---------------------------------------------------------------------------
# 4. python venv: training deps (verl) + repo package (toolserver deps)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" != "1" ]; then
    "$ROOT/training/launch/setup_remote.sh"
fi
[ -x "$VENV/bin/python" ] || { log "FATAL: $VENV missing (run setup_remote.sh)"; exit 1; }
if ! "$VENV/bin/python" -c 'import fastapi, uvicorn, qdrant_client, neo4j' 2>/dev/null; then
    log "installing repo package into $VENV (toolserver deps; torch already satisfied by verl)"
    "$VENV/bin/pip" install -q -e "$ROOT"
fi

# ---------------------------------------------------------------------------
# 5. toolserver (multi-worker uvicorn: rollout does 1.5k-6k calls/step)
# ---------------------------------------------------------------------------
start_service toolserver 7801 "$RUN_DIR/toolserver.pid" "$RUN_DIR/toolserver.log" \
    env PYTHONPATH="$ROOT:$ROOT/src" \
        ${TOOLSERVER_CUDA:+CUDA_VISIBLE_DEVICES="$TOOLSERVER_CUDA"} \
        "$VENV/bin/uvicorn" toolserver.app:app \
        --host 127.0.0.1 --port 7801 --workers "$TOOLSERVER_WORKERS"

# warm the lazy embedder in every worker (first /vector_search per worker is slow)
log "warming embedder across $TOOLSERVER_WORKERS workers"
for _ in $(seq 1 "$((TOOLSERVER_WORKERS * 2))"); do
    curl -sf -m 120 -X POST localhost:7801/vector_search \
        -H 'content-type: application/json' -d '{"query":"warmup","k":1}' >/dev/null || true
done

# ---------------------------------------------------------------------------
# 6. health summary
# ---------------------------------------------------------------------------
status
log "done. Next: $VENV/bin/python -m training.rollout_smoke   (live stack)"
log "      then: training/launch/loadtest_toolserver.py --duration 30"
log "      then: training/launch/run_validate.sh"
