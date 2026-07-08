#!/usr/bin/env bash
# Vast.ai instance setup: neo4j + qdrant + toolserver as SUPERVISOR services.
#
# Differences vs setup_runpod.sh (read /etc/vast-agents-guide.md on the box --
# it is the operating guide and this script follows it):
#   * Long-running services go under **supervisor** (wrapper script in
#     /opt/supervisor-scripts/ + /etc/supervisor/conf.d/ entry, foreground,
#     bound to 127.0.0.1, logs at /var/log/portal/<name>.log) -- NOT loose
#     nohup/setsid. No /etc/portal.yaml entries: these services are
#     localhost-only, never exposed through the Caddy edge.
#   * Python: the image's default venv /venv/main (torch preinstalled) hosts
#     the toolserver + ingest deps via `uv pip install -e .`; the verl training
#     venv (.venv-train) is separate and unchanged (setup_remote.sh).
#   * PERSISTENCE WARNING: if `vast-capabilities | jq .instance.workspace_is_volume`
#     is false (typical), NOTHING survives a recycle/destroy -- stop/start is
#     safe, recycle is not. Keep anything irreplaceable synced off-box; the
#     data here is rebuildable via ingest/ (that is the plan on this box).
#
# Data: this box builds its own corpus (ingest/download.py -> build_sidecar.py
# -> load_neo4j.py --csv-only + native import -> embed_qdrant.py). --restore
# and --import-csv are still supported for the tarball route.
#
# Usage (repo root, e.g. /workspace/entity_component_search):
#   training/launch/setup_vast.sh [--restore FILE] [--import-csv]
#                                 [--skip-train-setup] [--status] [--stop]
# Idempotent: re-run freely.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${ECS_DATA_DIR:-$ROOT/data}"
SERVICES_DIR="${ECS_SERVICES_DIR:-$ROOT/.services}"
MAIN_VENV="${ECS_MAIN_VENV:-/venv/main}"

QDRANT_VERSION="${QDRANT_VERSION:-v1.18.2}"
NEO4J_VERSION="${NEO4J_VERSION:-5.26.0}"
NEO4J_PASSWORD="${ECS_NEO4J_PASSWORD:-ecs-local-dev}"
NEO4J_HEAP="${NEO4J_HEAP:-8G}"
NEO4J_PAGECACHE="${NEO4J_PAGECACHE:-12G}"
TOOLSERVER_WORKERS="${TOOLSERVER_WORKERS:-4}"

NEO4J_HOME="$SERVICES_DIR/neo4j-community-$NEO4J_VERSION"
QDRANT_BIN="$SERVICES_DIR/qdrant/qdrant"
SUP_SCRIPTS=/opt/supervisor-scripts
SUP_CONF=/etc/supervisor/conf.d

log() { echo "[setup_vast] $*"; }
port_up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

status() {
    supervisorctl status ecs-qdrant ecs-neo4j ecs-toolserver 2>/dev/null || true
    for svc in "qdrant 6333" "neo4j 7687" "toolserver 7801"; do
        set -- $svc
        if port_up "$2"; then echo "$1  UP    :$2"; else echo "$1  DOWN  :$2"; fi
    done
    [ -x "$MAIN_VENV/bin/python" ] && (cd "$ROOT" && "$MAIN_VENV/bin/python" scripts/status.py) || true
}

RESTORE_TARBALL="" IMPORT_CSV=0 SKIP_TRAIN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --restore) RESTORE_TARBALL="$2"; shift 2 ;;
        --import-csv) IMPORT_CSV=1; shift ;;
        --skip-train-setup) SKIP_TRAIN=1; shift ;;
        --stop) supervisorctl stop ecs-toolserver ecs-neo4j ecs-qdrant 2>/dev/null || true; exit 0 ;;
        --status) status; exit 0 ;;
        *) log "unknown arg: $1"; exit 2 ;;
    esac
done

mkdir -p "$SERVICES_DIR" "$DATA_DIR"

# --- persistence check (vast guide §3) ----------------------------------------
if command -v vast-capabilities >/dev/null 2>&1; then
    if [ "$(vast-capabilities | jq -r '.instance.workspace_is_volume' 2>/dev/null)" != "true" ]; then
        log "WARNING: \$WORKSPACE is NOT a volume -- nothing survives recycle/destroy."
        log "         stop/start is safe; data here is rebuildable via ingest/."
    fi
fi

# --- OS deps (logged; vast guide asks installs be accounted for) ---------------
if ! command -v java >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1; then
    log "apt-get install: openjdk-21-jre-headless zstd (for neo4j + data bundles)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq openjdk-21-jre-headless zstd \
        || apt-get install -y -qq openjdk-17-jre-headless zstd
fi
java -version 2>&1 | head -1

# --- optional data restore ------------------------------------------------------
if [ -n "$RESTORE_TARBALL" ]; then
    [ -f "$RESTORE_TARBALL" ] || { log "FATAL: $RESTORE_TARBALL not found"; exit 1; }
    if port_up 6333 || port_up 7687; then log "FATAL: --stop services before restore"; exit 1; fi
    tar -I zstd -xf "$RESTORE_TARBALL" -C "$ROOT"
    if [ -f "$DATA_DIR/data_bundle.sha256" ] && [ "${ECS_SKIP_VERIFY:-0}" != "1" ]; then
        (cd "$ROOT" && sha256sum --quiet -c "$DATA_DIR/data_bundle.sha256") || { log "FATAL: checksum mismatch"; exit 1; }
    fi
fi

# --- qdrant binary ----------------------------------------------------------------
if [ ! -x "$QDRANT_BIN" ]; then
    log "downloading qdrant $QDRANT_VERSION"
    mkdir -p "$SERVICES_DIR/qdrant"
    url_base="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION"
    curl -fsSL "$url_base/qdrant-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/qdrant.tgz \
        || curl -fsSL "$url_base/qdrant-x86_64-unknown-linux-musl.tar.gz" -o /tmp/qdrant.tgz
    tar -xzf /tmp/qdrant.tgz -C "$SERVICES_DIR/qdrant" && rm -f /tmp/qdrant.tgz
fi
mkdir -p "$DATA_DIR/qdrant"

# --- neo4j tarball ------------------------------------------------------------------
if [ ! -x "$NEO4J_HOME/bin/neo4j" ]; then
    log "downloading neo4j-community-$NEO4J_VERSION"
    curl -fsSL "https://dist.neo4j.org/neo4j-community-$NEO4J_VERSION-unix.tar.gz" -o /tmp/neo4j.tgz
    tar -xzf /tmp/neo4j.tgz -C "$SERVICES_DIR" && rm -f /tmp/neo4j.tgz
fi
conf="$NEO4J_HOME/conf/neo4j.conf"
set_conf() { grep -q "^$1=" "$conf" && sed -i "s|^$1=.*|$1=$2|" "$conf" || echo "$1=$2" >> "$conf"; }
set_conf server.directories.data "$DATA_DIR/neo4j/data"
set_conf server.directories.import "$DATA_DIR/neo4j/import"
set_conf server.default_listen_address 127.0.0.1
set_conf server.memory.heap.initial_size "$NEO4J_HEAP"
set_conf server.memory.heap.max_size "$NEO4J_HEAP"
set_conf server.memory.pagecache.size "$NEO4J_PAGECACHE"
mkdir -p "$DATA_DIR/neo4j/data" "$DATA_DIR/neo4j/import"

if [ "$IMPORT_CSV" = "1" ]; then
    [ -f "$DATA_DIR/neo4j/import/entities.csv" ] || { log "FATAL: entities.csv missing"; exit 1; }
    if port_up 7687; then log "FATAL: stop neo4j before --import-csv (supervisorctl stop ecs-neo4j)"; exit 1; fi
    log "neo4j-admin bulk import (same command as ingest/load_neo4j.py)"
    "$NEO4J_HOME/bin/neo4j-admin" database import full neo4j \
        --nodes=Entity="$DATA_DIR/neo4j/import/entities.csv" \
        --relationships="$DATA_DIR/neo4j/import/triples.csv" \
        --overwrite-destination --verbose
fi
if [ ! -d "$DATA_DIR/neo4j/data/dbms" ]; then
    "$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD" || true
fi

# --- python deps into the image venv (toolserver + ingest) ---------------------------
if ! "$MAIN_VENV/bin/python" -c 'import fastapi, uvicorn, qdrant_client, neo4j, sentence_transformers' 2>/dev/null; then
    log "uv pip install -e $ROOT into $MAIN_VENV (toolserver + ingest deps)"
    (source "$MAIN_VENV/bin/activate" && uv pip install -q -e "$ROOT")
fi

# --- supervisor services (vast guide §7) ----------------------------------------------
write_service() {  # name command...
    local name="$1"; shift
    cat > "$SUP_SCRIPTS/$name.sh" <<EOF
#!/bin/bash
# ECS $name -- generated by training/launch/setup_vast.sh (internal, 127.0.0.1 only)
utils=$SUP_SCRIPTS/utils
. "\${utils}/logging.sh" 2>/dev/null || true
. "\${utils}/environment.sh" 2>/dev/null || true
cd "$ROOT"
exec $* 2>&1
EOF
    chmod +x "$SUP_SCRIPTS/$name.sh"
    cat > "$SUP_CONF/$name.conf" <<EOF
[program:$name]
environment=PROC_NAME="%(program_name)s"
command=$SUP_SCRIPTS/$name.sh
autostart=true
autorestart=unexpected
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
EOF
}

write_service ecs-qdrant \
    env QDRANT__STORAGE__STORAGE_PATH="$DATA_DIR/qdrant" \
        QDRANT__SERVICE__HOST=127.0.0.1 \
        QDRANT__SERVICE__HTTP_PORT=6333 QDRANT__SERVICE__GRPC_PORT=6334 \
        QDRANT__TELEMETRY_DISABLED=true \
        "$QDRANT_BIN"

write_service ecs-neo4j "$NEO4J_HOME/bin/neo4j" console

write_service ecs-toolserver \
    env PYTHONPATH="$ROOT:$ROOT/src" \
        "$MAIN_VENV/bin/uvicorn" toolserver.app:app \
        --host 127.0.0.1 --port 7801 --workers "$TOOLSERVER_WORKERS"

supervisorctl reread >/dev/null
supervisorctl update
for _ in $(seq 1 60); do port_up 6333 && port_up 7687 && break; sleep 2; done

# --- training venv (verl) ----------------------------------------------------------------
if [ "$SKIP_TRAIN" != "1" ]; then
    "$ROOT/training/launch/setup_remote.sh"
fi

status
log "done. Ingest next (this box builds its own data):"
log "  $MAIN_VENV/bin/python ingest/download.py"
log "  $MAIN_VENV/bin/python ingest/build_sidecar.py"
log "  $MAIN_VENV/bin/python ingest/load_neo4j.py --csv-only"
log "  supervisorctl stop ecs-neo4j && training/launch/setup_vast.sh --import-csv --skip-train-setup"
log "  $MAIN_VENV/bin/python ingest/embed_qdrant.py"
