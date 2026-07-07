#!/usr/bin/env bash
# Build data_bundle.tar.zst on the LOCAL box for shipping to a RunPod pod.
#
#   training/launch/make_data_tarball.sh [-o /path/out.tar.zst] [--no-checksums] [--allow-hot]
#
# Contents (restored by setup_runpod.sh --restore):
#   data/qdrant/              qdrant storage dir  (binary; pod must run qdrant 1.18.x)
#   data/neo4j/data/          neo4j store         (binary; pod must run neo4j 5.26.x)
#   data/neo4j/import/*.csv   entities/triples CSVs (version-proof re-import route)
#   data/sidecar.db           sqlite sidecar (abstracts/aliases/relations)
#   data/questions/*.jsonl    training questions (if present)
#   data/data_bundle_manifest.json + data/data_bundle.sha256
#
# COLD-COPY REQUIREMENT: this script stops nothing. qdrant and neo4j must NOT
# be running while their dirs are copied (page-level torn writes otherwise):
#     docker compose stop qdrant neo4j     # ... make tarball ... then:
#     docker compose start qdrant neo4j
# It also refuses while the embedding run is still writing to qdrant
# (data/embed_full.pid) -- the tarball must be made AFTER embedding completes.
# The neo4j store and sidecar.db are already stable. --allow-hot overrides all
# of this if you know the services are idle (NOT recommended for qdrant).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$ROOT/data"
OUT="$DATA_DIR/data_bundle.tar.zst"
DO_CHECKSUMS=1 ALLOW_HOT=0

while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUT="$2"; shift 2 ;;
        --no-checksums) DO_CHECKSUMS=0; shift ;;
        --allow-hot) ALLOW_HOT=1; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

command -v zstd >/dev/null || { echo "FATAL: zstd not installed (apt install zstd)"; exit 1; }

# --- safety gates ------------------------------------------------------------
if [ -f "$DATA_DIR/embed_full.pid" ] && kill -0 "$(cat "$DATA_DIR/embed_full.pid")" 2>/dev/null; then
    echo "FATAL: embedding run still in progress (pid $(cat "$DATA_DIR/embed_full.pid"))."
    echo "The qdrant dir is being written; make the tarball AFTER embedding completes."
    exit 1
fi
running="$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^ecs-(qdrant|neo4j)$' || true)"
if [ -n "$running" ] && [ "$ALLOW_HOT" != "1" ]; then
    echo "FATAL: services running: $(echo "$running" | tr '\n' ' ')"
    echo "Cold-copy required:  docker compose stop qdrant neo4j"
    echo "(restart afterwards: docker compose start qdrant neo4j)   [--allow-hot to override]"
    exit 1
fi

# --- collect contents ----------------------------------------------------------
cd "$ROOT"
declare -a paths=()
for p in data/qdrant data/neo4j/data data/neo4j/import data/sidecar.db data/questions; do
    [ -e "$p" ] && paths+=("$p") || echo "WARN: $p missing -- skipping"
done
[ ${#paths[@]} -gt 0 ] || { echo "FATAL: nothing to bundle"; exit 1; }

# --- manifest ------------------------------------------------------------------
qdrant_img="$(docker inspect ecs-qdrant --format '{{.Config.Image}}' 2>/dev/null || echo unknown)"
neo4j_img="$(docker inspect ecs-neo4j --format '{{.Config.Image}}' 2>/dev/null || echo unknown)"
qdrant_ver="$(curl -sf -m 2 localhost:6333/ 2>/dev/null | sed -n 's/.*"version":"\([^"]*\)".*/\1/p' || true)"
{
    echo "{"
    echo "  \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"host\": \"$(hostname)\","
    echo "  \"qdrant_image\": \"$qdrant_img\","
    echo "  \"qdrant_api_version\": \"${qdrant_ver:-not-running}\","
    echo "  \"neo4j_image\": \"$neo4j_img\","
    echo "  \"restore\": \"training/launch/setup_runpod.sh --restore <this file>; binary route needs qdrant 1.18.x + neo4j 5.26.x on the pod, else --import-csv\","
    echo "  \"contents\": ["
    for i in "${!paths[@]}"; do
        sz="$(du -sb "${paths[$i]}" | cut -f1)"
        sep=","; [ "$i" = "$((${#paths[@]} - 1))" ] && sep=""
        echo "    {\"path\": \"${paths[$i]}\", \"bytes\": $sz}$sep"
    done
    echo "  ]"
    echo "}"
} > "$DATA_DIR/data_bundle_manifest.json"
cat "$DATA_DIR/data_bundle_manifest.json"

# --- checksums (over every file; ~30-40GB takes a few minutes) ------------------
if [ "$DO_CHECKSUMS" = "1" ]; then
    echo "computing sha256 checksums..."
    find "${paths[@]}" -type f -print0 | xargs -0 sha256sum > "$DATA_DIR/data_bundle.sha256"
    echo "$(wc -l < "$DATA_DIR/data_bundle.sha256") files checksummed"
else
    rm -f "$DATA_DIR/data_bundle.sha256"
fi

# --- tar -------------------------------------------------------------------------
extras=(data/data_bundle_manifest.json)
[ -f "$DATA_DIR/data_bundle.sha256" ] && extras+=(data/data_bundle.sha256)
echo "writing $OUT (zstd -T0)..."
tar -I 'zstd -T0 -6' -cf "$OUT" "${extras[@]}" "${paths[@]}"
ls -lh "$OUT"
echo "ship it:  rsync -avP --inplace $OUT root@<pod-ip>:/workspace/"
echo "restore:  training/launch/setup_runpod.sh --restore /workspace/$(basename "$OUT")"
