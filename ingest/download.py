"""Download Wikidata5M (transductive) into data/wikidata5m/.

Source: the Hugging Face mirror `intfloat/wikidata5m` (the SimKGC mirror of the
original DeepGraphLearning/GraphVite release; the original Dropbox links on the
KEPLER project page are flaky). Override with ECS_WIKIDATA5M_REPO.

Idempotent + resumable:
  - hf_hub_download resumes partial downloads and skips completed ones.
  - Extraction is skipped when the target file already exists and is non-empty.

Files produced (per DESIGN.md):
  data/wikidata5m/wikidata5m_transductive_train.txt   (~20.6M triples, TSV Q P Q)
  data/wikidata5m/wikidata5m_transductive_valid.txt
  data/wikidata5m/wikidata5m_transductive_test.txt
  data/wikidata5m/wikidata5m_text.txt                 (~4.8M lines, QID \t abstract)
  data/wikidata5m/wikidata5m_entity.txt               (entity aliases, QID \t alias \t ...)
  data/wikidata5m/wikidata5m_relation.txt             (relation aliases, PID \t alias \t ...)

Usage:
  .venv/bin/python ingest/download.py [--verify-only]
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402

REPO_ID = os.environ.get("ECS_WIKIDATA5M_REPO", "intfloat/wikidata5m")
OUT_DIR = os.path.join(config.DATA_DIR, "wikidata5m")

ARCHIVES = {
    "wikidata5m_transductive.tar.gz": [
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive_valid.txt",
        "wikidata5m_transductive_test.txt",
    ],
    "wikidata5m_alias.tar.gz": [
        "wikidata5m_entity.txt",
        "wikidata5m_relation.txt",
    ],
    "wikidata5m_text.txt.gz": ["wikidata5m_text.txt"],
}

# sanity thresholds (approximate known sizes)
MIN_LINES = {
    "wikidata5m_transductive_train.txt": 20_000_000,
    "wikidata5m_text.txt": 4_500_000,
    "wikidata5m_entity.txt": 4_500_000,
    "wikidata5m_relation.txt": 800,
}


def _present(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def download() -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(OUT_DIR, exist_ok=True)
    for archive, members in ARCHIVES.items():
        targets = [os.path.join(OUT_DIR, m) for m in members]
        if all(_present(t) for t in targets):
            print(f"[skip] {archive}: all extracted files present")
            continue
        print(f"[download] {archive} from {REPO_ID} ...")
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=archive,
            local_dir=OUT_DIR,
        )
        print(f"[extract] {local}")
        if archive.endswith(".tar.gz"):
            with tarfile.open(local, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if name not in members:
                        continue
                    dst = os.path.join(OUT_DIR, name)
                    if _present(dst):
                        continue
                    src = tar.extractfile(member)
                    assert src is not None
                    with open(dst + ".part", "wb") as out:
                        shutil.copyfileobj(src, out)
                    os.replace(dst + ".part", dst)
                    print(f"  -> {dst}")
        elif archive.endswith(".txt.gz"):
            dst = os.path.join(OUT_DIR, members[0])
            with gzip.open(local, "rb") as src, open(dst + ".part", "wb") as out:
                shutil.copyfileobj(src, out)
            os.replace(dst + ".part", dst)
            print(f"  -> {dst}")


def verify() -> bool:
    ok = True
    for members in ARCHIVES.values():
        for m in members:
            path = os.path.join(OUT_DIR, m)
            if not _present(path):
                print(f"[FAIL] missing: {path}")
                ok = False
                continue
            n = 0
            with open(path, "rb") as f:
                for _ in f:
                    n += 1
            size_mb = os.path.getsize(path) / 1e6
            status = "OK"
            if m in MIN_LINES and n < MIN_LINES[m]:
                status = f"FAIL (expected >= {MIN_LINES[m]:,} lines)"
                ok = False
            print(f"[{status}] {m}: {n:,} lines, {size_mb:,.1f} MB")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        download()
    if not verify():
        sys.exit(1)


if __name__ == "__main__":
    main()
