from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from src.models.virchow2_encoder import Virchow2Encoder, build_virchow2_preprocess

    build_virchow2_preprocess()
    _ = Virchow2Encoder()

    payload = {
        "status": "loaded",
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        "TORCH_HOME": os.environ.get("TORCH_HOME", ""),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ""),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
