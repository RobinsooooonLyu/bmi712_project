from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def check_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "yaml": "yaml",
        "openslide": "openslide",
        "PIL": "PIL",
        "huggingface_hub": "huggingface_hub",
        "safetensors": "safetensors",
    }
    checks: dict[str, dict[str, str | bool]] = {}
    for label, module_name in modules.items():
        ok, detail = check_import(module_name)
        checks[label] = {"ok": ok, "detail": detail}

    runtime: dict[str, object] = {}
    try:
        import torch

        runtime["cuda_available"] = torch.cuda.is_available()
        runtime["cuda_device_count"] = torch.cuda.device_count()
        runtime["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception as exc:  # pragma: no cover
        runtime["cuda_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from src.models.virchow2_encoder import build_virchow2_preprocess

        build_virchow2_preprocess()
        runtime["virchow2_preprocess_ok"] = True
    except Exception as exc:  # pragma: no cover
        runtime["virchow2_preprocess_ok"] = False
        runtime["virchow2_preprocess_error"] = f"{type(exc).__name__}: {exc}"

    runtime["HF_HOME"] = os.environ.get("HF_HOME", "")
    runtime["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "")
    runtime["HUGGINGFACE_HUB_CACHE"] = os.environ.get("HUGGINGFACE_HUB_CACHE", "")
    runtime["TORCH_HOME"] = os.environ.get("TORCH_HOME", "")
    runtime["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", "")

    print(json.dumps({"imports": checks, "runtime": runtime}, indent=2))

    failed_imports = [name for name, payload in checks.items() if not payload["ok"]]
    if failed_imports:
        print(f"FAILED_IMPORTS: {', '.join(failed_imports)}")
        return 1
    if not runtime.get("virchow2_preprocess_ok", False):
        print("FAILED_RUNTIME: virchow2_preprocess")
        return 1

    print("ENVIRONMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
