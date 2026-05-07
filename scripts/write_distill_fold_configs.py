from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml


FOLD_DIR_KEYS = {
    "frozen_latent_dir",
    "ffpe_latent_dir",
    "output_dir",
    "frozen_checkpoint",
}


def build_split(fold: int, n_folds: int = 5) -> tuple[list[int], list[int], list[int]]:
    test_fold = fold
    val_fold = (fold - 1) % n_folds
    train_folds = [idx for idx in range(n_folds) if idx not in {test_fold, val_fold}]
    return train_folds, [val_fold], [test_fold]


def fold_path(path: str, fold: int) -> str:
    if not path:
        return path
    if re.search(r"(^|/)fold\d+(/|$)", path):
        return re.sub(r"(^|/)fold\d+(/|$)", lambda match: f"{match.group(1)}fold{fold}{match.group(2)}", path)
    parent = str(path).rsplit("/", 1)[0]
    return f"{parent}/fold{fold}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    for fold in range(args.n_folds):
        payload = dict(base)
        for key in FOLD_DIR_KEYS:
            if key in base and base[key] is not None:
                payload[key] = fold_path(str(base[key]), fold)
        if {"train_folds", "val_folds", "test_folds"}.issubset(payload):
            train_folds, val_folds, test_folds = build_split(fold, args.n_folds)
            payload["train_folds"] = train_folds
            payload["val_folds"] = val_folds
            payload["test_folds"] = test_folds
        outpath = outdir / f"{base_config_path.stem}_fold{fold}.yaml"
        with open(outpath, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        print(outpath)


if __name__ == "__main__":
    main()
