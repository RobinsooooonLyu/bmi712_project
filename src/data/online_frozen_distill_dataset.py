from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.binary_wsi_dataset import TCGALUADBinaryPatientBagDataset


@dataclass
class FFPELatentTarget:
    label: int
    embedding: torch.Tensor


def load_ffpe_latent_targets(path: str | Path) -> dict[str, FFPELatentTarget]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"case_ids", "labels", "patient_embeddings"}
    missing = required_keys.difference(payload)
    if missing:
        raise KeyError(f"Missing keys in FFPE latent export {path}: {sorted(missing)}")

    targets: dict[str, FFPELatentTarget] = {}
    for case_id, label, embedding in zip(
        payload["case_ids"],
        payload["labels"].tolist(),
        payload["patient_embeddings"],
    ):
        targets[str(case_id)] = FFPELatentTarget(
            label=int(label),
            embedding=embedding.detach().clone().float(),
        )
    return targets


def collate_online_frozen_distill(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_ids": [item["case_id"] for item in batch],
        "bags": [item["slide_tiles"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "ffpe_embeddings": torch.stack([item["ffpe_embedding"] for item in batch], dim=0),
        "pathologic_stage_group": [item["pathologic_stage_group"] for item in batch],
        "clinical_stage_hybrid": [item["clinical_stage_hybrid"] for item in batch],
    }


class OnlineFrozenDistillDataset(Dataset):
    """Online frozen WSI bags paired with fixed fold-specific FFPE teacher latents."""

    def __init__(
        self,
        frozen_dataset: TCGALUADBinaryPatientBagDataset,
        ffpe_latent_path: str | Path,
        max_cases: int | None = None,
    ):
        self.frozen_dataset = frozen_dataset
        self.targets = load_ffpe_latent_targets(ffpe_latent_path)

        missing_case_ids: list[str] = []
        label_mismatches: list[str] = []
        for sample in self.frozen_dataset.samples:
            target = self.targets.get(sample.case_id)
            if target is None:
                missing_case_ids.append(sample.case_id)
                continue
            if int(sample.label) != target.label:
                label_mismatches.append(
                    f"{sample.case_id}: frozen={int(sample.label)} ffpe={target.label}"
                )

        if missing_case_ids:
            raise ValueError(
                f"{len(missing_case_ids)} frozen case_ids missing from FFPE latent export "
                f"{ffpe_latent_path}: {missing_case_ids[:5]}"
            )
        if label_mismatches:
            raise ValueError(
                f"{len(label_mismatches)} label mismatches between frozen split and FFPE latents: "
                f"{label_mismatches[:5]}"
            )

        if max_cases is not None:
            self.frozen_dataset.samples = self.frozen_dataset.samples[:max_cases]

    @property
    def samples(self):
        return self.frozen_dataset.samples

    def __len__(self) -> int:
        return len(self.frozen_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.frozen_dataset[index]
        target = self.targets[item["case_id"]]
        item["ffpe_embedding"] = target.embedding
        return item
