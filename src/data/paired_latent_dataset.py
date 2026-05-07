from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class PairedLatentSample:
    case_id: str
    label: int
    frozen_embedding: torch.Tensor
    ffpe_embedding: torch.Tensor


def collate_paired_latents(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_ids": [item["case_id"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "frozen_embeddings": torch.stack([item["frozen_embedding"] for item in batch], dim=0),
        "ffpe_embeddings": torch.stack([item["ffpe_embedding"] for item in batch], dim=0),
    }


def _load_latent_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"case_ids", "labels", "patient_embeddings"}
    missing = required_keys.difference(payload)
    if missing:
        raise KeyError(f"Missing keys in latent export {path}: {sorted(missing)}")
    return payload


class PairedLatentDataset(Dataset):
    def __init__(self, frozen_path: str | Path, ffpe_path: str | Path):
        frozen_payload = _load_latent_payload(frozen_path)
        ffpe_payload = _load_latent_payload(ffpe_path)

        ffpe_index = {
            case_id: (
                int(label),
                embedding.clone().float(),
            )
            for case_id, label, embedding in zip(
                ffpe_payload["case_ids"],
                ffpe_payload["labels"].tolist(),
                ffpe_payload["patient_embeddings"],
            )
        }

        self.samples: list[PairedLatentSample] = []
        missing_case_ids: list[str] = []
        for case_id, label, embedding in zip(
            frozen_payload["case_ids"],
            frozen_payload["labels"].tolist(),
            frozen_payload["patient_embeddings"],
        ):
            if case_id not in ffpe_index:
                missing_case_ids.append(case_id)
                continue
            ffpe_label, ffpe_embedding = ffpe_index[case_id]
            if int(label) != ffpe_label:
                raise ValueError(
                    f"Label mismatch for case_id={case_id}: frozen={int(label)} ffpe={ffpe_label}"
                )
            self.samples.append(
                PairedLatentSample(
                    case_id=case_id,
                    label=int(label),
                    frozen_embedding=embedding.clone().float(),
                    ffpe_embedding=ffpe_embedding,
                )
            )

        if missing_case_ids:
            raise ValueError(
                f"{len(missing_case_ids)} frozen case_ids missing from FFPE export: "
                f"{missing_case_ids[:5]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "case_id": sample.case_id,
            "label": sample.label,
            "frozen_embedding": sample.frozen_embedding,
            "ffpe_embedding": sample.ffpe_embedding,
        }
