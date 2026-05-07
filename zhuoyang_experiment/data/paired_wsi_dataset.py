from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from src.data.binary_wsi_dataset import TCGALUADBinaryPatientBagDataset


def collate_paired_patient_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_ids": [item["case_id"] for item in batch],
        "ffpe_bags": [item["ffpe_slide_tiles"] for item in batch],
        "frozen_bags": [item["frozen_slide_tiles"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
    }


class PairedFFPEFrozenPatientDataset(Dataset):
    """Loads an FFPE bag and a frozen bag for the same patient, index-aligned.

    Wraps two `TCGALUADBinaryPatientBagDataset` instances constructed from the
    same split and folds. FFPE and frozen are different physical sections of
    the same case, so pairing is at the case level, not tile level.
    """

    def __init__(
        self,
        patient_manifest_path: str | Path,
        split_path: str | Path,
        ffpe_wsi_root: str | Path,
        ffpe_coords_root: str | Path,
        frozen_wsi_root: str | Path,
        frozen_coords_root: str | Path,
        folds: list[int],
        is_training: bool,
        ffpe_augmentor: Callable,
        frozen_augmentor: Callable,
        preprocess_transform: Callable,
        ffpe_tiles_per_slide: int = 1024,
        frozen_tiles_per_slide: int = 1024,
        max_slides_per_patient: int = 1,
        slide_selection: str = "largest",
        seed: int = 20260403,
    ):
        self.ffpe = TCGALUADBinaryPatientBagDataset(
            patient_manifest_path=patient_manifest_path,
            split_path=split_path,
            wsi_root=ffpe_wsi_root,
            coords_root=ffpe_coords_root,
            modality="ffpe",
            folds=folds,
            is_training=is_training,
            augmentor=ffpe_augmentor,
            preprocess_transform=preprocess_transform,
            tiles_per_slide=ffpe_tiles_per_slide,
            max_slides_per_patient=max_slides_per_patient,
            slide_selection=slide_selection,
            seed=seed,
        )
        self.frozen = TCGALUADBinaryPatientBagDataset(
            patient_manifest_path=patient_manifest_path,
            split_path=split_path,
            wsi_root=frozen_wsi_root,
            coords_root=frozen_coords_root,
            modality="frozen",
            folds=folds,
            is_training=is_training,
            augmentor=frozen_augmentor,
            preprocess_transform=preprocess_transform,
            tiles_per_slide=frozen_tiles_per_slide,
            max_slides_per_patient=max_slides_per_patient,
            slide_selection=slide_selection,
            seed=seed,
        )

        if len(self.ffpe) != len(self.frozen):
            raise ValueError(
                f"Paired sub-datasets have different lengths: "
                f"ffpe={len(self.ffpe)} frozen={len(self.frozen)}"
            )
        for idx, (a, b) in enumerate(zip(self.ffpe.samples, self.frozen.samples)):
            if a.case_id != b.case_id:
                raise ValueError(
                    f"Case mismatch at index {idx}: ffpe={a.case_id} frozen={b.case_id}"
                )

    @property
    def samples(self):
        return self.frozen.samples

    def __len__(self) -> int:
        return len(self.frozen)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ffpe_item = self.ffpe[index]
        frozen_item = self.frozen[index]
        return {
            "case_id": frozen_item["case_id"],
            "label": frozen_item["label"],
            "ffpe_slide_tiles": ffpe_item["slide_tiles"],
            "frozen_slide_tiles": frozen_item["slide_tiles"],
        }
