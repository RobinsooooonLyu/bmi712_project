from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import openslide
    OPENSLIDE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    openslide = None
    OPENSLIDE_IMPORT_ERROR = exc


@dataclass
class BinaryPatientSample:
    case_id: str
    label: int
    slide_paths: list[Path]
    coord_paths: list[Path]
    pathologic_stage_group: str
    clinical_stage_hybrid: str


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def collate_binary_patient_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_ids": [item["case_id"] for item in batch],
        "bags": [item["slide_tiles"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "pathologic_stage_group": [item["pathologic_stage_group"] for item in batch],
        "clinical_stage_hybrid": [item["clinical_stage_hybrid"] for item in batch],
    }


class TCGALUADBinaryPatientBagDataset(Dataset):
    def __init__(
        self,
        patient_manifest_path: str | Path,
        split_path: str | Path,
        wsi_root: str | Path,
        coords_root: str | Path,
        modality: str,
        folds: list[int],
        is_training: bool,
        augmentor: Callable[[Image.Image], Image.Image],
        preprocess_transform: Callable[[Image.Image], torch.Tensor],
        tiles_per_slide: int = 1024,
        max_slides_per_patient: int = 1,
        slide_selection: str = "largest",
        seed: int = 20260403,
    ):
        if openslide is None:  # pragma: no cover
            raise RuntimeError(
                f"openslide-python is required to load SVS tiles. Import error: {OPENSLIDE_IMPORT_ERROR}"
            )
        if modality not in {"ffpe", "frozen"}:
            raise ValueError(f"Unsupported modality: {modality}")
        if slide_selection not in {"largest", "manifest", "random"}:
            raise ValueError(f"Unsupported slide_selection: {slide_selection}")

        self.wsi_root = Path(wsi_root)
        self.coords_root = Path(coords_root)
        self.modality = modality
        self.is_training = is_training
        self.augmentor = augmentor
        self.preprocess_transform = preprocess_transform
        self.tiles_per_slide = tiles_per_slide
        self.max_slides_per_patient = max_slides_per_patient
        self.slide_selection = slide_selection
        self.seed = seed
        self.random = random.Random(seed)

        manifest_rows = {row["case_id"]: row for row in read_tsv(patient_manifest_path)}
        split_rows = [row for row in read_tsv(split_path) if int(row["fold"]) in set(folds)]
        slide_key = "ffpe_filenames" if modality == "ffpe" else "frozen_filenames"

        self.samples: list[BinaryPatientSample] = []
        for row in split_rows:
            case_id = row["case_id"]
            manifest = manifest_rows[case_id]
            slide_names = [name for name in manifest[slide_key].split("|") if name]
            slide_paths = [self.wsi_root / name for name in slide_names]
            coord_paths = [self.coords_root / f"{name}.json" for name in slide_names]
            self.samples.append(
                BinaryPatientSample(
                    case_id=case_id,
                    label=int(row["label"]),
                    slide_paths=slide_paths,
                    coord_paths=coord_paths,
                    pathologic_stage_group=row.get("pathologic_stage_group", ""),
                    clinical_stage_hybrid=row.get("clinical_stage_hybrid", ""),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        slide_pairs = list(zip(sample.slide_paths, sample.coord_paths))
        slide_pairs = self._select_slides(slide_pairs)

        slide_tiles = []
        for slide_path, coord_path in slide_pairs:
            payload = self._load_coords(coord_path)
            selected = self._select_coords(payload["coordinates"], coord_path)
            read_level = int(payload.get("selected_level", 0))
            read_size = int(payload.get("tile_size_level", payload.get("tile_size", 224)))
            slide_tiles.append(self._read_tiles(slide_path, selected, read_level, read_size))

        return {
            "case_id": sample.case_id,
            "slide_tiles": slide_tiles,
            "label": sample.label,
            "pathologic_stage_group": sample.pathologic_stage_group,
            "clinical_stage_hybrid": sample.clinical_stage_hybrid,
        }

    def _select_slides(self, slide_pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
        if self.slide_selection == "random" and self.is_training:
            self.random.shuffle(slide_pairs)
            return slide_pairs[: self.max_slides_per_patient]
        if self.slide_selection == "largest":
            slide_pairs = sorted(slide_pairs, key=lambda pair: self._num_tiles(pair[1]), reverse=True)
            return slide_pairs[: self.max_slides_per_patient]
        return slide_pairs[: self.max_slides_per_patient]

    def _num_tiles(self, coord_path: Path) -> int:
        try:
            payload = self._load_coords(coord_path)
        except FileNotFoundError:
            return 0
        return int(payload.get("num_tiles", len(payload.get("coordinates", []))))

    def _load_coords(self, path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _select_coords(self, coords: list[dict[str, int]], coord_path: Path) -> list[dict[str, int]]:
        if len(coords) <= self.tiles_per_slide:
            return coords
        if self.is_training:
            return self.random.sample(coords, self.tiles_per_slide)

        digest = hashlib.sha1(f"{self.seed}:{coord_path.name}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        indices = sorted(rng.sample(range(len(coords)), self.tiles_per_slide))
        return [coords[index] for index in indices]

    def _read_tiles(
        self,
        slide_path: Path,
        coords: list[dict[str, int]],
        read_level: int,
        read_size: int,
    ) -> torch.Tensor:
        slide = openslide.OpenSlide(str(slide_path))
        tile_tensors = []
        try:
            for coord in coords:
                x = int(coord["x"])
                y = int(coord["y"])
                tile = slide.read_region((x, y), read_level, (read_size, read_size)).convert("RGB")
                tile = self.augmentor(tile)
                tile_tensors.append(self.preprocess_transform(tile))
        finally:
            slide.close()
        return torch.stack(tile_tensors, dim=0)
