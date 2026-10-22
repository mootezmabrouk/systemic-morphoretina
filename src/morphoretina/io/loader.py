"""Image loading (EF-1), IDRiD labels and DIARETDB1 dataset reader.

Place at: src/morphoretina/io/loader.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SUPPORTED = {".png", ".jpg", ".jpeg", ".dcm"}

IDRID_DIR = Path("data/raw/idrid")

# Lesion code -> ground-truth folder name in DIARETDB1 v1.1
LESION_DIRS = {
    "MA": "redsmalldots",
    "HE": "hemorrhages",
    "EX": "hardexudates",
    "SE": "softexudates",
}

# Minimum mask value counted as lesion. Masks encode expert agreement
# (steps of 63 for HE/EX/SE, 21 for MA). PROVISIONAL values: confirm
# against doc/diaretdb1_techreport_v_1_1.pdf and move to configs/default.yaml.
DEFAULT_THRESHOLDS = {"MA": 63, "HE": 189, "EX": 189, "SE": 189}


class ImageLoadError(ValueError):
    """Raised for unreadable, corrupted or unsupported input files."""


def load_image(path: str | Path) -> np.ndarray:
    """Load a JPEG/PNG/DICOM fundus image as an RGB uint8 array (H, W, 3)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ImageLoadError(f"Unsupported format: {ext}")
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")

    if ext == ".dcm":
        try:
            import pydicom

            arr = pydicom.dcmread(str(path)).pixel_array
        except Exception as e:  # corrupted or non-image DICOM
            raise ImageLoadError(f"Cannot read DICOM {path.name}: {e}") from e
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    # np.fromfile + imdecode also handles non-ASCII Windows paths
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ImageLoadError(f"Cannot decode image (corrupted?): {path.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if m is None:
        raise ImageLoadError(f"Cannot decode mask: {path}")
    return m if m.ndim == 2 else m[..., 0]


def load_idrid_labels(
    csv_path: str | Path = IDRID_DIR / "idrid_labels.csv",
    images_dir: str | Path = IDRID_DIR / "Imagenes",
) -> pd.DataFrame:
    """IDRiD grading labels: id_code, grade (0-4), edema_risk, path, exists."""
    df = pd.read_csv(csv_path, usecols=[0, 1, 2])
    df.columns = ["id_code", "grade", "edema_risk"]
    df = df.dropna(subset=["id_code"]).reset_index(drop=True)
    df["grade"] = df["grade"].astype(int)
    df["path"] = df["id_code"].map(lambda i: Path(images_dir) / f"{i}.jpg")
    df["exists"] = df["path"].map(lambda p: p.is_file())
    return df


class Diaretdb1:
    """Reader for DIARETDB1 v1.1 (images, lesion masks, official split)."""

    def __init__(self, root: str | Path, thresholds: dict[str, int] | None = None):
        self.root = Path(root)
        res = self.root / "resources"
        self.img_dir = res / "images" / "ddb1_fundusimages"
        self.gt_dir = res / "images" / "ddb1_groundtruth"
        self.split_dir = res
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Not found: {self.img_dir}")

    def split(self, name: str) -> list[str]:
        """Image file names of the official 'train' (28) or 'test' (61) split."""
        sub, fname = {
            "train": ("traindatasets", "trainset.txt"),
            "test": ("testdatasets", "testset.txt"),
        }[name]
        lines = (self.split_dir / sub / fname).read_text().splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    def all_images(self) -> list[str]:
        return sorted(p.name for p in self.img_dir.glob("image*.png"))

    def load_image(self, name: str) -> np.ndarray:
        return load_image(self.img_dir / name)

    def load_raw_mask(self, name: str, lesion: str) -> np.ndarray:
        """Expert-agreement mask (uint8) for one lesion type."""
        return _read_gray(self.gt_dir / LESION_DIRS[lesion] / name)

    def load_masks(self, name: str) -> dict[str, np.ndarray]:
        """Binary bool masks per lesion type, using the agreement thresholds."""
        return {
            k: self.load_raw_mask(name, k) >= self.thresholds[k]
            for k in LESION_DIRS
        }