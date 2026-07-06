import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from thesis.config import (
    DATA_DIR,
    IMG_SIZE,
    METADATA_CSV,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_data() -> pd.DataFrame:
    return pd.read_csv(METADATA_CSV).reset_index(drop=True)


def split_data(df, val_frac=0.15, test_frac=0.15, random_state=42):
    """Split by crater_id so augmented variants never span train/val/test."""
    groups = df["crater_id"].to_numpy()

    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=val_frac + test_frac, random_state=random_state
    )
    train_idx, valtest_idx = next(gss1.split(df, groups=groups))
    train = df.iloc[train_idx]
    val_test = df.iloc[valtest_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_frac / (val_frac + test_frac),
        random_state=random_state,
    )
    val_idx, test_idx = next(
        gss2.split(val_test, groups=val_test["crater_id"].to_numpy())
    )
    val = val_test.iloc[val_idx]
    test = val_test.iloc[test_idx]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _load_image(rel_path: str, size: int = IMG_SIZE) -> torch.Tensor:
    """Load grayscale PNG, resize, expand to 3 channels, ImageNet-normalize."""
    rel = rel_path.removeprefix("data/")
    img = Image.open(DATA_DIR / rel).convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return (tensor.expand(3, -1, -1).contiguous() - IMAGENET_MEAN) / IMAGENET_STD


class CraterImageDataset(Dataset):
    """Unlabeled crater images."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = _load_image(self.df.iloc[idx]["image_path"])
        if self.transform is not None:
            img = self.transform(img)
        return img
