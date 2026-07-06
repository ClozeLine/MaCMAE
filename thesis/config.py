import os
from pathlib import Path

import torch


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default

# encoder / training
HF_MODEL_ID = "facebook/vit-mae-base"
IMG_SIZE = 224

BATCH_SIZE = 64
LR = 1.5e-4
WEIGHT_DECAY = 0.05
EPOCHS = 200  # keep constant once a run starts; the LR schedule is sized to it
WARMUP_EPOCHS = 10

CHECKPOINT_EVERY = 10

EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA_REL = 0.0005

LABEL_MAP = {"A": 0, "B": 1, "C": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = _env_path("THESIS_DATA_DIR", PACKAGE_DIR / "data")
CHECKPOINTS_DIR = _env_path("THESIS_CHECKPOINTS_DIR", PACKAGE_DIR / "checkpoints")
EMBEDDINGS_DIR = _env_path("THESIS_EMBEDDINGS_DIR", PACKAGE_DIR / "embeddings")
RESULTS_DIR = _env_path("THESIS_RESULTS_DIR", PACKAGE_DIR / "results")

# Image dataset dir under DATA_DIR (published on Zenodo; see README).
CRATER_SUBDIR = os.environ.get("THESIS_CRATER_SUBDIR", "augmented_data")
CRATER_DATA = DATA_DIR / CRATER_SUBDIR
METADATA_CSV = CRATER_DATA / "metadata.csv"
