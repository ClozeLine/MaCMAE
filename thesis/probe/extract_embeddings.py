import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesis.config import (
    BATCH_SIZE,
    DEVICE,
    EMBEDDINGS_DIR,
)
from thesis.dataset import CraterImageDataset, load_data
from thesis.pretrain.mae import load_mae


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MAE checkpoint .pt. Omit for the epoch-0 baseline (raw vit-mae-base).",
    )
    parser.add_argument(
        "--all-augmented",
        action="store_true",
        help="include augmented variants (default: originals only, one per crater).",
    )
    return parser.parse_args()

def load_checkpoint(model, checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", "?")
    print(f"loaded {checkpoint_path.name} (epoch {epoch})")
    return model

def build_model(checkpoint_path):
    model, _ = load_mae()
    if checkpoint_path is None:
        print("epoch-0 baseline: raw vit-mae-base (no checkpoint loaded)")
    else:
        model = load_checkpoint(model, checkpoint_path)
    model.to(DEVICE)
    model.eval()
    return model

def build_loader(all_augmented=False):
    df = load_data()
    if not all_augmented and "orientation" in df.columns:
        df = df[df["orientation"] == "id"].reset_index(drop=True)
        print(f"probe set: {len(df)} originals only (orientation=='id')")
    else:
        print(f"probe set: {len(df)} rows (all augmented)")
    dataset = CraterImageDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return df, loader

@torch.no_grad()
def extract(model, loader):
    all_embeddings = []
    for batch in tqdm(loader, desc="extracting"):
        batch = batch.to(DEVICE, non_blocking=True)

        outputs = model.vit(
            pixel_values=batch,
            output_hidden_states=False,
            interpolate_pos_encoding=False
        )
        tokens = outputs.last_hidden_state
        patch_tokens = tokens[:, 1:, :]
        pooled = patch_tokens.mean(dim=1)

        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()

def save(df, embeddings, checkpoint_path) -> Path:
    df = df.copy()
    df["embeddings"] = list(embeddings)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    # no checkpoint -> mae_epoch_000 (the x=0 baseline in a probe-vs-epoch sweep)
    stem = "mae_epoch_000" if checkpoint_path is None else checkpoint_path.stem
    out_path = EMBEDDINGS_DIR / f"{stem}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"saved {out_path} ({len(df)} craters, dim={embeddings.shape[1]})")
    return out_path

def main():
    args = parse_args()
    model = build_model(args.checkpoint)
    df, loader = build_loader(all_augmented=args.all_augmented)
    embeddings = extract(model, loader)
    save(df, embeddings, args.checkpoint)

if __name__ == "__main__":
    main()