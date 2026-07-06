import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from thesis.config import (
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    CHECKPOINTS_DIR,
    DEVICE,
    EARLY_STOP_MIN_DELTA_REL,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from thesis.dataset import CraterImageDataset, load_data, split_data
from thesis.pretrain.mae import load_mae

LATEST_CKPT = CHECKPOINTS_DIR / "latest.pt"
BEST_CKPT = CHECKPOINTS_DIR / "mae_best.pt"


def main():
    model, _ = load_mae()
    model.to(DEVICE)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, _ = split_data(df)

    train_loader = DataLoader(
        CraterImageDataset(train_df),
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        CraterImageDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
        num_training_steps=EPOCHS * steps_per_epoch,
    )

    # resume state (defaults = fresh run)
    start_epoch = 0
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    fresh = os.environ.get("FRESH_START") == "1"
    if LATEST_CKPT.exists() and not fresh:
        ck = torch.load(LATEST_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"]
        best_val = ck["best_val"]
        best_epoch = ck["best_epoch"]
        epochs_no_improve = ck["epochs_no_improve"]
        if ck.get("epochs_total") != EPOCHS:
            print(
                f"WARNING: checkpoint was built with EPOCHS={ck.get('epochs_total')} "
                f"but config now says {EPOCHS}. The LR schedule will be inconsistent. "
                f"Set FRESH_START=1 to start over, or restore EPOCHS."
            )
        if ck.get("steps_per_epoch") not in (None, steps_per_epoch):
            print(
                f"WARNING: checkpoint steps_per_epoch={ck.get('steps_per_epoch')} "
                f"but current is {steps_per_epoch} (BATCH_SIZE or dataset size changed). "
                f"The cosine LR schedule will be corrupted. Set FRESH_START=1 "
                f"to start over, or restore the original BATCH_SIZE / dataset."
            )
        print(
            f"Resumed from {LATEST_CKPT.name}: continuing at epoch {start_epoch + 1}/"
            f"{EPOCHS} | best val {best_val:.5f} @ ep{best_epoch} | "
            f"patience {epochs_no_improve}/{EARLY_STOP_PATIENCE}"
        )
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Already early-stopped in a prior run. Nothing to do.")
            return
        if start_epoch >= EPOCHS:
            print("Epoch ceiling already reached in a prior run. Nothing to do.")
            return
    else:
        kind = "forced fresh" if fresh else "fresh"
        print(f"{kind} run from vit-mae-base. Ceiling {EPOCHS} epochs "
              f"| LR {LR} | warmup {WARMUP_EPOCHS}.")

    def save_latest(epochs_done):
        """Write full resume state atomically (temp then rename)."""
        payload = {
            "epoch": epochs_done,
            "epochs_total": EPOCHS,
            "steps_per_epoch": steps_per_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
        }
        tmp = LATEST_CKPT.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(LATEST_CKPT)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE, non_blocking=True)
            outputs = model(pixel_values=batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches
        print(f"epoch {epoch + 1:03d}/{EPOCHS} | loss {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  val {epoch + 1}"):
                batch = batch.to(DEVICE, non_blocking=True)
                outputs = model(pixel_values=batch)
                val_loss += outputs.loss.item()
                n_val += 1
        avg_val_loss = val_loss / n_val
        print(f"epoch {epoch + 1:03d}/{EPOCHS} | val loss {avg_val_loss:.4f}")

        # "improved" = beats best by more than the relative margin (ignores noise)
        improved = avg_val_loss < best_val * (1.0 - EARLY_STOP_MIN_DELTA_REL)
        if improved:
            best_val = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state": model.state_dict(),
                    "avg_loss": avg_loss,
                    "val_loss": avg_val_loss,
                },
                BEST_CKPT,
            )
            print(f"  new best val {best_val:.5f} @ epoch {best_epoch} -> mae_best.pt")
        else:
            epochs_no_improve += 1
            print(
                f"  no improve ({epochs_no_improve}/{EARLY_STOP_PATIENCE}); "
                f"best {best_val:.5f} @ epoch {best_epoch}"
            )

        # only arm early stopping after warmup (LR ramps from ~0 during it)
        warmup_done = (epoch + 1) > WARMUP_EPOCHS
        is_final = (epoch + 1) == EPOCHS
        stop_now = warmup_done and epochs_no_improve >= EARLY_STOP_PATIENCE

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or is_final or stop_now:
            save_latest(epoch + 1)
            # permanent per-epoch snapshot for the probe-vs-epoch curve
            snap = CHECKPOINTS_DIR / f"mae_epoch_{epoch + 1:03d}.pt"
            torch.save(
                {"epoch": epoch + 1, "model_state": model.state_dict(),
                 "avg_loss": avg_loss, "val_loss": avg_val_loss},
                snap,
            )
            print(f"  saved {snap.name} + latest.pt")

        if stop_now:
            print(
                f"\nEarly stop: no val improvement >"
                f"{EARLY_STOP_MIN_DELTA_REL * 100:.2f}% for {EARLY_STOP_PATIENCE} "
                f"epochs. Best val {best_val:.5f} @ epoch {best_epoch}. "
                f"Stopped at epoch {epoch + 1}/{EPOCHS}."
            )
            break
    else:
        print(f"\nReached epoch ceiling {EPOCHS}.")

    print(f"\nDone. Best val loss {best_val:.5f} @ epoch {best_epoch} (mae_best.pt).")


if __name__ == "__main__":
    main()
