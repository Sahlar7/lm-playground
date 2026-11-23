"""Generic training loop. Knows nothing about tokenizers or a specific architecture:
it only needs a model that maps inputs -> logits and datasets of (input, target) pairs.
Fine-tuning = load a checkpoint into the model, then call train() with different data."""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    batch_size:  int   = 64
    lr:          float = 1e-4
    n_epochs:    int   = 50
    num_workers: int   = 4
    out_dir:     str   = "runs/default"  # best.pt is saved here


def train(model, train_dataset, val_dataset, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn   = nn.CrossEntropyLoss()
    model.to(device)

    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,  num_workers=config.num_workers)
    val_loader    = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"

    best_val_loss = float('inf')

    for epoch in range(config.n_epochs):
        # --- training pass ---
        model.train()
        total_train_loss = 0

        for input_ids, target_ids in train_loader:
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss   = loss_fn(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- validation pass ---
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids  = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                loss   = loss_fn(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"epoch {epoch:3d}  train loss {avg_train_loss:.4f}  val loss {avg_val_loss:.4f}")

        # save best checkpoint based on val loss, not train loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved new best (val loss {avg_val_loss:.4f})")

    return checkpoint_path
