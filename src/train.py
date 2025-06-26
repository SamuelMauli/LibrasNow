import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from dataset import get_data_loaders
from engine import train_step, validation_step
from model import SignLanguageTransformer
from optimizer import SAM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from config import *


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "model_hidden_units": HIDDEN_UNITS,
        },
    )

    train_loader, val_loader, _, num_classes = get_data_loaders()
    if not train_loader:
        return

    model = SignLanguageTransformer(
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=LEARNING_RATE, rho=SAM_RHO)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.base_optimizer, "min", patience=3, factor=0.5
    )
    early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)

    best_val_accuracy = 0.0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_acc = train_step(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validation_step(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "baseline.pth"))
            print("Model saved as best.")

        scheduler.step(val_loss)
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered.")
            break

    wandb.finish()


if __name__ == "__main__":
    main()
