import copy
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from config import *
from dataset import get_data_loaders
from engine import validation_step
from model import SignLanguageTransformer


def measure_model_size(model, filepath="temp.pth"):
    torch.save(model.state_dict(), filepath)
    size = os.path.getsize(filepath) / 1e6
    os.remove(filepath)
    return size


def evaluate_model(model, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validation_step(model, val_loader, criterion, device)
    print(f"Validation Accuracy: {val_acc:.2%}, Validation Loss: {val_loss:.4f}")
    return val_acc


if __name__ == "__main__":
    train_loader, val_loader, _, num_classes = get_data_loaders()
    if val_loader is None:
        exit()

    baseline_model_path = os.path.join(MODEL_SAVE_DIR, "baseline.pth")
    if not os.path.exists(baseline_model_path):
        print("Baseline model not found. Please train the model first.")
        exit()

    model = SignLanguageTransformer(
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(baseline_model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("--- Evaluating Baseline Model ---")
    size_original = measure_model_size(model)
    print(f"Original model size: {size_original:.2f} MB")
    evaluate_model(model, val_loader, DEVICE)

    print("\n--- Applying Structured Pruning ---")
    pruned_model = copy.deepcopy(model)
    pruned_model = apply_structured_pruning(pruned_model, amount=PRUNING_AMOUNT)
    size_pruned = measure_model_size(pruned_model)
    print(f"Pruned model size: {size_pruned:.2f} MB")
    evaluate_model(pruned_model, val_loader, DEVICE)

    print("\n--- Applying Post-Training Dynamic Quantization ---")
    pruned_model_cpu = pruned_model.to("cpu")
    quantized_model = torch.quantization.quantize_dynamic(
        pruned_model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    size_quantized = measure_model_size(quantized_model, "temp_quant.pth")
    print(f"Pruned and Quantized model size: {size_quantized:.2f} MB")

    # Evaluation for quantized model must be done on CPU
    print("Evaluating quantized model on CPU...")
    # NOTE: Dataloader num_workers might need to be 0 for CPU evaluation on some systems
    _, val_loader_cpu, _, _ = get_data_loaders(force_cpu=True)
    evaluate_model(quantized_model, val_loader_cpu, "cpu")

    compressed_model_path = os.path.join(
        MODEL_SAVE_DIR, "compressed_model_quantized.pth"
    )
    torch.save(quantized_model.state_dict(), compressed_model_path)
    print(f"\nCompressed model saved to {compressed_model_path}")


# Helper function in compress.py to apply pruning
def apply_structured_pruning(model, amount=0.4):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model


# Update dataset.py to handle CPU evaluation
# Add a `force_cpu` parameter to get_data_loaders
def get_data_loaders(force_cpu=False):
    # ... (existing code) ...
    num_workers = 0 if force_cpu else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not force_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not force_cpu,
    )
    # ... (return statement)
