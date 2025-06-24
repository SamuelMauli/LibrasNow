import copy
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from config import *
from dataset import get_data_loaders
from model import SignLanguageTransformer


def measure_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size


def apply_structured_pruning(model, amount=0.4):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model


def apply_dynamic_quantization(model):
    model_quantized = torch.quantization.quantize_dynamic(
        model.to("cpu"), {nn.Linear}, dtype=torch.qint8
    )
    return model_quantized


if __name__ == "__main__":
    _, _, _, num_classes = get_data_loaders()

    baseline_model_path = os.path.join(MODEL_SAVE_DIR, "baseline.pth")
    if not os.path.exists(baseline_model_path):
        print("Baseline model not found. Please train the model first.")
        exit()

    # --- Load Baseline Model ---
    model = SignLanguageTransformer(
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(baseline_model_path))
    model.eval()

    size_original = measure_model_size(model)
    print(f"Original model size: {size_original:.2f} MB")

    # --- Pruning ---
    pruned_model = copy.deepcopy(model)
    pruned_model = apply_structured_pruning(pruned_model, amount=PRUNING_AMOUNT)
    size_pruned = measure_model_size(pruned_model)
    print(f"Pruned model size: {size_pruned:.2f} MB")

    # --- Quantization ---
    quantized_model = apply_dynamic_quantization(copy.deepcopy(pruned_model))
    size_quantized = measure_model_size(quantized_model)
    print(f"Pruned and Quantized model size: {size_quantized:.2f} MB")

    # --- Save Compressed Model ---
    compressed_model_path = os.path.join(
        MODEL_SAVE_DIR, "compressed_model_quantized.pth"
    )
    torch.save(quantized_model.state_dict(), compressed_model_path)
    print(f"Compressed model saved to {compressed_model_path}")
