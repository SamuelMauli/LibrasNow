import copy
import os
import pickle
import platform
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import QuantizedSignLanguageTransformer, SignLanguageTransformer
from pruning_utils import saliency_based_pruning

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from config import *


class SignLanguageDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def get_data_loaders(batch_size_override=None, force_cpu=False):
    try:
        with open(PROCESSED_DATA_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(
            f"Erro: Arquivo de dados processados não encontrado em {PROCESSED_DATA_FILE}"
        )
        return None, None, None, 0

    X, y, actions, num_classes = (
        data["sequences"],
        data["labels"],
        data["actions"],
        len(data["actions"]),
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)

    num_workers = 0 if platform.system() == "Windows" or force_cpu else 4
    batch_size = batch_size_override if batch_size_override else BATCH_SIZE

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not force_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not force_cpu,
    )

    return train_loader, val_loader, actions, num_classes


def validation_step(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc=f"Validating on {device}"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def measure_model_size(model, filepath="temp.pth"):
    torch.save(model.state_dict(), filepath, _use_new_zipfile_serialization=False)
    size = os.path.getsize(filepath) / 1e6
    os.remove(filepath)
    return size


def fine_tune_model(model, train_loader, val_loader, device, title, epochs=5, lr=1e-5):
    print(f"\n--- Starting {title} ---")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for sequences, labels in tqdm(
            train_loader, desc=f"{title} Epoch {epoch+1}/{epochs}"
        ):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        evaluate_model(
            model, val_loader, device, title=f"Validation after {title} Epoch {epoch+1}"
        )
    return model


def evaluate_model(model, val_loader, device, title="Model Evaluation"):
    print(f"\n--- {title} ---")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validation_step(model, val_loader, criterion, device)
    print(f"Validation Accuracy: {val_acc:.2%}, Validation Loss: {val_loss:.4f}")
    return val_acc


def main():
    train_loader, val_loader, _, num_classes = get_data_loaders(batch_size_override=32)
    if val_loader is None:
        return

    baseline_model_path = os.path.join(MODEL_SAVE_DIR, "baseline.pth")
    if not os.path.exists(baseline_model_path):
        print("Baseline model not found. Please train the model first.")
        return

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

    evaluate_model(model, val_loader, DEVICE, title="Evaluating Baseline Model")
    size_original = measure_model_size(model)
    print(f"Original model size: {size_original:.2f} MB")

    pruned_model = saliency_based_pruning(
        copy.deepcopy(model),
        PRUNING_AMOUNT,
        train_loader,
        nn.CrossEntropyLoss(),
        DEVICE,
    )
    evaluate_model(
        pruned_model,
        val_loader,
        DEVICE,
        title="Evaluating Pruned Model (Before Fine-Tuning)",
    )

    finetuned_model = fine_tune_model(
        pruned_model,
        train_loader,
        val_loader,
        DEVICE,
        "Fine-Tuning after Pruning",
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
    )
    size_pruned = measure_model_size(finetuned_model)
    print(f"Pruned and Fine-Tuned model size: {size_pruned:.2f} MB")

    print("\n--- Creating Final Quantized Model ---")
    finetuned_model.to("cpu").eval()
    quantized_model = QuantizedSignLanguageTransformer(finetuned_model)
    quantized_model.eval()

    size_quantized = measure_model_size(quantized_model, "temp_quant.pth")
    print(f"Pruned, Fine-Tuned, and Quantized model size: {size_quantized:.2f} MB")

    _, val_loader_cpu, _, _ = get_data_loaders(force_cpu=True, batch_size_override=32)
    evaluate_model(
        quantized_model,
        val_loader_cpu,
        "cpu",
        title="Evaluating Final Quantized Model on CPU",
    )

    compressed_model_path = os.path.join(
        MODEL_SAVE_DIR, "compressed_model_quantized.pth"
    )
    torch.save(quantized_model.state_dict(), compressed_model_path)
    print(f"\nFinal compressed model saved to {compressed_model_path}")


if __name__ == "__main__":
    main()
