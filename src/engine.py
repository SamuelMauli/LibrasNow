import torch
import wandb
from tqdm import tqdm


def train_step(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for sequences, labels in tqdm(data_loader, desc="Training"):
        sequences, labels = sequences.to(device), labels.to(device)

        # SAM first step
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step
        criterion(model(sequences), labels).backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validation_step(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc="Validating"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
