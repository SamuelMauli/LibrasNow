import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, PROCESSED_DATA_FILE


class SignLanguageDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def get_data_loaders():
    try:
        with open(PROCESSED_DATA_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_FILE}")
        print("Run data_processing.py first.")
        return None, None, None, 0

    X = data["sequences"]
    y = data["labels"]
    actions = data["actions"]
    num_classes = len(actions)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, actions, num_classes
