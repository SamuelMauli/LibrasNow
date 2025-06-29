import collections
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, PROCESSED_DATA_FILE


class SignLanguageDataset(Dataset):
    """
    Classe Dataset do PyTorch para carregar as sequências de keypoints e seus rótulos.
    Esta classe implementa a interface que o DataLoader do PyTorch usa para
    iterar sobre os dados em batches durante o treinamento.
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class DataManager:
    """
    Gerencia o carregamento dos dados processados, a divisão em conjuntos de
    treino/validação e a criação dos DataLoaders.
    """

    def __init__(self, data_path: str = PROCESSED_DATA_FILE):
        self.data_path = data_path
        self._load_and_diagnose_data()

    def _load_and_diagnose_data(self):
        """Carrega e diagnostica o arquivo .pkl com os dados processados."""
        try:
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
            self.sequences: np.ndarray = data["sequences"]
            self.labels: np.ndarray = data["labels"]
            self.actions: List[str] = data["actions"]
            self.num_classes: int = len(self.actions)
            print(
                f"Dados carregados com sucesso. Encontradas {len(self.sequences)} sequências."
            )

            # --- Bloco de Diagnóstico ---
            print("\n--- DIAGNÓSTICO DA DISTRIBUIÇÃO DE CLASSES ---")
            counts = collections.Counter(self.labels)
            is_valid = True
            for label_idx, count in sorted(counts.items()):
                action_name = self.actions[label_idx]
                if count < 2:
                    print(
                        f"!!! PROBLEMA: A classe '{action_name}' (label {label_idx}) tem apenas {count} amostra(s)."
                    )
                    is_valid = False
                else:
                    print(
                        f"    OK: Classe '{action_name}' (label {label_idx}) tem {count} amostras."
                    )

            if not is_valid:
                print("\nERRO DE DADOS: Pelo menos uma classe tem menos de 2 amostras.")
                print(
                    "Para corrigir, adicione mais dados às pastas problemáticas em 'data/raw_data/' ou remova-as."
                )
                raise ValueError(
                    "Dataset inválido para divisão estratificada. Verifique o diagnóstico acima."
                )
            print("--- Diagnóstico Concluído: O dataset é válido para divisão. ---\n")

        except FileNotFoundError:
            print(
                f"ERRO: Arquivo de dados processados não encontrado em '{self.data_path}'."
            )
            print("Execute o script de processamento de dados primeiro.")
            raise

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        X_train, X_val, y_train, y_val = train_test_split(
            self.sequences,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels,
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

        return train_loader, val_loader
