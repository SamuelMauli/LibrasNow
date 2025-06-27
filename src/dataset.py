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
        self._load_data()

    def _load_data(self):
        """Carrega o arquivo .pkl com os dados processados."""
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
        except FileNotFoundError:
            print(
                f"ERRO: Arquivo de dados processados não encontrado em '{self.data_path}'."
            )
            print("Execute o script de processamento de dados primeiro.")
            raise

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Cria e retorna os DataLoaders de treino e validação.

        # Fundamento Acadêmico: Divisão Estratificada (Stratified Split)
        # O que faz: Ao dividir os dados, a opção `stratify=y` garante que a
        # proporção de cada classe (cada sinal de Libras) seja a mesma tanto no
        # conjunto de treino quanto no de validação. Isso é crucial para evitar
        # viés na avaliação do modelo, especialmente se o dataset for desbalanceado,
        # e assegura que a métrica de acurácia de validação seja confiável.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            self.sequences,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels,
        )

        train_dataset = SignLanguageDataset(X_train, y_train)
        val_dataset = SignLanguageDataset(X_val, y_val)

        # Otimização: num_workers e pin_memory aceleram o carregamento dos dados
        # para a GPU, mantendo-a ocupada e otimizando o tempo de treinamento.
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
