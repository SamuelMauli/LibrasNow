import copy
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.optimizer.sam import SAM


class TrainingEngine:
    """
    Encapsula toda a lógica de treinamento, validação e fine-tuning.
    Gerencia o loop de épocas, otimização, logging, early stopping e
    salvamento do melhor modelo.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _train_one_epoch(self, optimizer) -> Tuple[float, float]:
        """Executa uma única época de treinamento."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for sequences, labels in tqdm(self.train_loader, desc="Training Epoch"):
            sequences, labels = sequences.to(self.device), labels.to(self.device)

            # --- Lógica do Otimizador SAM ---
            # # Fundamento Acadêmico: Sharpness-Aware Minimization (SAM)
            # # Referência: Foret et al., "Sharpness-Aware Minimization for
            # # Efficiently Improving Generalization" (2020).
            # # O que faz: SAM busca encontrar parâmetros que não apenas tenham baixa
            # # perda (loss), mas que estejam em uma vizinhança "plana" do espaço de
            # # perda. Isso melhora a generalização do modelo, tornando-o mais
            # # robusto a pequenas perturbações nos dados de teste.

            # 1. Primeiro passo: calcula a perda e sobe o "morro"
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 2. Segundo passo: calcula a perda no ponto perturbado e otimiza
            self.criterion(self.model(sequences), labels).backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        return total_loss / total_samples, total_correct / total_samples

    def _validate_one_epoch(self) -> Tuple[float, float]:
        """Executa uma única época de validação."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for sequences, labels in tqdm(self.val_loader, desc="Validation Epoch"):
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        return total_loss / total_samples, total_correct / total_samples

    def _save_checkpoint(self, path: Path):
        """Salva o estado do modelo no disco."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Modelo salvo com sucesso em '{path}'")

    def train(self, epochs: int, lr: float, save_path: Path, patience: int):
        """
        Orquestra o processo completo de treinamento por N épocas.
        """
        print(
            f"Iniciando treinamento por {epochs} épocas no dispositivo: {self.device}"
        )
        base_optimizer = torch.optim.Adam
        optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr)

        for epoch in range(epochs):
            train_loss, train_acc = self._train_one_epoch(optimizer)
            val_loss, val_acc = self._validate_one_epoch()

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )

            # --- Lógica de Early Stopping e Salvamento do Modelo ---
            # # Fundamento Acadêmico: Early Stopping
            # # O que faz: Monitora a perda de validação e interrompe o treinamento
            # # se ela não melhorar por um número definido de épocas (`patience`).
            # # Isso previne o overfitting, que ocorre quando o modelo começa a
            # # memorizar os dados de treino em vez de aprender padrões generalizáveis.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(
                        f"Early stopping ativado na época {epoch+1}. Melhor perda de validação: {self.best_val_loss:.4f}"
                    )
                    break

        # Carrega o melhor modelo salvo ao final do treinamento
        print("Carregando o melhor modelo encontrado durante o treinamento...")
        self.model.load_state_dict(torch.load(save_path))

    def evaluate(self) -> float:
        """Apenas executa a validação e retorna a acurácia."""
        print("Avaliando o modelo no conjunto de validação...")
        val_loss, val_acc = self._validate_one_epoch()
        print(
            f"Resultado da Avaliação -> Perda: {val_loss:.4f}, Acurácia: {val_acc:.2%}"
        )
        return val_acc
