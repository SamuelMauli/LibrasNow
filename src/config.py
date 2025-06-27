from pathlib import Path

import torch

# -----------------------------------------------------------------------------
# PRINCÍPIO DA RESPONSABILIDADE ÚNIC
# Este arquivo centraliza todas as configurações estáticas e constantes do
# projeto. O objetivo é fornecer um único "ponto da verdade" para
# hiperparâmetros de arquitetura, caminhos de diretório e valores fixos,
# facilitando a manutenção e a reprodutibilidade.
# -----------------------------------------------------------------------------

# --- Estrutura de Diretórios (Paths) ---
# Define a estrutura de diretórios do projeto de forma agnóstica ao sistema
# operacional, garantindo que o código seja portável.
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_SAVE_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "sequences_and_labels.pkl"


# --- Processamento e Definição dos Dados ---
# Parâmetros que definem a forma dos dados de entrada para a rede neural.
SEQUENCE_LENGTH = 30
INPUT_SIZE = 21 * 3  # 21 landmarks da MediaPipe, cada um com 3 coordenadas (x, y, z)
VIDEO_FILE_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --- Arquitetura do Modelo Transformer ---
# Hiperparâmetros que definem a arquitetura do modelo base. Estes valores
# são fixos para todos os experimentos para garantir uma comparação justa.
HIDDEN_UNITS = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.2


# --- Parâmetros de Treinamento Base ---
# Configurações para o treinamento inicial do modelo denso (baseline).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10


# --- Parâmetros de Inferência ---
# Limiar de confiança para que uma predição seja considerada válida e exibida
# na interface. Ajuda a filtrar previsões de baixa certeza.
INFERENCE_CONFIDENCE_THRESHOLD = 0.85


# --- Logging e Rastreabilidade (WandB) ---
# Configurações para o Weights & Biases, utilizado para rastrear os
# experimentos, métricas e artefatos gerados.
WANDB_PROJECT = "QPruner-Libras-TCC"
