# run_pipeline.py

import os
import sys

# Garante que o diretório src/ seja visível para importação
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from compress import main as run_compression
from data_processor import main as run_preprocessing
from train import main as run_training

if __name__ == "__main__":
    print("\n--- Etapa 1: Pré-processamento dos dados ---")
    run_preprocessing()

    print("\n--- Etapa 2: Treinamento do modelo ---")
    run_training()

    print("\n--- Etapa 3: Compressão e quantização do modelo ---")
    run_compression()
