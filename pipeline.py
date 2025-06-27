import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import torch

import wandb

# Importações dos componentes refatorados
from src.config import *
from src.data_processor import DataProcessor
from src.dataset import DataManager
from src.engine import TrainingEngine
from src.model import build_quantized_model, create_model
from src.pruning_strategies import (
    GlobalMagnitudePruning,
    L1StructuredPruning,
    L2StructuredPruning,
    LotteryTicketPruning,
    PruningStrategy,
    RandomStructuredPruning,
    RandomUnstructuredPruning,
)
from src.ui.dashboard_app import DashboardApp
from src.ui.inference_app import InferenceApp


def handle_process_data():
    """Lida com a ação de processamento de dados."""
    print("--- Iniciando Etapa: Processamento de Dados ---")
    processor = DataProcessor()
    processor.process()
    print("--- Etapa Concluída: Processamento de Dados ---")


def handle_train_baseline():
    """Lida com a ação de treinamento do modelo base."""
    print("--- Iniciando Etapa: Treinamento do Modelo Base ---")
    data_manager = DataManager()
    train_loader, val_loader = data_manager.get_data_loaders()

    wandb.init(project=WANDB_PROJECT, name="baseline-training")

    model = create_model(num_classes=data_manager.num_classes)

    # Salva o estado inicial para o Lottery Ticket
    torch.save(model.state_dict(), MODEL_SAVE_DIR / "initial_weights.pth")

    engine = TrainingEngine(model, train_loader, val_loader, DEVICE)
    engine.train(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path=MODEL_SAVE_DIR / "baseline.pth",
        patience=EARLY_STOPPING_PATIENCE,
    )
    wandb.finish()
    print("--- Etapa Concluída: Treinamento do Modelo Base ---")


def measure_model_size(model: torch.nn.Module) -> float:
    """Mede o tamanho do modelo em MB."""
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = Path("temp_model.pth").stat().st_size / 1e6
    Path("temp_model.pth").unlink()
    return size_mb


def handle_run_experiments():
    """Lida com a ação de executar a bancada de testes de pruning."""
    print("--- Iniciando Etapa: Bancada de Testes de Pruning ---")

    # Definição dos experimentos
    strategies_to_test = [
        GlobalMagnitudePruning,
        L1StructuredPruning,
        L2StructuredPruning,
        RandomUnstructuredPruning,
        RandomStructuredPruning,
        LotteryTicketPruning,
    ]
    pruning_levels = [0.2, 0.4, 0.6, 0.8]
    finetune_epochs = 20
    finetune_lr = 1e-5

    # Carrega os dados e o modelo base
    data_manager = DataManager()
    train_loader, val_loader = data_manager.get_data_loaders()
    baseline_model = create_model(data_manager.num_classes)
    baseline_model.load_state_dict(torch.load(MODEL_SAVE_DIR / "baseline.pth"))

    initial_weights = torch.load(MODEL_SAVE_DIR / "initial_weights.pth")

    all_results: Dict[str, Dict[str, Any]] = {}
    RESULTS_DIR.mkdir(exist_ok=True)

    for strategy_class in strategies_to_test:
        for amount in pruning_levels:
            strategy = strategy_class(amount=amount)
            experiment_name = f"{strategy}_{int(amount*100)}p"
            print(f"\n>>> Executando Experimento: {experiment_name} <<<\n")

            wandb.init(project=WANDB_PROJECT, name=experiment_name, reinit=True)

            # 1. Aplicar Pruning
            model_to_prune = copy.deepcopy(baseline_model)
            pruned_model = strategy.apply(
                model_to_prune, initial_state_dict=initial_weights
            )

            # 2. Fine-tuning do modelo podado
            engine = TrainingEngine(pruned_model, train_loader, val_loader, DEVICE)
            save_path = MODEL_SAVE_DIR / f"model_{experiment_name}.pth"
            engine.train(
                epochs=finetune_epochs, lr=finetune_lr, save_path=save_path, patience=5
            )

            # 3. Avaliação e Coleta de Resultados
            final_accuracy = engine.evaluate()
            final_size_mb = measure_model_size(engine.model)

            all_results[experiment_name] = {
                "accuracy": final_accuracy,
                "size_mb": final_size_mb,
                "strategy": str(strategy),
                "amount": amount,
            }
            wandb.finish()

    # Salvar resultados consolidados
    results_path = RESULTS_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(
        f"\n--- Bancada de Testes Concluída. Resultados salvos em '{results_path}' ---"
    )


def handle_inference():
    """Lida com a ação de iniciar a aplicação de inferência."""
    print("--- Iniciando Aplicação de Inferência ---")
    app = InferenceApp()
    app.run()


def handle_dashboard():
    """Lida com a ação de iniciar o dashboard de resultados."""
    print("--- Iniciando Dashboard de Resultados ---")
    print("Execute o seguinte comando em um novo terminal:")
    print(f"streamlit run {BASE_DIR / 'app_dashboard.py'}")


def main():
    """Ponto de entrada principal do projeto."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Treinamento e Análise para o TCC de Libras."
    )
    parser.add_argument(
        "action",
        choices=[
            "process-data",
            "train-baseline",
            "run-experiments",
            "inference",
            "dashboard",
        ],
        help="A ação a ser executada pelo pipeline.",
    )
    args = parser.parse_args()

    action_handlers = {
        "process-data": handle_process_data,
        "train-baseline": handle_train_baseline,
        "run-experiments": handle_run_experiments,
        "inference": handle_inference,
        "dashboard": handle_dashboard,
    }

    handler = action_handlers.get(args.action)
    if handler:
        handler()
    else:
        print(f"Ação '{args.action}' desconhecida.")


if __name__ == "__main__":
    main()
