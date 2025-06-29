# tests/test_pruning.py
import os

import pytest
import torch
import torch.nn.utils.prune as prune

# Importe sua classe de modelo
from src.model import MyModel  # Substitua MyModel pelo nome da sua classe de modelo


def test_pruning_functionality():
    """
    Testa se a poda de modelo pode ser aplicada e o modelo salvo.
    """
    # Crie um modelo de exemplo para o teste
    num_features_in = 66
    hidden_size = 128
    num_classes = 29
    model = MyModel(num_features_in, hidden_size, num_classes)

    # Definir os parâmetros a serem podados
    parameters_to_prune = (
        (model.fc1, "weight"),
        (model.fc2, "weight"),
    )

    # Defina a quantidade de poda (ex: 20%)
    pruning_amount = 0.2

    # Salve uma cópia do modelo antes da poda para comparação
    original_model_path = "temp_original_model.pth"
    torch.save(model.state_dict(), original_model_path)

    # Aplique a poda
    try:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )
        print("\n✅ Poda aplicada com sucesso!")

    except Exception as e:
        pytest.fail(f"Erro ao aplicar a poda: {e}")

    # Verifique se a máscara de poda foi adicionada
    assert hasattr(model.fc1, "weight_mask"), "Máscara de poda não encontrada no fc1."

    # Salve o modelo podado
    pruned_model_path = "temp_pruned_model.pth"
    try:
        # Remova permanentemente os parâmetros podados para salvar o modelo comprimido
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        torch.save(model.state_dict(), pruned_model_path)

        # Verifique se o arquivo foi criado
        assert os.path.exists(
            pruned_model_path
        ), "O arquivo do modelo podado não foi salvo."

    except Exception as e:
        pytest.fail(f"Erro ao salvar o modelo podado: {e}")

    # Limpeza: remova os arquivos temporários
    os.remove(original_model_path)
    os.remove(pruned_model_path)

    print("\n✅ Teste de poda e salvamento de modelo concluído com sucesso!")
