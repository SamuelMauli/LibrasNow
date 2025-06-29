# tests/test_model.py
import pytest
import torch

# Importe sua classe de modelo, presumindo que está em src.model
from src.model import MyModel  # Substitua MyModel pelo nome da sua classe de modelo


def test_model_loading_and_inference():
    """
    Testa se o modelo pré-treinado pode ser carregado com sucesso e se
    ele consegue fazer uma inferência em um tensor de entrada.
    """
    model_path = "C:/TCC/LibrasNow/models/baseline.pth"

    # Carregue o modelo
    try:
        # Instancie o modelo com a mesma arquitetura que você usou para treinar
        # Substitua 'num_features_in', 'hidden_size', e 'num_classes' pelos valores corretos.
        # Exemplo:
        num_features_in = 66  # Ou qualquer número que corresponda aos seus dados
        hidden_size = 128
        num_classes = 29
        model = MyModel(num_features_in, hidden_size, num_classes)

        # Carregue os pesos do modelo
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()  # Coloque o modelo em modo de avaliação

    except FileNotFoundError:
        pytest.fail(
            f"Arquivo do modelo não encontrado: {model_path}. Por favor, verifique o caminho."
        )
    except Exception as e:
        pytest.fail(f"Erro ao carregar o modelo: {e}")

    # Crie um tensor de entrada de teste (dummy)
    # Assuma o formato (batch_size, sequence_length, num_features)
    # Ajuste o shape conforme o seu modelo espera
    dummy_input = torch.randn(1, 100, num_features_in)

    # Execute uma inferência
    with torch.no_grad():
        try:
            output = model(dummy_input)

            # Verifique o shape da saída
            # A saída deve ser (batch_size, num_classes)
            assert output.shape == (1, num_classes)

        except Exception as e:
            pytest.fail(f"Erro durante a inferência do modelo: {e}")

    print("\n✅ Teste de carregamento e inferência do modelo concluído com sucesso!")
