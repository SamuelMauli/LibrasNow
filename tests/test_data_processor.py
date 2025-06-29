# tests/test_data_processor.py
import pytest
import torch
from torch.utils.data import DataLoader

# Importe a classe do seu dataset, presumindo que ela está em src.data_processor
# Ajuste o caminho de importação conforme a estrutura do seu projeto.
from src.data_processor import SignDataset


def test_data_loading_and_iteration():
    """
    Testa se o dataset é carregado com sucesso e se o DataLoader pode iterar
    sobre ele sem levantar erros.
    """
    # Certifique-se de que o caminho para o seu arquivo de dados CSV está correto.
    # Exemplo: 'C:/TCC/LibrasNow/data/your_dataset.csv'
    # Ajuste o caminho abaixo para corresponder à sua estrutura de pastas.
    try:
        # Substitua 'caminho/para/seu/dataset.csv' pelo caminho real do seu arquivo.
        dataset = SignDataset(csv_file="data/test_data.csv")
    except FileNotFoundError:
        # Se o arquivo não for encontrado, crie um dummy para o teste.
        # Isso é uma boa prática para testes unitários, pois evita dependências externas.
        # Para este teste de integração, vamos supor que o arquivo existe.
        pytest.fail("Dataset CSV não encontrado. Por favor, verifique o caminho.")

    # Crie um DataLoader para testar a iteração
    # Usamos um batch_size pequeno e num_workers=0 para simplicidade.
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Verifique se o dataset não está vazio
    assert len(dataset) > 0, "O dataset não deve estar vazio."

    # Itere sobre alguns batches para garantir que não há erros
    # e que os dados têm o formato esperado.
    num_batches_to_test = 5
    for i, (features, labels) in enumerate(data_loader):
        # features deve ser um tensor de Torch e ter o shape correto
        assert isinstance(features, torch.Tensor)
        assert features.ndim == 3  # (batch_size, num_frames, num_features)

        # labels deve ser um tensor de Torch e ter o shape correto
        assert isinstance(labels, torch.Tensor)
        assert labels.ndim == 1  # (batch_size)

        if i >= num_batches_to_test:
            break

    print("\n✅ Teste de carregamento e iteração de dados concluído com sucesso!")
