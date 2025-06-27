import copy
import math
from typing import Dict, Type

import torch
import torch.nn as nn

from src.config import DROPOUT, HIDDEN_UNITS, INPUT_SIZE, NUM_HEADS, NUM_LAYERS


class PositionalEncoding(nn.Module):
    """
    # Fundamento Acadêmico: Positional Encoding
    # Referência: Vaswani et al., "Attention Is All You Need" (2017).
    # O que faz: Injeta informações sobre a posição relativa ou absoluta dos tokens
    # na sequência. Como o mecanismo de auto-atenção do Transformer não processa a
    # ordem sequencial dos dados por si só, o Positional Encoding é essencial para
    # que o modelo entenda a ordem temporal dos keypoints da mão.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    """
    Modelo Transformer para classificação de sequências de keypoints de Libras.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # A projeção de entrada e a escala são passos padrão para preparar os dados
        # para o encoder do Transformer.
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        # # Fundamento Acadêmico: Global Average Pooling
        # # O que faz: Em vez de usar o output de um token específico (como o [CLS] no BERT),
        # # tiramos a média dos outputs de todos os tokens da sequência. Isso cria um vetor
        # # de características de tamanho fixo que representa a sequência inteira,
        # # tornando o modelo mais robusto a pequenas variações no comprimento da sequência.
        output = output.mean(dim=1)

        output = self.classifier(output)
        return output


def create_model(num_classes: int) -> SignLanguageTransformer:
    """
    Fábrica (Factory) para criar uma instância do modelo com as configurações do projeto.
    Isso desacopla a criação do modelo do script de treinamento.
    """
    return SignLanguageTransformer(
        num_classes=num_classes,
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )


def build_quantized_model(
    model_fp32: SignLanguageTransformer,
) -> SignLanguageTransformer:
    """
    # Fundamento Acadêmico: Quantização Dinâmica
    # Referência: Documentação do PyTorch e artigos sobre otimização de modelos.
    # O que faz: Converte os pesos das camadas lineares de ponto flutuante (32-bit)
    # para inteiros (8-bit), reduzindo drasticamente o tamanho do modelo e acelerando
    # a inferência na CPU. A quantização é "dinâmica" porque as ativações são
    # quantizadas em tempo real durante o forward pass.
    """
    model_int8 = copy.deepcopy(model_fp32)
    model_int8.eval()
    model_int8.to("cpu")

    # Aplica a quantização dinâmica apenas às camadas que se beneficiam dela (Lineares)
    # e que são seguras de quantizar sem perda significativa de acurácia.
    quantized_layers: Dict[str, Type[nn.Module]] = {
        "input_projection": nn.Linear,
        "classifier": nn.Linear,
    }

    for name, layer_type in quantized_layers.items():
        if hasattr(model_int8, name):
            layer_to_quantize = getattr(model_int8, name)
            quantized_layer = torch.quantization.quantize_dynamic(
                layer_to_quantize, {layer_type}, dtype=torch.qint8
            )
            setattr(model_int8, name, quantized_layer)

    return model_int8
