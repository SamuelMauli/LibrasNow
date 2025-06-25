import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_classes,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(SignLanguageTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.classifier(output)
        return output


# --- NOVA CLASSE PARA QUANTIZAÇÃO SEGURA ---
class QuantizedSignLanguageTransformer(nn.Module):
    def __init__(self, float_model: SignLanguageTransformer):
        super(QuantizedSignLanguageTransformer, self).__init__()
        # Garante que o modelo a ser quantizado está em modo de avaliação e na CPU
        float_model.eval()
        float_model.to("cpu")

        # Quantiza as camadas que são seguras (Linear)
        self.input_projection = torch.quantization.quantize_dynamic(
            float_model.input_projection, {nn.Linear}, dtype=torch.qint8
        )
        self.classifier = torch.quantization.quantize_dynamic(
            float_model.classifier, {nn.Linear}, dtype=torch.qint8
        )

        # Mantém as camadas complexas como estavam (em float)
        self.pos_encoder = float_model.pos_encoder
        self.transformer_encoder = float_model.transformer_encoder
        self.d_model = float_model.d_model

    def forward(self, src):
        # O forward pass agora usa os módulos corretos (quantizados e float)
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.classifier(output)
        return output
