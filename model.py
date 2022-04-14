import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    From the classic: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        new_x = x + self.pe[:, : x.size(1)]
        return new_x


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        nlayers=6,
        nhead=4,
        d_model=128,
        d_hid=128,
        dropout=0.1,
    ):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Embedding(n_tokens + 2, d_model),
            PositionalEncoding(d_model),
        )

        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        self.to_logits = nn.Linear(d_model, n_tokens)

        self.device = torch.device("cuda")

    def to(self, device):
        self.device = device
        return super().to(device)

    def __call__(self, data):
        input_data = data.input_ids.cuda()
        mask = data.attention_mask.cuda()

        embedded = self.embed(mask * input_data)
        encoded = self.encoder(embedded, src_key_padding_mask=~mask.bool())

        out = encoded[:, 0]
        logits = self.to_logits(out)
        return logits
