import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import BertModel, BertConfig


class PositionalEncoding(nn.Module):
    """
    From the classic: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = 20 * torch.arange(max_len).unsqueeze(1)
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


def positionalencoding1d(d_model, length):
    """
    This is from: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py

    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        nlayers=6,
        nhead=8,
        d_model=128,
        d_hid=128,
        dropout=0.1,
        use_xavier=True,
    ):
        super().__init__()

        self.pe = positionalencoding1d(d_model, 25).to("cuda")

        self.embed = nn.Sequential(
            nn.Embedding(n_tokens + 2, d_model),
        )

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layers, nlayers)

        self.to_logits = nn.Sequential(
            nn.Linear(d_model, 2),
        )

        self.device = torch.device("cuda")

        if use_xavier:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, data):
        input_data = data.input_ids.cuda()
        mask = data.attention_mask.cuda()
        embedded = self.embed(input_data) + self.pe[: input_data.shape[1]]
        encoded = self.encoder(embedded)  # , src_key_padding_mask=~mask.bool())

        out = encoded[:, 0]
        return self.to_logits(out)


class BERT(nn.Module):
    """
    ever heard of it?
    """

    def __init__(self):
        super().__init__()

        hidden_size = 768
        config = BertConfig(vocab_size=4, num_hidden_layers=2, hidden_size=hidden_size)
        self.model = BertModel(config).cuda()
        self.head = nn.Linear(hidden_size, 2)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        x = x.to(torch.device("cuda"))
        out = self.model(**x)
        return self.head(out.pooler_output)


class LSTM(nn.Module):
    def __init__(
        self,
        n_tokens=4,
        nlayers=6,
        d_model=128,
        d_hid=128,
    ):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_model).cuda()
        self.lstm = nn.LSTM(
            input_size=d_model,
            num_layers=nlayers,
            hidden_size=d_hid,
            bidirectional=False,
            batch_first=True,
        ).cuda()
        self.head = nn.Linear(d_model, 2).cuda()
        self.device = torch.device("cuda")

    def to(self, device):
        self.device = device
        return super().to(device)

    def __call__(self, data):
        data = data.input_ids.to(self.device)
        lengths = torch.sum(data != -1, -1)
        mask = data != -1

        embedded_data = self.embed(mask * data)

        packed_embedded_data = pack_padded_sequence(
            embedded_data, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_embedded_data)
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out_final = padded_out[:, -1]
        rep = self.head(out_final)
        return rep
