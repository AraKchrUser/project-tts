import math

from so_vits.modules.attentions import Encoder

import torch
from torch import nn
from torch.nn import Module


class TextEncoder(Module):

    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, 
                 n_heads, n_layers, kernel_size, p_dropout):
        
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.encoder = Encoder(
            hidden_channels, filter_channels, 
            n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        
        x      = self.emb(x) * math.sqrt(self.hidden_channels)
        x      = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(self._sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x      = self.encoder(x * x_mask, x_mask)
        
        stats   = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask

    def _sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)