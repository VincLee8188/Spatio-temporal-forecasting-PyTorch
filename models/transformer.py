import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GConv(nn.Module):
    # Spectral-based graph convolution function.
    # x: tensor, [batch_size, c_in, time_step, n_route].
    # theta: tensor, [ks*c_in, c_out], trainable kernel parameters.
    # ks: int, kernel size of graph convolution.
    # c_in: int, size of input channel.
    # c_out: int, size of output channel.
    # return: tensor, [batch_size, c_out, time_step, n_route].

    def __init__(self, ks, c_in, c_out, graph_kernel):
        super(GConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.graph_kernel = graph_kernel
        self.theta = nn.Linear(ks*c_in, c_out)

    def forward(self, x):
        # graph kernel: tensor, [n_route, ks*n_route]
        kernel = self.graph_kernel
        # time_step, n_route
        _, _, t, n = x.shape
        # x:[batch_size, c_in, time_step, n_route] -> [batch_size, time_step, c_in, n_route]
        x_tmp = x.transpose(1, 2).contiguous()
        # x_ker = x_tmp * ker -> [batch_size, time_step, c_in, ks*n_route]
        x_ker = torch.matmul(x_tmp, kernel)
        # -> [batch_size, time_step, c_in*ks, n_route] -> [batch_size, time_step, n_route, c_in*ks]
        x_ker = x_ker.reshape(-1, t, self.c_in * self.ks, n).transpose(2, 3)
        # -> [batch_size, time_step, n_route, c_out]
        x_fig = self.theta(x_ker)
        # -> [batch_size, c_out, time_step, n_route]
        return x_fig.permute(0, 3, 1, 2).contiguous()


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)
        self.ninp = ninp

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tar, tgt_mask=None):
        if tgt_mask is None:
            device = tar.device
            tgt_mask = self._generate_square_subsequent_mask(tar.shape[0]).to(device)

        src = self.pos_encoder(src)
        tar = self.pos_encoder(tar)
        memory = self.transformer_encoder(src)
        output = self.decoder(tar, memory, tgt_mask)
        return output


class ShiftTransformer(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(ShiftTransformer, self).__init__()
        self.fc_1 = nn.Linear(1, ninp)
        self.fc_2 = nn.Linear(ninp, 1)
        self.transformer = TransformerModel(ninp, nhead, nhid, nlayers, dropout)

    def forward(self, src, tar, tgt_mask=None):
        _, b, nodes = src.shape
        src = src.reshape(-1, b*nodes, 1)
        tar = tar.reshape(-1, b*nodes, 1)
        src = self.fc_1(src)
        tar = self.fc_1(tar)
        output = self.transformer(src, tar)
        output = self.fc_2(output)
        return output.reshape(-1, b, nodes)


class GCNShiftTransformer(nn.Module):
    def __init__(self, ks, graph_kernel, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(GCNShiftTransformer, self).__init__()
        self.fc_1 = nn.Linear(ninp, 1)
        self.gcn_1 = GConv(ks, 1, ninp, graph_kernel)
        self.transformer = TransformerModel(ninp, nhead, nhid, nlayers, dropout)
        self.ninp = ninp

    def forward(self, src, tar):
        _, b, nodes = src.shape
        src = src.view(-1, b, nodes, 1).permute(1, 3, 0, 2)
        tar = tar.view(-1, b, nodes, 1).permute(1, 3, 0, 2)
        src = self.gcn_1(src)
        tar = self.gcn_1(tar)
        src = src.permute(2, 0, 3, 1).reshape(-1, b * nodes, self.ninp)
        tar = tar.permute(2, 0, 3, 1).reshape(-1, b * nodes, self.ninp)
        output = self.transformer(src, tar)
        output = self.fc_1(output)
        return output.reshape(-1, b, nodes)
